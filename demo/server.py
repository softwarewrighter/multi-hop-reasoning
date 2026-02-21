"""
Demo server for KG reward visualization.

Serves the web UI and provides API endpoints for run data.
Includes live inference endpoint for interactive demo.
"""

import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
WEB_DIR = Path(__file__).parent / "web"

# Lazy-loaded model state
_model_state = {
    "model": None,
    "tokenizer": None,
    "kg": None,
    "entity_vocab": None,
    "loaded_adapter": None
}


def get_model():
    """Lazy-load the inference model."""
    if _model_state["model"] is None:
        print("Loading inference model (first request)...")

        # Try to find the best available adapter (prefer distribution-matched RSFT)
        adapter_paths = [
            DATA_DIR / "runs" / "run_360m" / "models" / "rsft_eval",  # 360M RSFT on hard (67%)
            DATA_DIR / "runs" / "run_360m" / "models" / "rsft",       # 360M RSFT on easy (27%)
            DATA_DIR / "runs" / "run_0001" / "models" / "rsft",       # 135M RSFT
            DATA_DIR / "runs" / "run_360m" / "models" / "sft",        # 360M SFT (37%)
            DATA_DIR / "runs" / "run_0001" / "models" / "sft",        # 135M SFT
        ]

        adapter_path = None
        model_id = "HuggingFaceTB/SmolLM-135M-Instruct"  # Default

        for path in adapter_paths:
            if path.exists() and (path / "adapters.safetensors").exists():
                adapter_path = path
                # Determine model based on path
                if "360m" in str(path):
                    model_id = "HuggingFaceTB/SmolLM-360M-Instruct"
                break

        if adapter_path:
            print(f"  Using adapter: {adapter_path}")
            print(f"  Base model: {model_id}")
        else:
            print(f"  No adapter found, using base model: {model_id}")

        try:
            from core.infer import load_mlx_model
            _model_state["model"], _model_state["tokenizer"] = load_mlx_model(
                model_id, adapter_path
            )
            _model_state["loaded_adapter"] = str(adapter_path) if adapter_path else None
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    # Also load KG if needed
    if _model_state["kg"] is None:
        from core.kg import load_kg, get_entity_vocab
        kg_path = DATA_DIR / "kg.json"
        if kg_path.exists():
            _model_state["kg"] = load_kg(kg_path)
            _model_state["entity_vocab"] = get_entity_vocab(_model_state["kg"])

    return _model_state


class DemoHandler(SimpleHTTPRequestHandler):
    """Handler for demo server with API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        # API endpoints
        if parsed.path == "/api/runs":
            self.send_json(self.list_runs())
        elif parsed.path.startswith("/api/run/"):
            run_id = parsed.path.split("/")[-1]
            self.send_json(self.get_run(run_id))
        elif parsed.path == "/api/kg":
            self.send_json(self.get_kg())
        elif parsed.path.startswith("/api/episodes/"):
            run_id = parsed.path.split("/")[-1]
            self.send_json(self.get_episodes(run_id))
        elif parsed.path == "/api/comparison":
            self.send_json(self.get_comparison())
        elif parsed.path == "/api/model-status":
            self.send_json(self.get_model_status())
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests for inference."""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/infer":
            self.handle_infer()
        else:
            self.send_error(404, "Not Found")

    def handle_infer(self):
        """Run live inference on a question."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            question = data.get("question", "")
            if not question:
                self.send_json({"error": "Missing 'question' field"})
                return

            # Format as MCQ prompt
            prompt = self.format_question_prompt(question)

            # Get model (lazy load)
            try:
                state = get_model()
            except Exception as e:
                self.send_json({
                    "error": f"Model not available: {str(e)}",
                    "hint": "Run 'make train-360m' or 'make sft' first to train a model"
                })
                return

            # Run inference
            from core.infer import run_inference
            from core.reward import compute_reward, parse_completion

            completion = run_inference(
                state["model"],
                state["tokenizer"],
                prompt,
                max_tokens=256,
                temperature=0.0
            )

            # Parse and score
            valid, trace, answer = parse_completion(completion)
            parsed = {"valid": valid, "trace": trace, "answer": answer}

            # Get path entities from question if available
            path_entities = data.get("path_entities", [])
            if not path_entities and state["entity_vocab"]:
                # Try to extract entities mentioned in question
                path_entities = [
                    e for e in state["entity_vocab"]
                    if e.lower() in question.lower()
                ]

            reward = compute_reward(
                completion=completion,
                answer_star="A",  # Dummy, we don't know correct answer for free-form
                path_entities=path_entities,
                entity_vocab=state["entity_vocab"] or set()
            )

            self.send_json({
                "completion": completion,
                "parsed": parsed,
                "reward": {
                    "path_coverage": reward["path_coverage"],
                    "total": reward["total"]
                },
                "model_info": {
                    "adapter": state["loaded_adapter"]
                }
            })

        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON in request body"})
        except Exception as e:
            self.send_json({"error": str(e)})

    def format_question_prompt(self, question: str) -> str:
        """Format a free-form question as an MCQ prompt."""
        # For free-form questions, we'll generate a simple prompt
        # The model was trained on MCQ format, but can still reason
        return f"""You are a DevOps expert. Answer the following question by reasoning through the causal chain.

Question: {question}

Think step by step about what causes what, then provide your answer.

Format your response as:
TRACE: [your reasoning showing the causal chain]
ANSWER: [your final answer]"""

    def get_comparison(self):
        """Get pre-recorded comparison data for static demo."""
        comparison_path = DATA_DIR / "distribution_comparison.json"
        if not comparison_path.exists():
            return {"error": "Comparison data not found. Run 'make generate-comparison' first."}

        with open(comparison_path) as f:
            return json.load(f)

    def get_model_status(self):
        """Get current model loading status."""
        return {
            "loaded": _model_state["model"] is not None,
            "adapter": _model_state["loaded_adapter"],
            "kg_loaded": _model_state["kg"] is not None
        }

    def send_json(self, data):
        """Send JSON response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def list_runs(self):
        """List available runs."""
        runs_dir = DATA_DIR / "runs"
        if not runs_dir.exists():
            return {"runs": []}

        runs = []
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                meta_path = run_dir / "meta.json"
                metrics_path = run_dir / "metrics.json"

                run_info = {"id": run_dir.name}

                if meta_path.exists():
                    with open(meta_path) as f:
                        run_info["meta"] = json.load(f)

                if metrics_path.exists():
                    with open(metrics_path) as f:
                        run_info["metrics"] = json.load(f)

                runs.append(run_info)

        return {"runs": runs}

    def get_run(self, run_id: str):
        """Get full run data."""
        run_dir = DATA_DIR / "runs" / run_id

        if not run_dir.exists():
            return {"error": f"Run {run_id} not found"}

        result = {"id": run_id}

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["meta"] = json.load(f)

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                result["metrics"] = json.load(f)

        return result

    def get_kg(self):
        """Get knowledge graph."""
        kg_path = DATA_DIR / "kg.json"
        if not kg_path.exists():
            return {"error": "kg.json not found"}

        with open(kg_path) as f:
            return json.load(f)

    def get_episodes(self, run_id: str):
        """Get episodes for a run."""
        episodes_path = DATA_DIR / "runs" / run_id / "episodes.jsonl"

        if not episodes_path.exists():
            return {"error": f"Episodes not found for {run_id}"}

        episodes = []
        with open(episodes_path) as f:
            for line in f:
                episodes.append(json.loads(line))

        return {"episodes": episodes}


def main():
    """Run the demo server."""
    # Port 3519 chosen to avoid conflicts with common services on 8080
    port = 3519
    server = HTTPServer(("localhost", port), DemoHandler)

    print(f"Demo server running at http://localhost:{port}")
    print(f"Serving web files from: {WEB_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print("")
    print("Features:")
    print("  - Static demo (Training/Inference tabs)")
    print("  - Live inference (Try It tab) - model loads on first request")
    print("  - Distribution comparison (Distribution tab)")
    print("\nPress Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

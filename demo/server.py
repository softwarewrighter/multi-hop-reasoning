"""
Demo server for KG reward visualization.

Serves the web UI and provides API endpoints for run data.
"""

import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse


DATA_DIR = Path(__file__).parent.parent / "data"
WEB_DIR = Path(__file__).parent / "web"


class DemoHandler(SimpleHTTPRequestHandler):
    """Handler for demo server with API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

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
        else:
            # Serve static files
            super().do_GET()

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
    port = 8080
    server = HTTPServer(("localhost", port), DemoHandler)

    print(f"Demo server running at http://localhost:{port}")
    print(f"Serving web files from: {WEB_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print("\nPress Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

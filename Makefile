# KG Reward Demo - Makefile
# Multi-hop reasoning with MLX-first training
# See spec/agent1-core.md for full task descriptions

.PHONY: help setup data infer_base sft rsft eval serve clean clean-all test train-360m generate-comparison \
        sft-unsloth rsft-unsloth train-unsloth train-360m-unsloth setup-unsloth

# Configuration
PYTHON ?= python3
MODEL ?= HuggingFaceTB/SmolLM-135M-Instruct
RUN_ID ?= run_0001
RUN_DIR = data/runs/$(RUN_ID)
MAX_EXAMPLES ?= 50

help:
	@echo "KG Reward Demo - Multi-hop Reasoning with MLX"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  setup      - Create Python environment and install dependencies"
	@echo "  data       - Generate KG and training/eval datasets"
	@echo "  infer_base - Run baseline inference and log episodes"
	@echo "  sft        - Train SFT LoRA adapters"
	@echo "  rsft       - Run rejection sampling fine-tuning"
	@echo "  eval       - Generate metrics from episodes"
	@echo "  serve      - Start demo web server (with live inference)"
	@echo "  train-360m - Train SmolLM-360M model (MLX/macOS)"
	@echo "  generate-comparison - Generate static comparison data"
	@echo ""
	@echo "Linux/CUDA (Unsloth):"
	@echo "  setup-unsloth      - Setup Unsloth environment"
	@echo "  train-unsloth      - Full training with Unsloth"
	@echo "  train-360m-unsloth - Train 360M with Unsloth"
	@echo ""
	@echo "Other:"
	@echo "  test       - Run tests"
	@echo "  clean      - Remove generated model files"
	@echo "  clean-all  - Remove all generated data"
	@echo ""
	@echo "Quick start (full pipeline):"
	@echo "  make setup && make data && make infer_base && make sft && make rsft && make eval"
	@echo ""
	@echo "Live demo setup:"
	@echo "  make train-360m && make serve"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL=$(MODEL)"
	@echo "  RUN_ID=$(RUN_ID)"
	@echo "  MAX_EXAMPLES=$(MAX_EXAMPLES)"

# Environment setup (using uv for fast, reliable installs)
setup:
	@echo "Setting up Python environment with uv..."
	uv venv .venv
	@echo "Installing dependencies..."
	. .venv/bin/activate && uv pip install -e ".[dev]"
	@echo ""
	@echo "Environment ready! Activate with:"
	@echo "  source .venv/bin/activate"

# Install MLX dependencies (macOS only)
setup-mlx: setup
	@echo "Installing MLX dependencies..."
	. .venv/bin/activate && uv pip install mlx mlx-lm
	@echo "MLX installed successfully!"

# Data generation
data:
	@echo "Generating KG and datasets..."
	@mkdir -p data
	$(PYTHON) -m core.dataset generate \
		--kg data/kg.json \
		--train data/train.jsonl \
		--eval data/eval.jsonl \
		--n-train 500 \
		--n-eval 100 \
		--seed 42
	@echo ""
	@echo "Generated:"
	@echo "  - data/kg.json (Knowledge Graph)"
	@echo "  - data/train.jsonl (1-3 hop training examples)"
	@echo "  - data/eval.jsonl (4-5 hop evaluation examples)"

# Baseline inference
infer_base:
	@echo "Running baseline inference..."
	@mkdir -p $(RUN_DIR)
	$(PYTHON) -m core.infer \
		--examples data/eval.jsonl \
		--kg data/kg.json \
		--output $(RUN_DIR)/episodes.jsonl \
		--model $(MODEL) \
		--phase base \
		--step 0 \
		--max-examples $(MAX_EXAMPLES)
	@echo ""
	@echo "Baseline episodes logged to: $(RUN_DIR)/episodes.jsonl"

# SFT training
sft:
	@echo "Training SFT LoRA adapters..."
	@mkdir -p $(RUN_DIR)/models/sft
	$(PYTHON) -m core.mlx_sft \
		--train data/train.jsonl \
		--output $(RUN_DIR)/models/sft \
		--model $(MODEL) \
		--iters 500 \
		--batch-size 4 \
		--lr 1e-4 \
		--lora-rank 16
	@echo ""
	@echo "SFT adapters saved to: $(RUN_DIR)/models/sft"

# SFT evaluation
infer_sft: sft
	@echo "Running SFT inference..."
	$(PYTHON) -m core.infer \
		--examples data/eval.jsonl \
		--kg data/kg.json \
		--output $(RUN_DIR)/episodes.jsonl \
		--model $(MODEL) \
		--adapter $(RUN_DIR)/models/sft \
		--phase sft \
		--step 500 \
		--max-examples $(MAX_EXAMPLES)
	@echo ""
	@echo "SFT episodes logged to: $(RUN_DIR)/episodes.jsonl"

# RSFT training
rsft:
	@echo "Running rejection sampling fine-tuning..."
	@mkdir -p $(RUN_DIR)/models/rsft
	$(PYTHON) -m core.rsft \
		--examples data/train.jsonl \
		--kg data/kg.json \
		--sft-adapter $(RUN_DIR)/models/sft \
		--output $(RUN_DIR)/models/rsft \
		--model $(MODEL) \
		--k-samples 8 \
		--keep-top 1 \
		--temperature 0.7 \
		--iters 300 \
		--max-examples $(MAX_EXAMPLES)
	@echo ""
	@echo "RSFT adapters saved to: $(RUN_DIR)/models/rsft"

# RSFT evaluation
infer_rsft: rsft
	@echo "Running RSFT inference..."
	$(PYTHON) -m core.infer \
		--examples data/eval.jsonl \
		--kg data/kg.json \
		--output $(RUN_DIR)/episodes.jsonl \
		--model $(MODEL) \
		--adapter $(RUN_DIR)/models/rsft \
		--phase rsft \
		--step 1000 \
		--max-examples $(MAX_EXAMPLES)
	@echo ""
	@echo "RSFT episodes logged to: $(RUN_DIR)/episodes.jsonl"

# Evaluation
eval:
	@echo "Generating metrics..."
	$(PYTHON) -m core.eval \
		--run-dir $(RUN_DIR) \
		--run-id $(RUN_ID)
	@echo ""
	@echo "Metrics saved to: $(RUN_DIR)/metrics.json"

# Full training pipeline
train: infer_base infer_sft infer_rsft eval
	@echo ""
	@echo "Full training pipeline complete!"
	@echo "Results in: $(RUN_DIR)/"

# Train 360M model (larger model for demo)
train-360m:
	@echo "Training SmolLM-360M model..."
	$(MAKE) sft MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m
	$(MAKE) infer_sft MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m
	$(MAKE) rsft MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m
	$(MAKE) infer_rsft MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m
	$(MAKE) eval RUN_ID=run_360m
	@echo ""
	@echo "360M model training complete!"
	@echo "Models in: data/runs/run_360m/"

# ========== UNSLOTH TRAINING (Linux/CUDA) ==========

# SFT with Unsloth
sft-unsloth:
	@echo "Training SFT with Unsloth..."
	@mkdir -p $(RUN_DIR)/models/sft
	$(PYTHON) -m core.unsloth_sft \
		--train data/train.jsonl \
		--output $(RUN_DIR)/models/sft \
		--model $(MODEL) \
		--epochs 3 \
		--batch-size 4 \
		--lr 2e-4 \
		--lora-rank 16
	@echo ""
	@echo "SFT adapters saved to: $(RUN_DIR)/models/sft"

# RSFT with Unsloth (use eval.jsonl for distribution matching!)
rsft-unsloth:
	@echo "Running RSFT with Unsloth..."
	@mkdir -p $(RUN_DIR)/models/rsft
	$(PYTHON) -m core.unsloth_rsft \
		--examples data/eval.jsonl \
		--kg data/kg.json \
		--sft-adapter $(RUN_DIR)/models/sft \
		--output $(RUN_DIR)/models/rsft \
		--model $(MODEL) \
		--k-samples 8 \
		--keep-top 1 \
		--temperature 0.7 \
		--epochs 2
	@echo ""
	@echo "RSFT adapters saved to: $(RUN_DIR)/models/rsft"

# Full Unsloth training pipeline
train-unsloth:
	@echo "Training with Unsloth (Linux/CUDA)..."
	$(MAKE) sft-unsloth RUN_ID=run_unsloth
	$(MAKE) rsft-unsloth RUN_ID=run_unsloth
	$(MAKE) eval RUN_ID=run_unsloth
	@echo ""
	@echo "Unsloth training complete!"
	@echo "Models in: data/runs/run_unsloth/"

# Train 360M with Unsloth
train-360m-unsloth:
	@echo "Training SmolLM-360M with Unsloth..."
	$(MAKE) sft-unsloth MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m_unsloth
	$(MAKE) rsft-unsloth MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m_unsloth
	$(MAKE) eval RUN_ID=run_360m_unsloth
	@echo ""
	@echo "360M Unsloth training complete!"
	@echo "Models in: data/runs/run_360m_unsloth/"

# Setup for Linux/CUDA
setup-unsloth:
	@echo "Setting up Unsloth environment..."
	uv venv .venv
	. .venv/bin/activate && uv pip install -e ".[dev]"
	. .venv/bin/activate && uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	. .venv/bin/activate && uv pip install --no-deps trl peft accelerate bitsandbytes
	@echo ""
	@echo "Unsloth environment ready! Activate with:"
	@echo "  source .venv/bin/activate"

# ========== END UNSLOTH ==========

# Generate comparison data for static demo
generate-comparison:
	@echo "Generating distribution comparison data..."
	$(PYTHON) -m core.generate_comparison
	@echo "Comparison data saved to: data/distribution_comparison.json"

# Demo server
serve:
	@echo "Starting demo server..."
	@echo "Open http://localhost:8000 in your browser"
	$(PYTHON) demo/server.py

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v

# Test reward function
test-reward:
	@echo "Testing reward function..."
	$(PYTHON) -c "from core.reward import compute_reward; \
		result = compute_reward('TRACE: TLSHandshakeError is caused by ClockSkew\nANSWER: B', \
			'B', ['TLSHandshakeError', 'ClockSkew', 'NTPsync'], \
			{'TLSHandshakeError', 'ClockSkew', 'NTPsync', 'CertificateExpired'}); \
		print('Reward:', result)"

# Validate KG
validate-kg:
	@echo "Validating knowledge graph..."
	$(PYTHON) -c "from core.kg import load_kg, sample_path; \
		kg = load_kg('data/kg.json'); \
		print(f'Entities: {len(kg[\"entities\"])}'); \
		print(f'Edges: {len(kg[\"edges\"])}'); \
		print(f'Relations: {len(kg[\"relations\"])}'); \
		path = sample_path(kg, 3, seed=42); \
		print(f'Sample 3-hop path: {path[\"entities\"]}')"

# Cleanup
clean:
	rm -rf $(RUN_DIR)/models
	rm -f $(RUN_DIR)/episodes.jsonl
	rm -f $(RUN_DIR)/metrics.json
	@echo "Cleaned generated files (kept datasets)"

clean-all:
	rm -rf data/runs
	rm -f data/*.jsonl
	@echo "Cleaned all generated data (kept kg.json)"

# Create run directory structure
init-run:
	@mkdir -p $(RUN_DIR)/models/base
	@mkdir -p $(RUN_DIR)/models/sft
	@mkdir -p $(RUN_DIR)/models/rsft
	@echo '{"run_id": "$(RUN_ID)", "model": "$(MODEL)"}' > $(RUN_DIR)/meta.json
	@echo "Initialized run directory: $(RUN_DIR)"

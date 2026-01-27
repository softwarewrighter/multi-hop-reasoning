# KG Reward Demo - Makefile
# Multi-hop reasoning with MLX-first training
# See spec/agent1-core.md for full task descriptions

.PHONY: help setup data infer_base sft rsft eval serve clean clean-all test

# Configuration
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
	@echo "  serve      - Start demo web server"
	@echo "  test       - Run tests"
	@echo "  clean      - Remove generated model files"
	@echo "  clean-all  - Remove all generated data"
	@echo ""
	@echo "Quick start (full pipeline):"
	@echo "  make setup && make data && make infer_base && make sft && make rsft && make eval"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL=$(MODEL)"
	@echo "  RUN_ID=$(RUN_ID)"
	@echo "  MAX_EXAMPLES=$(MAX_EXAMPLES)"

# Environment setup
setup:
	@echo "Setting up Python environment..."
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo ""
	@echo "Environment ready! Activate with:"
	@echo "  source .venv/bin/activate"

# Install MLX dependencies (macOS only)
setup-mlx: setup
	@echo "Installing MLX dependencies..."
	.venv/bin/pip install mlx mlx-lm
	@echo "MLX installed successfully!"

# Data generation
data:
	@echo "Generating KG and datasets..."
	@mkdir -p data
	python -m core.dataset generate \
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
	python -m core.infer \
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
	python -m core.mlx_sft \
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
	python -m core.infer \
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
	python -m core.rsft \
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
	python -m core.infer \
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
	python -m core.eval \
		--run-dir $(RUN_DIR) \
		--run-id $(RUN_ID)
	@echo ""
	@echo "Metrics saved to: $(RUN_DIR)/metrics.json"

# Full training pipeline
train: infer_base infer_sft infer_rsft eval
	@echo ""
	@echo "Full training pipeline complete!"
	@echo "Results in: $(RUN_DIR)/"

# Demo server
serve:
	@echo "Starting demo server..."
	@echo "Open http://localhost:8000 in your browser"
	python demo/server.py

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Test reward function
test-reward:
	@echo "Testing reward function..."
	python -c "from core.reward import compute_reward; \
		result = compute_reward('TRACE: TLSHandshakeError is caused by ClockSkew\nANSWER: B', \
			'B', ['TLSHandshakeError', 'ClockSkew', 'NTPsync'], \
			{'TLSHandshakeError', 'ClockSkew', 'NTPsync', 'CertificateExpired'}); \
		print('Reward:', result)"

# Validate KG
validate-kg:
	@echo "Validating knowledge graph..."
	python -c "from core.kg import load_kg, sample_path; \
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

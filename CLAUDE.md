# Claude Code Instructions

## Environment Setup

This project uses `uv` for Python package management. **Never use `pip` or `pip3` directly.**

---

## macOS (Apple Silicon) - MLX

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
uv pip install mlx mlx-lm

# Or use Make
make setup-mlx
source .venv/bin/activate
```

### Training (MLX)
```bash
make train-360m  # Train SmolLM-360M
make serve       # Run demo server
```

---

## Linux (NVIDIA CUDA) - Unsloth

### Setup
```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Install Unsloth (CUDA)
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
uv pip install --no-deps trl peft accelerate bitsandbytes

# Or use Make
make setup-unsloth
source .venv/bin/activate
```

### Training (Unsloth)
```bash
# Full pipeline (SFT → RSFT → eval)
make train-360m-unsloth

# Or individual steps
make sft-unsloth MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=my_run
make rsft-unsloth MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=my_run
make eval RUN_ID=my_run
```

### GPU: RTX 5060 16GB
- Batch size 4 should work well
- 4-bit quantization enabled by default
- Gradient checkpointing enabled

---

## Running the Demo

```bash
source .venv/bin/activate
python3 demo/server.py
# Open http://localhost:3519
```

---

## Key Files

### Core ML Pipeline
- `core/infer.py` - Model inference (auto-detects MLX vs PyTorch)
- `core/mlx_sft.py` - SFT training with MLX (macOS)
- `core/unsloth_sft.py` - SFT training with Unsloth (Linux/CUDA)
- `core/rsft.py` - RSFT with MLX
- `core/unsloth_rsft.py` - RSFT with Unsloth
- `core/reward.py` - Reward function (shared)
- `core/kg.py` - Knowledge graph utilities (shared)

### Demo
- `demo/server.py` - Web server with live inference
- `demo/web/` - Frontend (HTML/JS/CSS)

### Data
- `data/kg.json` - Knowledge graph
- `data/train.jsonl` - Training examples (1-3 hops)
- `data/eval.jsonl` - Eval examples (4-5 hops) - **USE THIS FOR RSFT!**
- `data/runs/` - Training run outputs

---

## Critical: Distribution Matching

The key finding of this project:
- **RSFT on easy examples (train.jsonl)**: 20% accuracy (WORSE than SFT!)
- **RSFT on hard examples (eval.jsonl)**: 75% accuracy

Always use `eval.jsonl` for RSFT to match the evaluation distribution.

---

## Current Status (Feb 2026)

### What's Working
- ✅ SmolLM-360M trained with SFT (37% accuracy)
- ✅ RSFT trained on eval distribution (**67% accuracy**)
- ✅ Live inference demo with Try It tab (using best model)
- ✅ Distribution visualization tab
- ✅ All four demo tabs functional
- ✅ Distribution matching demonstrated and documented

### Results Summary
| Phase | Accuracy | Notes |
|-------|----------|-------|
| SFT | 37% | Format learning |
| RSFT (easy) | 27% | Distribution mismatch |
| **RSFT (hard)** | **67%** | ✅ Distribution matched |

---

## Next Steps

### ✅ Completed: Distribution Fix
RSFT retrained on eval.jsonl → **67% accuracy** (up from 27%)

### Priority 1: Linux/Unsloth Training

If continuing on Linux with NVIDIA GPU:

```bash
# 1. Setup
make setup-unsloth
source .venv/bin/activate

# 2. Generate data if needed
make data

# 3. Train with correct distribution
# Edit core/unsloth_rsft.py or Makefile to use eval.jsonl
make train-360m-unsloth

# 4. Verify
cat data/runs/run_360m_unsloth/metrics.json

# 5. Run demo
python3 demo/server.py
```

### Priority 3: Demo Improvements

- [ ] Update demo to load rsft_eval adapter (once trained)
- [ ] Generate static comparison data: `make generate-comparison`
- [ ] Add loading indicator during model init
- [ ] Consider adding side-by-side SFT vs RSFT comparison

---

## Quick Reference

### Training Commands
```bash
# macOS
make train-360m

# Linux
make train-360m-unsloth
```

### Demo
```bash
source .venv/bin/activate
python3 demo/server.py
# Open http://localhost:3519
```

### Key Insight
**Always use eval.jsonl for RSFT** - training distribution must match eval distribution!

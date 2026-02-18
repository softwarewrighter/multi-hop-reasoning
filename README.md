# Multi-Hop Reasoning Demo

MLX-first pipeline for training language models on multi-hop reasoning tasks with knowledge graph grounding.

## Paper

Based on: [Knowledge Graph-Guided Retrieval Augmented Generation](https://arxiv.org/html/2601.15160v1)

## Overview

This project implements a complete pipeline for training small language models to perform multi-hop reasoning over knowledge graphs:

- **Knowledge Graph**: Software troubleshooting domain with ~200 entities and ~600 edges
- **Dataset Generation**: MCQ tasks with 1-3 hop paths (train) and 4-5 hop paths (eval)
- **Reward Function**: Correctness + path coverage scoring with anti-spam penalties
- **SFT**: Supervised Fine-Tuning with LoRA in MLX
- **RSFT**: Rejection Sampling Fine-Tuning (RL-lite approach)
- **Demo UI**: Interactive visualization with live inference and distribution analysis
- **Cross-Platform**: MLX for macOS, Unsloth for Linux/CUDA

## Key Finding: Distribution Matters

Training on examples that don't match your evaluation distribution can make performance **worse**:

| Approach | Training Data | Eval Accuracy |
|----------|--------------|---------------|
| SFT | Easy (1-3 hop) | 30% |
| RSFT Easy | Easy (1-3 hop) | 20% ↓ |
| RSFT Hard | Hard (4-5 hop) | **75%** ★ |

RSFT on easy examples performed *worse* than SFT baseline because the model learned to match its training distribution, failing to generalize to harder eval questions.

## Results

### SmolLM-360M Training Results

| Phase | Accuracy | Path Coverage | Notes |
|-------|----------|---------------|-------|
| Base | 0% | 0% | No format compliance |
| SFT (500 iters) | 37% | 32% | Learns TRACE + ANSWER format |
| RSFT (train distribution) | 27% | 30% | Distribution mismatch hurts! |

The RSFT accuracy dropped because it trained on easy examples (1-3 hop) but was evaluated on hard examples (4-5 hop). This demonstrates the key finding about distribution matching.

See [documentation/training-status.md](documentation/training-status.md) for details on model storage and continuing training.

## Demo

**Live Demo:** [https://softwarewrighter.github.io/multi-hop-reasoning/](https://softwarewrighter.github.io/multi-hop-reasoning/)

Or run locally with live inference:
```bash
source .venv/bin/activate
make train-360m  # Train the 360M model (first time only)
python3 demo/server.py
# Open http://localhost:3519
```

### Demo Tabs

| Tab | Description |
|-----|-------------|
| **Training** | Visualizes SFT→RSFT training with knowledge graph scoring |
| **Inference** | Shows model reasoning without graph access |
| **Try It** | Live inference - ask questions and see reasoning traces |
| **Distribution** | Interactive visualization of the key finding |

**Training View:** Knowledge graph scores model outputs in real-time
![Training Demo](images/screenshot-training.png?ts=1769921411000)

**Inference View:** Model reasons without graph access
![Inference Demo](images/screenshot-testing.png?ts=1769921411000)

### Try It (Live Inference)

Ask DevOps troubleshooting questions and watch the model reason:

```
Question: What causes TLSHandshakeError?

TRACE: TLSHandshakeError is caused by ConnectionTimeout,
and ConnectionTimeout leads to ConnectionRefused,
and ConnectionRefused is diagnosed by VerifyCert...
```

## Quick Start

```bash
# Setup environment (macOS with Apple Silicon)
# Requires: uv (https://docs.astral.sh/uv/)
make setup-mlx
source .venv/bin/activate

# Generate data
make data

# Run full training pipeline
make train
```

### Manual Setup (without Make)

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
uv pip install mlx mlx-lm  # macOS only
```

Or run steps individually:

```bash
make infer_base  # Baseline evaluation
make sft         # Supervised fine-tuning
make rsft        # Rejection sampling fine-tuning
make eval        # Generate metrics
```

## Project Structure

```
multi-hop-reasoning/
├── core/                  # Core library
│   ├── kg.py              # Knowledge graph loading & path sampling
│   ├── dataset.py         # MCQ generation
│   ├── reward.py          # Reward computation
│   ├── infer.py           # Model inference (MLX + PyTorch)
│   ├── mlx_sft.py         # LoRA training with MLX (macOS)
│   ├── rsft.py            # Rejection sampling fine-tuning (MLX)
│   ├── unsloth_sft.py     # SFT with Unsloth (Linux/CUDA)
│   ├── unsloth_rsft.py    # RSFT with Unsloth (Linux/CUDA)
│   └── eval.py            # Metrics generation
├── data/                  # Data files
│   ├── kg.json            # Knowledge graph
│   ├── train.jsonl        # Training examples (1-3 hops)
│   ├── eval.jsonl         # Evaluation examples (4-5 hops)
│   └── runs/              # Training run outputs
│       └── run_360m/      # Trained SmolLM-360M model
├── demo/                  # Demo web application
│   ├── server.py          # Server with live inference API
│   └── web/               # Frontend (Training, Inference, Try It, Distribution)
├── spec/                  # Specifications
├── documentation/         # Documentation
├── CLAUDE.md              # AI agent instructions for continuing work
└── tests/                 # Test suite
```

## Models

| Model | Parameters | Command |
|-------|------------|---------|
| SmolLM-135M | 135M | `make train` (default) |
| SmolLM-360M | 360M | `make train-360m` (recommended for demo) |

The 360M model produces better reasoning traces and is used by the live inference demo.

## Output Format

Models must output in this strict format:

```
TRACE: <one or two sentences explaining reasoning>
ANSWER: A|B|C|D
```

## Reward Function

- **R_corr**: +1.0 correct, -2.0 incorrect
- **R_path**: Entity coverage of path entities in TRACE (min 2 hits required)
- **P_spam**: -0.5 penalty if any entity repeated >2 times
- **Total**: `W_corr * R_corr + W_path * R_path - P_spam`

See [spec/reward.md](spec/reward.md) for full specification.

## Documentation

- [documentation/eli5.md](documentation/eli5.md) - ELI5 explanation of the project and key findings
- [documentation/training-status.md](documentation/training-status.md) - Current results, model storage, continuing training
- [spec/schemas.md](spec/schemas.md) - Data file schemas (kg.json, train.jsonl, episodes.jsonl, etc.)
- [spec/reward.md](spec/reward.md) - Reward function specification
- [documentation/process.md](documentation/process.md) - Development process and TDD guidelines
- [documentation/tools.md](documentation/tools.md) - Development tools
- [documentation/ai_agent_instructions.md](documentation/ai_agent_instructions.md) - AI coding agent guidelines

## Cross-Platform Support

### macOS (Apple Silicon) - MLX
```bash
make setup-mlx
make train-360m
```

### Linux (CUDA) - Unsloth
```bash
make setup-unsloth
make train-360m-unsloth
```

Unsloth provides 2x faster training with 60% less memory on NVIDIA GPUs. See [CLAUDE.md](CLAUDE.md) for detailed setup instructions.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Apple Silicon Mac (for MLX) or CUDA GPU (for Unsloth/PyTorch)
- Dependencies: mlx, mlx-lm, transformers, numpy, tqdm

## License

MIT License - see [LICENSE](LICENSE) for details.

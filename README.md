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
- **Demo UI**: Interactive visualization of training progress

## Results

| Phase | Accuracy | Notes |
|-------|----------|-------|
| Base (SmolLM-135M) | 0% | No format compliance |
| SFT (200 iters) | 30% | Learns TRACE + ANSWER format |
| RSFT (eval distribution) | **75%** | Distribution-matched training |

See [docs/training-status.md](docs/training-status.md) for details on model storage and continuing training.

## Demo

```bash
python demo/server.py
# Open http://localhost:3519
```

The demo visualizes training (with knowledge graph scoring) and inference (without graph access).

## Quick Start

```bash
# Setup environment (macOS with Apple Silicon)
make setup-mlx

# Generate data
make data

# Run full training pipeline
make train
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
├── core/              # Core library
│   ├── kg.py          # Knowledge graph loading & path sampling
│   ├── dataset.py     # MCQ generation
│   ├── reward.py      # Reward computation
│   ├── infer.py       # Model inference
│   ├── mlx_sft.py     # LoRA training with MLX
│   ├── rsft.py        # Rejection sampling fine-tuning
│   └── eval.py        # Metrics generation
├── data/              # Data files
│   ├── kg.json        # Knowledge graph
│   ├── train.jsonl    # Training examples (1-3 hops)
│   ├── eval.jsonl     # Evaluation examples (4-5 hops)
│   └── runs/          # Training run outputs
├── demo/              # Demo web application
├── spec/              # Specifications
│   ├── schemas.md     # Data schemas
│   └── reward.md      # Reward function spec
├── docs/              # Documentation
├── tests/             # Test suite
└── video/             # Video production assets
```

## Model

Default: **SmolLM-135M-Instruct** (`HuggingFaceTB/SmolLM-135M-Instruct`)

Can be changed via: `make sft MODEL=HuggingFaceTB/SmolLM-360M-Instruct`

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

- [docs/eli5.md](docs/eli5.md) - ELI5 explanation of the project and key findings
- [docs/training-status.md](docs/training-status.md) - Current results, model storage, continuing training
- [spec/schemas.md](spec/schemas.md) - Data file schemas (kg.json, train.jsonl, episodes.jsonl, etc.)
- [spec/reward.md](spec/reward.md) - Reward function specification
- [docs/process.md](docs/process.md) - Development process and TDD guidelines
- [docs/tools.md](docs/tools.md) - Development tools
- [docs/ai_agent_instructions.md](docs/ai_agent_instructions.md) - AI coding agent guidelines

## Requirements

- Python 3.10+
- Apple Silicon Mac (for MLX) or CUDA GPU (for Unsloth/PyTorch)
- Dependencies: mlx, mlx-lm, transformers, numpy, tqdm

## License

MIT License - see [LICENSE](LICENSE) for details.

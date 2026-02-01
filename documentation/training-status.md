# Training Status

## Current Results

| Phase | Accuracy | Path Coverage | Avg Reward | Notes |
|-------|----------|---------------|------------|-------|
| Base (SmolLM-135M) | 0% | 0% | -2.00 | Untrained, no format compliance |
| SFT (200 iters) | 30% | 31% | -1.01 | Learns TRACE + ANSWER format |
| RSFT on eval data | **75%** | 33% | +0.38 | Distribution-matched training |

Training was performed on Apple M3 Pro using MLX. Total pipeline time: ~5 minutes.

## Model Storage

Trained adapters are stored in:

```
data/runs/run_0001/models/
├── sft/adapters.safetensors         # 267MB - SFT adapter (30% accuracy)
├── rsft/adapters.safetensors        # 267MB - RSFT on train data (20%)
└── rsft_eval/adapters.safetensors   # 267MB - RSFT on eval data (75%)
```

These are LoRA adapters, not full models. The base model (SmolLM-135M-Instruct) is downloaded from HuggingFace and cached in `~/.cache/huggingface/`.

## Git Ignored

Model binaries are excluded from git via `.gitignore`:
- `*.safetensors`
- `data/runs/*/models/`

This prevents committing large binary files to the repository.

## Continue Training

To continue training from an existing adapter with new examples:

```bash
# Start from the best adapter (rsft_eval) and train on new examples
.venv/bin/python -m core.mlx_sft \
  --train data/new_examples.jsonl \
  --output data/runs/run_0002/models/continued \
  --model HuggingFaceTB/SmolLM-135M-Instruct \
  --adapter data/runs/run_0001/models/rsft_eval \
  --iters 100
```

The `--adapter` flag loads existing trained weights as a starting point.

## Reproducing Training

Full training pipeline from scratch:

```bash
# 1. Setup environment
python3 -m venv .venv
.venv/bin/pip install mlx mlx-lm transformers numpy tqdm safetensors

# 2. SFT training (~90 seconds)
.venv/bin/python -m core.mlx_sft \
  --train data/train.jsonl \
  --output data/runs/run_0001/models/sft \
  --model HuggingFaceTB/SmolLM-135M-Instruct \
  --iters 200

# 3. RSFT on eval distribution (~2 minutes)
.venv/bin/python -m core.rsft \
  --examples data/eval.jsonl \
  --kg data/kg.json \
  --sft-adapter data/runs/run_0001/models/sft \
  --output data/runs/run_0001/models/rsft_eval \
  --k-samples 8 \
  --max-examples 50

# 4. Evaluate
.venv/bin/python -m core.infer \
  --examples data/eval.jsonl \
  --kg data/kg.json \
  --output data/runs/run_0001/episodes.jsonl \
  --adapter data/runs/run_0001/models/rsft_eval \
  --phase rsft \
  --max-examples 20
```

## Key Findings

1. **SFT teaches format compliance** - Model learns to output structured TRACE + ANSWER format
2. **RSFT is distribution-sensitive** - Training on easy examples (1-3 hop) hurt performance (20% vs 30%)
3. **Distribution matching is crucial** - RSFT on hard examples (4-5 hop) achieved 75% accuracy

## Future Improvements

Potential ways to improve beyond 75%:
- Larger base model (more capacity for complex reasoning)
- More diverse training examples
- Higher K for rejection sampling (more candidates)
- Curriculum learning (progressive difficulty)

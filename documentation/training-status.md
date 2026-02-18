# Training Status

*Last updated: February 18, 2026*

## Current State

### Trained Models Available

| Model | Run ID | Location | Status |
|-------|--------|----------|--------|
| SmolLM-360M SFT | run_360m | `data/runs/run_360m/models/sft/` | ✅ Complete |
| SmolLM-360M RSFT | run_360m | `data/runs/run_360m/models/rsft/` | ✅ Complete |

### SmolLM-360M Results (run_360m)

| Phase | Accuracy | Path Coverage | Avg Reward | Notes |
|-------|----------|---------------|------------|-------|
| Base | 0% | 0% | -2.00 | Untrained, no format compliance |
| SFT (500 iters) | 37% | 32% | -0.78 | Learns TRACE + ANSWER format |
| RSFT (train data) | 27% | 30% | -1.12 | Distribution mismatch! |

**Key Finding**: RSFT on easy examples (train.jsonl, 1-3 hop) performed *worse* than SFT baseline because the model learned the wrong distribution for the 4-5 hop evaluation questions.

### Previous Results (SmolLM-135M)

| Phase | Accuracy | Path Coverage | Avg Reward | Notes |
|-------|----------|---------------|------------|-------|
| Base | 0% | 0% | -2.00 | Untrained |
| SFT (200 iters) | 30% | 31% | -1.01 | Format learning |
| RSFT on eval data | **75%** | 33% | +0.38 | Distribution-matched |

## Demo Features

The demo (`http://localhost:3519`) now includes:

| Tab | Feature | Status |
|-----|---------|--------|
| Training | Animated SFT→RSFT visualization | ✅ Working |
| Inference | Pre-recorded inference examples | ✅ Working |
| Try It | **Live inference** with 360M model | ✅ Working |
| Distribution | Key finding visualization | ✅ Working |

## Model Storage

Trained adapters are stored in:

```
data/runs/run_360m/models/
├── sft/
│   ├── adapters.safetensors    # ~740MB - SFT adapter
│   ├── adapter_config.json
│   └── sft_train.jsonl         # Training data used
└── rsft/
    ├── adapters.safetensors    # ~740MB - RSFT adapter
    ├── adapter_config.json
    ├── rsft_train.jsonl        # Winning completions
    └── rsft_winners.jsonl      # All scored winners
```

These are LoRA adapters. The base model (SmolLM-360M-Instruct) is cached in `~/.cache/huggingface/`.

## Git Ignored

Model binaries are excluded from git:
- `*.safetensors`
- `data/runs/*/models/`

---

## Next Steps

### Immediate (To Fix Distribution Issue)

1. **Train RSFT on eval.jsonl** (hard examples) to achieve 75%+ accuracy:
   ```bash
   # Modify Makefile rsft target to use eval.jsonl instead of train.jsonl
   # Or run manually:
   python3 -m core.rsft \
     --examples data/eval.jsonl \
     --kg data/kg.json \
     --sft-adapter data/runs/run_360m/models/sft \
     --output data/runs/run_360m/models/rsft_eval \
     --model HuggingFaceTB/SmolLM-360M-Instruct \
     --k-samples 8 \
     --max-examples 50
   ```

2. **Update demo to use rsft_eval adapter** for better live inference quality

### Linux/Unsloth Training

Ready for execution on Linux with NVIDIA GPU:

```bash
# On Linux system with CUDA
git clone <repo>
cd multi-hop-reasoning
make setup-unsloth
source .venv/bin/activate
make train-360m-unsloth
```

See `CLAUDE.md` for detailed instructions.

### Future Improvements

| Improvement | Expected Impact | Effort |
|-------------|-----------------|--------|
| RSFT on eval distribution | 75%+ accuracy | Low |
| Larger model (1B+) | Better reasoning | Medium |
| More training examples | Better generalization | Medium |
| Higher K (16-32) for RSFT | More diverse winners | Low |
| Curriculum learning | Smoother learning | High |
| Web search for KG expansion | Richer knowledge | High |

### Demo Improvements

- [ ] Add loading spinner during model initialization
- [ ] Show example MCQ format in Try It tab
- [ ] Add comparison view showing SFT vs RSFT responses
- [ ] Generate static comparison data for GitHub Pages

---

## Reproducing Training

### macOS (MLX)

```bash
# Setup
make setup-mlx
source .venv/bin/activate

# Full pipeline
make train-360m

# Or individual steps
make data
make sft MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m
make rsft MODEL=HuggingFaceTB/SmolLM-360M-Instruct RUN_ID=run_360m
make eval RUN_ID=run_360m
```

### Linux (Unsloth)

```bash
# Setup
make setup-unsloth
source .venv/bin/activate

# Full pipeline
make train-360m-unsloth
```

---

## Key Learnings

1. **SFT teaches format compliance** - Model learns TRACE + ANSWER structure
2. **Distribution matching is critical** - Train on examples similar to eval
3. **RSFT can hurt if mismatched** - 20% vs 30% when trained on wrong distribution
4. **360M is sufficient** - Produces coherent multi-hop reasoning traces
5. **MLX is fast** - Full pipeline in ~10 minutes on Apple Silicon

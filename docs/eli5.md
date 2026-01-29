# Multi-Hop Reasoning: ELI5

## The Problem

Large language models (LLMs) are good at sounding confident, but they often make things up. When you ask "Why is my server timing out?", the model might give a plausible-sounding answer without actually reasoning through the problem.

We want models that can **think step by step** through a chain of causes and effects—and we want to **verify** that their reasoning is grounded in real relationships.

## The Solution: Knowledge Graphs + Rewards

### What's a Knowledge Graph?

Think of it as a map of "what causes what" in a specific domain.

```
TLSHandshakeError --[caused by]--> ClockSkew --[fixed by]--> NTPSync
```

Each arrow is a relationship we know to be true. The graph captures expert knowledge about how problems connect to causes connect to solutions.

### What's Multi-Hop Reasoning?

Following multiple arrows to get from a symptom to a root cause or fix.

**1-hop**: Server timeout → Check network
**3-hop**: Server timeout → High latency → Network congestion → Add bandwidth
**5-hop**: TLS error → Clock skew → NTP failure → Firewall blocking → Port closed → Open port 123

More hops = harder reasoning. The model needs to chain together multiple facts correctly.

### What's the Reward?

We score model outputs on two things:

1. **Correctness**: Did you pick the right answer? (+1 for right, -2 for wrong)
2. **Path Coverage**: Did your reasoning mention the actual entities in the knowledge graph path?

A model that gets the right answer **and** shows its work along the correct path gets the highest reward.

## The Key Insight: Training vs Inference

This is the crucial part that makes this approach powerful:

### During Training: Graph is the Teacher

The knowledge graph acts as a **reward signal**. We use it to score whether the model's reasoning follows valid paths:

```
Model output: "TLSHandshakeError is caused by ClockSkew..."
Graph check:  Does edge (TLSHandshakeError → ClockSkew) exist? ✓
Reward:       +path_coverage bonus
```

The graph tells us if the model is "cheating" (right answer, wrong reasoning) or genuinely learning the causal structure.

### During Inference: Graph is Removed

Once trained, the model runs **without access to the knowledge graph**. It must have internalized the reasoning patterns:

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING                         INFERENCE                 │
│  ─────────                        ─────────                 │
│  Model ←──reward──← KG            Model (alone)             │
│    │                  │              │                      │
│    └─► output ───────►│              └─► output             │
│                       │                                     │
│  Graph scores the     │           No graph available.       │
│  reasoning path.      │           Model must have learned   │
│                       │           the paths internally.     │
└─────────────────────────────────────────────────────────────┘
```

This is why it matters: we're not building a system that looks up answers in a graph. We're teaching a model to **reason like the graph** so it can generalize to new situations.

## The Three Training Phases

| Phase | What It Is | Expected Performance |
|-------|------------|---------------------|
| **Base** | Untrained model, just guessing | ~25% accuracy, poor reasoning |
| **SFT** | Supervised fine-tuning on good examples | ~55% accuracy, better reasoning |
| **RSFT** | Rejection sampling—keep only high-reward outputs | ~65% accuracy, best reasoning |

### How RSFT Works (RL-lite)

Instead of complex reinforcement learning:
1. Generate K completions per question (e.g., K=8)
2. Score each with the reward function
3. Keep only the best ones as new training examples
4. Fine-tune on these "winners"

This is simpler than PPO but captures the key idea: use rewards to select better training data.

## What the Demo Shows

### The Visualization

**Left Panel: Knowledge Graph**
- Nodes = entities (errors, causes, tools, fixes)
- Gold edges = the correct reasoning path
- Green nodes = entities the model mentioned in its trace
- Gray nodes = entities the model missed

**Right Panel: Episode Details**
- Question: The multiple-choice problem
- Model Output: What the model said (TRACE + ANSWER)
- Reward Bars: Visual breakdown of the score

### Reading the Reward Bars

```
Correctness:  [████████████] +1.0   ← Got it right
Path Coverage:[████████    ] 0.67   ← Mentioned 2/3 path entities
Spam Penalty: [            ] 0.0    ← Didn't repeat entities
─────────────────────────────────────
Total:        [█████████   ] 1.34   ← Good score!
```

A negative correctness bar means wrong answer (-2.0).

### What to Look For

1. **Base phase**: Model often picks wrong answers, traces are vague or miss the path
2. **SFT phase**: More correct answers, traces start mentioning relevant entities
3. **RSFT phase**: Best accuracy, traces follow the knowledge graph path closely

## Project Status

This project has two modes:

### Simulation Mode (Current Demo)

The demo visualization shows **simulated data** to illustrate the concept:
- Episodes are generated with randomized rewards matching expected distributions
- No actual model inference has been run
- Useful for understanding the UI and reward mechanics

### Real Training Mode

The training pipeline is fully implemented but not yet executed:

```bash
# Run actual training (takes 1-2 hours on M3 Pro)
make sft      # Supervised fine-tuning with LoRA
make rsft     # Rejection sampling fine-tuning
make eval     # Generate real episodes with trained model
```

After running these, the demo will show **real model outputs** instead of simulated data.

### What's Implemented

| Component | Status |
|-----------|--------|
| Knowledge graph + dataset generation | ✅ Complete |
| Reward function with path scoring | ✅ Complete |
| MLX LoRA training code | ✅ Complete |
| Inference pipeline | ✅ Complete |
| Demo visualization | ✅ Complete |
| Trained model weights | ⏳ Not yet run |
| Real episode data | ⏳ Not yet run |

## Why This Matters

Traditional training just teaches models to mimic human text. This approach:

1. **Grounds reasoning** in a verifiable knowledge structure
2. **Rewards the process**, not just the final answer
3. **Makes reasoning auditable**—you can see if the model followed a valid path
4. **Removes the crutch**—at inference, the model must reason independently

The goal: models that don't just guess right, but **reason right**, and can do so without external lookup.

## Glossary

| Term | Meaning |
|------|---------|
| **KG** | Knowledge Graph—a network of entities and relationships |
| **Hop** | One step/edge in the graph |
| **Trace** | The model's explanation of its reasoning |
| **SFT** | Supervised Fine-Tuning—learning from labeled examples |
| **RSFT** | Rejection Sampling Fine-Tuning—filtering for high-reward outputs |
| **LoRA** | Low-Rank Adaptation—efficient fine-tuning that only updates small adapter weights |
| **MLX** | Apple's ML framework optimized for Apple Silicon |

## Running the Demo

```bash
# Start the visualization server
python demo/server.py

# Open http://localhost:3519
```

Use the phase dropdown to switch between base/sft/rsft and see how reasoning quality changes across training phases.

## Training Time Estimates

On Apple M3 Pro with MLX:
- **SFT**: ~30-60 minutes (1000 iterations)
- **RSFT**: ~30 minutes (500 iterations, K=8 samples)
- **Eval**: ~5 minutes (50 examples)

Model: `SmolLM-135M-Instruct` (small enough to train fast, large enough to follow instructions)

# Agent 1: Core Library + MLX Training

## Mission

Deliver a working MLX-first pipeline that produces:

1. Dataset (train/eval)
2. Baseline inference logs
3. SFT LoRA model
4. RSFT "RL-lite" model
5. Metrics + episode logs for the demo UI

## Platform

- **Primary**: Apple Silicon (M3 Pro) + MLX
- **Secondary**: Arch Linux + Unsloth/CUDA on RTX 5060 16GB (then RTX 3090 24GB)

MLX-first because it's fastest path to "it runs." Use CUDA/Unsloth as Phase 2 (optional "scale-up").

## Core Concept (Paper Reproduction)

1. KG → sample path (1–3 hop train, 4–5 hop eval)
2. MCQ generation (deterministic templates; no external LLM required)
3. Model outputs strict schema:
   ```
   TRACE: ...
   ANSWER: A|B|C|D
   ```
4. Reward:
   - R_corr: +1 correct, -2 incorrect
   - R_path: entity coverage of path entities in TRACE, with anti-spam penalty
5. Training:
   - SFT (LoRA) in MLX
   - RSFT (Rejection Sampling Fine-Tuning) for RL-lite approach

---

## Milestones

### M0 — Environment + Reproducibility

- Create python venv/uv env
- Ensure MLX + mlx-lm works on M3 Pro
- Add Makefile or justfile targets:
  - `make data`
  - `make infer_base`
  - `make sft`
  - `make rsft`
  - `make eval`

### M1 — KG + Dataset Generator (Deterministic)

**Domain**: Software troubleshooting KG

Build `data/kg.json` with:
- ~200 entities
- ~600 edges
- Relations like `caused_by`, `fixed_by`, `diagnosed_by`

Implement `core/dataset.py`:
- Sample 1–3 hop paths for train, 4–5 for eval
- Generate MCQs with near-miss distractors
- Generate a reference trace template that mentions all path entities

### M2 — Reward Function

Implement `core/reward.py`:
- Parse completion (strict schema)
- Extract entities by exact match from a known vocabulary (fast + deterministic)
- R_corr = +1/-2
- R_path = min(1.0, coverage) but:
  - Require min_hits >= 2
  - Apply spam penalty if entity repetition ratio too high
- Export reward breakdown

### M3 — Baseline Inference + Logging

`core/infer.py`:
- Run base model on N eval examples
- Write `episodes.jsonl` for phase=base

### M4 — SFT LoRA in MLX

`core/mlx_sft.py`:
- Train LoRA on `train.jsonl` using reference traces
- Produce model artifacts under `data/runs/run_xxxx/models/sft`
- Rerun inference, log phase=sft

### M5 — RSFT (Rejection Sampling Fine-Tuning) = RL-lite

This is the pragmatic trick:

For each prompt:
1. Sample K completions (K=4–16)
2. Score by total reward
3. Keep top-1 (or top-2) completions as "pseudo-labels"
4. Fine-tune LoRA further on these winners

`core/rsft.py`:
- Generate winners dataset `rsft.jsonl`
- Continue LoRA training
- Log phase=rsft

### M6 — Evaluation Report

`core/eval.py`:
- Report:
  - Accuracy on eval (4–5 hop)
  - Avg path coverage
  - Spam score
- Output `metrics.json`

---

## Why RSFT is "Good Enough" for the First Demo

It communicates the paper's core:
- You're improving behavior using a reward that includes path alignment
- The model "learns to mention the right intermediate entities" while improving correctness

And it works without implementing PPO in MLX.

---

## Model Choice

**Recommended**: SmolLM-135M-Instruct for speed and low VRAM.

If it's too weak to "follow the game," move up slightly:
- SmolLM-360M or Qwen2.5-0.5B/1.5B Instruct (still manageable with LoRA + 4-bit)

**Tradeoff**: Smaller model = faster + more "wow, it learned something"
But too small can fail to format outputs reliably.

---

## Tools & Libraries

**Python (MLX)**:
- mlx
- mlx-lm
- transformers (for tokenizer/config)

**Optional for CUDA Phase 2**:
- transformers
- peft
- trl (if using PPO)
- bitsandbytes (optional 4-bit)
- accelerate

---

# Data Schemas

## Directory Layout

```
kg_reward_demo/
  data/
    kg.json
    train.jsonl
    eval.jsonl
    runs/
      run_0001/
        meta.json
        metrics.json
        episodes.jsonl
        models/
          base/      # optional pointer only
          sft/       # MLX adapter / weights
          rsft/      # MLX adapter / weights
```

## File: data/kg.json

Defines the KG: entities and directed labeled edges.

```json
{
  "version": "1.0",
  "domain": "software_troubleshooting",
  "entities": [
    {"id": "TLSHandshakeError", "label": "TLS Handshake Error", "type": "symptom"},
    {"id": "ClockSkew", "label": "Clock Skew", "type": "cause"},
    {"id": "NTPsync", "label": "NTP Sync", "type": "fix"}
  ],
  "relations": [
    {"id": "caused_by", "label": "caused by"},
    {"id": "fixed_by", "label": "fixed by"},
    {"id": "diagnosed_by", "label": "diagnosed by"}
  ],
  "edges": [
    {"src": "TLSHandshakeError", "rel": "caused_by", "dst": "ClockSkew"},
    {"src": "ClockSkew", "rel": "fixed_by", "dst": "NTPsync"}
  ]
}
```

**Constraints**:
- `entities[*].id` must be unique
- `relations[*].id` must be unique
- Each edge references existing `src`, `dst`, and `rel`
- Graph may contain cycles; path sampling must prevent repeats in a sampled path

## File: data/train.jsonl and data/eval.jsonl

Self-contained MCQ tasks derived from KG paths. One JSON object per line.

```json
{
  "id": "ex_000123",
  "split": "train",
  "hop_len": 3,

  "question": "Your service shows TLSHandshakeError. What is the most likely underlying cause?",
  "options": {
    "A": "CertificateExpired",
    "B": "ClockSkew",
    "C": "PacketLoss",
    "D": "DiskFull"
  },
  "answer_star": "B",

  "path_star": {
    "entities": ["TLSHandshakeError", "ClockSkew", "NTPsync"],
    "edges": [
      {"src": "TLSHandshakeError", "rel": "caused_by", "dst": "ClockSkew"},
      {"src": "ClockSkew", "rel": "fixed_by", "dst": "NTPsync"}
    ]
  },

  "prompt": "You must follow the exact output format.\n\nOUTPUT FORMAT:\nTRACE: <one or two sentences>\nANSWER: <A|B|C|D>\n\nQuestion: ...\nA) ...\nB) ...\nC) ...\nD) ...\n",

  "ref": {
    "trace": "TRACE: TLSHandshakeError is often caused by ClockSkew, and ClockSkew is fixed by NTPsync.\nANSWER: B",
    "style_rules": [
      "Mention at least 2 path entities.",
      "Do not list all options.",
      "Do not repeat an entity more than twice."
    ]
  },

  "meta": {
    "topic": "tls",
    "difficulty": "easy",
    "distractor_type": "near_miss"
  }
}
```

**Constraints**:
- `options` must have exactly keys A,B,C,D
- `answer_star` must be one of A,B,C,D
- `path_star.entities` is ordered and should reflect the path traversal order
- `prompt` must contain the output format rules exactly
- `ref.trace` must be parseable with the same parser used for model completions

## File: data/runs/run_xxxx/meta.json

Reproducibility + provenance.

```json
{
  "run_id": "run_0001",
  "created_at": "2026-01-27T12:00:00-08:00",
  "platform": "apple_m3pro_mlx",
  "model": {
    "base": "HuggingFaceTB/SmolLM-135M-Instruct",
    "tokenizer": "HuggingFaceTB/SmolLM-135M-Instruct"
  },
  "training": {
    "seed": 1234,
    "phases": ["base", "sft", "rsft"],
    "sft": {"lora_r": 16, "lora_alpha": 16, "lr": 1e-4, "steps": 1000, "batch_size": 8},
    "rsft": {"k_samples": 8, "keep_top": 1, "steps": 500}
  },
  "reward": {
    "version": "1.0",
    "weights": {"correctness": 1.0, "path": 0.5},
    "correct": 1.0,
    "incorrect": -2.0,
    "min_hits": 2,
    "max_path_reward": 1.0,
    "spam_penalty": {"repeat_entity_max": 2, "repeat_penalty": 0.5}
  }
}
```

## File: data/runs/run_xxxx/metrics.json

Simple aggregated metrics for UI + charts.

```json
{
  "run_id": "run_0001",
  "splits": {
    "eval": {
      "base": {"accuracy": 0.28, "avg_path_coverage": 0.11, "avg_total_reward": -1.2},
      "sft":  {"accuracy": 0.52, "avg_path_coverage": 0.34, "avg_total_reward": 0.1},
      "rsft": {"accuracy": 0.61, "avg_path_coverage": 0.48, "avg_total_reward": 0.6}
    }
  },
  "curves": {
    "steps": [0, 50, 100],
    "eval_accuracy": [0.28, 0.55, 0.61],
    "eval_path_coverage": [0.11, 0.41, 0.48]
  }
}
```

## File: data/runs/run_xxxx/episodes.jsonl

Playback log: the UI replays examples across phases. One JSON object per line.

```json
{
  "phase": "base",
  "step": 0,
  "ex_id": "ex_000123",
  "split": "eval",

  "prompt": "....",
  "completion": "TRACE: ...\nANSWER: B",

  "parsed": {
    "answer": "B",
    "trace_text": "TLSHandshakeError is often caused by ClockSkew ...",
    "trace_entities": ["TLSHandshakeError", "ClockSkew"],
    "path_entities": ["TLSHandshakeError", "ClockSkew", "NTPsync"],
    "valid_format": true
  },

  "reward": {
    "correctness": 1.0,
    "path_coverage": 0.67,
    "path_reward": 0.67,
    "spam_penalty": 0.0,
    "total": 1.34
  },

  "debug": {
    "hits": 2,
    "unique_entity_mentions": 2,
    "entity_repeat_counts": {"TLSHandshakeError": 1, "ClockSkew": 1}
  }
}
```

**Constraints**:
- `phase` ∈ {base, sft, rsft}
- `parsed.valid_format=false` is allowed; in that case reward should be strongly negative
- UI should be robust to missing `debug`

---

# Reward Specification

## Goal

Compute a scalar reward for a model completion that encourages:
1. Choosing the correct MCQ option
2. Grounding the trace in the KG path (mentioning key entities)

While resisting:
- Entity-spam reward hacking
- Malformed output

## Input

- **Example object** from train/eval.jsonl:
  - `answer_star`
  - `path_star.entities` (ordered list)
- **Model completion string** (raw text)

## Output

Reward breakdown object:

```json
{
  "correctness": 1.0,
  "path_coverage": 0.67,
  "path_reward": 0.67,
  "spam_penalty": 0.0,
  "total": 1.34
}
```

## Parsing Contract (Strict)

Model completion must match:
1. Must contain a line starting with `TRACE:` (case-sensitive)
2. Must contain a line starting with `ANSWER:` (case-sensitive)
3. `ANSWER:` value must be exactly one of `A|B|C|D` (strip whitespace)

**If parsing fails**:
- `valid_format=false`
- Set:
  - `correctness = -2.0`
  - `path_reward = 0.0`
  - `spam_penalty = 0.0`
  - `total = -2.0`

Rationale: we need deterministic training signals.

## Entity Extraction

We do dictionary matching, not embeddings, not fuzzy matching (toy demo).

1. Build `ENTITY_VOCAB = set(kg.entities[*].id)` plus optionally labels
2. Normalize:
   - For IDs: match exact token substrings with word boundaries when possible
   - For labels: optional (if used), normalize to lowercase and strip punctuation

**Recommended toy approach**: Include entities in prompts and ref traces by id to make matching easy.

**Extracted sets**:
- `E_path = set(path_star.entities)`
- `E_trace = set(entities found in TRACE text)`

## Correctness Reward (R_corr)

Let `y` = predicted answer, `y*` = answer_star.

- If `y == y*`: R_corr = **+1.0**
- Else: R_corr = **-2.0**

These values are tunable but should remain asymmetric (wrong hurts more).

## Path Coverage & Reward (R_path)

Compute:
- `hits = |E_trace ∩ E_path|`
- `coverage = hits / max(1, |E_path|)`

Apply minimum-hit constraint:
- If `hits < MIN_HITS` (default 2): `R_path_raw = 0.0`
- Else: `R_path_raw = coverage`

Cap:
- `R_path = min(R_path_raw, MAX_PATH_REWARD)` (default MAX=1.0)

## Anti-Spam Penalty (P_spam)

Goal: penalize "entity chanting."

Compute entity repeat counts within TRACE:
- `count[e] = number of occurrences of entity e in TRACE`

If any `count[e] > REPEAT_ENTITY_MAX` (default 2):
- `P_spam = REPEAT_PENALTY` (default 0.5)

Else:
- `P_spam = 0.0`

## Total Reward

Weights from meta.json:
- `W_corr` (default 1.0)
- `W_path` (default 0.5)

Compute:
```
total = (W_corr * R_corr) + (W_path * R_path) - P_spam
```

## Debug Fields (Recommended)

Return:
- `hits`
- `coverage`
- `entity_repeat_counts`
- `valid_format`

These are extremely useful for UI and video overlays.

## RSFT Selection Score

When doing rejection sampling:
- Use `total` as the selection score
- Ties broken by:
  1. Higher correctness
  2. Higher coverage
  3. Shorter TRACE (encourage concise reasoning)

---

## Acceptance Tests

Reward test cases (must be automated):

1. **Correct answer, mentions 2 path entities, no spam** → positive total
2. **Correct answer, mentions 0–1 path entities** → total ~ +1 (path bonus 0)
3. **Wrong answer but mentions all entities** → should still be negative overall
4. **Correct answer but spams one entity 6 times** → penalty applied
5. **Missing ANSWER: line** → total = -2 and valid_format=false

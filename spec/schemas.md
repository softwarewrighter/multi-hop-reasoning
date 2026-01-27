# Shared Data Schemas

Single source of truth for data interchange between all agents.

All files are newline-delimited JSON (.jsonl) unless stated otherwise.
All strings are UTF-8.

---

## Directory Layout

```
kg_reward_demo/
  data/
    kg.json           # Knowledge graph definition
    train.jsonl       # Training examples (1-3 hop)
    eval.jsonl        # Evaluation examples (4-5 hop)
    runs/
      run_0001/
        meta.json     # Run configuration/provenance
        metrics.json  # Aggregated metrics
        episodes.jsonl # Per-example inference logs
        models/
          base/       # Pointer to base model
          sft/        # SFT LoRA adapters
          rsft/       # RSFT LoRA adapters
```

---

## kg.json

Knowledge graph with entities and directed labeled edges.

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
- `entities[*].id` - unique identifier (used for matching)
- `entities[*].label` - human-readable name
- `entities[*].type` - category (symptom, cause, fix, tool, etc.)
- `relations[*].id` - unique relation identifier
- Edges reference valid src/dst entity IDs and rel relation ID

---

## train.jsonl / eval.jsonl

One JSON object per line. MCQ tasks derived from KG paths.

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
  "prompt": "OUTPUT FORMAT:\nTRACE: <reasoning>\nANSWER: <A|B|C|D>\n\nQuestion: ...",
  "ref": {
    "trace": "TRACE: TLSHandshakeError is caused by ClockSkew...\nANSWER: B",
    "style_rules": ["Mention at least 2 path entities", "No spam"]
  },
  "meta": {
    "topic": "tls",
    "difficulty": "easy"
  }
}
```

---

## meta.json

Run configuration for reproducibility.

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
    "sft": {"lora_r": 16, "lora_alpha": 16, "lr": 1e-4, "steps": 1000},
    "rsft": {"k_samples": 8, "keep_top": 1, "steps": 500}
  },
  "reward": {
    "version": "1.0",
    "weights": {"correctness": 1.0, "path": 0.5},
    "correct": 1.0,
    "incorrect": -2.0,
    "min_hits": 2,
    "spam_penalty": {"repeat_entity_max": 2, "repeat_penalty": 0.5}
  }
}
```

---

## metrics.json

Aggregated results for UI display.

```json
{
  "run_id": "run_0001",
  "splits": {
    "eval": {
      "base": {"accuracy": 0.28, "avg_path_coverage": 0.11, "avg_total_reward": -1.2},
      "sft": {"accuracy": 0.52, "avg_path_coverage": 0.34, "avg_total_reward": 0.1},
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

---

## episodes.jsonl

Per-example inference log for UI playback. One JSON object per line.

```json
{
  "phase": "base",
  "step": 0,
  "ex_id": "ex_000123",
  "split": "eval",
  "prompt": "...",
  "completion": "TRACE: ...\nANSWER: B",
  "parsed": {
    "answer": "B",
    "trace_text": "TLSHandshakeError is caused by ClockSkew...",
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
    "entity_repeat_counts": {"TLSHandshakeError": 1}
  }
}
```

**phase** must be one of: `base`, `sft`, `rsft`

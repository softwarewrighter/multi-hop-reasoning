# Agent 2: Visual Demo App

## Mission

Create an interactive demo that:

1. Loads a `data/runs/run_xxxx/episodes.jsonl`
2. Shows the KG graph
3. Highlights the ground-truth path
4. Shows model TRACE/ANSWER with entity highlights
5. Shows reward breakdown bars
6. Can autoplay as a "demo mode" for video capture

## Stack Choice

**Quickest approach**:
- Pure static web (HTML/JS/CSS) + small Python server
- Graph rendering: simple canvas force-directed layout (or d3-force)
- Highlight path edges

---

## Milestones

### D1 — Run Browser + Picker

- UI loads run folders and allows selection
- Shows `metrics.json` summary
- Lists available phases (base, sft, rsft)

### D2 — Episode Player

- Next/prev buttons
- Autoplay toggle with configurable speed
- Scrubber/slider to jump to specific episodes
- Step counter display

### D3 — Graph + Highlight

- Render subgraph around the path entities (don't render the entire KG at once)
- Highlight:
  - **True path** in bright stroke (e.g., gold/yellow)
  - **Entities mentioned in TRACE** in another color (e.g., green)
  - **Missed path entities** in muted color (e.g., gray)
- Display in a way that reads cleanly in a 1080p recording
- Node labels must be legible

### D4 — Reward HUD

Display reward breakdown as visual bars:

```
Correctness:  [████████████] +1.0
Path Coverage:[████████    ] 0.67
Spam Penalty: [            ] 0.0
─────────────────────────────────
Total:        [█████████   ] 1.34
```

- Color-coded (green for positive, red for negative/penalty)
- Animate transitions between episodes

### D5 — "Before / After" Compare

- 3 tabs or dropdown: **base / sft / rsft**
- Load same example ID across phases if possible
- Side-by-side or tabbed comparison view
- Show metrics diff (accuracy improvement, coverage improvement)

---

## Data Dependencies (from Agent 1)

### episodes.jsonl Schema

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

### metrics.json Schema

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

### kg.json Schema

```json
{
  "version": "1.0",
  "domain": "software_troubleshooting",
  "entities": [
    {"id": "TLSHandshakeError", "label": "TLS Handshake Error", "type": "symptom"}
  ],
  "relations": [
    {"id": "caused_by", "label": "caused by"}
  ],
  "edges": [
    {"src": "TLSHandshakeError", "rel": "caused_by", "dst": "ClockSkew"}
  ]
}
```

---

## UI Layout (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│  Run: run_0001  │  Phase: [base ▼]  │  ◀ │ ▶ │ ▶▶ Auto │ 12/50 │
├─────────────────────────────────────────────────────────────────┤
│                        │                                        │
│   ┌─────────────────┐  │  Question:                             │
│   │                 │  │  Your service shows TLSHandshakeError. │
│   │   GRAPH VIEW    │  │  What is the most likely fix?          │
│   │                 │  │                                        │
│   │  [nodes/edges]  │  │  Options:                              │
│   │                 │  │  A) Restart service                    │
│   │                 │  │  B) NTP sync  ✓                        │
│   └─────────────────┘  │  C) Rotate cert                        │
│                        │  D) Increase timeout                   │
│                        ├────────────────────────────────────────│
│                        │  Model Output:                         │
│                        │  TRACE: [TLSHandshakeError] is caused  │
│                        │  by [ClockSkew], fixed by [NTPsync].   │
│                        │  ANSWER: B                             │
│                        ├────────────────────────────────────────│
│                        │  Reward:                               │
│                        │  Correct: ████████████ +1.0            │
│                        │  Path:    ████████     0.67            │
│                        │  Penalty:              0.0             │
│                        │  ────────────────────                  │
│                        │  Total:   █████████    1.34            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tools & Libraries

**Python**:
- `http.server` or FastAPI/Flask for simple static file serving
- JSON file loading

**JavaScript**:
- d3-force (graph layout)
- OR simple canvas-based force simulation
- Plain HTML/JS is fine (keeps it easy)

**CSS**:
- Dark theme for video recording (high contrast)
- Large, readable fonts

---

## Deliverable

`python demo/server.py` opens a local URL that's record-ready:

1. Shows KG + highlighted ground-truth path
2. Shows model completion with entity highlighting
3. Shows reward breakdown
4. Can autoplay a "training story"
5. Works at 1920x1080 resolution for clean screen capture

---

## Demo Mode Features (for Agent 3)

- **Autoplay**: Cycles through episodes at configurable interval (2-3 seconds each)
- **Phase transitions**: Automatically switch from base → sft → rsft with visual indicator
- **Pause on interesting cases**: Option to pause on:
  - First correct answer after training
  - Big reward improvements
  - 5-hop examples
- **Export frame data**: Optional JSON dump of current state for overlay generation

---

## Acceptance Criteria

1. Can load and display any valid `episodes.jsonl`
2. Graph renders without overlapping node labels
3. Path highlighting is visually distinct
4. Reward bars animate smoothly
5. Autoplay works without user interaction
6. Renders cleanly at 1080p
7. Works in Chrome/Firefox/Safari

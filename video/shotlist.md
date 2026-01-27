# Shot List

## Required Footage

### Phase: Base (Before Training)

| Shot | Description | Duration | Notes |
|------|-------------|----------|-------|
| B1 | Wrong answer, trace mentions random entities | 3s | Show red reward bars |
| B2 | Another wrong, confident but wrong trace | 2s | Entity spam visible |
| B3 | Graph view with no path highlighting | 2s | Gray nodes, no gold |

### Phase: SFT (After Supervised Training)

| Shot | Description | Duration | Notes |
|------|-------------|----------|-------|
| S1 | Better format, some path entities mentioned | 3s | Partial green in path bar |
| S2 | Correct answer, medium coverage | 3s | First green correctness |
| S3 | Graph with 2/3 path nodes lit | 2s | Progress visible |

### Phase: RSFT (After RL-lite)

| Shot | Description | Duration | Notes |
|------|-------------|----------|-------|
| R1 | Correct + grounded, high coverage | 4s | All bars green |
| R2 | Hard 5-hop question, model succeeds | 5s | Hero shot |
| R3 | Full path lit up in graph | 3s | Gold line complete |

### Metrics & Curves

| Shot | Description | Duration | Notes |
|------|-------------|----------|-------|
| M1 | Training curve animation | 4s | Accuracy rising |
| M2 | Coverage curve animation | 3s | Path coverage rising |
| M3 | Final metrics comparison | 3s | base vs rsft |

### Comparison

| Shot | Description | Duration | Notes |
|------|-------------|----------|-------|
| C1 | Side-by-side base vs rsft, same question | 5s | Split screen or tabs |
| C2 | Reward breakdown comparison | 3s | Bar chart side by side |

---

## Overlay Graphics

| Overlay | Where Used | Description |
|---------|------------|-------------|
| "Before Training" | B1-B3 | Bottom left label |
| "After SFT" | S1-S3 | Bottom left label |
| "After RL-lite" | R1-R3 | Bottom left label |
| "Correctness Reward" | Solution section | Arrow pointing to green bar |
| "Path Alignment" | Solution section | Arrow pointing to blue bar |
| "Penalty" | When spam shown | Arrow pointing to red bar |
| Reward formula | Solution section | `R = correct + path - spam` |

---

## Audio Cues

| Cue | Trigger | Sound |
|-----|---------|-------|
| Correct answer | Green answer appears | Subtle ding |
| Wrong answer | Red answer appears | Soft buzz |
| Path lights up | Node highlights | Soft chime |
| Full path complete | All nodes gold | Success fanfare |

---

## B-Roll Needs

- [ ] Abstract neural network visualization (optional)
- [ ] Knowledge graph zoom out showing full structure
- [ ] Code scrolling (training loop) - brief flash

---

## Shot Priority

**Must have** (core story):
- B1, S2, R2, M1, C1

**Nice to have** (polish):
- B2, S1, R1, R3, M2, M3

**Optional** (if time):
- B3, S3, C2

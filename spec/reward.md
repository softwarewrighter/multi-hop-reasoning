# Reward Specification

## Goal

Compute a scalar reward that encourages:
1. Correct MCQ answers
2. Grounded reasoning (mentioning path entities)

While penalizing:
- Entity spam (reward hacking)
- Malformed output

---

## Formula

```
total = (W_corr × R_corr) + (W_path × R_path) - P_spam
```

Default weights:
- W_corr = 1.0
- W_path = 0.5

---

## Parsing

Model output must match:
```
TRACE: <reasoning text>
ANSWER: <A|B|C|D>
```

**If parsing fails**: `valid_format=false`, `total=-2.0`

---

## R_corr (Correctness)

| Condition | Value |
|-----------|-------|
| Correct answer | +1.0 |
| Wrong answer | -2.0 |

Asymmetric to discourage guessing.

---

## R_path (Path Alignment)

```python
E_path = set(path_star.entities)
E_trace = set(entities_in_trace)
hits = len(E_path & E_trace)
coverage = hits / len(E_path)

if hits < MIN_HITS:  # default: 2
    R_path = 0.0
else:
    R_path = min(coverage, MAX_PATH_REWARD)  # default max: 1.0
```

---

## P_spam (Anti-Spam Penalty)

```python
for entity in trace:
    if count(entity) > REPEAT_MAX:  # default: 2
        return REPEAT_PENALTY  # default: 0.5
return 0.0
```

---

## Entity Matching

- Dictionary matching using `kg.entities[*].id`
- Exact substring match with word boundaries
- No fuzzy matching (keep toy simple)

---

## Output Schema

```json
{
  "correctness": 1.0,
  "path_coverage": 0.67,
  "path_reward": 0.67,
  "spam_penalty": 0.0,
  "total": 1.34
}
```

---

## Test Cases

| Case | Expected |
|------|----------|
| Correct + 2 path entities + no spam | positive total |
| Correct + 0-1 path entities | ~1.0 (no path bonus) |
| Wrong + all path entities | negative (correctness dominates) |
| Correct + entity spammed 6x | penalty applied |
| Missing ANSWER: line | total=-2.0, valid_format=false |

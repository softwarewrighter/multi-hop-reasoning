# Video Script: KG as Reward Model

## YouTube Short (60-75 seconds)

---

### HOOK (0-5s)
**Visual**: Graph visualization with glowing golden path
**VO**: "We turned a knowledge graph into a reward model."

---

### PROBLEM (5-15s)
**Visual**: Base model example - wrong answer, random trace
**VO**: "Language models can answer questions..."

**Visual**: Highlight gibberish entities in trace
**VO**: "...but their reasoning? Often completely made up."

**Visual**: Zoom on negative reward bar
**VO**: "Wrong answer. Wrong path. Negative reward."

---

### SOLUTION (15-30s)
**Visual**: KG with path highlighted in gold
**VO**: "What if we rewarded the model for following actual chains of facts?"

**Visual**: Reward breakdown appears
**VO**: "Not just for the right answer..."

**Visual**: Path coverage bar fills
**VO**: "...but for the right reasoning."

**Visual**: Formula overlay: `reward = correctness + path_alignment`

---

### TRAINING MONTAGE (30-45s)
**Visual**: Accuracy curve rising, fast-forward
**VO**: "We trained on short chains - just 1 to 3 hops..."

**Visual**: SFT examples showing improvement
**VO**: "First supervised learning to learn the format..."

**Visual**: RSFT examples, high rewards
**VO**: "Then reward-based selection to learn the logic."

---

### PAYOFF (45-60s)
**Visual**: 5-hop question appears
**VO**: "Then we tested on longer chains it never saw."

**Visual**: Model answers, path lights up one node at a time
**VO**: "Five hops. All correct. Path fully grounded."

**Visual**: Full path glowing, reward bars maxed
**VO**: "It learned to compose facts, not just memorize."

---

### CTA (60-75s)
**Visual**: Side-by-side base vs trained metrics
**VO**: "28% accuracy to 61%. Path coverage up 4x."

**Visual**: Subscribe animation
**VO**: "Next: scaling this to bigger models. Subscribe to see what happens."

---

## Timing Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| Hook | 5s | 0:05 |
| Problem | 10s | 0:15 |
| Solution | 15s | 0:30 |
| Training | 15s | 0:45 |
| Payoff | 15s | 1:00 |
| CTA | 15s | 1:15 |

**Total: 75 seconds** (trim to 60s if needed by tightening training montage)

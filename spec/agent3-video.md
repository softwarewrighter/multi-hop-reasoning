# Agent 3: Video Production

## Mission

Turn the demo into educational video content:

1. **YouTube Short** (60–75 seconds) - primary deliverable
2. **Long-form explainer** (6–10 min) - optional follow-up

---

## Deliverables

### V1 — Script (Short)

**Structure** (60-75 seconds):

| Time | Section | Content |
|------|---------|---------|
| 0–5s | Hook | "We turned a knowledge graph into a reward model." |
| 5–15s | Problem | Model guesses; reasoning is fluff; show wrong answer example |
| 15–30s | Solution | Reward = correctness + path alignment; visual breakdown |
| 30–55s | Results | Before vs after; training montage; hard 5-hop question |
| 55–65s | Payoff | Model nails it; path highlights light up |
| 65–75s | CTA | "Next: we scale it up to bigger models." / Subscribe |

**Script file**: `video/script.md`

### V2 — Shot List

**Screen capture sequences from demo UI**:

| Shot # | Phase | Description | Duration |
|--------|-------|-------------|----------|
| 1 | base | Wrong answer, low path coverage, negative reward | 3s |
| 2 | base | Another wrong, random trace entities | 2s |
| 3 | sft | Better format, some path entities mentioned | 3s |
| 4 | sft | Correct answer, medium coverage | 3s |
| 5 | rsft | Correct + grounded, high coverage, low spam | 4s |
| 6 | rsft | Hard 5-hop question, model succeeds | 5s |
| 7 | metrics | Training curve animation (accuracy rising) | 4s |
| 8 | compare | Side-by-side base vs rsft same question | 5s |

**Overlays to add**:

| Overlay | Description |
|---------|-------------|
| "Correctness reward" | Arrow pointing to green bar |
| "Path reward" | Arrow pointing to blue bar |
| "Penalty" | Arrow pointing to red bar (when applicable) |
| "Before training" | Label on base phase |
| "After SFT" | Label on sft phase |
| "After RL-lite" | Label on rsft phase |

**Shot list file**: `video/shotlist.md`

### V3 — Capture Checklist

**Pre-capture setup**:

- [ ] Window size: 1920x1080 (or 1080x1920 for vertical Short)
- [ ] Browser zoom: 100%
- [ ] Clear browser tabs/bookmarks bar
- [ ] Dark mode enabled
- [ ] Demo server running
- [ ] Load run with best results

**Recording settings**:

- [ ] 60fps (for smooth scrubbing)
- [ ] High bitrate (50+ Mbps for lossless editing)
- [ ] System audio OFF
- [ ] Mouse cursor visible but not distracting

**Quality checks**:

- [ ] No tiny text (minimum 24px effective size)
- [ ] High contrast (dark background)
- [ ] KG graph nodes readable
- [ ] Reward bars clearly visible
- [ ] No notification pop-ups during recording

**Capture checklist file**: `video/capture_checklist.md`

### V4 — Edit Notes

**Pacing**:
- 2-3 seconds per example is ideal
- Speed ramp training curves (fast in middle, slow at start/end)
- Quick cuts between phases

**Audio**:
- Add sound cue when model hits correct answer (subtle ding)
- Background music: subtle, non-distracting
- Voiceover sync points marked in script

**Visual effects**:
- Highlight flash when correct answer appears
- Smooth transitions between phases
- Zoom on reward bars during explanation

**Cuts to make**:
- Remove any loading/buffering
- Cut anything not visually legible
- Remove redundant examples (keep best 2 per phase)

**Text overlays**:
- Sans-serif font (Inter, Roboto, etc.)
- High contrast (white on dark or with shadow)
- Position: lower third or upper corners
- Duration: 2-3 seconds minimum

**Edit notes file**: `video/edit_notes.md`

---

## Narrative Arc

### The Story (Short version)

1. **Problem**: LLMs guess answers but their "reasoning" is often made-up
2. **Insight**: A knowledge graph contains verified chains of facts
3. **Solution**: Use the KG path as a reward signal during training
4. **Mechanism**: Reward = did you get it right + did you follow the path
5. **Result**: Model learns to reason along the right path
6. **Proof**: Works on harder (longer) chains it never saw in training

### Key Visual Moments

| Moment | Visual | Emotional Beat |
|--------|--------|----------------|
| Problem reveal | Wrong answer with confident trace | Frustration |
| Aha moment | KG path highlighted, reward formula shown | Understanding |
| Training montage | Accuracy curve rising | Anticipation |
| Payoff | 5-hop correct with full path lit up | Satisfaction |

---

## Technical Requirements for Agent 2 (Demo)

To support video production, the demo should provide:

1. **Clean autoplay mode** - no user interaction needed during recording
2. **Phase labels** - clear visual indicator of current phase
3. **Episode counter** - "12/50" style display
4. **Pause capability** - manual pause for specific shots
5. **Dark theme** - essential for video quality
6. **Large fonts** - readable in 1080p without zooming
7. **Smooth transitions** - animated reward bars, fading between examples

---

## Recording Workflow

### For YouTube Short (Vertical 1080x1920)

1. Resize browser to 1080x1920
2. Adjust demo layout for vertical (stack elements)
3. Run demo autoplay
4. Record 3-4 minutes of footage (more than needed)
5. Edit down to 60-75 seconds

### For Landscape (1920x1080)

1. Standard browser window
2. Demo layout as designed (side-by-side)
3. Same capture process

---

## Example Script Draft

```markdown
# video/script.md

## Hook (0-5s)
[Graph visualization with glowing path]
VO: "We turned a knowledge graph into a reward model."

## Problem (5-15s)
[Show base model example - wrong answer]
VO: "Language models can answer questions, but their reasoning? Often made up."
[Highlight: random entities in trace, none on the true path]

## Solution (15-30s)
[Show KG with path highlighted]
VO: "What if we rewarded the model for following actual chains of facts?"
[Show reward breakdown: correctness + path alignment]
VO: "Not just for the right answer, but for the right reasoning."

## Training (30-45s)
[Montage: accuracy curve rising, examples improving]
VO: "We trained on short chains..."
[Show 1-2 hop examples succeeding]

## Payoff (45-60s)
[Show 5-hop question]
VO: "...and tested on longer ones it never saw."
[Model answers correctly, full path lights up]
VO: "It learned to compose facts, not just memorize answers."

## CTA (60-75s)
[Metrics comparison: base vs trained]
VO: "Next: scaling this up. Subscribe to see what happens."
```

---

## File Structure

```
video/
  script.md           # Full voiceover script with timing
  shotlist.md         # Detailed shot-by-shot breakdown
  capture_checklist.md # Pre-recording checklist
  edit_notes.md       # Post-production guidance
  assets/             # (optional) overlay graphics, thumbnails
```

---

## Success Criteria

1. **Script** is complete and timed (60-75s for Short)
2. **Shot list** covers all necessary footage
3. **Checklist** ensures quality capture
4. **Edit notes** enable efficient post-production
5. **All files** are in `video/` directory
6. Content is educational and engaging
7. Technical accuracy maintained (paper concepts correct)

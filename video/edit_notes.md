# Edit Notes

## General Guidelines

### Pacing
- 2-3 seconds per example is ideal
- Don't linger on any single screen
- Quick cuts between phases keep energy up
- Speed ramp curves: fast in middle, slow at inflection points

### Cuts to Make
- Remove all loading/buffering moments
- Cut anything not immediately legible
- Remove redundant examples (keep best 2 per phase)
- Trim dead air between actions

---

## Visual Effects

### Transitions
- Simple crossfades between phases (0.3s)
- Hard cuts within phases
- No spinning/3D transitions (distracting)

### Highlights
- Flash effect when correct answer appears
- Subtle glow on path nodes as they light up
- Zoom on reward bars during explanation (1.2x)

### Color Grading
- Boost contrast slightly
- Ensure dark background stays deep black
- Verify colors match between clips

---

## Text Overlays

### Style
- Font: Inter, Roboto, or similar sans-serif
- Color: White (#FFFFFF) with drop shadow
- Size: Minimum 48px for short, 36px for long-form
- Position: Lower third or upper corners
- Duration: Minimum 2 seconds

### Labels

| Label | When | Position |
|-------|------|----------|
| "Before Training" | Base phase clips | Bottom left |
| "After SFT" | SFT phase clips | Bottom left |
| "After RL-lite" | RSFT phase clips | Bottom left |
| "Correctness" | Reward explanation | Arrow to bar |
| "Path Alignment" | Reward explanation | Arrow to bar |
| "+1.0" / "-2.0" | When discussing values | Near bars |

### Formula Overlay
```
Reward = Correctness + Path Alignment - Penalty
```
- Appear word by word or highlight each term
- Duration: 3-4 seconds total

---

## Audio

### Voiceover
- Record separately in quiet environment
- Match pacing to visuals
- Leave room for breath/pauses at transitions
- Sync points marked in script

### Sound Effects

| Sound | Trigger | Volume |
|-------|---------|--------|
| Subtle ding | Correct answer appears | 30% |
| Soft buzz | Wrong answer appears | 20% |
| Chime | Each path node lights | 25% |
| Success fanfare | Full path complete | 40% |

- All SFX should be subtle, not jarring
- Consider removing if they feel annoying

### Music
- Subtle electronic/ambient
- Low volume (barely audible)
- No lyrics
- Fade out during key explanations
- Royalty-free (check license)

---

## Structure

### YouTube Short Structure

| Time | Content | Edit Notes |
|------|---------|------------|
| 0-5s | Hook + graph animation | Fast, attention-grabbing |
| 5-15s | Problem examples | Quick cuts, 2-3 clips |
| 15-30s | Solution explanation | Slow down, show formula |
| 30-45s | Training montage | Speed ramp curve animation |
| 45-60s | Payoff - 5 hop success | Build tension, release |
| 60-75s | CTA + metrics | Clean ending |

### Key Moments to Nail
1. **Hook** (0-3s): Must grab attention
2. **First correct** (around 35s): Emotional beat
3. **5-hop success** (around 50s): Climax
4. **Final metrics** (around 65s): Proof

---

## Export Settings

### YouTube Short
- Resolution: 1080x1920 (9:16)
- Frame rate: 60fps
- Bitrate: 20-30 Mbps
- Format: H.264 MP4
- Duration: 60-75 seconds max

### Landscape Version
- Resolution: 1920x1080 (16:9)
- Same other settings

---

## Review Checklist

Before final export:

- [ ] All text readable at mobile size
- [ ] No audio clipping
- [ ] Smooth transitions (no jarring cuts)
- [ ] Pacing feels right (not rushed, not slow)
- [ ] Hook grabs attention immediately
- [ ] Story arc is clear
- [ ] CTA is present
- [ ] Total duration within limits

---

## Version Control

- Save project after each major section
- Export drafts as `short_v1.mp4`, `short_v2.mp4`, etc.
- Keep notes on what changed between versions
- Get feedback before final export

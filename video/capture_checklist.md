# Capture Checklist

## Pre-Recording Setup

### System
- [ ] Close all unnecessary applications
- [ ] Disable notifications (Do Not Disturb)
- [ ] Disable screen savers and sleep
- [ ] Check disk space (need 10GB+ free for raw footage)

### Display
- [ ] Set resolution to 1920x1080 (landscape) or 1080x1920 (vertical Short)
- [ ] Browser zoom: 100%
- [ ] Clear browser history/bookmarks bar
- [ ] Hide URL bar if possible

### Demo Application
- [ ] Demo server running (`python demo/server.py`)
- [ ] Load run with best results
- [ ] Verify all phases have data (base, sft, rsft)
- [ ] Test autoplay works
- [ ] Test phase switching works

### Theme & Appearance
- [ ] Dark mode enabled
- [ ] High contrast colors visible
- [ ] All text readable at 1080p
- [ ] No tiny fonts in graph nodes

---

## Recording Settings

### Software
- [ ] OBS Studio or similar
- [ ] 60fps (for smooth scrubbing in edit)
- [ ] High bitrate: 50+ Mbps
- [ ] Format: MP4 or MOV
- [ ] Encoder: Hardware (NVENC/AMD/Apple) if available

### Audio
- [ ] System audio: OFF
- [ ] Microphone: OFF (record VO separately)

### Screen
- [ ] Window capture (not full screen) for flexibility
- [ ] Mouse cursor: Visible but not distracting
- [ ] No recording indicator overlay

---

## During Recording

### Quality Checks
- [ ] Verify recording is actually capturing
- [ ] No loading spinners or lag
- [ ] No notification pop-ups
- [ ] Graph renders smoothly
- [ ] Reward bars animate smoothly

### Capture Order

1. **Autoplay full run** (base → sft → rsft)
   - Let it run 3-4 minutes
   - More footage than needed

2. **Manual hero shots**
   - Base: Find worst examples
   - SFT: Find first correct answers
   - RSFT: Find 5-hop successes

3. **Metrics views**
   - Pause on metrics footer
   - Capture each phase's metrics

4. **Phase transitions**
   - Switch phases manually
   - Capture the tab/dropdown changing

5. **Graph close-ups**
   - Pause on good graph examples
   - Full path visible

---

## Post-Recording Verification

- [ ] Footage plays back correctly
- [ ] No dropped frames
- [ ] Audio is silent (as expected)
- [ ] Colors look correct
- [ ] Text is legible when viewed at 1080p
- [ ] File saved to correct location

---

## File Organization

```
video/
  raw/
    01_autoplay_full.mp4
    02_base_examples.mp4
    03_sft_examples.mp4
    04_rsft_examples.mp4
    05_metrics.mp4
    06_transitions.mp4
  assets/
    overlays/
    audio/
  exports/
    short_v1.mp4
```

---

## Backup

- [ ] Copy raw footage to backup drive
- [ ] Note timestamps of best shots
- [ ] Save project file after each session

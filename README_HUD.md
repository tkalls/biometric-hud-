# Real-Time Biometric Tracking HUD
> Dual-person face and hand tracking system with emotion detection, adaptive motion prediction, and a cyberpunk surveillance HUD — built entirely with OpenCV and MediaPipe, no game engine.

<!-- Drop your demo GIF here — it's the first thing people see -->
![HUD demo](assets/demo.gif)

---

## What it does

Welcome Buddy tracks two people simultaneously through a single webcam, renders a live biometric overlay for each, and classifies their emotional state in real time from 52 facial muscle coefficients. Every hand in frame gets a glowing landmark skeleton with ghosting trails and a predicted trajectory that physically adapts to how fast your hands are moving — slow hands get tight, conservative predictions; fast hands get a long arc projected ahead. The whole stack — tracking, inference, prediction, and rendering — runs in a single Python process at camera framerate with no external game engine or GPU requirement.

---

## Features

- **Dual-person tracking** — `FaceLandmarker` + `HandLandmarker` running async in `LIVE_STREAM` mode; detected faces sorted left-to-right and matched to registered user profiles each frame
- **Emotion classification** — 52 blendshape coefficients scored per face; EMA smoothing eliminates flicker between frames; outputs Happy · Sad · Angry · Surprised · Neutral
- **Adaptive motion prediction** — linear regression (`np.linalg.lstsq`) over recent index-tip history; history window length and prediction distance both scale dynamically with measured hand speed
- **4 prediction profiles** — swap live with `P`; each profile independently tunes ghost intensity, lookahead distance, history depth, and line colour (see table below)
- **Dynamic info panels** — each face gets its own panel; panels auto-flip left or right to whichever side has more room, with a straight connector line and 130 px gap
- **Ghosting + glow** — `cv2.addWeighted` trail blending; joint glow rings; profile-specific ghost alpha
- **Post-processing pipeline** — vignette, scanlines, and three background filters (Greyscale · Colour · Thermal) all applied in-frame with NumPy masks
- **CLI typewriter boot sequence** — the login form builds itself character-by-character before the HUD starts

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Face & hand tracking** | MediaPipe 0.10 Tasks API — `FaceLandmarker`, `HandLandmarker`, `LIVE_STREAM` async callbacks |
| **Emotion detection** | 52 facial blendshape coefficients, per-face `_FaceResultSlice` isolation, EMA smoothing |
| **Motion prediction** | `np.linalg.lstsq` linear regression, speed-scaled history window and lookahead distance |
| **Rendering** | OpenCV — frame capture, overlays, vignette, scanlines, colour filters |
| **HUD text** | Pillow — TrueType font rendering at multiple sizes (crisp at any resolution) |
| **Runtime** | Python 3.11 |

---

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python3.11 -m pip install -r requirements.txt
python3.11 camera_tracking_v2.py
```

> **macOS:** System Settings → Privacy & Security → Camera → enable Terminal on first run.
> Model files (`.task`) are downloaded automatically — no manual setup needed.

---

## Motion Prediction Profiles

Switch profiles live with `P`. Each profile is a different trade-off between stability and aggression.

| Profile | History window | Max lookahead | Ghost alpha | Best for |
|---|---|---|---|---|
| **Precise** | 4–8 frames | 2.0× | 0.55 | Slow, deliberate movements — tightest tracking |
| **Balanced** *(default)* | 6–10 frames | 2.7× | 0.62 | General use — smooth blend of history and prediction |
| **Futuristic** | 7–14 frames | 4.2× | 0.68 | Long ghost trails, high arc — best visual effect |
| **Aggressive** | 5–12 frames | 5.5× | 0.72 | Fast gestures — maximum lookahead, quickest response |

History window and prediction distance both stretch dynamically at runtime based on measured hand speed — the numbers above are baselines, not hard limits.

---

## Controls

| Key | Action |
|---|---|
| `P` | Cycle prediction profile |
| `F` | Cycle background filter (Greyscale → Colour → Thermal) |
| `1` | Toggle face bounding boxes |
| `2` | Toggle hand skeleton |
| `3` | Toggle info panels |
| `4` | Toggle scanlines |
| `✌` gesture | Toggle face blur |
| `Q` | Quit |

---

## Demo

![HUD demo](assets/demo.gif)

---

## License

MIT

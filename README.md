# Computer Vision Projects

Real-time hand and face tracking experiments built with MediaPipe, OpenCV, and Python.  
No mouse. No keyboard. Just your hands and a webcam.

---

## Projects

### 1 · Welcome Buddy — Biometric HUD (`camera_tracking_v2.py`)

A dual-person surveillance-style HUD with a cyberpunk aesthetic. Tracks two people simultaneously — face bounding boxes, live emotion detection, and a glowing hand skeleton with adaptive motion prediction.

**Features**
- **2-person tracking** — faces sorted left-to-right, each matched to a registered user profile
- **Emotion detection** — blendshape-based classifier; live mood label per person (Happy · Sad · Angry · Surprised · Neutral)
- **HandVisualizer** — glowing skeleton with ghosting trails and 4 prediction profiles:
  - `Precise` — tight, low-latency tracking
  - `Balanced` *(default)* — smooth blend of history and prediction
  - `Futuristic` — long ghost trail, high prediction arc
  - `Aggressive` — maximum prediction distance, fast response
- **Adaptive motion prediction** — linear regression (`np.linalg.lstsq`) on index-tip history; prediction distance and history window scale with hand speed
- **Dynamic info panels** — per-face panels auto-flip to the roomier side; straight connector line with 130 px gap
- **CLI typewriter boot animation** — form fields build character-by-character on launch
- **Post-processing** — vignette, scanlines, greyscale / colour / thermal filters

**Keys**

| Key | Action |
|-----|--------|
| `P` | Cycle hand profile (Precise → Balanced → Futuristic → Aggressive) |
| `F` | Cycle background filter |
| `1` | Toggle face bounding box |
| `2` | Toggle hand skeleton |
| `3` | Toggle info panels |
| `4` | Toggle scanlines |
| `✌` gesture | Toggle face blur |
| `Q` | Quit |

**Tech stack**

| Library | Role |
|---------|------|
| MediaPipe 0.10 (Tasks API) | `FaceLandmarker` + `HandLandmarker` in `LIVE_STREAM` mode; blendshapes for emotion |
| OpenCV | Camera capture, frame rendering, HUD overlays, filters |
| NumPy | Linear regression, EMA smoothing, vignette/scanline masks |
| Pillow | TrueType font rendering (crisp HUD text at any resolution) |

```bash
python3.11 camera_tracking_v2.py
```

---

### 2 · Virtual Physics Lab v2 — "Chaos Edition" (`virtual_physics_lab_v2.py`)

A real-time gesture-controlled physics sandbox. Spawn rigid bodies, flip gravity, and throw objects — entirely with hand gestures.

**Gestures**

| Gesture | Effect |
|---------|--------|
| Pinch (thumb + index) | Spawn a random polygon |
| Fist | Flip gravity direction |
| Open palm thrust | Shockwave — radial impulse on nearby objects |
| Fast finger flick | Velocity kick — physically launch objects you swipe through |

**Tech stack**

| Library | Role |
|---------|------|
| MediaPipe | Real-time hand landmark detection (21 landmarks per hand) |
| Pymunk | 2D rigid-body physics (Chipmunk 2D wrapper) |
| OpenCV | Camera capture and frame rendering |
| NumPy | Frame buffer manipulation |

```bash
python3.11 virtual_physics_lab_v2.py
```

---

## Installation

> Requires **Python 3.9–3.11**. MediaPipe does not support 3.12+.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

MediaPipe model files (`.task`) are downloaded automatically on first run.

**macOS — camera permission**  
System Settings → Privacy & Security → Camera → enable Terminal.

---

## License

MIT

# Virtual Physics Lab v2 — "Chaos Edition"

A real-time, gesture-controlled physics sandbox built with **MediaPipe**, **Pymunk**, and **OpenCV**.  
Control the simulation entirely with your hands — no mouse, no keyboard, no controllers.

> **Live demo GIF** — record one with `python record_demo.py` (see below) and drop it here.

---

## Features

| Gesture | Effect |
|---------|--------|
| **Pinch** (thumb + index) | Spawn a random polygon — triangle, pentagon, hexagon, or box |
| **Fist** (all fingers closed) | Flip gravity direction (up ↔ down) |
| **Open palm thrust** | Shockwave blast — radial impulse on everything nearby |
| **Fast finger flick** | Velocity kick — physically launch objects you swipe through |

- Two hands tracked simultaneously and independently
- Particle explosion system — burst on every spawn and flick impact
- Up to 45 live rigid bodies; oldest are culled automatically
- Kinematic finger bodies collide physically with dynamic objects
- Gravity toggle with full particle feedback burst

---

## How It Works

### 1. Hand Tracking — MediaPipe

MediaPipe's `Hands` model detects 21 **3D landmarks** per hand from a single RGB frame.  
Key landmarks used:

```
Thumb tip   → landmark 4
Index tip   → landmark 8
Middle tip  → landmark 12
Wrist root  → landmark 0
PIP joints  → landmarks 6, 10, 14, 18  (used for extended-finger test)
```

Each landmark is a normalised `(x, y, z)` coordinate.  
We multiply `x` and `y` by frame width/height to get pixel positions.

### 2. Gesture Classification

All gesture detection is purely geometric — no ML classifier beyond MediaPipe itself.

| Gesture | Rule |
|---------|------|
| **Extended finger** | `tip.y < pip.y` (tip is above middle knuckle in screen coords) |
| **Fist** | 0 non-thumb fingers extended |
| **Open palm** | ≥ 4 non-thumb fingers extended |
| **Pinch** | Euclidean distance(thumb tip, index tip) < 50 px |
| **Flick** | Finger positional delta × FPS > 350 px/s |

Rising-edge detection prevents accidental repeated triggers:  
a fist only flips gravity **once per close**, and a pinch only spawns **once per squeeze**.

### 3. Physics — Pymunk (Chipmunk 2D wrapper)

- **Dynamic bodies** — shapes affected by gravity, collisions, and impulses
- **Kinematic bodies** — finger-tip circles driven by hand position, not forces.  
  Setting `velocity = Δpos / dt` each frame lets Pymunk resolve collision impulses correctly
- **Static segments** — boundary walls rebuilt once at first frame when resolution is known
- **Paddle** — a live `Segment` between thumb–index and index–middle creates a "hand bat"

Physics impulse for a velocity flick:
```
magnitude = (1 - dist/radius) × finger_speed × 0.09 × object_mass
direction = normalised velocity vector
body.apply_impulse_at_world_point(direction × magnitude, body.position)
```

Shockwave uses a radial version of the same formula with `direction = (Δx/dist, Δy/dist)`.

### 4. Particle System

Particles are pure Python structs (using `__slots__` for speed).  
Each tick:
- Position updated by velocity
- Horizontal velocity multiplied by 0.90 (air drag)
- Vertical velocity incremented by 420 px/s² (particle gravity, always downward)
- Colour channel multiplied by `life / max_life` for fade-to-black effect

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| [OpenCV](https://opencv.org/) | 4.13 | Camera capture, frame rendering, HUD |
| [MediaPipe](https://mediapipe.dev/) | 0.10 | Real-time hand landmark detection |
| [Pymunk](http://www.pymunk.org/) | 7.2 | 2D rigid-body physics (Chipmunk wrapper) |
| [NumPy](https://numpy.org/) | 2.4 | Frame buffer manipulation, polygon drawing |

---

## Installation

> **Requires Python 3.9–3.11.** MediaPipe does not yet support 3.12+.

```bash
git clone https://github.com/YOUR_USERNAME/virtual-physics-lab.git
cd virtual-physics-lab

# Create a virtual environment (use python3.11 if 3.12+ is your default)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### macOS — camera permission

System Settings → Privacy & Security → Camera → enable Terminal (or your IDE).

---

## Usage

```bash
source .venv/bin/activate
python virtual_physics_lab_v2.py
```

Press **Q** to quit.

---

## Project Structure

```
virtual-physics-lab/
├── virtual_physics_lab_v2.py   # Main application (single file)
│   ├── CONFIG                  # Tunable constants
│   ├── Particle system         # Lightweight spark emitter
│   ├── Physics space           # Pymunk space + wall builder
│   ├── Dynamic objects         # Shape spawning + culling
│   ├── Kinematic bodies        # Finger-tip collision proxies
│   ├── Gesture detection       # Pure-geometry classifiers
│   └── Main loop               # Capture → detect → physics → render
├── requirements.txt
└── README.md
```

---

## Tuning Constants

All physics feel is controlled by constants at the top of the file:

| Constant | Default | Effect |
|----------|---------|--------|
| `PINCH_THRESH` | 50 px | How close thumb/index must be to trigger pinch |
| `FLICK_THRESH` | 350 px/s | Minimum finger speed to trigger impulse kick |
| `PALM_COOLDOWN` | 1.2 s | Minimum delay between shockwaves |
| `SPAWN_COOLDOWN` | 0.7 s | Minimum delay between shape spawns |
| `MAX_OBJECTS` | 45 | Hard cap on simultaneous rigid bodies |
| `GRAVITY_NORMAL` | (0, 900) px/s² | Downward gravity strength |

---

## What I Learned

- **MediaPipe landmark coordinate system** — normalised `(x, y)` with `z` as relative depth; `y` increases downward so "finger extended" means `tip.y < pip.y`
- **Kinematic vs dynamic bodies** — kinematic bodies must have their `velocity` set (not just `position`) for Pymunk to resolve collision impulses correctly
- **Rising-edge gesture gating** — storing previous-frame gesture state prevents one deliberate gesture from firing dozens of times
- **Particle system performance** — `__slots__` + in-place `list.pop()` avoids GC pressure in the hot loop
- **Frame-rate independence** — capping `dt = min(elapsed, 0.05)` prevents physics explosions when the process is paused or the camera stalls

---

## Potential Extensions

- [ ] Record and export GIF demos automatically
- [ ] Sound effects tied to gesture events (collisions, shockwave)
- [ ] Save/load scene state (pickle the body positions/velocities)
- [ ] Web version using MediaPipe JS + Matter.js
- [ ] Colour themes switchable by gesture (e.g., two-hand pinch)
- [ ] Gesture sequence recognition (pinch → fist → palm = special combo)

---

## License

MIT — do whatever you like with it.

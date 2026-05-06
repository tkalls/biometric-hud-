# Handy — Complete Coding Lesson

> **Project:** Real-time biometric tracking HUD (`camera_tracking_v2.py`)
> **Stack:** Python 3.11 · MediaPipe 0.10 · OpenCV · NumPy · Pillow
> **Repo:** https://github.com/tkalls/biometric-hud-

---

## Table of Contents

1. [[#What We Built]]
2. [[#System Architecture]]
3. [[#Tech Stack — The Why]]
4. [[#Threading Model]]
5. [[#MediaPipe Tasks API]]
6. [[#Face Landmarks & Blendshapes]]
7. [[#Hand Landmarks]]
8. [[#Emotion Detection]]
9. [[#Gesture Recognition]]
10. [[#EMA Smoothing]]
11. [[#Motion Prediction — Linear Regression]]
12. [[#The Rendering Pipeline]]
13. [[#Multi-Person Tracking]]
14. [[#HandVisualizer — Deep Dive]]
15. [[#Key Design Decisions]]
16. [[#SRS — Software Requirements Specification]]
17. [[#Use Cases]]
18. [[#Glossary]]

---

## What We Built

A real-time HUD (heads-up display) that runs on a standard webcam and does five things simultaneously at camera framerate:

1. Detects up to **2 faces** and draws a bounding box around each
2. Reads **52 facial muscle coefficients** per face and classifies emotion
3. Tracks up to **4 hands** with a glowing skeleton and ghosting trail
4. **Predicts where each hand is moving** using linear regression
5. Renders all of this on top of a stylised background (greyscale / colour / thermal)

No GPU required. No cloud API. Everything runs locally in a single Python process.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MAIN LOOP (30 fps)                  │
│                                                          │
│  Camera.read() ──► Filter.apply() ──► HUD overlays      │
│       ▲                                      │           │
│       │                              cv2.imshow()        │
│  ┌────┴────────────────────────┐             │           │
│  │  Camera Thread (daemon)     │     cv2.waitKey(1)      │
│  │  VideoCapture.read() loop   │             │           │
│  │  writes to shared _frame    │         _key()          │
│  └─────────────────────────────┘                         │
│                                                          │
│  Tracker.submit(frame) ──► MediaPipe async callbacks     │
│       │                                                  │
│       ├── FaceLandmarker.detect_async() ──► _on_face()   │
│       └── HandLandmarker.detect_async() ──► _on_hand()   │
│                         ▲                                │
│              Results written to shared state             │
│              under threading.Lock — read next frame      │
└─────────────────────────────────────────────────────────┘
```

Three things run concurrently:
- **Camera thread** — captures frames as fast as the hardware allows
- **MediaPipe async** — ML inference running in its own internal thread pool
- **Main loop** — reads the latest frame and latest results, renders, displays

---

## Tech Stack — The Why

### MediaPipe 0.10 (Tasks API)
The current Tasks API (`mediapipe.tasks`) replaced the older legacy API (`mediapipe.solutions`). Key differences:

| Legacy API | Tasks API |
|---|---|
| `mp.solutions.hands.Hands()` | `HandLandmarker.create_from_options()` |
| Blocking — waits for result | Async — fires a callback |
| One model at a time | Multiple models concurrently |
| Limited config | Full control over confidence thresholds |

We use `RunningMode.LIVE_STREAM` which means: *"I'll send you frames as fast as I can, call my callback when you have a result, don't block."* This is what allows the main loop to stay at 30fps even if ML inference takes 40ms.

### OpenCV (`cv2`)
- Handles camera capture (`VideoCapture`)
- All 2D drawing — lines, rectangles, circles, filled polygons
- Frame transforms — flip, colour conversion, blur
- The display window and keyboard input (`imshow` / `waitKey`)

### NumPy
- Every frame is just a `numpy.ndarray` of shape `(H, W, 3)` — we manipulate pixels directly
- Used for vignette and scanline masks (computed once, applied every frame with multiplication)
- Used for linear regression (`np.linalg.lstsq`) in motion prediction

### Pillow (PIL)
OpenCV's built-in text (`cv2.putText`) uses a bitmap font — it looks pixelated. PIL can render TrueType fonts (we use Menlo, the system monospace font) and alpha-blend them into the frame. The result is crisp text at any size.

### Python threading
MediaPipe's async mode fires callbacks on its own thread. Camera capture also runs on its own thread. `threading.Lock` is used everywhere shared state is read/written to prevent race conditions.

---

## Threading Model

This is the most important architecture concept in the project.

### The problem
Camera capture is slow (waits for the hardware). ML inference is slow (neural network forward pass). If both run on the main thread, the UI freezes between frames.

### The solution: producer-consumer pattern

```python
class Camera:
    def __init__(self):
        self._frame = None          # shared state
        self._lock  = threading.Lock()

    def _loop(self):                # runs on daemon thread
        while self._running:
            ok, f = self._cap.read()
            if ok:
                with self._lock:   # write
                    self._frame = f

    def read(self):                 # called from main thread
        with self._lock:           # read
            return (True, self._frame) if self._frame else (False, None)
```

The camera thread writes frames as fast as the hardware allows. The main loop reads the *latest* frame whenever it's ready. The `Lock` ensures they never read/write simultaneously (which would corrupt the array).

### Why daemon=True?
A daemon thread is automatically killed when the main program exits. Without `daemon=True`, the camera thread would keep running even after you press Q, and the program would hang.

---

## MediaPipe Tasks API

### How to initialise a landmarker

```python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

landmarker = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options    = mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode    = mp_vision.RunningMode.LIVE_STREAM,
        result_callback = self._on_face,     # ← called when result is ready
        num_faces       = 2,
        output_face_blendshapes = True,      # ← needed for emotion detection
    )
)
```

### How to send a frame

```python
def submit(self, frame: np.ndarray, ts_ms: int):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV is BGR, MediaPipe wants RGB
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    self._face_det.detect_async(mp_img, ts_ms)        # non-blocking
```

`ts_ms` is a monotonically increasing timestamp in milliseconds. MediaPipe uses it to discard stale results — if a newer frame was already processed, it won't fire the callback for the old one.

### The callback

```python
def _on_face(self, result, _img, _ts):
    with self._lock:
        self._face_res = result    # store latest result
```

This runs on MediaPipe's internal thread. Always use a Lock when writing shared state from a callback.

---

## Face Landmarks & Blendshapes

### What are face landmarks?
MediaPipe's `FaceLandmarker` returns **478 3D points** mapped to the face surface — eyes, nose, lips, jaw, cheeks. Each landmark is a normalised `(x, y, z)` coordinate:
- `x` and `y` are in `[0, 1]` relative to frame width/height
- `z` is relative depth (negative = closer to camera)
- Multiply by `w` and `h` to get pixel coordinates

### What are blendshapes?
Blendshapes are **52 named float scores** that describe facial muscle activations. Each score is in `[0, 1]`. Examples:

| Name | What it measures |
|---|---|
| `mouthSmileLeft` | Left corner of mouth pulled up |
| `mouthSmileRight` | Right corner of mouth pulled up |
| `browDownLeft` | Left brow pulled down (frowning) |
| `eyeBlinkLeft` | Left eye closed |
| `jawOpen` | Mouth open |
| `cheekPuff` | Cheeks puffed out |

These come directly from the neural network — no geometry calculations needed.

### Getting the bounding box from landmarks

```python
@staticmethod
def _face_bbox(landmarks, w, h, pad=18):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return (max(0, int(min(xs)) - pad),   # x1
            max(0, int(min(ys)) - pad),   # y1
            min(w, int(max(xs)) + pad),   # x2
            min(h, int(max(ys)) + pad))   # y2
```

We just take the min/max of all landmark x/y coordinates, add a small padding, and clip to the frame boundary.

---

## Hand Landmarks

MediaPipe returns **21 landmarks** per hand, indexed 0–20:

```
                 8   12  16  20
                 |   |   |   |
                 7   11  15  19
                 |   |   |   |
                 6   10  14  18
                 |   |   |   |
            4    5    9  13  17
            |    |
       3    |
       |
  2    |
  |
  1
  |
  0 (wrist)
```

Key indices:
- `0` — wrist
- `4` — thumb tip
- `8` — index tip ← used for motion prediction
- `12` — middle tip
- `16` — ring tip
- `20` — pinky tip
- Even numbers (4,8,12,16,20) — fingertips
- The intermediate joints: tip=`n`, DIP=`n-1`, PIP=`n-2`, MCP=`n-3`

### Finger extended test
A finger is extended if its **tip is above its PIP joint** (remember: y increases downward in screen space):

```python
index_extended = lms[8].y < lms[6].y   # tip y < PIP y = extended
```

---

## Emotion Detection

### The classifier

```python
@staticmethod
def _classify(face_result) -> str:
    s     = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
    smile = (s.get("mouthSmileLeft", 0) + s.get("mouthSmileRight", 0)) * 0.5
    frown = (s.get("mouthFrownLeft", 0) + s.get("mouthFrownRight", 0)) * 0.5
    brow  = (s.get("browDownLeft",   0) + s.get("browDownRight",   0)) * 0.5
    if smile > 0.28:                return "HAPPY"
    if frown > 0.18 or brow > 0.30: return "SAD"
    return "NEUTRAL"
```

We average left/right versions of each score to account for asymmetric expressions, then apply threshold rules in priority order.

### Majority-vote smoothing
Raw classification flickers because individual frames are noisy. We keep a rolling window of the last 18 classifications and return the most common one:

```python
class EmotionDetector:
    def __init__(self, window=18):
        self._history = deque(maxlen=window)

    def update(self, face_result) -> str:
        self._history.append(self._classify(face_result))
        return Counter(self._history).most_common(1)[0][0]
```

`Counter` counts occurrences. `most_common(1)[0][0]` returns the label with the highest count.

---

## Gesture Recognition

All gesture detection is **pure geometry** — no secondary ML model needed. We just look at the relative positions of the 21 hand landmarks.

### Peace sign

```python
@staticmethod
def _is_peace(lms) -> bool:
    index_up   = lms[8].y  < lms[6].y    # index tip above PIP
    middle_up  = lms[12].y < lms[10].y   # middle tip above PIP
    ring_down  = lms[16].y > lms[14].y   # ring tip BELOW PIP
    pinky_down = lms[20].y > lms[18].y   # pinky tip BELOW PIP
    return index_up and middle_up and ring_down and pinky_down
```

### Debouncing (cooldown)
Without debouncing, holding a gesture fires it dozens of times per second. We track the last trigger time per gesture:

```python
if check(lms) and now - self._last.get(name, 0) >= self._cooldown:
    triggered.add(name)
    self._last[name] = now    # reset the clock
```

This is called **rising-edge detection** — the gesture fires once when it becomes true, then is suppressed for `cooldown` seconds.

---

## EMA Smoothing

### The problem
Face bounding boxes jitter frame-to-frame because landmark detection has small per-frame variance. Displaying raw coordinates looks shaky.

### Exponential Moving Average (EMA)

```python
smoothed = alpha * new_value + (1 - alpha) * old_smoothed
```

- `alpha` close to 1 → fast response, more jitter (tracks movement quickly)
- `alpha` close to 0 → slow response, very smooth (lags behind movement)
- We use `alpha = 0.22` — smooth but not laggy

```python
def _smooth(self, idx: int, raw: list) -> list:
    a = Config.BOX_ALPHA   # 0.22
    if self._smooth_boxes[idx] is None:
        self._smooth_boxes[idx] = [float(v) for v in raw]
    else:
        self._smooth_boxes[idx] = [a * r + (1 - a) * s
                                   for r, s in zip(raw, self._smooth_boxes[idx])]
    return [int(v) for v in self._smooth_boxes[idx]]
```

We maintain one smoothed box per face slot (`idx=0` = leftmost, `idx=1` = rightmost).

---

## Motion Prediction — Linear Regression

### The idea
If we know where the index fingertip has been for the last N frames, we can fit a line through that history and extend it to predict where it's going.

### Linear regression with `np.linalg.lstsq`

```python
# pts = array of (x, y) positions over recent frames, shape (n, 2)
n = len(pts)
t = np.arange(n).reshape(-1, 1)          # time steps: [0, 1, 2, ..., n-1]
A = np.hstack([t, np.ones((n, 1))])      # design matrix: [[0,1],[1,1],[2,1],...]

x = pts[:, 0].astype(float)
y = pts[:, 1].astype(float)

mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]   # slope and intercept for x
my, cy = np.linalg.lstsq(A, y, rcond=None)[0]   # slope and intercept for y

next_t  = n + pred_distance               # time step to predict
pred_pt = (int(mx * next_t + cx),         # predicted x
           int(my * next_t + cy))         # predicted y
```

`lstsq` solves `A @ [m, c] = values` for the best-fit `m` (slope) and `c` (intercept). We solve it separately for x and y coordinates.

### Speed-adaptive parameters
Slow hands don't need aggressive prediction. Fast hands do. We measure speed as the mean pixel distance between recent frames:

```python
def _get_hand_speed(self, hand_id):
    pts    = np.array(self.history[hand_id])
    diffs  = np.diff(pts, axis=0)               # frame-to-frame deltas
    speeds = np.hypot(diffs[:, 0], diffs[:, 1]) # Euclidean distance each step
    return float(np.mean(speeds[-5:]))           # mean of last 5 steps
```

Then scale history window and prediction distance by speed:

```python
history_len   = clip(base_history   + speed * speed_history_mult,  4, max_history)
pred_distance = clip(base_pred_dist + speed * speed_pred_mult,      1.1, max_pred_dist)
```

---

## The Rendering Pipeline

Each frame goes through these stages in order:

```
Raw frame (BGR numpy array)
       │
       ▼
Filter.apply()          ← greyscale / colour / thermal
       │
       ▼
HUD.apply_vignette()    ← darken edges (NumPy multiply)
       │
       ▼
HUD.apply_scanlines()   ← dim every other row (NumPy multiply)
       │
       ▼
_render_face()          ← face boxes + info panels (always vivid lime)
       │
       ▼
HandVisualizer          ← ghosting + glowing skeleton + prediction
       │
       ▼
SignalWidget.draw()     ← waveform in bottom-left
       │
       ▼
HUD.header()            ← FPS + filter name top-left
       │
       ▼
HUD.menu()              ← key hints bottom-right
       │
       ▼
cv2.imshow()            ← display
```

### Why vignette and scanlines are fast
Both effects are pre-computed as NumPy masks on the first frame and reused every frame. Applying them is just one multiply:

```python
# Vignette — float mask, 1.0 at centre, ~0.1 at corners
display[:] = (display.astype(np.float32) * self._vign_mask).astype(np.uint8)

# Scanlines — 1.0 on even rows, 0.70 on odd rows
display[:] = (display.astype(np.float32) * self._scan_mask).astype(np.uint8)
```

### PIL text blending
TrueType text is rendered onto a transparent RGBA canvas, then alpha-composited into the frame:

```python
alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
rgb   = crop[:, :, :3][:, :, ::-1].astype(np.float32)  # RGBA → BGR
roi   = img[y0:y1, x0:x1].astype(np.float32)
img[y0:y1, x0:x1] = (roi * (1.0 - alpha) + rgb * alpha).astype(np.uint8)
```

Standard alpha compositing: `output = background * (1 - alpha) + foreground * alpha`

---

## Multi-Person Tracking

### The challenge
MediaPipe returns faces in arbitrary order. We need to consistently assign face index 0 to the leftmost person and index 1 to the rightmost — otherwise names/panels swap when one person moves.

### Solution: sort by x-centre every frame

```python
faces = face_res.face_landmarks
order = sorted(range(len(faces)),
               key=lambda i: float(np.mean([lm.x for lm in faces[i]])))
# order[0] = index of leftmost face
# order[1] = index of rightmost face
```

### `_FaceResultSlice`
`EmotionDetector` always reads `face_blendshapes[0]`. For multi-face support, we need it to read the blendshapes of the correct face. `_FaceResultSlice` is a thin adapter:

```python
class _FaceResultSlice:
    __slots__ = ("face_blendshapes",)
    def __init__(self, face_res, idx: int):
        self.face_blendshapes = (
            [face_res.face_blendshapes[idx]]   # wrap the correct face's data at index 0
            if face_res.face_blendshapes and idx < len(face_res.face_blendshapes)
            else []
        )
```

When we call `self._emotions[slot].update(_FaceResultSlice(face_res, face_idx))`, the emotion detector sees its data at `[0]` as expected, but it's actually the correct face's data.

---

## HandVisualizer — Deep Dive

### Ghosting
```python
if self.prev_overlay is not None:
    blended = cv2.addWeighted(display, 1 - ghost_alpha,
                              self.prev_overlay, ghost_alpha, 0)
    np.copyto(display, blended)
```

`cv2.addWeighted(src1, alpha, src2, beta, gamma)` computes `alpha*src1 + beta*src2 + gamma` pixel-by-pixel. We blend the current frame with the previous frame's overlay — the hand skeleton "fades" over several frames.

### The 4 profiles

| Parameter | What it does |
|---|---|
| `max_history` | Max frames of tip position to store (deque maxlen) |
| `base_history` | Starting regression window at zero speed |
| `speed_history_mult` | How fast the window grows with speed |
| `base_pred_distance` | Starting lookahead at zero speed |
| `speed_pred_mult` | How fast lookahead grows with speed |
| `max_pred_distance` | Cap on lookahead (prevents wild predictions) |
| `ghost_alpha` | How strongly previous frame bleeds into current |
| `line_color` | BGR colour for the prediction line |

### Confidence rings
After predicting the tip position, we draw ring(s) at the predicted point to indicate tracking confidence:

```python
cv2.circle(display, pred_pt, 11, Config.LIME, 1)         # always
if confidence > 0.85:
    cv2.circle(display, pred_pt, 14, (200, 255, 255), 1)  # extra ring if high confidence
```

Confidence is measured by how many key joints are visible within the frame boundary:

```python
visible = sum(1 for i in [0, 4, 8, 12, 16, 20]
              if 0.03 < hand_lms[i].x < 0.97 and 0.03 < hand_lms[i].y < 0.97)
confidence = visible / 6.0
```

---

## Key Design Decisions

### 1. Decouple background from overlays
The single most important aesthetic decision. Overlays are drawn *after* the filter is applied, so they're always vivid lime regardless of what the background looks like. A thermal background with greyscale overlays would look wrong; thermal background + vivid lime overlays looks like a real IR HUD.

### 2. Async over synchronous
Using `LIVE_STREAM` + `detect_async` instead of `IMAGE` mode keeps the main loop from waiting for ML inference. The HUD runs at camera framerate; inference results arrive slightly delayed but never block the render loop.

### 3. One `threading.Lock` per shared resource
The camera has its own lock. The tracker has its own lock. They never contend with each other. This is simpler and safer than one global lock.

### 4. `deque(maxlen=N)` for all rolling windows
`deque` with a `maxlen` automatically discards old entries. Used for FPS timestamps, emotion history, hand tip history, and signal widget data. O(1) append and pop from both ends.

### 5. Panel side detection
The info panel auto-flips to whichever side of the face has more room:

```python
space_r  = w - x2      # pixels to the right of the face
space_l  = x1          # pixels to the left of the face
to_right = space_r >= space_l   # pick the roomier side

# Hard-clip: if chosen side still doesn't fit, flip
if to_right and x2 + GAP + PANEL_W > w: to_right = False
if not to_right and x1 - GAP - PANEL_W < 0: to_right = True
```

---

## SRS — Software Requirements Specification

### Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | System shall detect and track up to 2 faces simultaneously in real time |
| FR-02 | System shall detect and track up to 4 hands simultaneously |
| FR-03 | System shall classify emotion per tracked face (Happy / Sad / Neutral) |
| FR-04 | System shall render a named info panel tethered to each detected face |
| FR-05 | System shall draw a glowing hand skeleton with ghosting trail |
| FR-06 | System shall predict the trajectory of each index fingertip |
| FR-07 | System shall support 4 motion prediction profiles switchable at runtime |
| FR-08 | System shall recognise peace sign and thumbs-down gestures with 1.5s cooldown |
| FR-09 | System shall support 3 background filters: Greyscale, Colour, Thermal |
| FR-10 | System shall display FPS, current filter, and key hints in the HUD |
| FR-11 | System shall animate a CLI typewriter login sequence on startup |
| FR-12 | System shall support 2-user login with Name, Age, Location fields |

### Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | Main loop must sustain ≥ 25 fps on Apple Silicon MacBook Air |
| NFR-02 | ML inference must not block the render loop |
| NFR-03 | All model files must be auto-downloaded on first run |
| NFR-04 | System must run on Python 3.9–3.11 with no GPU requirement |
| NFR-05 | HUD text must be crisp (TrueType) at 1280×720 |

### Constraints
- MediaPipe 0.10 does not support Python 3.12+
- macOS requires explicit camera permission granted to Terminal
- Face blendshapes require `output_face_blendshapes=True` in `FaceLandmarkerOptions`

---

## Use Cases

### UC-01 — Two-person login and identification
**Actor:** Two users sit in front of the camera
**Flow:** System detects both faces, sorts them L→R, assigns User 1 / User 2, displays name panel for each. If one person leaves, only one panel renders.

### UC-02 — Live emotion monitoring
**Actor:** Single user in a study/work session
**Flow:** System continuously classifies facial blendshapes, smooths over 18 frames, displays current mood label. Could be extended to log mood over time.

### UC-03 — Gesture-controlled UI
**Actor:** User with hands in frame
**Flow:** Peace sign toggles face blur (privacy mode). Thumbs-down cycles background filter. 1.5s cooldown prevents accidental double-triggers.

### UC-04 — Motion prediction demo
**Actor:** Demonstrating computer vision to an audience
**Flow:** Switch to Futuristic or Aggressive profile with `P`. Move hands fast — the prediction arc becomes very visible, showing how the system extrapolates trajectory from history.

### UC-05 — Surveillance aesthetic recording
**Actor:** Content creator, portfolio builder
**Flow:** Run in Greyscale or Thermal mode, screen-record the output. The HUD aesthetic is the demo artefact.

### UC-06 — CV / portfolio showcase
**Actor:** Developer at an interview or on GitHub
**Flow:** The README + GIF shows: real-time ML inference, multi-threading, signal processing (EMA), linear algebra (regression), and custom rendering — all in one project.

---

## Extending the Project

### Ideas you can build next

```
[ ] Log emotion to CSV — time-series of mood over a session
[ ] Pinch gesture to take a screenshot
[ ] Add face recognition (face_recognition library) — actual ID instead of login form
[ ] WebSocket server — broadcast hand positions to a browser canvas
[ ] Add depth estimation — use the z blendshape coordinate to infer distance
[ ] Kalman filter — replace linear regression with a proper motion model
[ ] Multi-camera support — stitch two feeds together for wider FOV
[ ] Audio feedback — play a tone when a gesture fires
```

---

## Glossary

| Term | Definition |
|---|---|
| **Blendshape** | A named float score (0–1) representing a specific facial muscle activation |
| **Landmark** | A normalised (x, y, z) point on a face or hand surface |
| **EMA** | Exponential Moving Average — weighted average that decays older values |
| **LIVE_STREAM mode** | MediaPipe async mode for video — non-blocking, callback-based |
| **Daemon thread** | A background thread that dies automatically when the main program exits |
| **Race condition** | When two threads read/write shared data simultaneously, causing corruption |
| **lstsq** | Least squares — finds the line of best fit through a set of points |
| **Alpha compositing** | Blending two images using a transparency (alpha) channel |
| **Rising-edge detection** | Firing an event only on the transition from false→true, not while held |
| **Vignette** | Darkening the edges of an image — used in film and HUD aesthetics |
| **BGR** | OpenCV's default colour order: Blue-Green-Red (opposite of standard RGB) |
| **deque** | Double-ended queue — O(1) append/pop from both ends; `maxlen` auto-discards old items |

---

*Built May 2026 · Python 3.11 · MediaPipe 0.10 · OpenCV · NumPy · Pillow*

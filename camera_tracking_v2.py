"""
CyberVision HUD  —  camera_tracking_v2.py
==========================================
Real-time face + emotion + hand tracking.  Premium surveillance HUD aesthetic.

Visual design
  - GREYSCALE mode : greyscale background  +  vivid lime overlays  (default)
  - COLOR mode     : full-colour RGB feed  +  vivid lime overlays
  - THERMAL mode   : FLIR-Iron colourmap   +  vivid lime overlays on top
  Decoupling background from overlays is what makes this look premium
  (like a real IR camera HUD, not an Instagram filter).

  - Solid lime rectangle face box, tethered with a visible connector line
  - Panel: dark tint + all fields as 'Label: Value', thin lime border (no header bar)
  - HD text via PIL TrueType (Menlo) — crisp, not blurry OpenCV bitmap
  - Scanlines always on by default

Gestures  (1.5 s cooldown — no flicker from held gestures)
  ✌  Peace sign   → toggle face-blur (obscures face ROI behind HUD elements)
  👎  Thumbs down  → cycle background mode  GREYSCALE → COLOR → THERMAL

Keys
  F   Cycle filter  GREYSCALE → COLOR → THERMAL
  1   Face box
  2   Hands
  3   Info panel
  4   Scanlines
  M   Toggle menu
  Q   Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os
import urllib.request
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    WIN_TITLE = "CyberVision  ◈  Surveillance HUD"

    # colours (BGR)
    LIME       = (57,  255,  20)
    LIME_DIM   = (18,  110,   7)
    BLACK      = (  0,   0,   0)
    MOOD_COL   = {
        "HAPPY"  : (57,  255,  20),
        "NEUTRAL": (210, 210, 210),
        "SAD"    : (255, 130,  50),
    }

    CAM_INDEX, CAM_W, CAM_H, CAM_FPS = 0, 1280, 720, 30

    _DIR          = os.path.dirname(os.path.abspath(__file__))
    FACE_LM_MODEL = os.path.join(_DIR, "face_landmarker.task")
    HAND_MODEL    = os.path.join(_DIR, "hand_landmarker.task")
    FACE_LM_URL   = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                     "face_landmarker/float16/1/face_landmarker.task")
    HAND_URL      = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                     "hand_landmarker/float16/1/hand_landmarker.task")

    from mediapipe.tasks.python import vision as _mpv
    HAND_CONNECTIONS = [(c.start, c.end)
                        for c in _mpv.HandLandmarksConnections.HAND_CONNECTIONS]

    # Filter order — F key and thumbs-down gesture both cycle this list
    FILTERS          = ["GREYSCALE", "COLOR", "THERMAL"]
    BOX_ALPHA        = 0.22   # EMA alpha for face-box smoothing
    MOOD_WINDOW      = 18     # majority-vote window (frames)
    GESTURE_COOLDOWN = 1.5    # seconds between gesture triggers (debounce)

    FONT_PATHS = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  HD TEXT RENDERER  —  PIL TrueType, blended into the frame
# ══════════════════════════════════════════════════════════════════════════════
class TextRenderer:
    """
    Renders anti-aliased TrueType text directly into a numpy BGR frame.
    Only the per-string ROI is touched — no full-frame BGR↔RGB round-trips.
    """

    SIZES = {"xs": 12, "sm": 14, "md": 17, "lg": 20, "xl": 26, "hd": 32}

    def __init__(self):
        path = next((p for p in Config.FONT_PATHS if os.path.exists(p)), None)
        if path:
            try:
                self._fonts = {k: ImageFont.truetype(path, v, index=0)
                               for k, v in self.SIZES.items()}
                self._ok = True
                print(f"  TextRenderer: loaded '{os.path.basename(path)}'", flush=True)
                return
            except Exception as e:
                print(f"  TextRenderer: font load failed ({e}), using OpenCV fallback",
                      flush=True)
        self._fonts = {}
        self._ok    = False

    def put(self, img: np.ndarray, text: str,
            xy: tuple, size: str = "md",
            color: tuple = Config.LIME) -> None:
        """Draw HD text on img in-place.  xy = (x, bottom-y) like cv2.putText."""
        if not self._ok or not text:
            scale = {"xs": .35, "sm": .42, "md": .50, "lg": .60, "xl": .75, "hd": .90}
            cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                        scale.get(size, 0.5), color, 1, cv2.LINE_AA)
            return

        font = self._fonts[size]
        r, g, b = color[2], color[1], color[0]   # BGR → RGB
        bbox = font.getbbox(text)
        if bbox is None or bbox[2] <= bbox[0]:
            return
        tw = bbox[2] - bbox[0] + 2
        th = bbox[3] - bbox[1] + 2
        canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text(
            (1 - bbox[0], 1 - bbox[1]), text, font=font, fill=(r, g, b, 255))
        x = int(xy[0])
        y = int(xy[1]) - th + 2
        self._blend(img, np.array(canvas), x, y)

    def text_width(self, text: str, size: str = "md") -> int:
        if not self._ok:
            return len(text) * 8
        bbox = self._fonts[size].getbbox(text)
        return (bbox[2] - bbox[0]) if bbox else 0

    @staticmethod
    def _blend(img: np.ndarray, canvas_rgba: np.ndarray, x: int, y: int) -> None:
        h, w = img.shape[:2]
        ch, cw = canvas_rgba.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + cw), min(h, y + ch)
        if x1 <= x0 or y1 <= y0:
            return
        crop  = canvas_rgba[y0-y: y1-y, x0-x: x1-x]
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = crop[:, :, :3][:, :, ::-1].astype(np.float32)
        roi   = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (roi * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  FPS COUNTER
# ══════════════════════════════════════════════════════════════════════════════
class FPSCounter:
    def __init__(self, window: int = 45):
        self._ts: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA  —  dedicated background capture thread
# ══════════════════════════════════════════════════════════════════════════════
class Camera:
    def __init__(self):
        self._cap = cv2.VideoCapture(Config.CAM_INDEX)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_H)
        self._cap.set(cv2.CAP_PROP_FPS,          Config.CAM_FPS)
        if not self._cap.isOpened():
            raise RuntimeError("Camera unavailable — System Settings → Privacy → Camera")
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False

    def start(self) -> "Camera":
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def read(self):
        with self._lock:
            return (False, None) if self._frame is None else (True, self._frame)

    def stop(self) -> None:
        self._running = False
        self._cap.release()

    def _loop(self) -> None:
        while self._running:
            ok, f = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = f


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER  —  async MediaPipe LIVE_STREAM (non-blocking)
# ══════════════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self._lock     = threading.Lock()
        self._face_res = None
        self._hand_res = None

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._face_det = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options                  = mp_python.BaseOptions(
                                                    model_asset_path=Config.FACE_LM_MODEL),
                running_mode                  = mp_vision.RunningMode.LIVE_STREAM,
                result_callback               = self._on_face,
                num_faces                     = 2,
                min_face_detection_confidence = 0.55,
                min_face_presence_confidence  = 0.55,
                min_tracking_confidence       = 0.55,
                output_face_blendshapes       = True,
            )
        )
        self._hand_det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options                  = mp_python.BaseOptions(
                                                    model_asset_path=Config.HAND_MODEL),
                running_mode                  = mp_vision.RunningMode.LIVE_STREAM,
                result_callback               = self._on_hand,
                num_hands                     = 4,
                min_hand_detection_confidence = 0.60,
                min_hand_presence_confidence  = 0.60,
                min_tracking_confidence       = 0.60,
            )
        )

    def submit(self, frame: np.ndarray, ts_ms: int) -> None:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._face_det.detect_async(mp_img, ts_ms)
        self._hand_det.detect_async(mp_img, ts_ms)

    def results(self):
        with self._lock:
            return self._face_res, self._hand_res

    def close(self) -> None:
        self._face_det.close()
        self._hand_det.close()

    def _on_face(self, result, _img, _ts) -> None:
        with self._lock:
            self._face_res = result

    def _on_hand(self, result, _img, _ts) -> None:
        with self._lock:
            self._hand_res = result


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION DETECTOR  —  FaceLandmarker blendshapes + majority-vote smoothing
# ══════════════════════════════════════════════════════════════════════════════
class EmotionDetector:
    def __init__(self, window: int = Config.MOOD_WINDOW):
        self._history: deque = deque(maxlen=window)

    def update(self, face_result) -> str:
        self._history.append(self._classify(face_result))
        return Counter(self._history).most_common(1)[0][0]

    @staticmethod
    def _classify(face_result) -> str:
        if (not face_result
                or not face_result.face_blendshapes
                or not face_result.face_blendshapes[0]):
            return "NEUTRAL"
        s     = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
        smile = (s.get("mouthSmileLeft", 0) + s.get("mouthSmileRight", 0)) * 0.5
        frown = (s.get("mouthFrownLeft", 0) + s.get("mouthFrownRight", 0)) * 0.5
        brow  = (s.get("browDownLeft",   0) + s.get("browDownRight",   0)) * 0.5
        if smile > 0.28:               return "HAPPY"
        if frown > 0.18 or brow > 0.30: return "SAD"
        return "NEUTRAL"


# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE RECOGNIZER  —  peace sign + thumbs-down, debounced
# ══════════════════════════════════════════════════════════════════════════════
class GestureRecognizer:
    """
    Detects gestures from MediaPipe hand landmarks and returns triggered
    gesture names.  Each gesture has an independent cooldown so holding
    a gesture only fires once per cooldown window — no flickering.

    Supported gestures
    ------------------
    'peace'       — index + middle extended, ring + pinky curled
    'thumbs_down' — thumb pointing down, all 4 fingers curled
    """

    def __init__(self, cooldown: float = Config.GESTURE_COOLDOWN):
        self._cooldown = cooldown
        self._last: dict = {}      # gesture → last trigger timestamp

    def update(self, hand_result) -> set:
        """Return a set of gesture names triggered this frame."""
        if not hand_result or not hand_result.hand_landmarks:
            return set()
        triggered = set()
        now = time.time()
        for lms in hand_result.hand_landmarks:
            for name, check in [("peace",       self._is_peace),
                                 ("thumbs_down", self._is_thumbs_down)]:
                if check(lms) and now - self._last.get(name, 0) >= self._cooldown:
                    triggered.add(name)
                    self._last[name] = now
        return triggered

    # ── Gesture classifiers ───────────────────────────────────────────────────
    @staticmethod
    def _is_peace(lms) -> bool:
        """Index + middle up, ring + pinky down."""
        index_up   = lms[8].y  < lms[6].y
        middle_up  = lms[12].y < lms[10].y
        ring_down  = lms[16].y > lms[14].y
        pinky_down = lms[20].y > lms[18].y
        return index_up and middle_up and ring_down and pinky_down

    @staticmethod
    def _is_thumbs_down(lms) -> bool:
        """Thumb tip below thumb base; at least 3 of 4 fingers curled."""
        fingers_curled = sum(
            1 for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)]
            if lms[t].y > lms[p].y
        ) >= 3
        thumb_down = lms[4].y > lms[2].y   # tip below metacarpal
        return fingers_curled and thumb_down


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER  —  professional visual pipelines applied to the background layer
# ══════════════════════════════════════════════════════════════════════════════
class Filter:
    """
    GREYSCALE: greyscale base — lime overlays pop with maximum contrast.
    COLOR:     full RGB feed — useful for reference / recording.
    THERMAL:   percentile stretch → aggressive CLAHE → bilateral sharpen →
               FLIR-Iron LUT.  Bilateral filter preserves hard object/face edges
               while smoothing flat areas — exactly how real IR sensors look.
               No Gaussian blur = no smear = crisp colour boundaries.
    """

    # Two CLAHE instances — thermal uses tighter tiles for finer local contrast
    _clahe_night    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    _clahe_thermal  = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    _PHOTO_LUT: np.ndarray = None   # Apple Photo Booth palette (built once)

    @classmethod
    def _build_photobooth_lut(cls) -> np.ndarray:
        """
        1:1 Apple Photo Booth 'Thermal Camera' LUT.
        Maps grayscale luminance → Dark Blue → Magenta → Yellow → Green → Cyan.
        Stored as (256, 1, 3) uint8 BGR for cv2.LUT.
        """
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            if i < 64:
                b, g, r = 255, 0, i * 4
            elif i < 128:
                b = 255 - (i - 64) * 4
                g = (i - 64) * 4
                r = 255
            elif i < 192:
                b = 0
                g = 255
                r = 255 - (i - 128) * 4
            else:
                b = (i - 192) * 4
                g = 255
                r = 0
            lut[i, 0] = [b, g, r]   # BGR
        return lut

    @classmethod
    def apply(cls, frame: np.ndarray, mode: str) -> np.ndarray:
        if mode == "THERMAL":
            return cls._thermal(frame)
        if mode == "GREYSCALE":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Also apply CLAHE to greyscale so it looks punchy, not flat
            eq = cls._clahe_night.apply(gray)
            return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        return frame.copy()     # COLOR — untouched RGB

    @classmethod
    def _thermal(cls, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 1. Percentile histogram stretch (2 %–98 %) ────────────────────
        lo, hi = np.percentile(gray, (2, 98))
        if hi > lo:
            gray = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255,
                           0, 255).astype(np.uint8)

        # ── 2. Aggressive CLAHE ───────────────────────────────────────────
        eq = cls._clahe_thermal.apply(gray)

        # ── 3. Slight Gaussian blur BEFORE colormap ───────────────────────
        #    This is the secret to the Apple Photo Booth look — smooth
        #    gradient transitions, chunky vibrant colour boundaries.
        blurred = cv2.GaussianBlur(eq, (5, 5), sigmaX=1.5)

        # ── 4. Apple Photo Booth LUT (Blue→Magenta→Yellow→Green→Cyan) ────
        if cls._PHOTO_LUT is None:
            cls._PHOTO_LUT = cls._build_photobooth_lut()
        return cv2.LUT(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), cls._PHOTO_LUT)


# ══════════════════════════════════════════════════════════════════════════════
#  HUD  —  drawing engine
# ══════════════════════════════════════════════════════════════════════════════
class HUD:
    def __init__(self, tr: TextRenderer):
        self._tr = tr

    # ── Solid face bounding box ───────────────────────────────────────────────
    @staticmethod
    def face_box(img, x1, y1, x2, y2,
                 color=Config.LIME, t=2) -> None:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)

    # ── Info panel — matches reference: dark box, Label: Value, no header bar ─
    def info_panel(self, img, x: int, y: int,
                   info: dict, mood: str) -> tuple:
        """
        Returns (bx1, by1, bx2, by2) so the caller can draw the connector line.
        Layout:  dark semi-transparent box, thin lime border, all-lime text,
                 each row:  'Label: Value'  (matches Kyle Powis reference).
        """
        lh  = 26
        pad = 14
        pw  = 240

        rows = list(info.items())   # preserves insertion order
        ph   = len(rows) * lh + pad * 2

        H, W = img.shape[:2]
        x    = int(np.clip(x, 4, W - pw - 4))
        y    = int(np.clip(y, 4, H - ph - 4))

        bx1 = int(np.clip(x - pad,      0, W))
        bx2 = int(np.clip(x - pad + pw, 0, W))
        by1 = int(np.clip(y,            0, H))
        by2 = int(np.clip(y + ph,       0, H))

        # ── Dark semi-transparent background ──────────────────────────────
        roi = img[by1:by2, bx1:bx2]
        if roi.size:
            dark = np.zeros_like(roi)
            cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
            img[by1:by2, bx1:bx2] = roi

        # ── Rows: centered "Label: Value" ─────────────────────────────────
        panel_cx = (bx1 + bx2) // 2
        ty = by1 + pad + lh
        for k, v in rows:
            if k == "Mood":
                v   = mood
                col = Config.MOOD_COL.get(mood, Config.LIME)
            else:
                col = Config.LIME
            label = f"{k}: {v}"
            tx    = panel_cx - self._tr.text_width(label, "sm") // 2
            self._tr.put(img, label, (tx, ty), size="sm", color=col)
            ty += lh

        # ── Thin lime border ───────────────────────────────────────────────
        cv2.rectangle(img, (bx1, by1), (bx2, by2), Config.LIME, 1, cv2.LINE_AA)

        return bx1, by1, bx2, by2

    # ── Connector line: face-box edge ↔ panel ────────────────────────────────
    @staticmethod
    def connector(img,
                  fx1, fy1, fx2, fy2,
                  pbx1, pby1, pbx2, pby2,
                  side: str = "right") -> None:
        """
        Straight horizontal line from face-box mid-edge to panel mid-edge.
        Filled dots at both anchor points.
        side='right': face-box right → panel left
        side='left' : face-box left  → panel right
        """
        mid_y = (fy1 + fy2) // 2
        if side == "right":
            p0 = (fx2,  mid_y)
            p1 = (pbx1, (pby1 + pby2) // 2)
        else:
            p0 = (fx1,  mid_y)
            p1 = (pbx2, (pby1 + pby2) // 2)

        cv2.line(img, p0, p1, Config.LIME, 2, cv2.LINE_AA)
        cv2.circle(img, p0, 4, Config.LIME, -1)
        cv2.circle(img, p1, 4, Config.LIME, -1)

    # ── Hand skeleton ─────────────────────────────────────────────────────────
    @staticmethod
    def hand_skeleton(img, result, w: int, h: int) -> None:
        if not result or not result.hand_landmarks:
            return
        TIP = {0, 4, 8, 12, 16, 20}
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in Config.HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], Config.LIME, 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                r = 5 if i in TIP else 3
                cv2.circle(img, pt, r, Config.LIME,  -1)
                cv2.circle(img, pt, r, Config.BLACK,  1)   # thin black ring

    # ── Header bar ───────────────────────────────────────────────────────────
    def header(self, img, fps: float, filter_mode: str,
               face_blur: bool = False) -> None:
        W = img.shape[1]
        ts = time.strftime("%Y-%m-%d   %H:%M:%S")
        self._tr.put(img, ts, (14, 26), size="sm")

        label = f"[ {filter_mode} ]"
        if face_blur:
            label += "  [ FACE BLUR ]"
        lw = self._tr.text_width(label, "md")
        self._tr.put(img, label, (W // 2 - lw // 2, 26), size="md")

        fps_txt = f"FPS {fps:04.1f}"
        fw = self._tr.text_width(fps_txt, "sm")
        self._tr.put(img, fps_txt, (W - fw - 14, 26), size="sm")
        cv2.line(img, (0, 32), (W, 32), Config.LIME_DIM, 1)

    # ── Compact key menu ─────────────────────────────────────────────────────
    def menu(self, img, toggles: list) -> None:
        W = img.shape[1]
        x, y = W - 210, 52
        for key, label, active in toggles:
            col = Config.LIME if active else Config.LIME_DIM
            self._tr.put(img, f"[{key}] {label}", (x, y), size="xs", color=col)
            y += 20

    # ── Scanline effect ───────────────────────────────────────────────────────
    @staticmethod
    def apply_scanlines(img, mask: np.ndarray) -> None:
        # mask is float32 (0.0–1.0); use float multiply to avoid uint8 overflow
        tmp = img.astype(np.float32)
        tmp *= mask
        np.clip(tmp, 0, 255, out=tmp)
        np.copyto(img, tmp.astype(np.uint8))

    # ── Vignette ─────────────────────────────────────────────────────────────
    @staticmethod
    def apply_vignette(img, mask: np.ndarray) -> None:
        tmp = img.astype(np.float32)
        tmp *= mask
        np.clip(tmp, 0, 255, out=tmp)
        np.copyto(img, tmp.astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL WIDGET  —  bottom-left retro oscilloscope monitor
# ══════════════════════════════════════════════════════════════════════════════
class SignalWidget:
    """
    Retro CRT-style signal monitor (bottom-left corner).
    Scrolling waveform driven by face-detection confidence.
    Blinking cursor + dim grid + inner corner brackets = surveillance terminal look.
    """
    BOX_W   = 360
    BOX_H   = 78
    PAD     = 18
    HISTORY = 300    # rolling sample buffer

    def __init__(self):
        self._buf = deque([0.45] * self.HISTORY, maxlen=self.HISTORY)

    def push(self, value: float) -> None:
        """Push a 0–1 signal value (face confidence, or decay when lost)."""
        noise = np.random.normal(0, 0.012)
        self._buf.append(float(np.clip(value + noise, 0.03, 0.97)))

    def draw(self, img: np.ndarray, tr: TextRenderer) -> None:
        H_img, W_img = img.shape[:2]
        x1 = self.PAD
        y1 = H_img - self.BOX_H - self.PAD
        x2 = x1 + self.BOX_W
        y2 = y1 + self.BOX_H

        # ── Transparent background (20 % tint) ────────────────────────────
        roi = img[y1:y2, x1:x2]
        if roi.size:
            cv2.addWeighted(np.zeros_like(roi), 0.20, roi, 0.80, 0, roi)
            img[y1:y2, x1:x2] = roi

        # ── Outer border ──────────────────────────────────────────────────
        cv2.rectangle(img, (x1, y1), (x2, y2), Config.LIME, 1, cv2.LINE_AA)

        # ── Inner corner brackets (retro terminal aesthetic) ───────────────
        arm = 7
        for bx, by, dx, dy in [
            (x1+2, y1+2,  arm, 0), (x1+2, y1+2, 0,  arm),
            (x2-2, y1+2, -arm, 0), (x2-2, y1+2, 0,  arm),
            (x1+2, y2-2,  arm, 0), (x1+2, y2-2, 0, -arm),
            (x2-2, y2-2, -arm, 0), (x2-2, y2-2, 0, -arm),
        ]:
            cv2.line(img, (bx, by), (bx+dx, by+dy), Config.LIME, 1, cv2.LINE_AA)

        # ── Time + blinking cursor ─────────────────────────────────────────
        cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
        tr.put(img, time.strftime(f"%H:%M {cursor}"), (x1+10, y1+22), size="sm")
        tr.put(img, "SIG", (x2-32, y1+22), size="xs", color=Config.LIME_DIM)

        # ── Chart area ────────────────────────────────────────────────────
        cx1, cx2 = x1+8,  x2-8
        cy1, cy2 = y1+32, y2-8
        cw, ch   = cx2-cx1, cy2-cy1

        # Dim horizontal grid lines
        for frac in (0.33, 0.66):
            gy = int(cy1 + ch * frac)
            cv2.line(img, (cx1, gy), (cx2, gy), Config.LIME_DIM, 1)

        # Scrolling waveform — one sample per pixel column
        vals    = list(self._buf)[-cw:]
        prev_pt = None
        for i, v in enumerate(vals):
            px = cx1 + i
            py = int(cy2 - v * ch)
            py = max(cy1, min(cy2, py))
            cur_pt = (px, py)
            if prev_pt:
                cv2.line(img, prev_pt, cur_pt, Config.LIME, 1, cv2.LINE_AA)
            prev_pt = cur_pt

        # Bright dot at the live tip
        if prev_pt:
            cv2.circle(img, prev_pt, 2, Config.LIME, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  HAND VISUALIZER — Glassy + Multiple Prediction Profiles
# ══════════════════════════════════════════════════════════════════════════════

class HandVisualizer:
    def __init__(self, text_renderer):
        self.tr = text_renderer
        self.history = {}
        self.prev_overlay = None
        self.alpha = 0.62

        self.show_labels = True
        self.show_prediction = True

        # Current profile
        self.current_profile = "Balanced"   # Default

        # === PRESET PROFILES ===
        self.profiles = {
            "Precise": {
                "max_history": 8,
                "base_history": 4,
                "speed_history_mult": 0.04,
                "base_pred_distance": 1.1,
                "speed_pred_mult": 0.035,
                "max_pred_distance": 2.0,
                "ghost_alpha": 0.55,
                "line_color": (0, 255, 180)      # Classic lime-cyan
            },
            "Balanced": {
                "max_history": 10,
                "base_history": 6,
                "speed_history_mult": 0.085,
                "base_pred_distance": 1.45,
                "speed_pred_mult": 0.06,
                "max_pred_distance": 2.7,
                "ghost_alpha": 0.62,
                "line_color": (80, 255, 220)
            },
            "Futuristic": {
                "max_history": 14,
                "base_history": 7,
                "speed_history_mult": 0.14,
                "base_pred_distance": 1.8,
                "speed_pred_mult": 0.11,
                "max_pred_distance": 4.2,
                "ghost_alpha": 0.68,
                "line_color": (120, 255, 255)
            },
            "Aggressive": {
                "max_history": 12,
                "base_history": 5,
                "speed_history_mult": 0.18,
                "base_pred_distance": 2.2,
                "speed_pred_mult": 0.15,
                "max_pred_distance": 5.5,
                "ghost_alpha": 0.72,
                "line_color": (0, 240, 255)
            }
        }

    def set_profile(self, profile_name: str):
        """Change prediction profile."""
        if profile_name in self.profiles:
            self.current_profile = profile_name
            print(f"[HandVisualizer] Switched to {profile_name} profile", flush=True)
            return True
        return False

    def cycle_profile(self):
        """Advance to the next profile in order."""
        keys = list(self.profiles.keys())
        idx  = keys.index(self.current_profile)
        self.set_profile(keys[(idx + 1) % len(keys)])

    def get_current_profile(self):
        return self.profiles[self.current_profile]

    def update_and_draw(self, display, hand_result, w, h):
        if not hand_result or not hand_result.hand_landmarks:
            self.prev_overlay = None
            return

        profile = self.get_current_profile()

        # Ghosting with profile-specific alpha
        if self.prev_overlay is not None:
            blended = cv2.addWeighted(display, 1 - profile["ghost_alpha"],
                                      self.prev_overlay, profile["ghost_alpha"], 0)
            np.copyto(display, blended)

        for idx, hand_lms in enumerate(hand_result.hand_landmarks):
            hand_id = f"hand_{idx}"

            tip    = hand_lms[8]
            tip_pt = (int(tip.x * w), int(tip.y * h))

            if hand_id not in self.history:
                self.history[hand_id] = deque(maxlen=profile["max_history"])
            self.history[hand_id].append(tip_pt)

            self._draw_glowing_hand(display, hand_lms, w, h)

            if self.show_prediction:
                self._draw_adaptive_prediction(display, hand_id, tip_pt, hand_lms, profile)

            if self.show_labels:
                self._draw_joint_labels(display, hand_lms, w, h, idx)

        self.prev_overlay = display.copy()

    def _draw_adaptive_prediction(self, display, hand_id, tip_pt, hand_lms, profile):
        hist = self.history[hand_id]
        if len(hist) < 4:
            return

        speed = self._get_hand_speed(hand_id)

        history_len  = np.clip(profile["base_history"] + speed * profile["speed_history_mult"],
                               4, profile["max_history"])
        pred_distance = np.clip(profile["base_pred_distance"] + speed * profile["speed_pred_mult"],
                                1.1, profile["max_pred_distance"])

        pts = np.array(list(hist)[-int(history_len):])
        n   = len(pts)
        t   = np.arange(n).reshape(-1, 1)
        A   = np.hstack([t, np.ones((n, 1))])

        x = pts[:, 0].astype(float)
        y = pts[:, 1].astype(float)

        mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]
        my, cy = np.linalg.lstsq(A, y, rcond=None)[0]

        next_t  = n + pred_distance
        pred_pt = (int(mx * next_t + cx), int(my * next_t + cy))

        confidence = self._get_tracking_confidence(hand_lms)
        thickness  = 4 if confidence > 0.75 else 3
        color      = profile["line_color"]

        cv2.line(display, tip_pt, pred_pt, color, thickness, cv2.LINE_AA)
        cv2.line(display, tip_pt, pred_pt, (255, 255, 255), 1, cv2.LINE_AA)  # subtle highlight

        cv2.circle(display, pred_pt, 7, color, -1)
        cv2.circle(display, pred_pt, 11, Config.LIME, 1)

        if confidence > 0.85:
            cv2.circle(display, pred_pt, 14, (200, 255, 255), 1, cv2.LINE_AA)

    def _get_hand_speed(self, hand_id):
        hist = self.history[hand_id]
        if len(hist) < 3:
            return 0.0
        pts    = np.array(hist)
        diffs  = np.diff(pts, axis=0)
        speeds = np.hypot(diffs[:, 0], diffs[:, 1])
        return float(np.mean(speeds[-5:]))

    def _get_tracking_confidence(self, hand_lms):
        visible = sum(1 for i in [0, 4, 8, 12, 16, 20]
                      if 0.03 < hand_lms[i].x < 0.97 and 0.03 < hand_lms[i].y < 0.97)
        return visible / 6.0

    def _draw_glowing_hand(self, img, hand_lms, w, h):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in Config.HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], Config.LIME, 2, cv2.LINE_AA)

        TIP = {4, 8, 12, 16, 20}
        for i, pt in enumerate(pts):
            r = 7 if i in TIP else 4
            cv2.circle(img, pt, r + 4, (0, 255, 180), -1)
            cv2.circle(img, pt, r, Config.LIME, -1)
            cv2.circle(img, pt, r, (0, 0, 0), 1)

    def _draw_joint_labels(self, img, hand_lms, w, h, hand_idx):
        important = [0, 4, 8, 12, 16, 20]
        for i in important:
            lm    = hand_lms[i]
            x     = int(lm.x * w)
            y     = int(lm.y * h)
            label = f"{i}"
            offset_x = 14 if hand_idx == 0 else -self.tr.text_width(label, "xs") - 18
            self.tr.put(img, label, (x + offset_x, y - 10), size="xs", color=(200, 255, 200))


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL  —  credential login
# ══════════════════════════════════════════════════════════════════════════════

# Fields collected from each user at login
_CRED_FIELDS = ["Name", "Age", "Location"]
_CRED_HINTS  = ["e.g. Tobi Kallaghe", "e.g. 21", "e.g. Kuala Lumpur, MY"]

# Layout grid constants — shared across all render + animate functions
_LX    = 96    # left column x  (labels + section headers)
_DOT_X = 220   # dot separator column
_VAL_X = 246   # value / input column
_ROW_H = 44    # vertical spacing between field rows


def _load_fonts(sizes: dict) -> dict:
    """Return {key: ImageFont} for each key:size pair, falling back to default."""
    path = next((p for p in Config.FONT_PATHS if os.path.exists(p)), None)
    out  = {}
    for key, sz in sizes.items():
        if path:
            try:
                out[key] = ImageFont.truetype(path, sz, index=0)
                continue
            except Exception:
                pass
        out[key] = ImageFont.load_default()
    return out


def _tw(draw, text: str, font) -> int:
    """Pixel width of text string (draw arg unused — kept for call-site compat)."""
    bb = font.getbbox(text)
    return bb[2] - bb[0] if bb else 0


def _scanlines_pil(draw, width: int, height: int) -> None:
    for y in range(0, height, 2):
        draw.line([(0, y), (width, y)], fill=(0, 12, 0))


def _corner_brackets_pil(draw, width: int, height: int, color, arm: int = 18) -> None:
    m = 8
    for x, y, dx, dy in [
        (m, m, arm, 0), (m, m, 0, arm),
        (width-m, m, -arm, 0), (width-m, m, 0, arm),
        (m, height-m, arm, 0), (m, height-m, 0, -arm),
        (width-m, height-m, -arm, 0), (width-m, height-m, 0, -arm),
    ]:
        draw.line([(x, y), (x+dx, y+dy)], fill=color, width=1)


def _render_cred_form(width: int, height: int, fonts: dict,
                      user_num: int, completed: dict,
                      active_field: int, input_text: str,
                      error: str) -> np.ndarray:
    """Interactive credential form — lowercase, structured grid layout."""
    LIME     = (57, 255,  20)
    LIME_DIM = (18, 110,   7)
    DIM2     = (10,  55,   4)
    RED      = (255, 55,  55)

    img  = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    _scanlines_pil(draw, width, height)

    # ── Title ─────────────────────────────────────────────────────────────────
    title   = "welcome  buddy"
    title_w = _tw(draw, title, fonts["title"])
    title_x = (width - title_w) // 2
    title_y = 50
    draw.text((title_x, title_y), title, font=fonts["title"], fill=LIME)

    rule_y = title_y + fonts["title"].size + 10
    rule_x = (width - title_w - 40) // 2
    draw.line([(rule_x, rule_y), (rule_x + title_w + 40, rule_y)],
              fill=LIME_DIM, width=1)

    sub   = "face + hand tracking  ·  v3.0"
    sub_x = (width - _tw(draw, sub, fonts["hint"])) // 2
    draw.text((sub_x, rule_y + 14), sub, font=fonts["hint"], fill=LIME_DIM)

    # ── Section header ────────────────────────────────────────────────────────
    hdr_y = rule_y + 54
    draw.text((_LX, hdr_y),
              f"user  {user_num:02d}  ·  enter your details",
              font=fonts["label"], fill=LIME_DIM)
    draw.line([(_LX, hdr_y + 24), (width - _LX, hdr_y + 24)],
              fill=DIM2, width=1)

    # ── Fields grid ───────────────────────────────────────────────────────────
    field_y = hdr_y + 46

    for i, (field, hint) in enumerate(zip(_CRED_FIELDS, _CRED_HINTS)):
        fy          = field_y + i * _ROW_H
        is_active   = (i == active_field)
        is_complete = field in completed

        label_col = LIME if is_active else (LIME_DIM if not is_complete else DIM2)
        draw.text((_LX,    fy), field.lower(), font=fonts["label"], fill=label_col)
        draw.text((_DOT_X, fy), "·",           font=fonts["label"], fill=DIM2)

        if is_active:
            cur = "█" if int(time.time() * 2) % 2 == 0 else " "
            draw.text((_VAL_X, fy), f"{input_text}{cur}",
                      font=fonts["input"], fill=LIME)
        elif is_complete:
            draw.text((_VAL_X, fy), completed[field],
                      font=fonts["input"], fill=LIME)
        else:
            draw.text((_VAL_X, fy), hint, font=fonts["hint"], fill=DIM2)

    # ── Bottom rule + instruction ──────────────────────────────────────────────
    sep_y = field_y + len(_CRED_FIELDS) * _ROW_H + 20
    draw.line([(_LX, sep_y), (width - _LX, sep_y)], fill=LIME_DIM, width=1)

    if error:
        ex = (width - _tw(draw, error, fonts["hint"])) // 2
        draw.text((ex, sep_y + 14), error, font=fonts["hint"], fill=RED)
    else:
        tip   = "return  to confirm each field  ·  esc  to quit"
        tip_x = (width - _tw(draw, tip, fonts["hint"])) // 2
        draw.text((tip_x, sep_y + 14), tip, font=fonts["hint"], fill=LIME_DIM)

    _corner_brackets_pil(draw, width, height, DIM2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _render_second_user_prompt(width: int, height: int, fonts: dict,
                                user1: dict, input_text: str,
                                error: str) -> np.ndarray:
    """Ask whether to add a second user — same grid / lowercase style."""
    LIME     = (57, 255,  20)
    LIME_DIM = (18, 110,   7)
    DIM2     = (10,  55,   4)
    RED      = (255, 55,  55)

    img  = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    _scanlines_pil(draw, width, height)

    # ── Title ─────────────────────────────────────────────────────────────────
    title   = "welcome  buddy"
    title_w = _tw(draw, title, fonts["title"])
    title_x = (width - title_w) // 2
    draw.text((title_x, 50), title, font=fonts["title"], fill=LIME)

    rule_y = 50 + fonts["title"].size + 10
    rule_x = (width - title_w - 40) // 2
    draw.line([(rule_x, rule_y), (rule_x + title_w + 40, rule_y)],
              fill=LIME_DIM, width=1)

    sub   = "face + hand tracking  ·  v3.0"
    sub_x = (width - _tw(draw, sub, fonts["hint"])) // 2
    draw.text((sub_x, rule_y + 14), sub, font=fonts["hint"], fill=LIME_DIM)

    # ── User 1 confirmed summary ───────────────────────────────────────────────
    sy = rule_y + 54
    draw.text((_LX, sy), "user  01  confirmed",
              font=fonts["label"], fill=LIME_DIM)
    draw.line([(_LX, sy + 24), (width - _LX, sy + 24)], fill=DIM2, width=1)
    sy += 44
    for k, v in user1.items():
        draw.text((_LX, sy), f"{k.lower()}  ·  {v}", font=fonts["hint"], fill=DIM2)
        sy += 22

    # ── Prompt ────────────────────────────────────────────────────────────────
    sy += 22
    draw.line([(_LX, sy), (width - _LX, sy)], fill=LIME_DIM, width=1)
    sy += 20
    draw.text((_LX, sy), "add second user?", font=fonts["label"], fill=LIME)
    sy += 34

    cur = "█" if int(time.time() * 2) % 2 == 0 else " "
    draw.text((_LX, sy), f">  {input_text}{cur}", font=fonts["input"], fill=LIME)
    sy += 28

    if error:
        draw.text((_LX, sy), error, font=fonts["hint"], fill=RED)
    else:
        draw.text((_LX, sy),
                  "y  to add  ·  n  to continue  ·  return  to confirm",
                  font=fonts["hint"], fill=LIME_DIM)

    _corner_brackets_pil(draw, width, height, DIM2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _animate_cred_intro(window: str, width: int, height: int,
                         fonts: dict, user_num: int) -> None:
    """
    CLI typewriter intro — the welcome buddy form builds itself element by element
    before handing off to interactive input mode.  No separate boot screen.

    Sequence:  title → subtitle → section header → field labels → instructions
    Each string types character-by-character; structural rules appear once their
    anchor text is committed.
    """
    LIME     = (57, 255,  20)
    LIME_DIM = (18, 110,   7)
    DIM2     = (10,  55,   4)

    # ── Pre-compute layout (must match _render_cred_form exactly) ─────────────
    title   = "welcome  buddy"
    title_w = _tw(None, title, fonts["title"])
    title_x = (width - title_w) // 2
    title_y = 50
    rule_y  = title_y + fonts["title"].size + 10
    rule_x  = (width - title_w - 40) // 2
    sub     = "face + hand tracking  ·  v3.0"
    sub_x   = (width - _tw(None, sub, fonts["hint"])) // 2
    sub_y   = rule_y + 14
    hdr_y   = rule_y + 54
    hdr     = f"user  {user_num:02d}  ·  enter your details"
    field_y = hdr_y + 46
    sep_y   = field_y + len(_CRED_FIELDS) * _ROW_H + 20
    tip     = "return  to confirm each field  ·  esc  to quit"
    tip_x   = (width - _tw(None, tip, fonts["hint"])) // 2
    tip_y   = sep_y + 14

    # ── Ordered animation sequence: (text, x, y, font_key, colour) ────────────
    seq = [
        (title, title_x, title_y, "title", LIME),
        (sub,   sub_x,   sub_y,   "hint",  LIME_DIM),
        (hdr,   _LX,     hdr_y,   "label", LIME_DIM),
    ] + [
        (f.lower(), _LX, field_y + i * _ROW_H, "label", LIME_DIM)
        for i, f in enumerate(_CRED_FIELDS)
    ] + [
        (tip, tip_x, tip_y, "hint", LIME_DIM),
    ]

    committed = []   # fully typed elements

    def _frame(cur_text="", cx=0, cy=0, cf="sm", cc=LIME):
        img  = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        _scanlines_pil(draw, width, height)
        _corner_brackets_pil(draw, width, height, DIM2)

        n = len(committed)

        # Replay committed elements
        for t, x, y, fk, col in committed:
            draw.text((x, y), t, font=fonts[fk], fill=col)

        # Structural rules keyed to commit stages
        if n >= 1:   # title committed → title rule
            draw.line([(rule_x, rule_y), (rule_x + title_w + 40, rule_y)],
                      fill=LIME_DIM, width=1)
        if n >= 3:   # section header committed → header rule
            draw.line([(_LX, hdr_y + 24), (width - _LX, hdr_y + 24)],
                      fill=DIM2, width=1)
        if n >= len(seq):   # all text committed → bottom rule
            draw.line([(_LX, sep_y), (width - _LX, sep_y)],
                      fill=LIME_DIM, width=1)

        # Dot separators — appear once each field label is committed
        for fi in range(min(max(n - 3, 0), len(_CRED_FIELDS))):
            draw.text((_DOT_X, field_y + fi * _ROW_H), "·",
                      font=fonts["label"], fill=DIM2)

        # Currently-typing partial text
        if cur_text:
            draw.text((cx, cy), cur_text, font=fonts[cf], fill=cc)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ── Typewriter loop ───────────────────────────────────────────────────────
    for text, x, y, fk, col in seq:
        built = ""
        for ch in text:
            built += ch
            cv2.imshow(window, _frame(built, x, y, fk, col))
            cv2.waitKey(18)
        committed.append((text, x, y, fk, col))
        time.sleep(0.028)

    # Hold complete layout briefly before going interactive
    cv2.imshow(window, _frame())
    cv2.waitKey(220)


def _success_flash(window: str, width: int, height: int, fonts: dict,
                   users: list) -> None:
    names = "  ·  ".join(u["Name"] for u in users)
    msg   = f"identity confirmed  —  {names}"
    for alpha in [0.15, 0.40, 0.70, 1.0, 0.80, 0.50, 0.20]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = int(200 * alpha)
        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        mw   = _tw(draw, msg, fonts["label"])
        draw.text(((width - mw) // 2, height // 2 - 14),
                  msg, font=fonts["label"], fill=(57, 255, 20))
        cv2.imshow(window, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        cv2.waitKey(55)


def _collect_credentials(window: str, width: int, height: int,
                          fonts: dict, user_num: int) -> dict | None:
    """Interactive field-by-field form. Returns filled dict or None on ESC."""
    completed: dict = {}
    active    = 0
    inp       = ""
    error     = ""

    while active < len(_CRED_FIELDS):
        cv2.imshow(window,
                   _render_cred_form(width, height, fonts, user_num,
                                     completed, active, inp, error))
        key = cv2.waitKey(33) & 0xFF
        if key == 255:
            continue
        elif key == 27:
            return None
        elif key in (13, 10):
            val = inp.strip()
            if val:
                completed[_CRED_FIELDS[active]] = val
                active += 1
                inp    = ""
                error  = ""
            else:
                error = "field cannot be empty"
        elif key in (8, 127):
            inp   = inp[:-1]
            error = ""
        elif 32 <= key <= 126:
            inp  += chr(key)
            error = ""

    completed["Mood"] = "DETECTING…"
    return completed


def terminal_login(window_title: str,
                   width: int  = Config.CAM_W,
                   height: int = Config.CAM_H) -> list | None:
    """
    CLI intro animation + multi-user credential form.
    Returns list of user dicts (1 or 2), or None on ESC.
    """
    fonts = _load_fonts({
        "title": 32,   # welcome buddy — large, commanding
        "label": 15,   # field names / section headers
        "input": 16,   # live typing
        "hint":  12,   # placeholders / instructions
        "sm":    13,   # fallback compatibility
    })

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)

    # ── User 1: animate form in, then go interactive ───────────────────────────
    _animate_cred_intro(window_title, width, height, fonts, 1)
    user1 = _collect_credentials(window_title, width, height, fonts, 1)
    if user1 is None:
        return None

    # ── Ask about second user ─────────────────────────────────────────────────
    inp, error = "", ""
    while True:
        cv2.imshow(window_title,
                   _render_second_user_prompt(width, height, fonts, user1, inp, error))
        key = cv2.waitKey(33) & 0xFF
        if key == 255:
            continue
        elif key == 27:
            return None
        elif key in (13, 10):
            choice = inp.strip().lower()
            if choice in ("y", "n"):
                break
            error = "type  y  or  n  then press return"
        elif key in (8, 127):
            inp = inp[:-1]; error = ""
        elif 32 <= key <= 126:
            inp += chr(key); error = ""

    users = [user1]
    if choice == "y":
        # Animate form in again for user 2
        _animate_cred_intro(window_title, width, height, fonts, 2)
        user2 = _collect_credentials(window_title, width, height, fonts, 2)
        if user2 is None:
            return None
        users.append(user2)

    _success_flash(window_title, width, height, fonts, users)
    return users


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _ensure_models() -> None:
    for path, url in [(Config.FACE_LM_MODEL, Config.FACE_LM_URL),
                      (Config.HAND_MODEL,    Config.HAND_URL)]:
        if not os.path.exists(path):
            print(f"  Downloading {os.path.basename(path)}…", flush=True)
            urllib.request.urlretrieve(url, path)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class _FaceResultSlice:
    """
    Adapts a multi-face MediaPipe result so that EmotionDetector (which always
    reads face_blendshapes[0]) sees a single target face's blendshapes.
    """
    __slots__ = ("face_blendshapes",)

    def __init__(self, face_res, idx: int):
        self.face_blendshapes = (
            [face_res.face_blendshapes[idx]]
            if face_res.face_blendshapes and idx < len(face_res.face_blendshapes)
            else []
        )


# ══════════════════════════════════════════════════════════════════════════════
#  APP  —  orchestrator
# ══════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, users: list):
        _ensure_models()
        self._users    = users                                  # 1–2 user dicts
        self._camera   = Camera().start()
        self._tracker  = Tracker()
        self._emotions = [EmotionDetector() for _ in users]    # one per slot
        self._fps      = FPSCounter()
        self._tr       = TextRenderer()
        self._hud      = HUD(self._tr)
        self._gesture  = GestureRecognizer(cooldown=Config.GESTURE_COOLDOWN)
        self._signal   = SignalWidget()
        self._hand_viz = HandVisualizer(self._tr)
        self._start_t  = time.time()

        # toggle states
        self.show_face_box   = True
        self.show_hands      = True
        self.show_info_panel = True
        self.show_scanlines  = True    # ON by default
        self.show_menu       = True
        self._filter_idx     = 0
        self._face_blur      = False   # peace sign toggles this

        # per-face smoothing + mood  (slot 0 = leftmost face, slot 1 = rightmost)
        self._smooth_boxes = [None, None]
        self._moods        = ["NEUTRAL", "NEUTRAL"]

        # pre-computed effect masks (rebuilt on resolution change)
        self._vign_mask   = None
        self._scan_mask   = None
        self._last_shape  = None

    @property
    def _ts(self) -> int:
        return int((time.time() - self._start_t) * 1000)

    @property
    def _filter(self) -> str:
        return Config.FILTERS[self._filter_idx % len(Config.FILTERS)]

    def _ensure_masks(self, h: int, w: int) -> None:
        if (h, w) == self._last_shape:
            return
        self._last_shape = (h, w)
        cy, cx = h / 2.0, w / 2.0
        Y, X   = np.ogrid[:h, :w]
        dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        vig    = np.clip(1.0 - dist * 0.40, 0.10, 1.0).astype(np.float32)
        self._vign_mask = vig[:, :, np.newaxis]
        # float32 mask — 1.0 = full brightness, 0.70 = dimmed alternate rows
        scan = np.ones((h, 1, 1), dtype=np.float32)
        scan[::2] = 0.70
        self._scan_mask = scan

    def _smooth(self, idx: int, raw: list) -> list:
        """EMA smoothing for face bounding box at slot idx (0=left, 1=right)."""
        a = Config.BOX_ALPHA
        if self._smooth_boxes[idx] is None:
            self._smooth_boxes[idx] = [float(v) for v in raw]
        else:
            self._smooth_boxes[idx] = [a * r + (1 - a) * s
                                       for r, s in zip(raw, self._smooth_boxes[idx])]
        return [int(v) for v in self._smooth_boxes[idx]]

    @staticmethod
    def _face_bbox(landmarks, w, h, pad=18):
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
                min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad))

    def _render_face(self, display, face_res, w, h) -> float:
        """
        Render overlays for up to N detected faces matched to N registered users.
        Faces are sorted left-to-right; user slot 0 = leftmost face.
        Returns best signal confidence (0.0 if no faces detected).
        """
        if not face_res or not face_res.face_landmarks:
            return 0.0

        # Sort detected face indices by x-centre (ascending = left to right)
        faces = face_res.face_landmarks
        order = sorted(range(len(faces)),
                       key=lambda i: float(np.mean([lm.x for lm in faces[i]])))

        max_slots = min(len(order), len(self._users))
        best_conf = 0.0

        for slot, face_idx in enumerate(order[:max_slots]):
            lms             = faces[face_idx]
            x1, y1, x2, y2 = self._smooth(slot, list(self._face_bbox(lms, w, h)))

            # ── Emotion update (blendshapes for this specific face) ───────────
            self._moods[slot] = self._emotions[slot].update(
                _FaceResultSlice(face_res, face_idx))

            # ── Signal confidence ─────────────────────────────────────────────
            if (face_res.face_blendshapes
                    and face_idx < len(face_res.face_blendshapes)
                    and face_res.face_blendshapes[face_idx]):
                sc   = [c.score for c in face_res.face_blendshapes[face_idx]]
                conf = float(np.clip(np.mean(sc) * 3.5, 0.2, 0.95))
            else:
                conf = 0.6
            best_conf = max(best_conf, conf)

            # ── Face blur (peace-sign gesture) ────────────────────────────────
            if self._face_blur:
                roi = display[y1:y2, x1:x2]
                if roi.size:
                    display[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)

            # ── Face bounding box ─────────────────────────────────────────────
            if self.show_face_box:
                HUD.face_box(display, x1, y1, x2, y2)

            # ── Info panel + connector ────────────────────────────────────────
            if self.show_info_panel:
                # ~3.5 cm gap at 96 dpi ≈ 130 px — visible straight line
                GAP = 130
                PANEL_W = 240

                # Panel follows the face: goes to whichever side has more room.
                # Face centre left of frame centre → panel right; else → panel left.
                face_cx  = (x1 + x2) // 2
                space_r  = w - x2          # pixels available to the right
                space_l  = x1             # pixels available to the left
                to_right = space_r >= space_l   # pick the roomier side

                # Hard-clip: if chosen side doesn't fit, flip
                if to_right and x2 + GAP + PANEL_W > w:
                    to_right = False
                if not to_right and x1 - GAP - PANEL_W < 0:
                    to_right = True

                panel_x = x2 + GAP if to_right else x1 - GAP - PANEL_W
                panel_y = y1

                # Build info dict from registered user + live mood
                user = self._users[slot]
                info = {k: user.get(k, "—")
                        for k in ("Name", "Age", "Location")}
                info["Mood"] = self._moods[slot]

                pbx1, pby1, pbx2, pby2 = self._hud.info_panel(
                    display, panel_x, panel_y, info, self._moods[slot])

                HUD.connector(display,
                              x1, y1, x2, y2,
                              pbx1, pby1, pbx2, pby2,
                              side="right" if to_right else "left")

        return best_conf

    def _key(self, key: int) -> bool:
        if key == ord('q'):  return False
        if key == ord('1'):  self.show_face_box   ^= True
        if key == ord('2'):  self.show_hands       ^= True
        if key == ord('3'):  self.show_info_panel  ^= True
        if key == ord('4'):  self.show_scanlines   ^= True
        if key == ord('f'):  self._filter_idx      += 1
        if key == ord('p'):  self._hand_viz.cycle_profile()
        if key == ord('m'):  self.show_menu        ^= True
        return True

    def run(self) -> None:
        names = "  ·  ".join(u.get("Name", "User") for u in self._users)
        print(f"✅  HANDSY HUD running  —  {names}\n"
              "   Gestures: ✌ Peace=face-blur  👎 Thumbs-down=cycle-filter\n"
              "   Keys: F=filter  P=hand profile  1=face  2=hands  3=panel  4=scanlines  M=menu  Q=quit\n",
              flush=True)
        cv2.namedWindow(Config.WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(Config.WIN_TITLE, Config.CAM_W, Config.CAM_H)
        try:
            while True:
                ok, frame = self._camera.read()
                if not ok:
                    time.sleep(0.001)
                    continue

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                self._tracker.submit(frame, self._ts)
                face_res, hand_res = self._tracker.results()
                self._ensure_masks(h, w)

                # ── Process gestures (debounced) ──────────────────────────────
                gestures = self._gesture.update(hand_res)
                if "peace" in gestures:
                    self._face_blur ^= True
                if "thumbs_down" in gestures:
                    self._filter_idx += 1

                # ── Build display frame (background layer) ────────────────────
                display = Filter.apply(frame, self._filter)
                HUD.apply_vignette(display, self._vign_mask)
                if self.show_scanlines:
                    HUD.apply_scanlines(display, self._scan_mask)

                # ── Draw overlays (always vivid lime, regardless of filter) ───
                sig_conf = self._render_face(display, face_res, w, h)
                self._signal.push(sig_conf if sig_conf > 0 else 0.12)

                if self.show_hands:
                    self._hand_viz.update_and_draw(display, hand_res, w, h)

                # ── Signal widget (bottom-left) ───────────────────────────────
                self._signal.draw(display, self._tr)

                self._hud.header(display, self._fps.tick(),
                                 self._filter, self._face_blur)

                if self.show_menu:
                    self._hud.menu(display, [
                        ("1", f"Face Box   {'ON' if self.show_face_box   else '--'}", self.show_face_box),
                        ("2", f"Hands      {'ON' if self.show_hands      else '--'}", self.show_hands),
                        ("3", f"Info Panel {'ON' if self.show_info_panel else '--'}", self.show_info_panel),
                        ("4", f"Scanlines  {'ON' if self.show_scanlines  else '--'}", self.show_scanlines),
                        ("F", f"Filter  {self._filter:<10}",                          True),
                        ("P", f"Profile {self._hand_viz.current_profile:<10}",        True),
                        ("✌", f"FaceBlur {'ON' if self._face_blur else '--'}",        self._face_blur),
                        ("M", "Hide Menu",                                             True),
                    ])

                cv2.imshow(Config.WIN_TITLE, display)
                if not self._key(cv2.waitKey(1) & 0xFF):
                    break

        finally:
            self._camera.stop()
            self._tracker.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    users = terminal_login(Config.WIN_TITLE)
    if users:
        App(users).run()

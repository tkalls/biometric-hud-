"""
Virtual Physics Lab — v2.0 "Chaos Edition"
===========================================
Gesture-controlled physics sandbox using MediaPipe hand tracking + Pymunk rigid-body simulation.

Gestures:
  Pinch (thumb + index < 50 px)   → spawn a random polygon with particle burst
  Fist (0 fingers extended)        → flip gravity direction
  Open palm (4+ fingers extended)  → radial shockwave blast
  Fast finger move (> 350 px/s)    → velocity impulse kick on nearby objects

Controls:
  Q  →  quit

Requirements: pip install opencv-python mediapipe pymunk numpy
Run:          python virtual_physics_lab_v2.py
"""

import cv2
import mediapipe as mp
import pymunk
import numpy as np
import math
import time
import random

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GRAVITY_NORMAL  = (0,  900)   # px/s² — positive Y is downward in screen space
GRAVITY_FLIPPED = (0, -900)   # px/s² — negative Y inverts gravity

BALL_RADIUS    = 22
FINGER_R       = 15     # kinematic finger-circle radius (px)
PINCH_THRESH   = 50     # px — thumb/index gap that registers as a pinch
FLICK_THRESH   = 350    # px/s — finger speed that triggers an impulse kick
PALM_COOLDOWN  = 1.2    # seconds between shockwave triggers (per hand)
SPAWN_COOLDOWN = 0.7    # seconds between shape spawns (per hand)
FPS            = 30
MAX_OBJECTS    = 45     # oldest shapes are culled once this cap is exceeded
MAX_HANDS      = 2

# Landmark IDs we track for each finger tip
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12}

# BGR colour palette for spawned shapes
SHAPE_PALETTE = [
    ( 50, 100, 255),  # orange
    (100, 255, 100),  # green
    (255, 100, 100),  # blue
    (100, 255, 255),  # yellow
    (255, 100, 255),  # magenta
    (255, 255, 100),  # cyan
    (180,  60, 255),  # purple
    ( 60, 200, 255),  # gold
]

COLOURS = {
    "thumb":  (0,   220, 255),
    "index":  (0,   255,  80),
    "middle": (255,  60, 200),
    "ball":   (255,  80,   0),
    "paddle": (255, 230,  50),
    "pinch":  (0,     0, 255),
}


# ══════════════════════════════════════════════════════════════════════════════
# PARTICLE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
class Particle:
    """Lightweight spark/fragment for visual feedback."""
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "r")

    def __init__(self, x, y, vx, vy, life, color, r):
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.life = self.max_life = float(life)
        self.color = color  # (R, G, B) tuple — converted to BGR on draw
        self.r = r

particles: list[Particle] = []


def emit(x: float, y: float,
         n: int = 25,
         color: tuple = (255, 200, 60),
         spd: tuple = (100, 350)) -> None:
    """Radial burst of n particles from (x, y)."""
    for _ in range(n):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*spd)
        particles.append(Particle(
            x, y,
            math.cos(angle) * speed,
            math.sin(angle) * speed,
            random.uniform(0.25, 0.65),
            color,
            random.randint(2, 5),
        ))


def tick_particles(dt: float) -> None:
    """Advance all particles one physics step; remove dead ones in-place."""
    i = 0
    while i < len(particles):
        p = particles[i]
        p.x  += p.vx * dt
        p.y  += p.vy * dt
        p.vx *= 0.90        # horizontal air drag
        p.vy += 420.0 * dt  # particle gravity (independent of world gravity)
        p.life -= dt
        if p.life <= 0:
            particles.pop(i)
        else:
            i += 1


def draw_particles(frame: np.ndarray) -> None:
    """Draw particles with alpha fade to black as they age."""
    for p in particles:
        alpha = max(0.0, p.life / p.max_life)
        r, g, b = p.color
        cv2.circle(frame, (int(p.x), int(p.y)), p.r,
                   (int(b * alpha), int(g * alpha), int(r * alpha)), -1)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS SPACE
# ══════════════════════════════════════════════════════════════════════════════
space           = pymunk.Space()
space.gravity   = GRAVITY_NORMAL
gravity_flipped = False
walls_added     = False


def add_walls(space: pymunk.Space, w: int, h: int) -> None:
    """Static boundary segments: floor, left wall, right wall, ceiling."""
    for a, b in [((0, h), (w, h)), ((0, 0), (0, h)),
                 ((w, 0), (w, h)), ((0, 0), (w, 0))]:
        seg = pymunk.Segment(space.static_body, a, b, 4)
        seg.elasticity = 0.55
        seg.friction   = 0.50
        space.add(seg)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC OBJECTS
# ══════════════════════════════════════════════════════════════════════════════
dynamic_objects: list[dict] = []


def _ngon(n: int, r: float) -> list[tuple]:
    """Vertices of a regular n-gon centred at the origin."""
    # Triangles point up; even-sided polygons sit flat
    offset = -math.pi / 2 if n % 2 != 0 else math.pi / n
    return [
        (r * math.cos(2 * math.pi * i / n + offset),
         r * math.sin(2 * math.pi * i / n + offset))
        for i in range(n)
    ]


def spawn_shape(space: pymunk.Space, x: float, y: float) -> None:
    """
    Randomly pick a polygon type (triangle, pentagon, hexagon, or box),
    create a dynamic Pymunk body, and emit a matching particle burst.
    """
    kind  = random.choice(["triangle", "pentagon", "hexagon", "box"])
    color = random.choice(SHAPE_PALETTE)

    if kind == "box":
        body  = pymunk.Body(mass=1, moment=pymunk.moment_for_box(1, (36, 36)))
        shape = pymunk.Poly.create_box(body, (36, 36))
    else:
        n     = {"triangle": 3, "pentagon": 5, "hexagon": 6}[kind]
        verts = _ngon(n, 22)
        body  = pymunk.Body(mass=1, moment=pymunk.moment_for_poly(1, verts))
        shape = pymunk.Poly(body, verts)

    body.position         = (x, y)
    body.angular_velocity = random.uniform(-7, 7)  # random initial spin
    shape.elasticity = 0.65
    shape.friction   = 0.55
    space.add(body, shape)
    dynamic_objects.append({
        "body":  body,
        "shape": shape,
        "type":  "poly",
        "color": color,
        "kind":  kind,
    })
    emit(x, y, n=40, color=color, spd=(120, 400))


def spawn_ball(space: pymunk.Space, x: float, y: float) -> None:
    body  = pymunk.Body(mass=1, moment=pymunk.moment_for_circle(1, 0, BALL_RADIUS))
    body.position = (x, y)
    shape = pymunk.Circle(body, radius=BALL_RADIUS)
    shape.elasticity = 0.85
    shape.friction   = 0.40
    space.add(body, shape)
    dynamic_objects.append({"body": body, "shape": shape, "type": "ball"})


def cull_objects(space: pymunk.Space, h: int) -> None:
    """
    Two-pass cull:
      1. Remove objects that fell below the visible area.
      2. Evict oldest objects when MAX_OBJECTS is exceeded.
    """
    for obj in list(dynamic_objects):
        if obj["body"].position.y > h + 200:
            space.remove(obj["body"], obj["shape"])
            dynamic_objects.remove(obj)
    while len(dynamic_objects) > MAX_OBJECTS:
        obj = dynamic_objects.pop(0)
        space.remove(obj["body"], obj["shape"])


# Seed the scene with one ball so the simulation isn't empty on launch
spawn_ball(space, 320, 60)


# ══════════════════════════════════════════════════════════════════════════════
# KINEMATIC FINGER BODIES  (one set per hand)
# ══════════════════════════════════════════════════════════════════════════════
# Kinematic bodies are positioned by us every frame.  They don't respond to
# gravity or forces, but dynamic bodies DO collide against them — giving the
# physical "touching" effect that makes gestures feel real.

hand_bodies: list[dict] = []

for _ in range(MAX_HANDS):
    hb = {}
    for name in FINGER_TIPS:
        fb = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        fb.position = (-5000, -5000)          # park off-screen initially
        fs = pymunk.Circle(fb, radius=FINGER_R)
        fs.elasticity = 0.50
        fs.friction   = 0.40
        space.add(fb, fs)
        hb[name] = fb
    hand_bodies.append(hb)

# Per-hand paddle segments (line between finger tips) — rebuilt each frame
active_paddle: list[list] = [[], []]


def rebuild_paddle(hand_idx: int, fpos: dict) -> None:
    """
    Tear down last frame's paddle segments and build new ones connecting
    thumb → index → middle.
    """
    if active_paddle[hand_idx]:
        space.remove(*active_paddle[hand_idx])
        active_paddle[hand_idx] = []
    for a, b in [("thumb", "index"), ("index", "middle")]:
        if a in fpos and b in fpos:
            seg = pymunk.Segment(space.static_body, fpos[a], fpos[b], 5)
            seg.elasticity = 0.65
            seg.friction   = 0.50
            space.add(seg)
            active_paddle[hand_idx].append(seg)


# ══════════════════════════════════════════════════════════════════════════════
# GESTURE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def count_extended(lm) -> int:
    """
    Count non-thumb fingers that are extended.

    MediaPipe gives 21 landmarks per hand.  A finger is 'extended' when its
    tip (landmark 8/12/16/20) has a smaller Y value than its PIP (middle
    knuckle, landmark 6/10/14/18).  Screen Y increases downward, so a smaller
    Y means the tip is *above* the knuckle — i.e., the finger is open.
    """
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return sum(1 for t, p in zip(tips, pips)
               if lm.landmark[t].y < lm.landmark[p].y)


def is_fist(lm) -> bool:
    return count_extended(lm) == 0


def is_open_palm(lm) -> bool:
    return count_extended(lm) >= 4


def apply_velocity_impulse(pos: tuple, vx: float, vy: float,
                            speed: float) -> None:
    """
    When a finger moves faster than FLICK_THRESH, push dynamic objects within
    a radius in the same direction.  Force tapers linearly with distance and
    emits a small spark burst at each hit object.
    """
    if speed < FLICK_THRESH:
        return
    RADIUS = 130
    nx, ny = vx / speed, vy / speed   # unit vector in flick direction
    for obj in dynamic_objects:
        body = obj["body"]
        dx   = body.position.x - pos[0]
        dy   = body.position.y - pos[1]
        dist = math.hypot(dx, dy) + 1
        if dist < RADIUS:
            taper = 1.0 - dist / RADIUS
            mag   = taper * speed * 0.09 * body.mass
            body.apply_impulse_at_world_point((nx * mag, ny * mag),
                                              body.position)
            emit(body.position.x, body.position.y,
                 n=8, color=(220, 220, 255), spd=(50, 200))


def apply_shockwave(px: float, py: float) -> None:
    """
    Radial shockwave from palm position: every object within RADIUS receives
    an outward impulse proportional to its proximity, plus a large burst.
    """
    RADIUS = 280
    FORCE  = 70000
    for obj in dynamic_objects:
        body = obj["body"]
        dx   = body.position.x - px
        dy   = body.position.y - py
        dist = math.hypot(dx, dy) + 1
        if dist < RADIUS:
            strength = (1.0 - dist / RADIUS) * FORCE
            body.apply_impulse_at_world_point(
                (dx / dist * strength, dy / dist * strength),
                body.position,
            )
    emit(px, py, n=90, color=(100, 255, 255), spd=(200, 700))


# ══════════════════════════════════════════════════════════════════════════════
# PER-HAND MUTABLE STATE
# ══════════════════════════════════════════════════════════════════════════════
prev_fpos  = [{} for _ in range(MAX_HANDS)]  # previous finger positions (velocity calc)
fist_was   = [False] * MAX_HANDS             # fist state from previous frame
pinch_was  = [False] * MAX_HANDS             # pinch state from previous frame
last_spawn = [0.0]   * MAX_HANDS             # timestamp of last shape spawn
last_shock = [0.0]   * MAX_HANDS             # timestamp of last shockwave


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def draw_dynamic_objects(frame: np.ndarray) -> None:
    for obj in dynamic_objects:
        body  = obj["body"]
        shape = obj["shape"]

        if obj["type"] == "ball":
            x, y = int(body.position.x), int(body.position.y)
            cv2.circle(frame, (x, y), BALL_RADIUS, COLOURS["ball"], -1)
            cv2.circle(frame, (x - 6, y - 6), 5, (200, 220, 255), -1)   # specular

        elif obj["type"] == "poly":
            # Transform vertices from body-local space to world space:
            # world_pt = rotate(local_pt, body.angle) + body.position
            verts_w = [v.rotated(body.angle) + body.position
                       for v in shape.get_vertices()]
            pts = np.array([[int(v.x), int(v.y)] for v in verts_w], dtype=np.int32)
            col = obj["color"]
            cv2.fillPoly(frame, [pts], col)
            outline = tuple(min(255, c + 90) for c in col)
            cv2.polylines(frame, [pts], isClosed=True,
                          color=outline, thickness=2, lineType=cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA + MEDIAPIPE SETUP
# ══════════════════════════════════════════════════════════════════════════════
cap  = cv2.VideoCapture(0)
mp_h = mp.solutions.hands
hands_detector = mp_h.Hands(
    max_num_hands            = MAX_HANDS,
    min_detection_confidence = 0.70,
    min_tracking_confidence  = 0.70,
)
mp_draw = mp.solutions.drawing_utils

prev_t = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame   = cv2.flip(frame, 1)        # mirror so left/right feels natural
    h, w, _ = frame.shape
    now     = time.time()
    dt      = min(now - prev_t, 0.05)   # cap dt to prevent physics explosions
    prev_t  = now

    if not walls_added:
        add_walls(space, w, h)
        walls_added = True

    # Convert to RGB for MediaPipe (it expects RGB, not BGR)
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Park ALL kinematic bodies off-screen — detected ones are repositioned below
    for hb in hand_bodies:
        for fb in hb.values():
            fb.position = (-5000, -5000)

    active_count = 0

    if results.multi_hand_landmarks:
        for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
            active_count += 1
            mp_draw.draw_landmarks(frame, hand_lm, mp_h.HAND_CONNECTIONS)

            fpos: dict[str, tuple] = {}

            # ── Track three finger tips ───────────────────────────────────
            for name, lm_id in FINGER_TIPS.items():
                lm = hand_lm.landmark[lm_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                fpos[name] = (cx, cy)

                # Teleport the kinematic body to live finger position.
                # Setting velocity so pymunk resolves collision impulses correctly.
                body = hand_bodies[hand_idx][name]
                prev = body.position
                body.velocity = ((cx - prev.x) / dt, (cy - prev.y) / dt)
                body.position = (cx, cy)

                cv2.circle(frame, (cx, cy), FINGER_R, COLOURS[name], -1)
                cv2.circle(frame, (cx, cy), FINGER_R, (255, 255, 255), 2)

            # ── Palm paddle (thumb–index–middle line) ────────────────────
            rebuild_paddle(hand_idx, fpos)
            for a, b in [("thumb", "index"), ("index", "middle")]:
                if a in fpos and b in fpos:
                    cv2.line(frame, fpos[a], fpos[b],
                             COLOURS["paddle"], 4, cv2.LINE_AA)

            # ── Pinch → spawn random polygon ─────────────────────────────
            if "thumb" in fpos and "index" in fpos:
                tp, ip = fpos["thumb"], fpos["index"]
                dist   = math.hypot(tp[0] - ip[0], tp[1] - ip[1])
                pinch  = dist < PINCH_THRESH
                mid    = ((tp[0] + ip[0]) // 2, (tp[1] + ip[1]) // 2)

                if pinch:
                    cv2.circle(frame, mid, 14, COLOURS["pinch"], -1)
                    # Rising-edge guard: only spawn on the FIRST pinch frame
                    if (not pinch_was[hand_idx]
                            and now - last_spawn[hand_idx] > SPAWN_COOLDOWN):
                        spawn_shape(space, *mid)
                        last_spawn[hand_idx] = now

                pinch_was[hand_idx] = pinch

            # ── Fist → flip gravity (rising edge only) ───────────────────
            fist = is_fist(hand_lm)
            if fist and not fist_was[hand_idx]:
                gravity_flipped = not gravity_flipped
                space.gravity   = GRAVITY_FLIPPED if gravity_flipped else GRAVITY_NORMAL
                emit(w // 2, h // 2, n=100, color=(100, 220, 255), spd=(300, 750))
            fist_was[hand_idx] = fist

            if fist:
                label = "↑ GRAVITY FLIPPED" if gravity_flipped else "↓ GRAVITY NORMAL"
                cv2.putText(frame, label, (14, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (100, 220, 255), 2, cv2.LINE_AA)

            # ── Velocity impulse — fast finger = physical kick ───────────
            for name, pos in fpos.items():
                prev = prev_fpos[hand_idx].get(name)
                if prev:
                    vx    = (pos[0] - prev[0]) * FPS
                    vy    = (pos[1] - prev[1]) * FPS
                    speed = math.hypot(vx, vy)
                    apply_velocity_impulse(pos, vx, vy, speed)
            prev_fpos[hand_idx] = dict(fpos)

            # ── Open palm → shockwave blast ───────────────────────────────
            if (is_open_palm(hand_lm)
                    and now - last_shock[hand_idx] > PALM_COOLDOWN):
                wrist  = hand_lm.landmark[0]
                px, py = int(wrist.x * w), int(wrist.y * h)
                apply_shockwave(px, py)
                last_shock[hand_idx] = now
                cv2.circle(frame, (px, py), 60, (0, 255, 255), 3)
                cv2.putText(frame, "SHOCKWAVE!", (14, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2, cv2.LINE_AA)

    # Park paddles for any hand that wasn't detected this frame
    for i in range(active_count, MAX_HANDS):
        rebuild_paddle(i, {})

    # ── Physics step ──────────────────────────────────────────────────────
    space.step(1.0 / FPS)
    cull_objects(space, h)

    # ── Render scene ──────────────────────────────────────────────────────
    draw_dynamic_objects(frame)
    tick_particles(dt)
    draw_particles(frame)

    # ── HUD ───────────────────────────────────────────────────────────────
    grav_label = "↑ GRAVITY" if gravity_flipped else "↓ GRAVITY"
    cv2.putText(frame,
                f"Objects: {len(dynamic_objects)}  |  {grav_label}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,
                "Pinch=spawn  |  Fist=flip gravity  |  Open palm=shockwave  |  Q=quit",
                (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("Virtual Physics Lab v2", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

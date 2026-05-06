import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
from collections import deque
import numpy as np
import random

model_path = "gesture_recognizer.task"
if not os.path.exists(model_path):
    print("Downloading gesture recognizer model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
        model_path
    )
    print("Done.")


options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=8,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
INFER_W, INFER_H = 640, 360
print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
cv2.namedWindow("Gesture Canvas", cv2.WINDOW_NORMAL)

_cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
font_scale = _cap_w / 1280
font_thick = max(1, int(font_scale * 2))


# ── gesture metadata ───────────────────────────────────────────────────────────

GESTURE_COLORS = {
    "Open_Palm":   (180,  80, 255),
    "Closed_Fist": ( 50, 200, 255),
    "Victory":     ( 80, 220,  80),
    "Thumb_Up":    ( 80, 180, 255),
    "Pointing_Up": (255, 100, 180),
    "ILoveYou":    (120, 255, 180),
    "Thumb_Down":  ( 80,  80, 220),
}

GESTURE_LABELS = {
    "Open_Palm":   "Open Palm",
    "Closed_Fist": "Closed Fist",
    "Victory":     "Victory",
    "Thumb_Up":    "Thumbs Up",
    "Pointing_Up": "Pointing Up",
    "ILoveYou":    "I Love You",
    "Thumb_Down":  "Thumbs Down",
}

TIER2_GESTURES = {"Open_Palm", "Closed_Fist"}


# ── particle system ────────────────────────────────────────────────────────────

_SS = 4

def _make_canvas(size, draw_fn, blur_sigma=None):
    s   = size * _SS * 3
    img = np.zeros((s, s, 3), dtype=np.uint8)
    draw_fn(img, s // 2, s // 2, size * _SS)
    cv2.GaussianBlur(img, (0, 0), blur_sigma or _SS * 0.8, img)
    out = size * 2
    sm  = cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA)
    c   = np.zeros((out, out, 4), dtype=np.uint8)
    c[:, :, :3] = sm
    c[:, :, 3]  = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    return c

def _draw_heart(img, cx, cy, r):
    col = (80, 60, 240)
    cv2.circle(img, (cx - r // 3, cy - r // 5), r // 2, col, -1)
    cv2.circle(img, (cx + r // 3, cy - r // 5), r // 2, col, -1)
    cv2.fillPoly(img, [np.array(
        [[cx - r//2, cy + r//8], [cx + r//2, cy + r//8], [cx, cy + r*2//3]], np.int32)], col)

def _draw_star(img, cx, cy, r):
    col = (0, 215, 255)
    pts = [[int(cx + (r if k % 2 == 0 else int(r * .42)) * math.cos(math.pi * k / 5 - math.pi / 2)),
            int(cy + (r if k % 2 == 0 else int(r * .42)) * math.sin(math.pi * k / 5 - math.pi / 2))]
           for k in range(10)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], col)

def _draw_diamond(img, cx, cy, r):
    col = (120, 220, 80)
    cv2.fillPoly(img, [np.array(
        [[cx, cy - r], [cx + r//2, cy], [cx, cy + r], [cx - r//2, cy]], np.int32)], col)

def _draw_spark(img, cx, cy, r):
    cv2.circle(img, (cx, cy), max(2, r // 2), (255, 240, 60), -1)

def _draw_tear(img, cx, cy, r):
    col = (200, 80, 50)
    cv2.circle(img, (cx, cy - r // 4), r * 2 // 3, col, -1)
    cv2.fillPoly(img, [np.array(
        [[cx - r//3, cy + r//8], [cx + r//3, cy + r//8], [cx, cy + r*3//4]], np.int32)], col)

_HEARTS   = [_make_canvas(s, _draw_heart)          for s in (12, 16, 20, 24)]
_STARS    = [_make_canvas(s, _draw_star)           for s in (10, 14, 18)]
_DIAMONDS = [_make_canvas(s, _draw_diamond)        for s in (8,  11, 14)]
_SPARKS   = [_make_canvas(s, _draw_spark, _SS*2.0) for s in (5,   7,  9)]
_TEARS    = [_make_canvas(s, _draw_tear)           for s in (8,  11, 14)]

def _composite(frame, canvas, cx, cy, alpha):
    fh, fw = frame.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = cx - cw // 2, cy - ch // 2
    fx1 = max(0, x1);      fy1 = max(0, y1)
    fx2 = min(fw, x1 + cw); fy2 = min(fh, y1 + ch)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    sx, sy = fx1 - x1, fy1 - y1
    roi    = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    patch  = canvas[sy:sy + (fy2 - fy1), sx:sx + (fx2 - fx1)]
    a      = patch[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    frame[fy1:fy2, fx1:fx2] = np.clip(
        roi * (1 - a) + patch[:, :, :3].astype(np.float32) * a, 0, 255).astype(np.uint8)

class Particle:
    __slots__ = ('x', 'y', '_c', 'vx', 'vy', 'life', 'decay', 'gravity', 'scale', 'dscale')

    def __init__(self, x, y, canvas, vx, vy, decay, gravity=60.0, scale=1.0, dscale=0.0):
        self.x, self.y   = float(x), float(y)
        self._c          = canvas
        self.vx, self.vy = float(vx), float(vy)
        self.life        = 1.0
        self.decay       = decay
        self.gravity     = gravity
        self.scale       = scale
        self.dscale      = dscale

    def update(self, dt):
        self.x    += self.vx * dt
        self.y    += self.vy * dt
        self.vy   += self.gravity * dt
        self.life -= self.decay * dt
        self.scale += self.dscale * dt
        return self.life > 0.02 and self.scale > 0.05

    def draw(self, frame):
        c = self._c
        if abs(self.scale - 1.0) > 0.02:
            ph, pw = c.shape[:2]
            c = cv2.resize(c, (max(4, int(pw * self.scale)), max(4, int(ph * self.scale))),
                           interpolation=cv2.INTER_LINEAR)
        _composite(frame, c, int(self.x), int(self.y), self.life)

def _spawn_hearts(x, y):
    for _ in range(6):
        particles.append(Particle(x, y, random.choice(_HEARTS),
            vx=random.uniform(-80, 80), vy=random.uniform(-220, -100),
            decay=0.55, gravity=30, scale=random.uniform(0.8, 1.5)))

def _spawn_stars(x, y):
    for _ in range(7):
        a  = random.uniform(0, math.tau)
        sp = random.uniform(100, 240)
        particles.append(Particle(x, y, random.choice(_STARS),
            vx=sp * math.cos(a), vy=sp * math.sin(a) - 40,
            decay=0.8, gravity=70, scale=random.uniform(0.8, 1.3), dscale=-0.3))

def _spawn_diamonds(x, y):
    for _ in range(8):
        a  = random.uniform(0, math.tau)
        sp = random.uniform(80, 180)
        particles.append(Particle(x, y, random.choice(_DIAMONDS),
            vx=sp * math.cos(a), vy=sp * math.sin(a),
            decay=0.7, gravity=50, scale=random.uniform(0.7, 1.2)))

def _spawn_sparks(x, y):
    for _ in range(10):
        a  = random.uniform(0, math.tau)
        sp = random.uniform(150, 320)
        particles.append(Particle(x, y, random.choice(_SPARKS),
            vx=sp * math.cos(a), vy=sp * math.sin(a),
            decay=1.6, gravity=90, scale=random.uniform(0.9, 1.4)))

def _spawn_tears(x, y):
    for _ in range(5):
        particles.append(Particle(x, y, random.choice(_TEARS),
            vx=random.uniform(-30, 30), vy=random.uniform(-20, 20),
            decay=0.5, gravity=200, scale=random.uniform(0.8, 1.3)))

_SPAWNERS = {
    "ILoveYou":    _spawn_hearts,
    "Thumb_Up":    _spawn_stars,
    "Victory":     _spawn_diamonds,
    "Pointing_Up": _spawn_sparks,
    "Thumb_Down":  _spawn_tears,
}


# ── main loop ──────────────────────────────────────────────────────────────────

HAND_CONNECTIONS = list(mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

BURST_COOLDOWN     = 1.5
frame_count        = 0
hand_prev_raw      = {}
hand_burst_time    = {}
particles          = []
hand_gesture_state = {}
hand_effect_cd     = {}

fps_history     = deque(maxlen=30)
fps_sum         = 0.0
last_frame_time = time.time()

MODE = 1  # 1=Ambient  2=Spatial  3=Expressive

MODE_HINTS = {
    1: "Tier 1 — Ambient  |  Just show up",
    2: "Tier 2 — Spatial  |  Move left or right  |  Palm or Fist recognized",
    3: "Tier 3 — Expressive  |  Full gesture vocabulary  |  7 gestures",
}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    cv2.convertScaleAbs(frame, frame, alpha=0.6)

    small  = cv2.resize(frame, (INFER_W, INFER_H))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts     = int(time.time() * 1000)
    results = recognizer.recognize_for_video(mp_img, ts)

    h, w, _ = frame.shape
    frame_count += 1

    now = time.time()
    new_fps = 1.0 / max(now - last_frame_time, 1e-6)
    if len(fps_history) == fps_history.maxlen:
        fps_sum -= fps_history[0]
    fps_history.append(new_fps)
    fps_sum += new_fps
    last_frame_time = now
    fps = fps_sum / len(fps_history)
    _dt = min(1.0 / max(new_fps, 1.0), 0.05)
    hud = f"{w}x{h}  {fps:.0f} FPS"
    (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, font_thick)
    cv2.putText(frame, hud, (w - tw - 10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.7, (180, 180, 180), font_thick)

    if results.hand_landmarks:
        # Mode 1: single overlay blend for all hands
        if MODE == 1:
            pulse  = 0.5 + 0.5 * math.sin(frame_count * 0.08)
            radius = int(60 + 18 * math.sin(frame_count * 0.08))
            overlay = frame.copy()
            for hand in results.hand_landmarks:
                px = int(hand[9].x * w)
                py = int(hand[9].y * h)
                cv2.circle(overlay, (px, py), radius, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, pulse * 0.25, frame, 1.0 - pulse * 0.25, 0, frame)

        # Mode 2: single zone overlay blend, then per-hand labels
        elif MODE == 2:
            zone_overlay = frame.copy()
            for hand in results.hand_landmarks:
                if hand[9].x < 0.5:
                    cv2.rectangle(zone_overlay, (0, 0), (w // 2, h), (180, 80, 255), -1)
                else:
                    cv2.rectangle(zone_overlay, (w // 2, 0), (w, h), (50, 200, 255), -1)
            cv2.addWeighted(frame, 0.91, zone_overlay, 0.09, 0, frame)

        for i, hand in enumerate(results.hand_landmarks):
            hand_key = results.handedness[i][0].category_name
            points   = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for conn in HAND_CONNECTIONS:
                cv2.line(frame, points[conn.start], points[conn.end], (70, 70, 70), 1)
            for pt in points:
                cv2.circle(frame, pt, 3, (110, 110, 110), -1)

            px  = int(hand[9].x * w)
            py  = int(hand[9].y * h)
            raw = results.gestures[i][0].category_name if results.gestures[i] else None

            if raw != hand_gesture_state.get(i) and raw in _SPAWNERS:
                if now - hand_effect_cd.get(i, 0) > 0.8:
                    _SPAWNERS[raw](px, py)
                    hand_effect_cd[i] = now
            hand_gesture_state[i] = raw

            if MODE == 1:
                cv2.putText(frame, "Hand Detected", (px - 60, py - radius - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75,
                            (210, 210, 210), font_thick, cv2.LINE_AA)

            elif MODE == 2:
                pos_text = f"x: {hand[9].x:.0%}   y: {hand[9].y:.0%}"
                cv2.putText(frame, pos_text, (px + 15, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65,
                            (190, 190, 190), font_thick, cv2.LINE_AA)
                if raw in TIER2_GESTURES:
                    hand_prev_raw[hand_key] = raw

            elif MODE == 3:
                if raw in GESTURE_LABELS:
                    hand_prev_raw[hand_key] = raw

    # collect active gestures across all hands
    active_gestures = set()
    if results.hand_landmarks:
        for i in range(len(results.hand_landmarks)):
            if results.gestures[i]:
                active_gestures.add(results.gestures[i][0].category_name)

    # tier 3: legend (right side, lights up when active)
    if MODE == 3:
        leg_y = th + 40
        for g_raw, g_label in GESTURE_LABELS.items():
            g_color   = GESTURE_COLORS[g_raw]
            is_active = g_raw in active_gestures
            thickness = font_thick + 1 if is_active else max(1, font_thick - 1)
            dim_color = g_color if is_active else tuple(c // 3 for c in g_color)
            cv2.putText(frame, f"• {g_label}", (w - 210, leg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.58,
                        dim_color, thickness, cv2.LINE_AA)
            leg_y += int(28 * font_scale)

    # mode indicator
    mode_label = "[1] Ambient  [2] Spatial  [3] Expressive"
    cv2.putText(frame, mode_label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.65, (80, 80, 80), font_thick)
    cv2.putText(frame, f"Tier {MODE}", (10, th + int(32 * font_scale) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (210, 210, 210), font_thick)

    # gesture readout — shown below tier label for modes 2 and 3
    readout_y = th + int(65 * font_scale) + 10
    if MODE == 2 or MODE == 3:
        gesture_pool = active_gestures if MODE == 3 else active_gestures & TIER2_GESTURES
        if gesture_pool:
            x_cursor = 10
            for g in gesture_pool:
                if g not in GESTURE_LABELS:
                    continue
                label = GESTURE_LABELS[g]
                color = GESTURE_COLORS[g]
                cv2.putText(frame, label, (x_cursor, readout_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                            color, font_thick, cv2.LINE_AA)
                lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, font_thick)[0][0]
                x_cursor += lw + int(20 * font_scale)
        else:
            cv2.putText(frame, "—", (10, readout_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (70, 70, 70), font_thick)

    hint = MODE_HINTS[MODE]
    cv2.putText(frame, hint, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.65, (130, 130, 130), font_thick)

    particles[:] = [p for p in particles if p.update(_dt)]
    for p in particles:
        p.draw(frame)

    cv2.imshow("Gesture Canvas", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        MODE = 1
    elif key == ord('2'):
        MODE = 2
    elif key == ord('3'):
        MODE = 3
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

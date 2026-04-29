import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
import random
import numpy as np
from collections import deque

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
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
INFER_W, INFER_H = 640, 360   # downsample for ML — landmarks are normalized so coords are unaffected
print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
cv2.namedWindow("Gesture Canvas", cv2.WINDOW_NORMAL)


# ── symbol rendering ───────────────────────────────────────────────────────────

def make_heart_bgr(size, color):
    s = size * 2 + 4
    bgr = np.zeros((s, s, 3), dtype=np.uint8)
    cx, cy = s // 2, s // 2
    pts = []
    for deg in range(0, 360, 3):
        t = math.radians(deg)
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        pts.append((int(cx + x * size / 16), int(cy + y * size / 16)))
    cv2.fillPoly(bgr, [np.array(pts, np.int32)], color)
    return bgr

def make_cross_bgr(size, color):
    s = size * 2 + 4
    bgr = np.zeros((s, s, 3), dtype=np.uint8)
    cx, cy = s // 2, s // 2
    arm_v = size // 2
    arm_h = int(size * 0.42)
    stem  = max(4, size // 6)
    knob  = max(6, int(stem * 1.9))
    bar_y = cy - int(arm_v * 0.28)
    cv2.rectangle(bgr, (cx - stem, cy - arm_v), (cx + stem, cy + arm_v), color, -1)
    cv2.rectangle(bgr, (cx - arm_h, bar_y - stem), (cx + arm_h, bar_y + stem), color, -1)
    cv2.circle(bgr, (cx, cy - arm_v), knob, color, -1)
    cv2.circle(bgr, (cx, cy + arm_v), knob, color, -1)
    cv2.circle(bgr, (cx - arm_h, bar_y), knob, color, -1)
    cv2.circle(bgr, (cx + arm_h, bar_y), knob, color, -1)
    return bgr

def bgr_to_canvas(bgr):
    canvas = np.zeros((*bgr.shape[:2], 4), dtype=np.uint8)
    canvas[:, :, :3] = bgr
    canvas[:, :, 3] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    return canvas

def composite(frame, canvas, cx, cy, alpha):
    if alpha <= 0:
        return
    fh, fw = frame.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = cx - cw // 2, cy - ch // 2
    x2, y2 = x1 + cw, y1 + ch
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(fw, x2), min(fh, y2)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    sx1, sy1 = fx1 - x1, fy1 - y1
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)
    patch = canvas[sy1:sy2, sx1:sx2]
    mask  = (patch[:, :, 3] / 255.0 * alpha)[:, :, np.newaxis]
    roi   = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (patch[:, :, :3] * mask + roi * (1.0 - mask)).astype(np.uint8)


# ── 3D particle system ─────────────────────────────────────────────────────────

VANISH_Z = 500.0

HEART_COLORS = [(100, 60, 220), (80, 40, 255), (140, 80, 200), (60, 100, 240)]
CROSS_COLORS = [(30, 190, 255), (20, 210, 240), (50, 160, 255), (10, 220, 200)]

class Symbol:
    def __init__(self, x, y, kind, frame_w, frame_h):
        self.x        = float(x)
        self.y        = float(y)
        self.z        = 0.0
        self.vx       = random.uniform(-2.0, 2.0)   * frame_w / 1280
        self.vy       = random.uniform(-7.0, -4.5)  * frame_h / 720
        self.vz       = random.uniform(4.0, 9.0)
        self.rot      = random.uniform(0, 360)
        self.rot_spd  = random.uniform(-5.0, 5.0)
        self.base_size = random.randint(45, 80) * frame_w // 1280
        self.pop_scale = random.uniform(1.6, 2.2)   # start big, spring down
        self.kind      = kind
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        palette = HEART_COLORS if kind == "heart" else CROSS_COLORS
        self._color    = random.choice(palette)
        self.age       = 0
        self._make_fn  = make_heart_bgr if kind == "heart" else make_cross_bgr

    @property
    def _perspective(self):
        return max(0.0, 1.0 - self.z / VANISH_Z)

    @property
    def _scale(self):
        # spring pop: starts at pop_scale, settles to 1.0 over ~10 frames
        t = min(1.0, self.age / 10)
        ease = 1 - (1 - t) ** 3
        return self.pop_scale + (1.0 - self.pop_scale) * ease

    def update(self):
        self.x   += self.vx
        self.y   += self.vy
        self.z   += self.vz
        self.vy  *= 0.97
        self.rot += self.rot_spd
        self.age += 1

    def draw(self, frame):
        p = self._perspective
        if p <= 0:
            return
        vp_x = self.frame_w * 0.5
        vp_y = self.frame_h * 0.35
        dx = int(self.x + (vp_x - self.x) * (1.0 - p))
        dy = int(self.y + (vp_y - self.y) * (1.0 - p))
        drawn_size = max(5, int(self.base_size * p * self._scale))
        alpha = p ** 1.4

        M = lambda s: cv2.getRotationMatrix2D((s//2 + 2, s//2 + 2), self.rot, 1.0)

        # glow layer — blurred, larger, low alpha
        glow_size = max(8, int(drawn_size * 1.8))
        glow_bgr  = self._make_fn(glow_size, self._color)
        gs = glow_bgr.shape[0]
        glow_bgr  = cv2.GaussianBlur(glow_bgr, (21, 21), 10)
        glow_bgr  = cv2.warpAffine(glow_bgr, M(gs), (gs, gs))
        glow_c    = bgr_to_canvas(glow_bgr)
        composite(frame, glow_c, dx, dy, alpha * 0.45)

        # crisp layer
        bgr = self._make_fn(drawn_size, self._color)
        s   = bgr.shape[0]
        bgr = cv2.warpAffine(bgr, M(s), (s, s))
        composite(frame, bgr_to_canvas(bgr), dx, dy, alpha)


symbols: list[Symbol] = []


# ── screen flash ───────────────────────────────────────────────────────────────

flash_alpha = 0.0
flash_color = np.zeros((3,), dtype=np.float32)

def trigger_flash(kind):
    global flash_alpha, flash_color
    flash_alpha = 0.35
    flash_color[:] = (100, 60, 200) if kind == "heart" else (30, 180, 255)


# ── main loop ──────────────────────────────────────────────────────────────────

SPAWN_INTERVAL  = 7
BURST_COUNT     = 8
frame_count     = 0
hand_prev_kind  = {}   # hand_key -> last kind, for burst detection

fps_history     = deque(maxlen=30)
last_frame_time = time.time()
GESTURE_TO_KIND = {"Open_Palm": "heart", "Closed_Fist": "cross"}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # dim background so symbols pop
    cv2.convertScaleAbs(frame, frame, alpha=0.6)

    small  = cv2.resize(frame, (INFER_W, INFER_H))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts     = int(time.time() * 1000)
    results = recognizer.recognize_for_video(mp_img, ts)

    h, w, _ = frame.shape
    font_scale = w / 1280
    font_thick = max(1, int(font_scale * 2))
    frame_count += 1

    now = time.time()
    fps_history.append(1.0 / max(now - last_frame_time, 1e-6))
    last_frame_time = now
    fps = sum(fps_history) / len(fps_history)
    hud = f"{w}x{h}  {fps:.0f} FPS"
    (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, font_thick)
    cv2.putText(frame, hud, (w - tw - 10, th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.7, (180, 180, 180), font_thick)

    if results.hand_landmarks:
        for i, hand in enumerate(results.hand_landmarks):
            hand_key = results.handedness[i][0].category_name
            points   = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for conn in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                cv2.line(frame, points[conn.start], points[conn.end], (70, 70, 70), 1)
            for pt in points:
                cv2.circle(frame, pt, 3, (110, 110, 110), -1)

            raw  = results.gestures[i][0].category_name if results.gestures[i] else None
            kind = GESTURE_TO_KIND.get(raw)
            prev = hand_prev_kind.get(hand_key)

            px = int(hand[9].x * w)
            py = int(hand[9].y * h)

            if kind and kind != prev:
                # burst on first trigger
                trigger_flash(kind)
                for _ in range(BURST_COUNT):
                    symbols.append(Symbol(
                        px + random.randint(-50, 50),
                        py + random.randint(-30, 30),
                        kind, w, h
                    ))
            elif kind and frame_count % SPAWN_INTERVAL == 0:
                # steady trickle while held
                symbols.append(Symbol(
                    px + random.randint(-35, 35),
                    py + random.randint(-20, 20),
                    kind, w, h
                ))

            hand_prev_kind[hand_key] = kind

    # screen flash overlay
    if flash_alpha > 0.01:
        cv2.addWeighted(frame, 1.0 - flash_alpha,
                        np.full_like(frame, flash_color.astype(np.uint8)),
                        flash_alpha, 0, frame)
        flash_alpha = max(0.0, flash_alpha - 0.025)

    # update + draw symbols oldest first (painter's order)
    for sym in symbols:
        sym.update()
        sym.draw(frame)
    symbols = [s for s in symbols if s._perspective > 0]

    hint = "Open palm = hearts   |   Fist = crosses"
    cv2.putText(frame, hint, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.65, (130, 130, 130), font_thick)

    cv2.imshow("Gesture Canvas", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

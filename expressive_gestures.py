import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
from collections import deque

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        model_path
    )
    print("Done.")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
cv2.namedWindow("Expressive Gestures", cv2.WINDOW_NORMAL)

# ── gesture colors (BGR) ──────────────────────────────────────────────────────
GESTURE_COLORS = {
    "HEART":   (80,  80,  255),   # red-pink
    "DIAMOND": (255,  0,  200),   # magenta
    "X":       (0,  165,  255),   # orange
    "SPREAD":  (0,  255,  120),   # green
    "FRAME":   (255, 200,   0),   # yellow
}
GESTURE_LABELS = {
    "HEART":   "HEART  - two index tips touching at top, thumbs below",
    "DIAMOND": "DIAMOND - all four tips (index + thumb) forming a square",
    "X":       "X - wrists crossed in front of body",
    "SPREAD":  "SPREAD - both hands open, arms wide apart",
    "FRAME":   "FRAME - index fingers pointing inward, same height",
}

def _d(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

NEAR  = 0.08   # normalized units — ~touching
FAR   = 0.45   # ~arms-width apart

def classify(hands):
    """
    hands: dict of hand_key -> list of 21 landmarks
    Returns a gesture string or None.
    Checked in priority order so overlapping conditions don't conflict.
    """
    if len(hands) < 2:
        return None

    vals = list(hands.values())
    a, b = vals[0], vals[1]

    a_wrist, b_wrist   = a[0],  b[0]
    a_thumb, b_thumb   = a[4],  b[4]
    a_idx,   b_idx     = a[8],  b[8]
    a_mid,   b_mid     = a[12], b[12]

    wrist_dist = _d(a_wrist, b_wrist)
    idx_dist   = _d(a_idx,   b_idx)
    thumb_dist = _d(a_thumb, b_thumb)

    # HEART: index tips close at top, thumbs below forming the bottom point
    if (idx_dist < NEAR
            and a_idx.y < a_thumb.y
            and b_idx.y < b_thumb.y
            and thumb_dist < NEAR * 2.5):
        return "HEART"

    # DIAMOND: index tips close AND thumb tips close (all four fingertips tight)
    if idx_dist < NEAR and thumb_dist < NEAR:
        return "DIAMOND"

    # X: wrists brought together (arms crossing in front of body)
    if wrist_dist < NEAR * 2:
        return "X"

    # SPREAD: both hands open, arms spread wide
    a_open = _d(a_idx, a_wrist) > 0.2 and _d(a_mid, a_wrist) > 0.2
    b_open = _d(b_idx, b_wrist) > 0.2 and _d(b_mid, b_wrist) > 0.2
    if wrist_dist > FAR and a_open and b_open:
        return "SPREAD"

    # FRAME: index tips pointing inward at the same height (camera frame shape)
    a_points_inward = a_idx.x > a_wrist.x   # in mirrored frame: pointing right = inward for left hand
    b_points_inward = b_idx.x < b_wrist.x
    tips_level      = abs(a_idx.y - b_idx.y) < 0.12
    tips_not_too_far = abs(a_idx.x - b_idx.x) < 0.35
    if a_points_inward and b_points_inward and tips_level and tips_not_too_far:
        return "FRAME"

    return None

# ── smoothing ─────────────────────────────────────────────────────────────────
HOLD_FRAMES  = 5   # gesture must be stable this many frames before it fires
gesture_queue = deque(maxlen=HOLD_FRAMES)
active_gesture = None

fps_history    = deque(maxlen=30)
last_frame_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts    = int(time.time() * 1000)
    results = detector.detect_for_video(mp_img, ts)

    h, w, _ = frame.shape
    font_scale   = w / 1280
    font_thick   = max(1, int(font_scale * 2))

    now = time.time()
    fps_history.append(1.0 / max(now - last_frame_time, 1e-6))
    last_frame_time = now
    fps = sum(fps_history) / len(fps_history)
    hud = f"{w}x{h}  {fps:.0f} FPS"
    (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, font_thick)
    cv2.putText(frame, hud, (w - tw - 10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (200, 200, 200), font_thick)

    # ── draw landmarks ────────────────────────────────────────────────────────
    hands_by_key = {}
    if results.hand_landmarks:
        for i, hand in enumerate(results.hand_landmarks):
            key    = results.handedness[i][0].category_name
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            hands_by_key[key] = hand
            for conn in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    # ── gesture classification + temporal smoothing ───────────────────────────
    raw = classify(hands_by_key)
    gesture_queue.append(raw)
    if len(gesture_queue) == HOLD_FRAMES and len(set(gesture_queue)) == 1:
        active_gesture = gesture_queue[0]   # stable for HOLD_FRAMES frames

    # clear when hands leave
    if not results.hand_landmarks:
        gesture_queue.clear()
        active_gesture = None

    # ── gesture overlay ───────────────────────────────────────────────────────
    if active_gesture:
        color = GESTURE_COLORS[active_gesture]
        # big label centered at top of frame
        big_label = active_gesture
        big_scale = font_scale * 3
        big_thick = max(2, font_thick * 2)
        (lw, lh), _ = cv2.getTextSize(big_label, cv2.FONT_HERSHEY_SIMPLEX, big_scale, big_thick)
        cv2.putText(frame, big_label, ((w - lw) // 2, lh + 30), cv2.FONT_HERSHEY_SIMPLEX, big_scale, color, big_thick)
        # hint line below
        hint = GESTURE_LABELS[active_gesture]
        hint_scale = font_scale * 0.55
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hint_scale, font_thick)
        cv2.putText(frame, hint, ((w - hw) // 2, lh + 30 + lh + 10), cv2.FONT_HERSHEY_SIMPLEX, hint_scale, color, font_thick)

    # ── legend (always visible) ───────────────────────────────────────────────
    legend_y = h - int(20 * font_scale) * len(GESTURE_LABELS) - 10
    for name, desc in GESTURE_LABELS.items():
        color = GESTURE_COLORS[name]
        dim   = (0, 0, 0) if active_gesture == name else None
        cv2.putText(frame, desc, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5,
                    color if active_gesture != name else (255, 255, 255), font_thick)
        legend_y += int(22 * font_scale)

    cv2.imshow("Expressive Gestures", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

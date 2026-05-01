import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
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




# ── main loop ──────────────────────────────────────────────────────────────────

HAND_CONNECTIONS = list(mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

BURST_COOLDOWN  = 1.5
frame_count     = 0
hand_prev_raw   = {}
hand_burst_time = {}

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

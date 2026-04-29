import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

GESTURE_MAP = {"Open_Palm": "OPEN", "Closed_Fist": "FIST", "Pointing_Up": "POINT"}
GESTURE_COLORS = {"OPEN": (0, 255, 120), "FIST": (0, 80, 255), "POINT": (255, 200, 0)}

speed_history = {}
SMOOTH_FRAMES = 5
prev_points = {}
prev_time = {}
NORM_TO_M = 1.0
fps_history = deque(maxlen=30)
last_frame_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    results = recognizer.recognize_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    font_scale = w / 1280
    font_thickness = max(1, int(font_scale * 2))

    now = time.time()
    fps_history.append(1.0 / max(now - last_frame_time, 1e-6))
    last_frame_time = now
    fps = sum(fps_history) / len(fps_history)
    hud = f"{w}x{h}  {fps:.0f} FPS"
    (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, font_thickness)
    cv2.putText(frame, hud, (w - tw - 10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (200, 200, 200), font_thickness)

    if results.hand_landmarks:
        for i, hand in enumerate(results.hand_landmarks):
            hand_key = results.handedness[i][0].category_name
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            for connection in connections:
                cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)

            raw_gesture = results.gestures[i][0].category_name if results.gestures[i] else None
            gesture = GESTURE_MAP.get(raw_gesture)
            confidence = round(results.handedness[i][0].score * 100, 1)
            cx, cy = points[0]
            label = f"{hand_key} {confidence}%"
            if gesture:
                label += f"  [{gesture}]"
            color = GESTURE_COLORS.get(gesture, (255, 255, 255))
            cv2.putText(frame, label, (cx, cy - int(20 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, font_thickness)

            lm9 = hand[9]
            cx, cy = points[9]
            current_time = time.time()
            if hand_key in prev_points:
                px_norm, py_norm = prev_points[hand_key]
                dt = current_time - prev_time[hand_key]
                if dt > 0:
                    vx = (lm9.x - px_norm) / dt
                    vy = (lm9.y - py_norm) / dt
                    speed = (vx**2 + vy**2) ** 0.5
                    current_avg = sum(speed_history[hand_key]) / len(speed_history[hand_key]) if hand_key in speed_history and speed_history[hand_key] else speed
                    if speed < max(5 * current_avg, 3.0):
                        if hand_key not in speed_history:
                            speed_history[hand_key] = deque(maxlen=SMOOTH_FRAMES)
                        speed_history[hand_key].append(speed)
                    if hand_key in speed_history and speed_history[hand_key]:
                        smooth_speed = sum(speed_history[hand_key]) / len(speed_history[hand_key])
                        speed_ms = smooth_speed * NORM_TO_M
                        cv2.putText(frame, f"Speed: {smooth_speed:.2f}/s ({speed_ms:.2f}m/s)", (cx, cy - int(40 * font_scale)), cv2.FONT_HERSHEY_COMPLEX, font_scale * 0.7, (255, 255, 0), font_thickness)

            prev_points[hand_key] = (lm9.x, lm9.y)
            prev_time[hand_key] = current_time

    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

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

face_model_path = "face_landmarker.task"
if not os.path.exists(face_model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        face_model_path
    )
    print("Done.")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=face_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=4,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

FLC = mp.tasks.vision.FaceLandmarksConnections
FACE_CONTOUR_CONNS = FLC.FACE_LANDMARKS_CONTOURS
FACE_IRIS_CONNS = list(FLC.FACE_LANDMARKS_LEFT_IRIS) + list(FLC.FACE_LANDMARKS_RIGHT_IRIS)

speed_history = {}
SMOOTH_FRAMES = 10
prev_points = {}
prev_time = {}
NORM_TO_M = 1.0  # 1.0 normalized unit ≈ 1 m (typical webcam view at arm's length)
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
    results = detector.detect_for_video(mp_image, timestamp_ms)
    face_results = face_detector.detect_for_video(mp_image, timestamp_ms)

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

    if face_results.face_landmarks:
        for face in face_results.face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face]
            for conn in FACE_CONTOUR_CONNS:
                cv2.line(frame, pts[conn.start], pts[conn.end], (255, 180, 0), 1)
            for conn in FACE_IRIS_CONNS:
                cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 255), 1)

    if results.hand_landmarks:
        
        for i, hand in enumerate(results.hand_landmarks):
            hand_key = results.handedness[i][0].category_name  # "Left" or "Right" — stable across frames
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            for connection in connections:
                cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)
            for classification in results.handedness[i]:
                confidence = round(classification.score * 100, 1)
                cx, cy = points[0]
                cv2.putText(frame, f"{classification.category_name} {confidence}%", (cx, cy - int(20 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), font_thickness)

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
                    if speed < max(5 * current_avg, 3.0):  # reject tracking glitch spikes
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
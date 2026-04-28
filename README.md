# Hand & Face Tracking

Real-time hand and face tracking using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) and OpenCV. Tracks up to 2 hands and 4 faces simultaneously via webcam.

## Features

- Hand landmark detection with left/right classification and confidence score
- Per-hand speed display in normalized units/s and m/s
- Face mesh overlay with contour and iris connections
- FPS and resolution HUD
- Spike-filtered, smoothed speed history

## Setup

```bash
git clone <repo-url>
cd hand-tracking
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python hand_tracking.py
```

The required MediaPipe models (`hand_landmarker.task`, `face_landmarker.task`) are downloaded automatically on first run.

Press `Q` to quit.

## Requirements

- Python 3.9+
- Webcam
- Windows (tested on Windows 11)

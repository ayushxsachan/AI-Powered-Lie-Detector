# AI Stress & Behavioral Analysis System

Educational, experimental webcam application for real-time facial stress and behavioral analysis. It is not a lie detector and does not claim deception accuracy.

## Features

- MediaPipe Face Mesh tracking with the base 468 landmarks, plus iris refinement when available.
- EAR blink detection with blink count, blink frequency, rapid blinking, long closures, and live EAR display.
- Heuristic eye-contact estimate from iris position and head pose.
- Facial stress proxies for eyebrow tension, lip compression, jaw stiffness, and head shaking.
- Cyberpunk PyQt5 HUD with neon overlays, face mesh visualization, scan lines, alerts, FPS, and animated trend graph.
- Multi-person tracking with primary-face dashboard metrics.
- Optional microphone voice-stress proxy using librosa and sounddevice.
- Optional TensorFlow emotion model loading from `models/emotion_model.keras` or `models/emotion_model.h5`.
- Session recording with annotated MP4, CSV metrics, behavioral summary, and manual CSV/plot export.

## Project Structure

```text
.
├── main.py
├── face_tracker.py
├── blink_detector.py
├── stress_analyzer.py
├── ui_dashboard.py
├── voice_analysis.py
├── utils.py
├── assets/
├── models/
├── requirements.txt
└── README.md
```

## Setup

Python 3.11 or 3.12 is recommended for the smoothest current TensorFlow, MediaPipe, and PyQt5 compatibility.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If Windows blocks the webcam or microphone, enable camera and microphone permissions for desktop apps in system privacy settings.

## Optional Emotion Model

Place a Keras model at one of these paths:

- `models/emotion_model.keras`
- `models/emotion_model.h5`

Optionally add `models/emotion_labels.txt` with one label per line. If no model is present, the application uses transparent heuristics for labels such as Calm, Focused, Nervous, Guarded, Tense, and Avoidant.

## Output Files

Recording creates a timestamped folder under `sessions/`:

- `behavior_metrics.csv`
- `annotated_session.mp4`
- `behavior_summary.txt`

The `SAVE CSV` button also writes a manual report folder with a CSV and a blink/stress trend plot.

## Important Note

The "Truth Probability" value is a heuristic stress-confidence meter derived from observable facial and voice proxies. It should be used only for demos, education, HCI experiments, and visualization work. It must not be used for real investigative, employment, medical, legal, or security decisions.

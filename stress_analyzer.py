from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from blink_detector import BlinkMetrics
from face_tracker import FaceTrack
from utils import RollingWindow, clamp, euclidean, safe_ratio


BROW_POINTS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LIP_POINTS = [61, 291, 13, 14, 0, 17, 78, 308]
JAW_POINTS = [152, 172, 136, 150, 149, 176, 148, 377, 400, 378, 379, 365, 397]


@dataclass
class BehaviorMetrics:
    face_id: int
    faces: int
    truth_probability: float
    stress_score: float
    stress_level: str
    blink_rate_per_min: float
    blink_count: int
    ear: float
    eye_contact_percent: float
    eye_contact_now: bool
    look_away_count: int
    emotion: str
    eyebrow_tension: float
    lip_compression: float
    jaw_stiffness: float
    head_shake_score: float
    head_yaw: float
    head_pitch: float
    gaze_direction: str
    rapid_blinking: bool
    long_eye_closure: bool
    warning: str
    fps: float = 0.0
    voice_stress: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class TensorFlowEmotionModel:
    """Optional Keras model loader. If no model exists, the analyzer uses heuristics."""

    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model = None
        self.labels = ["Neutral", "Calm", "Focused", "Nervous", "Tense", "Agitated"]
        self.available = False
        self.status = "No TensorFlow emotion model loaded"
        self._load()

    def _load(self) -> None:
        candidates = [
            self.model_dir / "emotion_model.keras",
            self.model_dir / "emotion_model.h5",
            self.model_dir / "emotion_model",
        ]
        model_path = next((path for path in candidates if path.exists()), None)
        if model_path is None:
            return
        labels_path = self.model_dir / "emotion_labels.txt"
        if labels_path.exists():
            self.labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(model_path)
            self.available = True
            self.status = f"Loaded emotion model: {model_path.name}"
        except Exception as exc:
            self.status = f"TensorFlow model unavailable: {exc}"
            self.available = False

    def predict(self, face_crop_bgr: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        if not self.available or self.model is None or face_crop_bgr is None or face_crop_bgr.size == 0:
            return None, 0.0
        try:
            input_shape = self.model.input_shape
            height = int(input_shape[1] or 48)
            width = int(input_shape[2] or 48)
            channels = int(input_shape[3] or 1) if len(input_shape) >= 4 else 1
            resized = cv2.resize(face_crop_bgr, (width, height))
            if channels == 1:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                tensor = resized.astype("float32")[None, :, :, None] / 255.0
            else:
                tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")[None, :, :, :] / 255.0
            prediction = np.asarray(self.model.predict(tensor, verbose=0))[0]
            index = int(np.argmax(prediction))
            label = self.labels[index] if index < len(self.labels) else f"Class {index}"
            return label, float(prediction[index])
        except Exception:
            return None, 0.0


class StressAnalyzer:
    def __init__(self, emotion_model: Optional[TensorFlowEmotionModel] = None) -> None:
        self.emotion_model = emotion_model or TensorFlowEmotionModel()
        self._state: Dict[int, Dict[str, object]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(
        self,
        face_id: int,
        faces: int,
        track: FaceTrack,
        blink: BlinkMetrics,
        timestamp: float,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> BehaviorMetrics:
        state = self._state.setdefault(
            face_id,
            {
                "baselines": {},
                "jaw_window": deque(maxlen=90),
                "yaw_window": deque(),
                "contact_window": RollingWindow(60.0),
                "last_contact": True,
                "look_away_count": 0,
            },
        )
        landmarks = track.landmarks
        baselines: Dict[str, float] = state["baselines"]  # type: ignore[assignment]
        face_height = max(euclidean(landmarks[10][:2], landmarks[152][:2]), 1e-4) if len(landmarks) > 152 else 1.0
        face_width = max(euclidean(landmarks[234][:2], landmarks[454][:2]), 1e-4) if len(landmarks) > 454 else 1.0

        brow_to_eye = self._average_brow_eye_distance(landmarks) / face_height
        inner_brow = euclidean(landmarks[107][:2], landmarks[336][:2]) / face_width if len(landmarks) > 336 else 0.45
        mouth_width = euclidean(landmarks[61][:2], landmarks[291][:2]) if len(landmarks) > 291 else 1e-4
        mouth_open = euclidean(landmarks[13][:2], landmarks[14][:2]) if len(landmarks) > 14 else 0.0
        lip_ratio = safe_ratio(mouth_open, mouth_width, 0.0)

        brow_base = self._baseline(baselines, "brow_to_eye", brow_to_eye, alpha=0.008)
        inner_brow_base = self._baseline(baselines, "inner_brow", inner_brow, alpha=0.008)
        lip_base = self._baseline(baselines, "lip_ratio", max(lip_ratio, 0.018), alpha=0.008)

        eyebrow_drop = clamp((brow_base - brow_to_eye) / max(brow_base * 0.28, 1e-4), 0.0, 1.0)
        brow_pinch = clamp((inner_brow_base - inner_brow) / max(inner_brow_base * 0.22, 1e-4), 0.0, 1.0)
        eyebrow_tension = clamp(eyebrow_drop * 0.62 + brow_pinch * 0.38, 0.0, 1.0)

        lip_compression = clamp((lip_base * 0.82 - lip_ratio) / max(lip_base * 0.52, 1e-4), 0.0, 1.0)
        if lip_ratio < 0.028:
            lip_compression = max(lip_compression, 0.35)

        jaw_window: Deque[Tuple[float, float]] = state["jaw_window"]  # type: ignore[assignment]
        jaw_open = mouth_open / face_height
        jaw_window.append((timestamp, jaw_open))
        jaw_values = [value for ts, value in jaw_window if timestamp - ts <= 4.0]
        jaw_motion = float(np.std(jaw_values)) if len(jaw_values) >= 4 else 0.012
        jaw_stiffness = clamp((0.012 - jaw_motion) / 0.012, 0.0, 1.0) * (0.45 + 0.55 * lip_compression)

        yaw = track.head_pose.yaw
        pitch = track.head_pose.pitch
        yaw_window: Deque[Tuple[float, float]] = state["yaw_window"]  # type: ignore[assignment]
        yaw_window.append((timestamp, yaw))
        while yaw_window and yaw_window[0][0] < timestamp - 4.0:
            yaw_window.popleft()
        head_shake_score = self._head_shake_score([value for _, value in yaw_window])

        eye_contact_now, gaze_direction = self._estimate_eye_contact(landmarks, yaw, pitch, blink.long_eye_closure)
        contact_window: RollingWindow = state["contact_window"]  # type: ignore[assignment]
        contact_window.append(timestamp, eye_contact_now)
        contact_values = contact_window.values()
        eye_contact_percent = 100.0 * safe_ratio(sum(1 for item in contact_values if item), len(contact_values), 1.0)
        if bool(state["last_contact"]) and not eye_contact_now:
            state["look_away_count"] = int(state["look_away_count"]) + 1
        state["last_contact"] = eye_contact_now

        blink_stress = clamp((blink.blink_rate_per_min - 18.0) / 24.0, 0.0, 1.0)
        lookaway_stress = clamp((78.0 - eye_contact_percent) / 78.0, 0.0, 1.0)
        closure_stress = 1.0 if blink.long_eye_closure else 0.0

        stress_norm = (
            eyebrow_tension * 0.21
            + lip_compression * 0.19
            + jaw_stiffness * 0.14
            + head_shake_score * 0.16
            + blink_stress * 0.15
            + lookaway_stress * 0.15
            + closure_stress * 0.08
        )
        stress_score = clamp(stress_norm * 100.0, 0.0, 100.0)
        truth_probability = clamp(
            96.0
            - stress_score * 0.58
            - lookaway_stress * 11.0
            - (7.0 if blink.rapid_blinking else 0.0)
            - (8.0 if blink.long_eye_closure else 0.0),
            3.0,
            99.0,
        )

        stress_level = self._stress_level(stress_score)
        model_emotion, confidence = self._model_emotion(frame_bgr, track)
        emotion = model_emotion if model_emotion and confidence >= 0.5 else self._heuristic_emotion(
            stress_score,
            blink,
            eye_contact_percent,
            lip_compression,
            head_shake_score,
        )
        warning = self._warning_text(stress_score, blink, eye_contact_percent, head_shake_score)

        return BehaviorMetrics(
            face_id=face_id,
            faces=faces,
            truth_probability=round(truth_probability, 1),
            stress_score=round(stress_score, 1),
            stress_level=stress_level,
            blink_rate_per_min=round(blink.blink_rate_per_min, 1),
            blink_count=blink.blink_count,
            ear=round(blink.ear, 3),
            eye_contact_percent=round(eye_contact_percent, 1),
            eye_contact_now=eye_contact_now,
            look_away_count=int(state["look_away_count"]),
            emotion=emotion,
            eyebrow_tension=round(eyebrow_tension, 3),
            lip_compression=round(lip_compression, 3),
            jaw_stiffness=round(jaw_stiffness, 3),
            head_shake_score=round(head_shake_score, 3),
            head_yaw=round(yaw, 1),
            head_pitch=round(pitch, 1),
            gaze_direction=gaze_direction,
            rapid_blinking=blink.rapid_blinking,
            long_eye_closure=blink.long_eye_closure,
            warning=warning,
        )

    def _baseline(self, baselines: Dict[str, float], key: str, value: float, alpha: float) -> float:
        if key not in baselines or not math.isfinite(baselines[key]):
            baselines[key] = float(value)
        else:
            baselines[key] = baselines[key] * (1.0 - alpha) + float(value) * alpha
        return baselines[key]

    def _average_brow_eye_distance(self, landmarks: np.ndarray) -> float:
        if len(landmarks) <= 386:
            return 0.05
        distances = [
            euclidean(landmarks[105][:2], landmarks[159][:2]),
            euclidean(landmarks[334][:2], landmarks[386][:2]),
            euclidean(landmarks[66][:2], landmarks[158][:2]),
            euclidean(landmarks[296][:2], landmarks[385][:2]),
        ]
        return float(np.mean(distances))

    def _head_shake_score(self, yaw_values: List[float]) -> float:
        if len(yaw_values) < 6:
            return 0.0
        amplitude = max(yaw_values) - min(yaw_values)
        diffs = np.diff(yaw_values)
        signs = np.sign(diffs[np.abs(diffs) > 0.8])
        sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        amplitude_score = clamp((amplitude - 10.0) / 22.0, 0.0, 1.0)
        oscillation_score = clamp(sign_changes / 5.0, 0.0, 1.0)
        return clamp(amplitude_score * 0.65 + oscillation_score * 0.35, 0.0, 1.0)

    def _estimate_eye_contact(
        self,
        landmarks: np.ndarray,
        yaw: float,
        pitch: float,
        long_eye_closure: bool,
    ) -> Tuple[bool, str]:
        gaze_offset = 0.0
        gaze_direction = "Center"
        if len(landmarks) >= 478:
            left_iris = np.mean(landmarks[468:473, :2], axis=0)
            right_iris = np.mean(landmarks[473:478, :2], axis=0)
            left_min = min(landmarks[33][0], landmarks[133][0])
            left_max = max(landmarks[33][0], landmarks[133][0])
            right_min = min(landmarks[362][0], landmarks[263][0])
            right_max = max(landmarks[362][0], landmarks[263][0])
            left_ratio = safe_ratio(left_iris[0] - left_min, left_max - left_min, 0.5)
            right_ratio = safe_ratio(right_iris[0] - right_min, right_max - right_min, 0.5)
            gaze_ratio = (left_ratio + right_ratio) / 2.0
            gaze_offset = abs(gaze_ratio - 0.5) * 2.0
            if gaze_ratio < 0.42:
                gaze_direction = "Left"
            elif gaze_ratio > 0.58:
                gaze_direction = "Right"
        else:
            gaze_offset = (abs(yaw) / 28.0 + abs(pitch) / 24.0) / 2.0
            if yaw > 14:
                gaze_direction = "Right"
            elif yaw < -14:
                gaze_direction = "Left"
            elif pitch > 14:
                gaze_direction = "Down"
            elif pitch < -14:
                gaze_direction = "Up"

        contact = gaze_offset < 0.42 and abs(yaw) < 17.0 and abs(pitch) < 18.0 and not long_eye_closure
        return bool(contact), gaze_direction

    def _model_emotion(self, frame_bgr: Optional[np.ndarray], track: FaceTrack) -> Tuple[Optional[str], float]:
        if frame_bgr is None or not self.emotion_model.available:
            return None, 0.0
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = track.bbox
        pad = int(max(bw, bh) * 0.08)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        return self.emotion_model.predict(frame_bgr[y1:y2, x1:x2])

    def _heuristic_emotion(
        self,
        stress_score: float,
        blink: BlinkMetrics,
        eye_contact_percent: float,
        lip_compression: float,
        head_shake_score: float,
    ) -> str:
        if blink.long_eye_closure or eye_contact_percent < 35:
            return "Avoidant"
        if head_shake_score > 0.55 and stress_score > 50:
            return "Agitated"
        if blink.rapid_blinking and stress_score > 42:
            return "Nervous"
        if lip_compression > 0.55:
            return "Guarded"
        if stress_score < 24:
            return "Calm"
        if stress_score < 52:
            return "Focused"
        return "Tense"

    def _stress_level(self, stress_score: float) -> str:
        if stress_score >= 68:
            return "HIGH"
        if stress_score >= 38:
            return "MEDIUM"
        return "LOW"

    def _warning_text(
        self,
        stress_score: float,
        blink: BlinkMetrics,
        eye_contact_percent: float,
        head_shake_score: float,
    ) -> str:
        if stress_score >= 68:
            return "Suspicious behavior detected"
        if blink.long_eye_closure:
            return "Long eye closure detected"
        if blink.rapid_blinking:
            return "Rapid blinking anomaly"
        if eye_contact_percent < 35:
            return "Frequent gaze avoidance"
        if head_shake_score > 0.65:
            return "Head movement anomaly"
        return "Nominal behavioral baseline"


def stress_regions(metrics: Mapping[str, object]) -> List[Tuple[str, Iterable[int], float]]:
    return [
        ("brow", BROW_POINTS, float(metrics.get("eyebrow_tension", 0.0) or 0.0)),
        ("lips", LIP_POINTS, float(metrics.get("lip_compression", 0.0) or 0.0)),
        ("jaw", JAW_POINTS, float(metrics.get("jaw_stiffness", 0.0) or 0.0)),
    ]


def generate_behavioral_summary(metrics: Mapping[str, object]) -> str:
    if not metrics or metrics.get("status") == "No face detected":
        return "AI summary: waiting for a stable face track."
    stress = float(metrics.get("stress_score", 0.0) or 0.0)
    blink_rate = float(metrics.get("blink_rate_per_min", 0.0) or 0.0)
    eye_contact = float(metrics.get("eye_contact_percent", 0.0) or 0.0)
    cues: List[str] = []
    if float(metrics.get("eyebrow_tension", 0.0) or 0.0) > 0.45:
        cues.append("brow tension")
    if float(metrics.get("lip_compression", 0.0) or 0.0) > 0.45:
        cues.append("lip compression")
    if float(metrics.get("jaw_stiffness", 0.0) or 0.0) > 0.45:
        cues.append("jaw stiffness")
    if float(metrics.get("head_shake_score", 0.0) or 0.0) > 0.45:
        cues.append("head movement")
    if blink_rate > 24:
        cues.append("rapid blinking")
    if eye_contact < 45:
        cues.append("reduced eye contact")

    cue_text = ", ".join(cues) if cues else "no dominant stress cue"
    level = metrics.get("stress_level", "LOW")
    truth = metrics.get("truth_probability", 0)
    return (
        f"AI summary: {level} stress heuristic, truth-probability meter {truth}%. "
        f"Dominant cues: {cue_text}. Educational analysis only."
    )

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Sequence

import numpy as np

from utils import clamp, euclidean


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class BlinkMetrics:
    face_id: int
    ear_left: float
    ear_right: float
    ear: float
    blink_count: int
    blink_rate_per_min: float
    rapid_blinking: bool
    long_eye_closure: bool
    closed_now: bool
    closure_duration: float


class BlinkDetector:
    """EAR-based blink detector with rolling blink frequency metrics."""

    def __init__(
        self,
        ear_threshold: float = 0.205,
        min_blink_frames: int = 2,
        long_closure_seconds: float = 0.75,
        rapid_blink_rate: float = 25.0,
        rate_window_seconds: float = 60.0,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.min_blink_frames = min_blink_frames
        self.long_closure_seconds = long_closure_seconds
        self.rapid_blink_rate = rapid_blink_rate
        self.rate_window_seconds = rate_window_seconds
        self._state: Dict[int, Dict[str, object]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, face_id: int, landmarks: np.ndarray, timestamp: float) -> BlinkMetrics:
        state = self._state.setdefault(
            face_id,
            {
                "blink_count": 0,
                "closed_frames": 0,
                "closed_start": None,
                "blink_times": deque(),
                "session_start": timestamp,
            },
        )

        ear_left = calculate_ear(landmarks, LEFT_EYE)
        ear_right = calculate_ear(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0
        closed_now = ear < self.ear_threshold
        blink_times: Deque[float] = state["blink_times"]  # type: ignore[assignment]

        if closed_now:
            state["closed_frames"] = int(state["closed_frames"]) + 1
            if state["closed_start"] is None:
                state["closed_start"] = timestamp
        else:
            if int(state["closed_frames"]) >= self.min_blink_frames:
                state["blink_count"] = int(state["blink_count"]) + 1
                blink_times.append(timestamp)
            state["closed_frames"] = 0
            state["closed_start"] = None

        cutoff = timestamp - self.rate_window_seconds
        while blink_times and blink_times[0] < cutoff:
            blink_times.popleft()

        elapsed = max(8.0, timestamp - float(state["session_start"]))
        if elapsed < self.rate_window_seconds:
            blink_rate = len(blink_times) * 60.0 / elapsed
        else:
            blink_rate = len(blink_times) * 60.0 / self.rate_window_seconds

        closure_duration = 0.0
        if state["closed_start"] is not None:
            closure_duration = timestamp - float(state["closed_start"])

        long_eye_closure = closed_now and closure_duration >= self.long_closure_seconds
        return BlinkMetrics(
            face_id=face_id,
            ear_left=ear_left,
            ear_right=ear_right,
            ear=ear,
            blink_count=int(state["blink_count"]),
            blink_rate_per_min=float(blink_rate),
            rapid_blinking=blink_rate >= self.rapid_blink_rate,
            long_eye_closure=long_eye_closure,
            closed_now=closed_now,
            closure_duration=float(closure_duration),
        )


def calculate_ear(landmarks: np.ndarray, eye_indices: Sequence[int]) -> float:
    """Eye Aspect Ratio: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    if len(landmarks) <= max(eye_indices):
        return 0.0
    points = [landmarks[index][:2] for index in eye_indices]
    horizontal = euclidean(points[0], points[3])
    vertical_1 = euclidean(points[1], points[5])
    vertical_2 = euclidean(points[2], points[4])
    return clamp((vertical_1 + vertical_2) / (2.0 * horizontal + 1e-8), 0.0, 1.0)

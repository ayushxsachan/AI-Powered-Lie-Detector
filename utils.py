from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime by the app.
    cv2 = None


Color = Tuple[int, int, int]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t, 0.0, 1.0)


def euclidean(p1: Sequence[float], p2: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))


def safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if abs(denominator) < 1e-8:
        return fallback
    return numerator / denominator


def normalized_to_pixel(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    if landmarks.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    points = np.column_stack((landmarks[:, 0] * width, landmarks[:, 1] * height))
    return np.round(points).astype(np.int32)


class ExponentialSmoother:
    def __init__(self, alpha: float = 0.25, initial: Optional[float] = None) -> None:
        self.alpha = clamp(alpha, 0.0, 1.0)
        self.value = initial

    def update(self, sample: float) -> float:
        if self.value is None or math.isnan(float(self.value)):
            self.value = float(sample)
        else:
            self.value = self.value * (1.0 - self.alpha) + float(sample) * self.alpha
        return float(self.value)


class RollingWindow:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.samples: Deque[Tuple[float, Any]] = deque()

    def append(self, timestamp: float, value: Any) -> None:
        self.samples.append((timestamp, value))
        self.prune(timestamp)

    def prune(self, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.monotonic()
        cutoff = timestamp - self.seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def values(self) -> List[Any]:
        return [value for _, value in self.samples]

    def __len__(self) -> int:
        return len(self.samples)


def to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def draw_neon_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Color = (255, 255, 255),
    scale: float = 0.65,
    thickness: int = 1,
) -> None:
    if cv2 is None:
        return
    x, y = origin
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness + 1, cv2.LINE_AA)
    halo = tuple(int(c * 0.5) for c in color)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, halo, thickness, cv2.LINE_AA)


def draw_transparent_rect(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Color,
    alpha: float = 0.35,
    border_color: Optional[Color] = None,
    border: int = 1,
) -> None:
    if cv2 is None:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, clamp(alpha, 0.0, 1.0), frame, 1.0 - clamp(alpha, 0.0, 1.0), 0, frame)
    if border_color is not None and border > 0:
        cv2.rectangle(frame, pt1, pt2, border_color, border, cv2.LINE_AA)


def draw_scan_lines(frame: np.ndarray, offset: int = 0, color: Color = (255, 80, 80), spacing: int = 24) -> None:
    if cv2 is None:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for y in range(offset % spacing, h, spacing):
        cv2.line(overlay, (0, y), (w, y), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def draw_warning_frame(frame: np.ndarray, intensity: float) -> None:
    if cv2 is None:
        return
    intensity = clamp(intensity, 0.0, 1.0)
    if intensity <= 0.01:
        return
    h, w = frame.shape[:2]
    color = (0, 0, int(255 * intensity))
    thickness = max(2, int(9 * intensity))
    cv2.rectangle(frame, (6, 6), (w - 7, h - 7), color, thickness, cv2.LINE_AA)
    overlay = np.zeros_like(frame)
    overlay[:, :] = color
    cv2.addWeighted(overlay, 0.08 * intensity, frame, 1.0, 0, frame)


def draw_hud_corner_brackets(frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Color = (255, 230, 40)) -> None:
    if cv2 is None:
        return
    x, y, w, h = bbox
    length = max(12, min(w, h) // 5)
    thickness = 2
    corners = [
        ((x, y), (x + length, y), (x, y + length)),
        ((x + w, y), (x + w - length, y), (x + w, y + length)),
        ((x, y + h), (x + length, y + h), (x, y + h - length)),
        ((x + w, y + h), (x + w - length, y + h), (x + w, y + h - length)),
    ]
    for anchor, horizontal, vertical in corners:
        cv2.line(frame, anchor, horizontal, color, thickness, cv2.LINE_AA)
        cv2.line(frame, anchor, vertical, color, thickness, cv2.LINE_AA)


CSV_FIELDS = [
    "wall_time",
    "elapsed_sec",
    "face_id",
    "faces",
    "truth_probability",
    "stress_score",
    "stress_level",
    "blink_rate_per_min",
    "blink_count",
    "ear",
    "eye_contact_percent",
    "emotion",
    "eyebrow_tension",
    "lip_compression",
    "jaw_stiffness",
    "head_shake_score",
    "gaze_direction",
    "warning",
    "fps",
    "voice_stress",
]


def metric_row(metrics: Mapping[str, Any], start_time: float) -> Dict[str, Any]:
    now = time.time()
    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row["wall_time"] = datetime.now().isoformat(timespec="seconds")
    row["elapsed_sec"] = f"{now - start_time:.2f}"
    for field in CSV_FIELDS:
        if field in metrics:
            value = metrics[field]
            if isinstance(value, float):
                row[field] = f"{value:.3f}"
            else:
                row[field] = value
    return row


class SessionWriter:
    def __init__(self, base_dir: str | Path = "sessions", csv_interval: float = 0.5) -> None:
        self.base_dir = Path(base_dir)
        self.csv_interval = csv_interval
        self.session_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.video_path: Optional[Path] = None
        self._csv_file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._video: Optional[Any] = None
        self._last_csv_time = 0.0
        self._start_wall = time.time()

    @property
    def is_open(self) -> bool:
        return self._csv_file is not None

    def start(self, frame_size: Tuple[int, int], fps: float = 24.0) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"session_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.session_dir / "behavior_metrics.csv"
        self.video_path = self.session_dir / "annotated_session.mp4"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._start_wall = time.time()
        self._last_csv_time = 0.0

        if cv2 is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video = cv2.VideoWriter(str(self.video_path), fourcc, max(10.0, fps), frame_size)
            if not self._video.isOpened():
                self._video.release()
                self._video = None
        return self.session_dir

    def write(self, frame_bgr: Optional[np.ndarray], metrics: Mapping[str, Any]) -> None:
        if not self.is_open:
            return
        if frame_bgr is not None and self._video is not None:
            self._video.write(frame_bgr)
        now = time.time()
        if self._writer is not None and now - self._last_csv_time >= self.csv_interval:
            self._writer.writerow(metric_row(metrics, self._start_wall))
            self._last_csv_time = now
            if self._csv_file is not None:
                self._csv_file.flush()

    def stop(self, summary: str = "") -> Optional[Path]:
        output_dir = self.session_dir
        if summary and output_dir is not None:
            (output_dir / "behavior_summary.txt").write_text(summary, encoding="utf-8")
        if self._video is not None:
            self._video.release()
        if self._csv_file is not None:
            self._csv_file.close()
        self._video = None
        self._csv_file = None
        self._writer = None
        self.session_dir = None
        return output_dir


def export_history_csv(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        start_time = time.time()
        for row in rows:
            writer.writerow(metric_row(row, start_time))
    return output


def save_metric_plot(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Optional[Path]:
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    xs = list(range(len(rows)))
    truth = [float(row.get("truth_probability", 0.0) or 0.0) for row in rows]
    stress = [float(row.get("stress_score", 0.0) or 0.0) for row in rows]
    blink = [float(row.get("blink_rate_per_min", 0.0) or 0.0) for row in rows]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4), facecolor="#06080f")
    ax = plt.gca()
    ax.set_facecolor("#06080f")
    ax.plot(xs, truth, label="Truth Probability", color="#20e7ff", linewidth=2)
    ax.plot(xs, stress, label="Stress Score", color="#ff365e", linewidth=2)
    ax.plot(xs, blink, label="Blink Rate/min", color="#ffd84d", linewidth=2)
    ax.set_ylim(0, 105)
    ax.grid(color="#1b2b38", alpha=0.7)
    ax.tick_params(colors="#b9faff")
    for spine in ax.spines.values():
        spine.set_color("#20e7ff")
    ax.legend(facecolor="#09111a", edgecolor="#20e7ff", labelcolor="#e8fdff")
    ax.set_title("Behavioral Analysis Session Trends", color="#e8fdff")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output

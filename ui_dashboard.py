from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from blink_detector import BlinkDetector
from face_tracker import FaceTrack, FaceTracker
from stress_analyzer import BehaviorMetrics, StressAnalyzer, generate_behavioral_summary, stress_regions
from utils import (
    ExponentialSmoother,
    SessionWriter,
    clamp,
    draw_neon_text,
    draw_scan_lines,
    draw_transparent_rect,
    draw_warning_frame,
    export_history_csv,
    save_metric_plot,
)
from voice_analysis import VoiceStressAnalyzer

# Import Qt after MediaPipe. On some Windows builds, loading PyQt5 first can
# make MediaPipe's native framework bindings fail DLL initialization.
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class AnalysisWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_ready = pyqtSignal(str)
    session_ready = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, max_faces: int = 2) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.max_faces = max_faces
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._recording = False
        self._interrogation = False
        self._latest_summary = "AI summary: waiting for analysis."

    def stop(self) -> None:
        self._running.clear()

    def set_recording(self, enabled: bool) -> None:
        with self._lock:
            self._recording = enabled

    def set_interrogation(self, enabled: bool) -> None:
        with self._lock:
            self._interrogation = enabled

    def run(self) -> None:
        cap = None
        tracker = None
        session = SessionWriter()
        try:
            cap = self._open_capture(self.camera_index)
            if not cap.isOpened():
                self.status_ready.emit("Camera unavailable")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            tracker = FaceTracker(max_num_faces=self.max_faces, refine_landmarks=True)
            blink_detector = BlinkDetector()
            stress_analyzer = StressAnalyzer()
            fps_smoother = ExponentialSmoother(alpha=0.18, initial=0.0)
            last_frame_time = time.monotonic()
            frame_index = 0
            last_metrics: Dict[str, object] = {"status": "Initializing camera"}

            self.status_ready.emit(stress_analyzer.emotion_model.status)
            while self._running.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.msleep(15)
                    continue

                frame = cv2.flip(frame, 1)
                timestamp = time.monotonic()
                dt = max(timestamp - last_frame_time, 1e-5)
                last_frame_time = timestamp
                fps = fps_smoother.update(1.0 / dt)

                tracks = tracker.process(frame)
                metric_pairs: List[Tuple[FaceTrack, BehaviorMetrics]] = []
                for track in tracks:
                    blink = blink_detector.update(track.face_id, track.landmarks, timestamp)
                    behavior = stress_analyzer.update(track.face_id, len(tracks), track, blink, timestamp, frame)
                    behavior.fps = round(float(fps), 1)
                    metric_pairs.append((track, behavior))

                annotated = frame.copy()
                tracker.draw_tracks(annotated, tracks)
                primary_metrics = self._select_primary_metrics(metric_pairs, fps)
                for track, behavior in metric_pairs:
                    self._draw_stress_heatmap(annotated, track, behavior.as_dict())

                with self._lock:
                    interrogation = self._interrogation
                    recording = self._recording

                self._draw_camera_overlay(annotated, primary_metrics, frame_index, interrogation)
                qimage = self._to_qimage(annotated)
                self.frame_ready.emit(qimage)
                self.metrics_ready.emit(primary_metrics)
                self._latest_summary = generate_behavioral_summary(primary_metrics)

                if recording and not session.is_open:
                    h, w = annotated.shape[:2]
                    output_dir = session.start((w, h), max(10.0, float(fps or 24.0)))
                    self.session_ready.emit(f"Recording: {output_dir}")
                if not recording and session.is_open:
                    output_dir = session.stop(self._latest_summary)
                    if output_dir is not None:
                        self.session_ready.emit(f"Saved session: {output_dir}")
                if recording and session.is_open:
                    session.write(annotated, primary_metrics)

                last_metrics = primary_metrics
                frame_index += 1
                self.msleep(1)
        except Exception as exc:
            self.status_ready.emit(f"Analysis error: {exc}")
        finally:
            if session.is_open:
                output_dir = session.stop(self._latest_summary)
                if output_dir is not None:
                    self.session_ready.emit(f"Saved session: {output_dir}")
            if tracker is not None:
                tracker.close()
            if cap is not None:
                cap.release()

    def _select_primary_metrics(
        self,
        metric_pairs: List[Tuple[FaceTrack, BehaviorMetrics]],
        fps: float,
    ) -> Dict[str, object]:
        if not metric_pairs:
            return {
                "status": "No face detected",
                "faces": 0,
                "truth_probability": 0.0,
                "stress_score": 0.0,
                "stress_level": "WAIT",
                "blink_rate_per_min": 0.0,
                "blink_count": 0,
                "ear": 0.0,
                "eye_contact_percent": 0.0,
                "emotion": "Scanning",
                "warning": "Acquire face target",
                "fps": round(float(fps), 1),
            }
        primary = max(metric_pairs, key=lambda pair: pair[0].bbox[2] * pair[0].bbox[3])[1]
        return primary.as_dict()

    def _open_capture(self, camera_index: int) -> cv2.VideoCapture:
        backend_candidates = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            if hasattr(cv2, name):
                backend_candidates.append(getattr(cv2, name))
        backend_candidates.append(None)

        for backend in backend_candidates:
            cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(camera_index)

    def _draw_camera_overlay(
        self,
        frame: np.ndarray,
        metrics: Mapping[str, object],
        frame_index: int,
        interrogation: bool,
    ) -> None:
        h, w = frame.shape[:2]
        draw_scan_lines(frame, offset=frame_index * 3, color=(255, 80, 80) if interrogation else (255, 255, 80))
        panel_color = (24, 12, 6) if interrogation else (22, 14, 4)
        border_color = (0, 0, 255) if interrogation else (255, 230, 40)
        draw_transparent_rect(frame, (18, 18), (360, 168), panel_color, 0.48, border_color, 1)
        truth = float(metrics.get("truth_probability", 0.0) or 0.0)
        stress = metrics.get("stress_level", "WAIT")
        blink = float(metrics.get("blink_rate_per_min", 0.0) or 0.0)
        contact = float(metrics.get("eye_contact_percent", 0.0) or 0.0)
        emotion = metrics.get("emotion", "Scanning")
        draw_neon_text(frame, f"TRUTH PROBABILITY  {truth:05.1f}%", (34, 50), (255, 230, 40), 0.62, 1)
        draw_neon_text(frame, f"STRESS LEVEL       {stress}", (34, 80), (60, 60, 255) if stress == "HIGH" else (120, 255, 160), 0.58, 1)
        draw_neon_text(frame, f"BLINK RATE         {blink:04.1f}/MIN", (34, 110), (255, 255, 120), 0.52, 1)
        draw_neon_text(frame, f"EYE CONTACT        {contact:04.1f}%", (34, 138), (255, 255, 120), 0.52, 1)
        draw_neon_text(frame, f"EMOTION            {emotion}", (34, 163), (180, 255, 255), 0.48, 1)
        fps = float(metrics.get("fps", 0.0) or 0.0)
        draw_neon_text(frame, f"FPS {fps:04.1f}", (w - 130, 34), (255, 230, 40), 0.56, 1)
        if interrogation and metrics.get("warning") != "Nominal behavioral baseline":
            pulse = 0.45 + 0.45 * abs(math.sin(frame_index * 0.22))
            draw_warning_frame(frame, pulse)
            draw_neon_text(frame, str(metrics.get("warning", "Alert")), (max(24, w // 2 - 230), h - 42), (30, 30, 255), 0.78, 2)

    def _draw_stress_heatmap(self, frame: np.ndarray, track: FaceTrack, metrics: Mapping[str, object]) -> None:
        overlay = frame.copy()
        for _, indices, intensity in stress_regions(metrics):
            if intensity < 0.22:
                continue
            radius = int(8 + intensity * 14)
            alpha = 0.08 + intensity * 0.25
            for index in indices:
                if index < len(track.pixel_landmarks):
                    point = tuple(int(v) for v in track.pixel_landmarks[index])
                    cv2.circle(overlay, point, radius, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _to_qimage(self, frame_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--", accent: str = "#20e7ff") -> None:
        super().__init__()
        self.setObjectName("MetricCard")
        self.accent = accent
        self.title_label = QLabel(title)
        self.value_label = QLabel(value)
        self.sub_label = QLabel("")
        self.title_label.setObjectName("CardTitle")
        self.value_label.setObjectName("CardValue")
        self.sub_label.setObjectName("CardSub")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)
        self.setStyleSheet(f"QFrame#MetricCard {{ border-color: {self.accent}; }}")

    def set_value(self, value: str, sub: str = "", accent: Optional[str] = None) -> None:
        self.value_label.setText(value)
        self.sub_label.setText(sub)
        if accent and accent != self.accent:
            self.accent = accent
            self.setStyleSheet(f"QFrame#MetricCard {{ border-color: {self.accent}; }}")


class SparklineWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(170)
        self.truth: Deque[float] = deque(maxlen=240)
        self.stress: Deque[float] = deque(maxlen=240)
        self.blink: Deque[float] = deque(maxlen=240)

    def add_metrics(self, metrics: Mapping[str, object]) -> None:
        self.truth.append(float(metrics.get("truth_probability", 0.0) or 0.0))
        self.stress.append(float(metrics.get("stress_score", 0.0) or 0.0))
        self.blink.append(clamp(float(metrics.get("blink_rate_per_min", 0.0) or 0.0) * 3.0, 0.0, 100.0))
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -22)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#07101a"))
        gradient.setColorAt(1.0, QColor("#100713"))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(QPen(QColor("#123648"), 1))
        for i in range(5):
            y = rect.top() + i * rect.height() / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        self._draw_series(painter, rect, list(self.truth), QColor("#20e7ff"))
        self._draw_series(painter, rect, list(self.stress), QColor("#ff365e"))
        self._draw_series(painter, rect, list(self.blink), QColor("#ffd84d"))
        painter.setPen(QColor("#bdfaff"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(12, self.height() - 8, "TRUTH   STRESS   BLINK")

    def _draw_series(self, painter: QPainter, rect, values: List[float], color: QColor) -> None:  # type: ignore[no-untyped-def]
        if len(values) < 2:
            return
        painter.setPen(QPen(color, 2))
        count = len(values)
        previous = None
        for i, value in enumerate(values):
            x = rect.left() + i * rect.width() / max(1, count - 1)
            y = rect.bottom() - clamp(value, 0.0, 100.0) / 100.0 * rect.height()
            point = (int(x), int(y))
            if previous is not None:
                painter.drawLine(previous[0], previous[1], point[0], point[1])
            previous = point


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Stress & Behavioral Analysis System")
        self.resize(1480, 860)
        self.worker: Optional[AnalysisWorker] = None
        self.voice = VoiceStressAnalyzer()
        self.history: Deque[Dict[str, object]] = deque(maxlen=3600)
        self.latest_metrics: Dict[str, object] = {}
        self._recording = False
        self._interrogation = False
        self._build_ui()
        self._apply_style()
        self.voice_timer = QTimer(self)
        self.voice_timer.timeout.connect(self._refresh_voice)
        self.voice_timer.start(900)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self.video_label = QLabel("INITIALIZING CAMERA")
        self.video_label.setObjectName("VideoPanel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(860, 520)
        root.addWidget(self.video_label, 3)

        side = QFrame()
        side.setObjectName("SidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)
        title = QLabel("AI STRESS & BEHAVIORAL ANALYSIS")
        title.setObjectName("Title")
        subtitle = QLabel("Educational heuristic system. Not a lie detector.")
        subtitle.setObjectName("Subtitle")
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.cards = {
            "truth": MetricCard("Truth Probability", "0%", "#20e7ff"),
            "stress": MetricCard("Stress Level", "WAIT", "#ff365e"),
            "blink": MetricCard("Blink Rate", "0/min", "#ffd84d"),
            "contact": MetricCard("Eye Contact", "0%", "#7dff9a"),
            "emotion": MetricCard("Emotion", "Scanning", "#d884ff"),
            "ear": MetricCard("EAR", "0.000", "#20e7ff"),
            "faces": MetricCard("Tracked Faces", "0", "#7dff9a"),
            "voice": MetricCard("Voice Stress", "OFF", "#ffd84d"),
        }
        for index, card in enumerate(self.cards.values()):
            grid.addWidget(card, index // 2, index % 2)
        side_layout.addLayout(grid)

        self.sparkline = SparklineWidget()
        self.sparkline.setObjectName("GraphPanel")
        side_layout.addWidget(self.sparkline)

        self.alert_label = QLabel("AI summary: waiting for analysis.")
        self.alert_label.setObjectName("AlertLabel")
        self.alert_label.setWordWrap(True)
        side_layout.addWidget(self.alert_label)

        controls = QHBoxLayout()
        self.camera_button = QPushButton("STOP")
        self.interrogation_button = QPushButton("INTERROGATE")
        self.record_button = QPushButton("RECORD")
        self.voice_button = QPushButton("VOICE")
        controls.addWidget(self.camera_button)
        controls.addWidget(self.interrogation_button)
        controls.addWidget(self.record_button)
        controls.addWidget(self.voice_button)
        side_layout.addLayout(controls)

        save_row = QHBoxLayout()
        self.save_button = QPushButton("SAVE CSV")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.status_label, 1)
        side_layout.addLayout(save_row)

        self.camera_button.clicked.connect(self.toggle_camera)
        self.interrogation_button.clicked.connect(self.toggle_interrogation)
        self.record_button.clicked.connect(self.toggle_recording)
        self.voice_button.clicked.connect(self.toggle_voice)
        self.save_button.clicked.connect(self.save_report)

        root.addWidget(side, 1)
        self.setCentralWidget(central)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #05070c;
                color: #dffcff;
                font-family: Consolas, Segoe UI, sans-serif;
                letter-spacing: 0px;
            }
            QLabel#VideoPanel {
                background: #02050a;
                border: 1px solid #20e7ff;
                border-radius: 8px;
                color: #20e7ff;
                font-size: 20px;
            }
            QFrame#SidePanel {
                background: rgba(5, 13, 22, 225);
                border: 1px solid #20e7ff;
                border-radius: 8px;
            }
            QLabel#Title {
                color: #e8feff;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#Subtitle {
                color: #7deeff;
                font-size: 11px;
            }
            QFrame#MetricCard {
                background: rgba(7, 18, 28, 210);
                border: 1px solid #20e7ff;
                border-radius: 8px;
            }
            QLabel#CardTitle {
                color: #7deeff;
                font-size: 11px;
            }
            QLabel#CardValue {
                color: #ffffff;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#CardSub {
                color: #9bb7c0;
                font-size: 10px;
            }
            QLabel#AlertLabel {
                background: rgba(18, 7, 14, 210);
                border: 1px solid #ff365e;
                border-radius: 8px;
                padding: 10px;
                color: #ffdbe3;
                font-size: 12px;
            }
            QLabel#StatusLabel {
                color: #9beeff;
                font-size: 11px;
            }
            QPushButton {
                background: rgba(8, 22, 32, 230);
                border: 1px solid #20e7ff;
                border-radius: 7px;
                color: #dffcff;
                padding: 9px 10px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(15, 43, 58, 240);
            }
            QPushButton:pressed {
                background: #20e7ff;
                color: #061018;
            }
            """
        )

    def start_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = AnalysisWorker(camera_index=0, max_faces=3)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.metrics_ready.connect(self._on_metrics)
        self.worker.status_ready.connect(self._on_status)
        self.worker.session_ready.connect(self._on_status)
        self.worker.start()
        self.camera_button.setText("STOP")
        self.status_label.setText("Camera running")

    def toggle_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None
            self.camera_button.setText("START")
            self.video_label.setText("CAMERA STOPPED")
            self.status_label.setText("Camera stopped")
        else:
            self.start_camera()

    def toggle_interrogation(self) -> None:
        self._interrogation = not self._interrogation
        if self.worker is not None:
            self.worker.set_interrogation(self._interrogation)
        self.interrogation_button.setText("NORMAL" if self._interrogation else "INTERROGATE")
        self.status_label.setText("Interrogation mode on" if self._interrogation else "Interrogation mode off")

    def toggle_recording(self) -> None:
        self._recording = not self._recording
        if self.worker is not None:
            self.worker.set_recording(self._recording)
        self.record_button.setText("STOP REC" if self._recording else "RECORD")
        self.status_label.setText("Recording session" if self._recording else "Recording stopped")

    def toggle_voice(self) -> None:
        latest = self.voice.latest()
        if latest.running:
            metrics = self.voice.stop()
        else:
            metrics = self.voice.start()
        self.status_label.setText(metrics.status)
        self._refresh_voice()

    def save_report(self) -> None:
        output_dir = Path("sessions") / f"manual_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_history_csv(list(self.history), output_dir / "behavior_metrics.csv")
        plot_path = save_metric_plot(list(self.history), output_dir / "blink_stress_trends.png")
        summary = generate_behavioral_summary(self.latest_metrics)
        (output_dir / "behavior_summary.txt").write_text(summary, encoding="utf-8")
        if plot_path:
            self.status_label.setText(f"Saved report: {output_dir}")
        else:
            self.status_label.setText(f"Saved CSV: {csv_path}")

    def _on_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_metrics(self, metrics: Dict[str, object]) -> None:
        self.latest_metrics = dict(metrics)
        voice = self.voice.latest()
        if voice.running:
            self.latest_metrics["voice_stress"] = voice.voice_stress
        self.history.append(dict(self.latest_metrics))
        self.sparkline.add_metrics(self.latest_metrics)
        truth = float(metrics.get("truth_probability", 0.0) or 0.0)
        stress_score = float(metrics.get("stress_score", 0.0) or 0.0)
        stress_level = str(metrics.get("stress_level", "WAIT"))
        stress_color = "#ff365e" if stress_level == "HIGH" else "#ffd84d" if stress_level == "MEDIUM" else "#7dff9a"
        self.cards["truth"].set_value(f"{truth:.1f}%", "heuristic meter", "#20e7ff")
        self.cards["stress"].set_value(stress_level, f"{stress_score:.1f}/100", stress_color)
        self.cards["blink"].set_value(f"{float(metrics.get('blink_rate_per_min', 0.0) or 0.0):.1f}/min", f"count {metrics.get('blink_count', 0)}")
        self.cards["contact"].set_value(f"{float(metrics.get('eye_contact_percent', 0.0) or 0.0):.1f}%", str(metrics.get("gaze_direction", "Center")))
        self.cards["emotion"].set_value(str(metrics.get("emotion", "Scanning")), str(metrics.get("warning", "")))
        self.cards["ear"].set_value(f"{float(metrics.get('ear', 0.0) or 0.0):.3f}", "live aspect ratio")
        self.cards["faces"].set_value(str(metrics.get("faces", 0)), f"FPS {float(metrics.get('fps', 0.0) or 0.0):.1f}")
        self.alert_label.setText(generate_behavioral_summary(self.latest_metrics))
        if metrics.get("warning") and metrics.get("warning") != "Nominal behavioral baseline":
            self.status_label.setText(str(metrics.get("warning")))

    def _on_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _refresh_voice(self) -> None:
        metrics = self.voice.latest()
        if metrics.running:
            self.cards["voice"].set_value(f"{metrics.voice_stress:.1f}", f"{metrics.pitch_hz:.0f} Hz")
        elif metrics.available:
            self.cards["voice"].set_value("READY", metrics.status)
        else:
            self.cards["voice"].set_value("OFF", metrics.status)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self.video_label.pixmap() is not None:
            self.video_label.setPixmap(self.video_label.pixmap().scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self.voice.stop()
        event.accept()


def launch() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    QTimer.singleShot(250, window.start_camera)
    return app.exec_()

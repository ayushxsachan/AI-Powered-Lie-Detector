from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Optional

import numpy as np

from utils import clamp


@dataclass
class VoiceMetrics:
    available: bool
    running: bool
    voice_stress: float
    pitch_hz: float
    pitch_variation: float
    pause_ratio: float
    tremor_score: float
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


class VoiceStressAnalyzer:
    """Optional microphone stress proxy using librosa pitch and pause statistics."""

    def __init__(self, sample_rate: int = 16000, window_seconds: float = 2.5) -> None:
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self._sd = None
        self._librosa = None
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: Deque[np.ndarray] = deque(maxlen=64)
        self._lock = threading.Lock()
        self._latest = VoiceMetrics(
            available=False,
            running=False,
            voice_stress=0.0,
            pitch_hz=0.0,
            pitch_variation=0.0,
            pause_ratio=0.0,
            tremor_score=0.0,
            status="Voice analysis off",
        )

    def start(self) -> VoiceMetrics:
        if self._running:
            return self.latest()
        try:
            import librosa
            import sounddevice as sd

            self._librosa = librosa
            self._sd = sd
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            self._running = True
            self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._thread.start()
            self._set_latest(status="Voice analysis running", available=True, running=True)
        except Exception as exc:
            self._running = False
            self._set_latest(status=f"Voice unavailable: {exc}", available=False, running=False)
        return self.latest()

    def stop(self) -> VoiceMetrics:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._set_latest(status="Voice analysis off", running=False)
        return self.latest()

    def latest(self) -> VoiceMetrics:
        with self._lock:
            return VoiceMetrics(**self._latest.as_dict())

    def _audio_callback(self, indata, frames, callback_time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            self._set_latest(status=str(status), available=True, running=self._running)
        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._lock:
            self._samples.append(samples)

    def _analysis_loop(self) -> None:
        while self._running:
            time.sleep(0.7)
            with self._lock:
                chunks = list(self._samples)
            if not chunks:
                continue
            samples = np.concatenate(chunks)
            needed = int(self.sample_rate * self.window_seconds)
            if len(samples) < self.sample_rate:
                continue
            samples = samples[-needed:]
            metrics = self._analyze_samples(samples)
            with self._lock:
                self._latest = metrics

    def _analyze_samples(self, samples: np.ndarray) -> VoiceMetrics:
        librosa = self._librosa
        if librosa is None:
            return self.latest()
        y = samples.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y = y / max(1.0, float(np.max(np.abs(y))))
        try:
            pitches = librosa.yin(y, fmin=65, fmax=420, sr=self.sample_rate)
            pitches = pitches[np.isfinite(pitches)]
            pitches = pitches[(pitches > 65) & (pitches < 420)]
            pitch_hz = float(np.median(pitches)) if len(pitches) else 0.0
            pitch_variation = float(np.std(pitches) / (np.mean(pitches) + 1e-6)) if len(pitches) else 0.0
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
            threshold = max(0.015, float(np.median(rms)) * 0.55)
            pause_ratio = float(np.mean(rms < threshold))
            pitch_diff = np.diff(pitches) if len(pitches) > 3 else np.array([0.0])
            tremor_score = clamp(float(np.std(pitch_diff) / 35.0), 0.0, 1.0)
            stress = clamp(pitch_variation * 120.0 + pause_ratio * 35.0 + tremor_score * 35.0, 0.0, 100.0)
            return VoiceMetrics(
                available=True,
                running=self._running,
                voice_stress=round(stress, 1),
                pitch_hz=round(pitch_hz, 1),
                pitch_variation=round(pitch_variation, 3),
                pause_ratio=round(pause_ratio, 3),
                tremor_score=round(tremor_score, 3),
                status="Voice analysis running",
            )
        except Exception as exc:
            return VoiceMetrics(
                available=True,
                running=self._running,
                voice_stress=0.0,
                pitch_hz=0.0,
                pitch_variation=0.0,
                pause_ratio=0.0,
                tremor_score=0.0,
                status=f"Voice analysis error: {exc}",
            )

    def _set_latest(
        self,
        status: Optional[str] = None,
        available: Optional[bool] = None,
        running: Optional[bool] = None,
        voice_stress: Optional[float] = None,
    ) -> None:
        with self._lock:
            data = self._latest.as_dict()
            if status is not None:
                data["status"] = status
            if available is not None:
                data["available"] = available
            if running is not None:
                data["running"] = running
            if voice_stress is not None:
                data["voice_stress"] = voice_stress
            self._latest = VoiceMetrics(**data)

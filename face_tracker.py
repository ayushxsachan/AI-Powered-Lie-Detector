from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import draw_hud_corner_brackets, draw_neon_text, euclidean, normalized_to_pixel


@dataclass
class HeadPose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    nose_origin: Tuple[int, int] = (0, 0)
    nose_target: Tuple[int, int] = (0, 0)
    solved: bool = False


@dataclass
class FaceTrack:
    face_id: int
    landmarks: np.ndarray
    pixel_landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    head_pose: HeadPose
    landmark_count: int


class FaceTracker:
    """MediaPipe Face Mesh wrapper with stable IDs and OpenCV drawing helpers."""

    def __init__(
        self,
        max_num_faces: int = 2,
        detection_confidence: float = 0.55,
        tracking_confidence: float = 0.55,
        refine_landmarks: bool = True,
    ) -> None:
        self.max_num_faces = max_num_faces
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._previous_centers: Dict[int, Tuple[float, float]] = {}
        self._next_id = 1
        self._connections = [
            (self._mp_face_mesh.FACEMESH_TESSELATION, (40, 95, 105), 1),
            (self._mp_face_mesh.FACEMESH_CONTOURS, (255, 225, 40), 1),
            (self._mp_face_mesh.FACEMESH_LEFT_EYE, (255, 70, 230), 1),
            (self._mp_face_mesh.FACEMESH_RIGHT_EYE, (255, 70, 230), 1),
            (self._mp_face_mesh.FACEMESH_LIPS, (50, 50, 255), 1),
        ]
        if hasattr(self._mp_face_mesh, "FACEMESH_IRISES"):
            self._connections.append((self._mp_face_mesh.FACEMESH_IRISES, (255, 255, 90), 1))

    def close(self) -> None:
        self._face_mesh.close()

    def process(self, frame_bgr: np.ndarray) -> List[FaceTrack]:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            self._previous_centers.clear()
            return []

        raw_tracks: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], HeadPose]] = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            pixels = normalized_to_pixel(landmarks, width, height)
            bbox = self._bbox_from_pixels(pixels, width, height)
            head_pose = self._estimate_head_pose(landmarks, pixels, width, height)
            raw_tracks.append((landmarks, pixels, bbox, head_pose))

        ids = self._assign_ids([bbox for _, _, bbox, _ in raw_tracks], width, height)
        tracks = [
            FaceTrack(
                face_id=face_id,
                landmarks=landmarks,
                pixel_landmarks=pixels,
                bbox=bbox,
                head_pose=head_pose,
                landmark_count=len(landmarks),
            )
            for face_id, (landmarks, pixels, bbox, head_pose) in zip(ids, raw_tracks)
        ]
        return tracks

    def draw_tracks(self, frame_bgr: np.ndarray, tracks: Sequence[FaceTrack]) -> None:
        for track in tracks:
            self._draw_mesh(frame_bgr, track.pixel_landmarks)
            draw_hud_corner_brackets(frame_bgr, track.bbox, (255, 230, 40))
            x, y, w, _ = track.bbox
            draw_neon_text(frame_bgr, f"FACE {track.face_id}", (x, max(24, y - 8)), (255, 230, 40), 0.55, 1)
            if track.head_pose.solved:
                cv2.line(
                    frame_bgr,
                    track.head_pose.nose_origin,
                    track.head_pose.nose_target,
                    (0, 80, 255),
                    2,
                    cv2.LINE_AA,
                )
                pose_text = f"Y {track.head_pose.yaw:+.0f} P {track.head_pose.pitch:+.0f}"
                draw_neon_text(frame_bgr, pose_text, (x, y + track.bbox[3] + 22), (100, 255, 255), 0.48, 1)

    def _bbox_from_pixels(self, pixels: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        if pixels.size == 0:
            return (0, 0, 0, 0)
        base_points = pixels[:468] if len(pixels) >= 468 else pixels
        x_min = int(np.clip(np.min(base_points[:, 0]) - 12, 0, width - 1))
        y_min = int(np.clip(np.min(base_points[:, 1]) - 12, 0, height - 1))
        x_max = int(np.clip(np.max(base_points[:, 0]) + 12, 0, width - 1))
        y_max = int(np.clip(np.max(base_points[:, 1]) + 12, 0, height - 1))
        return (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

    def _assign_ids(self, bboxes: Sequence[Tuple[int, int, int, int]], width: int, height: int) -> List[int]:
        centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in bboxes]
        max_distance = max(width, height) * 0.18
        used_previous = set()
        ids: List[int] = []
        next_centers: Dict[int, Tuple[float, float]] = {}

        for center in centers:
            best_id: Optional[int] = None
            best_distance = float("inf")
            for previous_id, previous_center in self._previous_centers.items():
                if previous_id in used_previous:
                    continue
                distance = euclidean(center, previous_center)
                if distance < best_distance:
                    best_distance = distance
                    best_id = previous_id
            if best_id is None or best_distance > max_distance:
                best_id = self._next_id
                self._next_id += 1
            used_previous.add(best_id)
            ids.append(best_id)
            next_centers[best_id] = center

        self._previous_centers = next_centers
        return ids

    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        pixels: np.ndarray,
        width: int,
        height: int,
    ) -> HeadPose:
        required = [1, 152, 33, 263, 61, 291]
        if len(pixels) <= max(required):
            return HeadPose()

        image_points = np.array([pixels[index] for index in required], dtype=np.float64)
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float64,
        )
        focal_length = float(width)
        camera_matrix = np.array(
            [[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return HeadPose()

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection)
        pitch, yaw, roll = [float(value) for value in euler.flatten()]

        nose_origin = tuple(int(v) for v in image_points[0])
        nose_end_3d = np.array([(0.0, 0.0, 85.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        nose_target = tuple(int(v) for v in nose_end_2d.reshape(-1, 2)[0])
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll, nose_origin=nose_origin, nose_target=nose_target, solved=True)

    def _draw_mesh(self, frame_bgr: np.ndarray, pixels: np.ndarray) -> None:
        if pixels.size == 0:
            return
        for connection_set, color, thickness in self._connections:
            self._draw_connections(frame_bgr, pixels, connection_set, color, thickness)

    @staticmethod
    def _draw_connections(
        frame_bgr: np.ndarray,
        pixels: np.ndarray,
        connections: Iterable[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        count = len(pixels)
        for start, end in connections:
            if start < count and end < count:
                cv2.line(frame_bgr, tuple(pixels[start]), tuple(pixels[end]), color, thickness, cv2.LINE_AA)

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, sqrt
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import serial
import time

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


TIP_IDS: Dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_IDS: Dict[str, int] = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

FINGER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "thumb": (55, 210, 255),
    "index": (80, 220, 120),
    "middle": (255, 210, 80),
    "ring": (255, 140, 90),
    "pinky": (190, 110, 255),
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class LandmarkPoint:
    x: int
    y: int


class HandTracker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._latest_result: Optional[HandLandmarkerResult] = None
        self._result_lock = Lock()
        self._model_path = self._resolve_model_path(model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._handle_result,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def process_frame(self, frame, timestamp_ms: int) -> Tuple[object, List[Dict[str, str]]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._landmarker.detect_async(mp_image, timestamp_ms)

        with self._result_lock:
            result = self._latest_result

        if result is None or not result.hand_landmarks:
            return frame, []

        return self._annotate_frame(frame, result)

    def _handle_result(
        self,
        result: HandLandmarkerResult,
        _output_image: mp.Image,
        _timestamp_ms: int,
    ) -> None:
        with self._result_lock:
            self._latest_result = result

    def _annotate_frame(
        self,
        frame,
        result: HandLandmarkerResult,
    ) -> Tuple[object, List[Dict[str, str]]]:
        frame_height, frame_width = frame.shape[:2]
        hand_summaries: List[Dict[str, str]] = []

        handedness_list = result.handedness or []
        for index, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks = self._to_pixel_landmarks(hand_landmarks, frame_width, frame_height)
            self._draw_hand_skeleton(frame, landmarks)
            finger_states = self._compute_finger_states(landmarks)
            average_bend_angle = self._compute_average_bend_angle(landmarks)

            label = "Hand"
            if index < len(handedness_list) and handedness_list[index]:
                label = handedness_list[index][0].category_name

            hand_summaries.append({"label": label, "index_angle": str(average_bend_angle), **finger_states})
            self._draw_overlay(frame, landmarks, label, finger_states, average_bend_angle)

        return frame, hand_summaries

    def _to_pixel_landmarks(
        self,
        hand_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Dict[int, LandmarkPoint]:
        points: Dict[int, LandmarkPoint] = {}
        for landmark_index, landmark in enumerate(hand_landmarks):
            points[landmark_index] = LandmarkPoint(
                x=int(landmark.x * frame_width),
                y=int(landmark.y * frame_height),
            )
        return points

    def _draw_hand_skeleton(self, frame, landmarks: Dict[int, LandmarkPoint]) -> None:
        for start_id, end_id in HAND_CONNECTIONS:
            start = landmarks[start_id]
            end = landmarks[end_id]
            cv2.line(
                frame,
                (start.x, start.y),
                (end.x, end.y),
                (90, 180, 255),
                2,
                cv2.LINE_AA,
            )

        for landmark_id, point in landmarks.items():
            radius = 6 if landmark_id in TIP_IDS.values() else 4
            color = FINGER_COLORS.get(self._finger_name_for_landmark(landmark_id), (220, 220, 220))
            cv2.circle(frame, (point.x, point.y), radius, color, -1)
            cv2.circle(frame, (point.x, point.y), radius + 2, (30, 30, 30), 1)

    def _finger_name_for_landmark(self, landmark_id: int) -> str:
        if landmark_id <= 4:
            return "thumb"
        if landmark_id <= 8:
            return "index"
        if landmark_id <= 12:
            return "middle"
        if landmark_id <= 16:
            return "ring"
        return "pinky"

    def _compute_finger_states(self, landmarks: Dict[int, LandmarkPoint]) -> Dict[str, str]:
        states: Dict[str, str] = {}

        thumb_tip = landmarks[TIP_IDS["thumb"]]
        thumb_joint = landmarks[PIP_IDS["thumb"]]
        states["thumb"] = "up" if thumb_tip.x > thumb_joint.x else "down"

        for finger_name in ("index", "middle", "ring", "pinky"):
            tip = landmarks[TIP_IDS[finger_name]]
            pip = landmarks[PIP_IDS[finger_name]]
            states[finger_name] = "up" if tip.y < pip.y else "down"

        return states

    def _compute_average_bend_angle(self, landmarks: Dict[int, LandmarkPoint]) -> int:
        finger_joints = {
            "thumb": (2, 3, 4),
            "index": (5, 6, 8),
            "middle": (9, 10, 12),
            "ring": (13, 14, 16),
            "pinky": (17, 18, 20),
        }
        bend_angles: List[int] = []
        for start_id, joint_id, tip_id in finger_joints.values():
            start = landmarks[start_id]
            joint = landmarks[joint_id]
            tip = landmarks[tip_id]
            joint_angle = self._angle_between_points(start, joint, tip)
            bend_angle = int(max(0.0, min(180.0, 180.0 - joint_angle)))
            bend_angles.append(bend_angle)

        return int(sum(bend_angles) / len(bend_angles))

    def _angle_between_points(
        self,
        point_a: LandmarkPoint,
        vertex: LandmarkPoint,
        point_c: LandmarkPoint,
    ) -> float:
        vector_a = (point_a.x - vertex.x, point_a.y - vertex.y)
        vector_c = (point_c.x - vertex.x, point_c.y - vertex.y)

        magnitude_a = sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
        magnitude_c = sqrt(vector_c[0] ** 2 + vector_c[1] ** 2)
        if magnitude_a == 0 or magnitude_c == 0:
            return 180.0

        dot_product = vector_a[0] * vector_c[0] + vector_a[1] * vector_c[1]
        cosine_value = dot_product / (magnitude_a * magnitude_c)
        cosine_value = max(-1.0, min(1.0, cosine_value))
        return degrees(acos(cosine_value))

    def _draw_overlay(
        self,
        frame,
        landmarks: Dict[int, LandmarkPoint],
        label: str,
        finger_states: Dict[str, str],
        index_angle: int,
    ) -> None:
        wrist = landmarks[0]
        cv2.putText(
            frame,
            label,
            (wrist.x - 20, wrist.y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for finger_name, tip_id in TIP_IDS.items():
            tip = landmarks[tip_id]
            color = FINGER_COLORS[finger_name]

            cv2.circle(frame, (tip.x, tip.y), 10, color, -1)
            cv2.circle(frame, (tip.x, tip.y), 14, (255, 255, 255), 2)

            cv2.putText(
                frame,
                finger_name,
                (tip.x + 10, tip.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                finger_states[finger_name],
                (tip.x + 10, tip.y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        index_tip = landmarks[TIP_IDS["index"]]
        cv2.putText(
            frame,
            f"Avg bend: {index_angle}",
            (index_tip.x + 10, index_tip.y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_status_banner(self, frame, fps: float, hand_summaries: List[Dict[str, str]]) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (22, 22, 22), -1)
        cv2.putText(
            frame,
            f"Finger Tracker  FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if hand_summaries:
            summary = hand_summaries[0]
            text = (
                f"{summary['label']} | "
                f"T:{summary['thumb']} I:{summary['index']} "
                f"M:{summary['middle']} R:{summary['ring']} P:{summary['pinky']} "
                f"| Avg bend: {summary['index_angle']}"
            )
        else:
            text = "No hand detected | Avg bend: 0"

        cv2.putText(
            frame,
            text,
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 230, 255),
            2,
            cv2.LINE_AA,
        )

    def _resolve_model_path(self, explicit_model_path: Optional[str]) -> Path:
        candidate_paths: List[Path] = []
        if explicit_model_path:
            candidate_paths.append(Path(explicit_model_path))

        project_root = Path(__file__).resolve().parents[2]
        candidate_paths.extend(
            [
                project_root / "models" / "hand_landmarker.task",
                project_root / "hand_landmarker.task",
            ]
        )

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        searched = "\n".join(f"- {path}" for path in candidate_paths)
        raise FileNotFoundError(
            "Could not find a MediaPipe hand landmarker model.\n"
            "Download `hand_landmarker.task` and place it in one of these locations:\n"
            f"{searched}"
        )


def run(camera_index: int = 0, model_path: Optional[str] = None) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and camera index.")

    mode = input("Test or arduino? ").strip().lower()
    use_arduino = mode in {"arduino", "a", "yes", "y"}
    arduino = None
    if use_arduino:
        arduino = serial.Serial("COM5", 9600)
        time.sleep(2)

    tracker = HandTracker(model_path=model_path)
    tick_frequency = cv2.getTickFrequency()
    previous_tick = cv2.getTickCount()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Failed to read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(cv2.getTickCount() / tick_frequency * 1000)
            processed_frame, hand_summaries = tracker.process_frame(frame, timestamp_ms)

            current_tick = cv2.getTickCount()
            elapsed = (current_tick - previous_tick) / tick_frequency
            previous_tick = current_tick
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            angle = int(hand_summaries[0]["index_angle"]) if hand_summaries else 0
            if arduino is not None:
                arduino.write((str(angle) + "\n").encode())

            tracker.draw_status_banner(processed_frame, fps, hand_summaries)
            cv2.imshow("Finger Tracker", processed_frame)
            time.sleep(0.02)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        if arduino is not None:
            arduino.close()
        capture.release()
        cv2.destroyAllWindows()

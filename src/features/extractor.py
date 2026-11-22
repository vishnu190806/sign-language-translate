from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np

from src.features.spec import FeatureSpec, DEFAULT_FEATURE_SPEC
from src.features.normalize import build_frame_features

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class LandmarkExtractor:
    def __init__(
        self,
        hand_model_path: str = "assets/hand_landmarker.task",
        pose_model_path: str = "assets/pose_landmarker.lite.task",
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.spec = spec
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._hand_callback,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._pose_callback,
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        self._latest_hand_result = None
        self._latest_pose_result = None
        self._timestamp_ms = 0

    def _hand_callback(self, result, output_image, timestamp_ms):
        self._latest_hand_result = result

    def _pose_callback(self, result, output_image, timestamp_ms):
        self._latest_pose_result = result

    def detect_async(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 1
        self._hand_landmarker.detect_async(mp_image, self._timestamp_ms)
        self._pose_landmarker.detect_async(mp_image, self._timestamp_ms)

    def get_latest_features(self) -> np.ndarray | None:
        left_hand = None
        right_hand = None

        if self._latest_hand_result and self._latest_hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(self._latest_hand_result.hand_landmarks):
                handedness = self._latest_hand_result.handedness[i][0].category_name
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
                )
                if handedness == "Left":
                    left_hand = coords
                else:
                    right_hand = coords

        pose_coords = None
        if self._latest_pose_result and self._latest_pose_result.pose_landmarks:
            all_pose = self._latest_pose_result.pose_landmarks[0]
            pose_coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in all_pose], dtype=np.float32
            )

        return build_frame_features(left_hand, right_hand, pose_coords, self.spec)

    def extract_from_frame(self, frame: np.ndarray, max_wait_ms: float = 5.0) -> np.ndarray:
        import time
        t_start = time.perf_counter()
        self.detect_async(frame)
        while (time.perf_counter() - t_start) * 1000 < max_wait_ms:
            if self._latest_hand_result is not None or self._latest_pose_result is not None:
                break
            time.sleep(0.0005)
        features = self.get_latest_features()
        if features is None:
            return np.zeros(self.spec.total_features, dtype=np.float32)
        return features

    def close(self) -> None:
        self._hand_landmarker.close()
        self._pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OfflineExtractor:
    def __init__(
        self,
        hand_model_path: str = "assets/hand_landmarker.task",
        pose_model_path: str = "assets/pose_landmarker.lite.task",
        spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.spec = spec
        self._hand_model_path = hand_model_path
        self._pose_model_path = pose_model_path
        self._num_hands = num_hands
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._ts_base_ms = 0
        self._create_landmarkers()

    def _create_landmarkers(self) -> None:
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._hand_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    def extract_from_video(
        self,
        video_path: str | Path,
        target_fps: int = 30,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(video_fps / target_fps))
        step_ms = int(1000 / target_fps)
        all_features = []
        frame_idx = 0
        ts_ms = self._ts_base_ms

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = self._hand_landmarker.detect_for_video(mp_image, ts_ms)
            pose_result = self._pose_landmarker.detect_for_video(mp_image, ts_ms)

            left_hand = None
            right_hand = None
            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[i][0].category_name
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32
                    )
                    if handedness == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords

            pose_coords = None
            if pose_result.pose_landmarks:
                all_pose = pose_result.pose_landmarks[0]
                pose_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in all_pose], dtype=np.float32
                )

            features = build_frame_features(left_hand, right_hand, pose_coords, self.spec)
            all_features.append(features)
            frame_idx += 1
            ts_ms += step_ms

        cap.release()
        self._ts_base_ms = ts_ms + step_ms
        return np.array(all_features, dtype=np.float16) if all_features else np.zeros((0, self.spec.total_features), dtype=np.float16)

    def reset(self) -> None:
        self.close()
        self._create_landmarkers()

    def close(self) -> None:
        self._hand_landmarker.close()
        self._pose_landmarker.close()

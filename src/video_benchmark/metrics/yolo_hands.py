"""YOLO11 hand/pose detection using ultralytics."""

from __future__ import annotations

import logging

import numpy as np

from video_benchmark.metrics.hand_detection import HandDetectionResult

logger = logging.getLogger(__name__)

_ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass

# COCO pose keypoint indices for wrists/hands
WRIST_LEFT = 9
WRIST_RIGHT = 10
# All keypoints we care about for hand visibility
HAND_KEYPOINTS = [WRIST_LEFT, WRIST_RIGHT]
# Total COCO keypoints
TOTAL_KEYPOINTS = 17


def is_available() -> bool:
    return _ULTRALYTICS_AVAILABLE


class YOLOHandMetric:
    """Hand/pose detection using YOLO11-pose.

    Falls back to MediaPipe HandDetectionMetric if ultralytics not installed.
    """

    def __init__(self, model_name: str = "yolo11n-pose.pt") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not _ULTRALYTICS_AVAILABLE:
            return False
        try:
            self._model = YOLO(self.model_name)
            logger.info(f"Loaded YOLO pose model: {self.model_name}")
            return True
        except Exception:
            logger.exception(f"Failed to load YOLO model: {self.model_name}")
            self._model = None
            return False

    def detect(self, frame: np.ndarray) -> HandDetectionResult | None:
        """Run YOLO pose estimation and extract hand/wrist keypoints.

        Returns HandDetectionResult or None if model unavailable.
        """
        if not self._ensure_model():
            return None

        try:
            results = self._model(frame, verbose=False)
            if not results or len(results) == 0:
                return HandDetectionResult(
                    detected=False, confidence=0.0, landmark_count=0
                )

            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                return HandDetectionResult(
                    detected=False, confidence=0.0, landmark_count=0
                )

            max_confidence = 0.0
            total_visible_keypoints = 0
            hand_detected = False

            # Iterate over detected persons
            for person_idx in range(len(result.keypoints)):
                kpts = result.keypoints[person_idx]

                # Check detection confidence from boxes
                if (
                    result.boxes is not None
                    and person_idx < len(result.boxes)
                ):
                    conf = float(result.boxes[person_idx].conf[0])
                    max_confidence = max(max_confidence, conf)

                # Check wrist keypoints visibility
                if kpts.data is not None and kpts.data.shape[-1] >= 3:
                    kpt_data = kpts.data[0]  # [17, 3] (x, y, conf)
                    for kp_idx in HAND_KEYPOINTS:
                        if kp_idx < len(kpt_data):
                            kp_conf = float(kpt_data[kp_idx, 2])
                            if kp_conf > 0.3:
                                hand_detected = True
                                total_visible_keypoints += 1

                    # Count all visible keypoints
                    for kp_idx in range(min(TOTAL_KEYPOINTS, len(kpt_data))):
                        if float(kpt_data[kp_idx, 2]) > 0.3:
                            total_visible_keypoints += 1

            return HandDetectionResult(
                detected=hand_detected,
                confidence=max_confidence,
                landmark_count=total_visible_keypoints,
            )

        except Exception:
            logger.debug("YOLO detection failed", exc_info=True)
            return None

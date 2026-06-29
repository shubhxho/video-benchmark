"""Hand detection metric — MediaPipe Hands detection rate and quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from video_benchmark.metrics.base import Metric

MAX_LANDMARKS_PER_HAND = 21


@dataclass
class HandDetectionResult:
    detected: bool
    confidence: float
    landmark_count: int


class HandDetectionMetric(Metric):
    name = "hand_detection"

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self.hands: Any | None = None
        solutions = getattr(mp, "solutions", None)
        hands_module = getattr(solutions, "hands", None) if solutions else None
        hands_cls = getattr(hands_module, "Hands", None) if hands_module else None
        if hands_cls is not None:
            self.hands = hands_cls(
                static_image_mode=True,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
            )

    def compute(self, frame: np.ndarray) -> float:
        """Return 1.0 if hands detected, 0.0 otherwise."""
        result = self.detect(frame)
        return 1.0 if result.detected else 0.0

    def detect(self, frame: np.ndarray) -> HandDetectionResult:
        """Run hand detection and return detailed results."""
        if self.hands is None:
            return HandDetectionResult(detected=False, confidence=0.0, landmark_count=0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return HandDetectionResult(detected=False, confidence=0.0, landmark_count=0)

        total_landmarks = 0
        max_confidence = 0.0

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            total_landmarks += len(hand_landmarks.landmark)
            if results.multi_handedness and i < len(results.multi_handedness):
                score = results.multi_handedness[i].classification[0].score
                max_confidence = max(max_confidence, score)

        return HandDetectionResult(
            detected=True,
            confidence=max_confidence,
            landmark_count=total_landmarks,
        )

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()

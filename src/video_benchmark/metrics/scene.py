"""CLIP-based zero-shot scene validation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CLIP_AVAILABLE = False
try:
    import open_clip
    import torch

    _CLIP_AVAILABLE = True
except ImportError:
    pass


SCENE_LABELS = [
    "a photo of a workspace with hands and tools",
    "a photo of hands working on a task",
    "a photo of a ceiling or overhead view",
    "a photo of a wall or flat surface",
    "a photo of a floor",
    "a blurry or obstructed camera image",
    "a dark or underexposed image",
]

# Labels that indicate valid workspace footage
VALID_LABELS = {
    "a photo of a workspace with hands and tools",
    "a photo of hands working on a task",
}


def is_available() -> bool:
    return _CLIP_AVAILABLE


class SceneValidator:
    """Zero-shot scene classification using CLIP.

    Validates whether a frame shows a valid workspace with hands.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._tokenizer = None
        self._preprocess = None
        self._text_features = None
        self._device = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not _CLIP_AVAILABLE:
            logger.warning(
                "open-clip-torch not installed — scene validation unavailable"
            )
            return False
        try:
            self._device = torch.device("cpu")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self._model = model.to(self._device).eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)

            # Pre-compute text features for all labels
            text_tokens = self._tokenizer(SCENE_LABELS).to(self._device)
            with torch.no_grad():
                self._text_features = self._model.encode_text(text_tokens)
                self._text_features /= self._text_features.norm(
                    dim=-1, keepdim=True
                )

            logger.info(f"Loaded CLIP model: {self.model_name}")
            return True
        except Exception:
            logger.exception("Failed to load CLIP model")
            self._model = None
            return False

    def classify(self, frame: np.ndarray) -> dict[str, float] | None:
        """Classify frame against scene labels.

        Returns label → confidence mapping, or None if unavailable.
        """
        if not self._ensure_model():
            return None

        try:
            import cv2
            from PIL import Image

            # BGR → RGB → PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Preprocess and encode
            img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                img_features = self._model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)

                # Cosine similarity
                similarity = (img_features @ self._text_features.T).squeeze(0)
                probs = torch.softmax(similarity * 100, dim=0)

            return {
                label: float(prob)
                for label, prob in zip(SCENE_LABELS, probs)
            }
        except Exception:
            logger.debug("CLIP classification failed", exc_info=True)
            return None

    def is_valid_workspace(
        self, frame: np.ndarray
    ) -> tuple[bool, str] | None:
        """Check if frame shows a valid workspace.

        Returns (is_valid, reason) or None if unavailable.
        """
        result = self.classify(frame)
        if result is None:
            return None

        top_label = max(result, key=result.get)  # type: ignore[arg-type]
        is_valid = top_label in VALID_LABELS
        reason = top_label if not is_valid else "valid_workspace"
        return is_valid, reason

    def compute_validity_score(self, frame: np.ndarray) -> float | None:
        """Return 0-100 score for workspace validity.

        Sum of probabilities for valid workspace labels × 100.
        """
        result = self.classify(frame)
        if result is None:
            return None
        valid_prob = sum(result.get(l, 0.0) for l in VALID_LABELS)
        return min(100.0, valid_prob * 100.0)

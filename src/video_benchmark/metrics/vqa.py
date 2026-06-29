"""Learned no-reference Video Quality Assessment via DOVER.

DOVER (ICCV 2023, VQAssessment) is the state-of-the-art disentangled
no-reference VQA model. Unlike every other metric in this project, which scores
individual frames, DOVER scores a *clip* and disentangles **technical** quality
(noise, blur, compression, exposure) from **aesthetic** quality (composition,
content), then fuses them into a single human-aligned score.

DOVER is an optional dependency. Install it and provide the model config +
weights, e.g.::

    uv pip install "dover @ git+https://github.com/VQAssessment/DOVER.git"
    # download DOVER.pth into ./pretrained_weights/ and point VB_DOVER_CONFIG
    # at the repo's dover.yml (or set VB_DOVER_WEIGHTS).

When unavailable, the metric is a no-op and scoring falls back to the
frame-level image-quality signal.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DOVER_AVAILABLE = False
try:
    import torch  # noqa: F401
    from dover.datasets import (
        UnifiedFrameSampler,
        spatial_temporal_view_decomposition,
    )
    from dover.models import DOVER

    _DOVER_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    pass

# ImageNet normalization used by DOVER.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def is_available() -> bool:
    return _DOVER_AVAILABLE


@dataclass
class VQAResult:
    technical: float  # 0-100
    aesthetic: float  # 0-100
    overall: float  # 0-100 (DOVER fused, human-aligned)


def _find_config() -> Path | None:
    """Locate the DOVER config YAML from env or common locations."""
    env = os.environ.get("VB_DOVER_CONFIG") or os.environ.get("DOVER_CONFIG")
    candidates = [Path(env)] if env else []
    candidates += [
        Path.cwd() / "dover.yml",
        Path.cwd() / "DOVER" / "dover.yml",
        Path.home() / "DOVER" / "dover.yml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


class DOVERVideoQualityMetric:
    """No-reference video quality assessment using DOVER."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self._model: Any | None = None
        self._samplers: dict[str, Any] = {}
        self._sample_types: dict[str, Any] = {}
        self._device: Any | None = None
        self._failed = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._failed or not _DOVER_AVAILABLE:
            return False
        try:
            import yaml

            cfg = Path(self.config_path) if self.config_path else _find_config()
            if cfg is None or not cfg.is_file():
                logger.warning(
                    "DOVER installed but no config found "
                    "(set VB_DOVER_CONFIG to dover.yml) — VQA disabled"
                )
                self._failed = True
                return False

            with open(cfg) as f:
                opt = yaml.safe_load(f)

            weights = os.environ.get("VB_DOVER_WEIGHTS") or opt.get("test_load_path")
            if not weights or not Path(weights).is_file():
                logger.warning(f"DOVER weights not found at {weights!r} — VQA disabled")
                self._failed = True
                return False

            self._device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
            model = DOVER(**opt["model"]["args"]).to(self._device)
            model.load_state_dict(torch.load(weights, map_location=self._device))
            self._model = model.eval()

            dopt = opt["data"]["val-l1080p"]["args"]
            self._sample_types = dopt["sample_types"]
            for stype, sopt in self._sample_types.items():
                if "t_frag" not in sopt:
                    self._samplers[stype] = UnifiedFrameSampler(
                        sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                    )
                else:
                    self._samplers[stype] = UnifiedFrameSampler(
                        sopt["clip_len"] // sopt["t_frag"],
                        sopt["t_frag"],
                        sopt["frame_interval"],
                        sopt["num_clips"],
                    )
            logger.info("Loaded DOVER video-quality model")
            return True
        except Exception:
            logger.exception("Failed to load DOVER model")
            self._failed = True
            self._model = None
            return False

    @staticmethod
    def _fuse(technical: float, aesthetic: float) -> float:
        """DOVER's learned fusion of technical+aesthetic → [0, 1]."""
        import numpy as np

        x = (technical - 0.1107) / 0.07355 * 0.6104 + (
            aesthetic + 0.08285
        ) / 0.03774 * 0.3896
        return float(1.0 / (1.0 + np.exp(-x)))

    def score_video(self, video_path: str) -> VQAResult | None:
        """Score a video clip. Returns 0-100 technical/aesthetic/overall."""
        if not self._ensure_model():
            return None

        try:
            assert self._model is not None
            mean = torch.tensor(_MEAN).reshape(3, 1, 1, 1)
            std = torch.tensor(_STD).reshape(3, 1, 1, 1)

            views, _ = spatial_temporal_view_decomposition(
                video_path, self._sample_types, self._samplers
            )
            for k, v in views.items():
                num_clips = self._sample_types[k].get("num_clips", 1)
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(self._device)
                )

            with torch.no_grad():
                results = [r.mean().item() for r in self._model(views)]

            technical_raw, aesthetic_raw = results[0], results[1]
            overall = self._fuse(technical_raw, aesthetic_raw)
            return VQAResult(
                technical=max(0.0, min(100.0, (technical_raw + 0.1) / 0.3 * 100.0)),
                aesthetic=max(0.0, min(100.0, (aesthetic_raw + 0.1) / 0.3 * 100.0)),
                overall=max(0.0, min(100.0, overall * 100.0)),
            )
        except Exception:
            logger.debug("DOVER scoring failed", exc_info=True)
            return None

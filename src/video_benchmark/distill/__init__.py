"""Knowledge-distillation of the heavy metric stack into one compact (<30 MB) model.

The production pipeline runs several large per-frame networks (learned IQA, a CLIP
scene model, DOVER, depth) plus classical OpenCV metrics. This subpackage distils
the *per-frame quality signals* into a single small backbone (MobileCLIP-S0) with
multi-task heads, so inference is one forward pass instead of a stack of models.

Targets are the per-frame, higher-is-better quality dimensions
(see :data:`teacher.TARGETS`). Inter-frame / per-clip signals (stability, temporal,
hands, audio) stay as cheap CV at inference and are out of scope here.
"""

from __future__ import annotations

from video_benchmark.distill.model import CompactQualityNet
from video_benchmark.distill.teacher import TARGETS, TeacherLabeler

__all__ = ["TARGETS", "CompactQualityNet", "TeacherLabeler"]

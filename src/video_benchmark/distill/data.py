"""Frame sampling, teacher labelling, and backbone-feature caching for distillation.

With a frozen backbone the student's trainable part is tiny, so we precompute the
backbone embedding for every sampled frame once and train heads on the cache. This
turns each training epoch into a cheap matrix op and keeps the whole run fast.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from video_benchmark.distill.model import CompactQualityNet
from video_benchmark.distill.teacher import TeacherLabeler

logger = logging.getLogger(__name__)


@dataclass
class FeatureCache:
    features: np.ndarray  # (N, feat_dim) backbone embeddings
    targets: np.ndarray  # (N, num_targets) teacher scores, 0..100
    groups: np.ndarray  # (N,) clip path per frame, for leakage-free splits

    def __len__(self) -> int:
        return int(self.features.shape[0])


def build_backbone_transform(backbone: object) -> Callable[..., torch.Tensor]:
    """Return the timm preprocessing transform for a backbone (PIL -> tensor)."""
    import timm

    cfg = timm.data.resolve_model_data_config(backbone)  # type: ignore[attr-defined,no-untyped-call]
    transform: Callable[..., torch.Tensor] = timm.data.create_transform(  # type: ignore[attr-defined]
        **cfg, is_training=False
    )
    return transform


def sample_frames(
    video_path: str, fps_sample: float = 2.0, max_frames: int = 400
) -> list[np.ndarray]:
    """Sample ~``fps_sample`` BGR frames per second from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(0.1, fps_sample))))
    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def _pyiqa_tensor(frames: list[np.ndarray], size: int = 224) -> torch.Tensor:
    """BGR frames -> [0,1] RGB tensor (N,3,size,size) for the pyiqa teacher."""
    out = []
    for f in frames:
        rgb = cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB)
        out.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


def build_feature_cache(
    video_paths: list[Path],
    model: CompactQualityNet,
    teacher: TeacherLabeler,
    device: str,
    fps_sample: float = 2.0,
    batch_size: int = 32,
) -> FeatureCache:
    """Sample frames from each clip, compute teacher targets + backbone features."""
    transform = build_backbone_transform(model.backbone)
    model.backbone.eval().to(device)

    feats: list[np.ndarray] = []
    targs: list[np.ndarray] = []
    groups: list[str] = []

    for vp in video_paths:
        frames = sample_frames(str(vp), fps_sample=fps_sample)
        if not frames:
            logger.warning("No frames sampled from %s", vp)
            continue
        logger.info("Labelling %d frames from %s", len(frames), vp.name)
        for i in range(0, len(frames), batch_size):
            chunk = frames[i : i + batch_size]
            # teacher targets (classical + deep IQA)
            iqa_in = _pyiqa_tensor(chunk)
            targs.append(teacher.label_frames(chunk, iqa_in))
            # backbone features (frozen)
            pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in chunk]
            x = torch.stack([transform(p) for p in pil]).to(device)
            with torch.no_grad():
                emb = model.backbone(x).detach().cpu().numpy()
            feats.append(emb)
            groups.extend([str(vp)] * len(chunk))

    return FeatureCache(
        features=np.concatenate(feats, axis=0),
        targets=np.concatenate(targs, axis=0),
        groups=np.asarray(groups),
    )


def split_by_clip(
    cache: FeatureCache, val_substr: str = "clip2"
) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-free split: frames from clips whose path contains ``val_substr`` are val."""
    is_val = np.array([val_substr in g for g in cache.groups])
    if not is_val.any() or is_val.all():
        # Fallback: deterministic 80/20 frame split.
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(cache))
        cut = int(0.8 * len(cache))
        train_idx = np.zeros(len(cache), dtype=bool)
        train_idx[perm[:cut]] = True
        return train_idx, ~train_idx
    return ~is_val, is_val

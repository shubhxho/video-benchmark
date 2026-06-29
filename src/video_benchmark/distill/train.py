"""Distillation training loop — fit the student heads to the teacher targets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from video_benchmark.distill.data import FeatureCache
from video_benchmark.distill.model import CompactQualityNet

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    best_val_loss: float
    epochs_run: int
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def train_heads(
    model: CompactQualityNet,
    cache: FeatureCache,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: str,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 40,
) -> TrainHistory:
    """Train trunk+heads on cached features; record the loss curve."""
    feats = torch.from_numpy(cache.features).float().to(device)
    targs = torch.from_numpy(cache.targets).float().to(device) / 100.0  # train in [0,1]

    xt, yt = feats[train_idx], targs[train_idx]
    xv, yv = feats[val_idx], targs[val_idx]

    trunk_and_heads = model.head_parameters()
    opt = torch.optim.AdamW(trunk_and_heads, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    bad = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    model.trunk.train()

    for epoch in range(epochs):
        model.trunk.train()
        opt.zero_grad()
        pred = model.forward_from_features(xt) / 100.0
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        sched.step()

        model.trunk.eval()
        with torch.no_grad():
            vpred = model.forward_from_features(xv) / 100.0
            vloss = float(loss_fn(vpred, yv))
        train_losses.append(float(loss.detach()))
        val_losses.append(vloss)
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if "backbone" not in k
            }
            bad = 0
        else:
            bad += 1
        if epoch % 25 == 0 or epoch == epochs - 1:
            logger.info("epoch %3d  train=%.4f  val=%.4f", epoch, float(loss.detach()), vloss)
        if bad >= patience:
            logger.info("early stop at epoch %d (best val=%.4f)", epoch, best_val)
            break

    if best_state:
        model.load_state_dict(best_state, strict=False)
    return TrainHistory(
        best_val_loss=best_val,
        epochs_run=epoch + 1,
        train_losses=train_losses,
        val_losses=val_losses,
    )

"""The compact multi-task student: one small backbone, several tiny heads.

A single MobileCLIP-S0 image tower (~10.9 M params, ~22 MB fp16 / ~11 MB int8)
produces a shared embedding; a small shared trunk feeds per-target linear heads.
One forward pass yields every per-frame quality dimension at once, replacing the
stack of teacher networks.
"""

from __future__ import annotations

import torch
from torch import nn

from video_benchmark.distill.teacher import TARGETS

DEFAULT_BACKBONE = "hf_hub:apple/mobileclip_s0_timm"


class CompactQualityNet(nn.Module):
    """MobileCLIP-S0 backbone + multi-task regression heads (outputs 0..100)."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        num_targets: int = len(TARGETS),
        trunk_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        feat_dim = int(self.backbone.num_features)  # type: ignore[arg-type]
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, trunk_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # One linear head per target keeps each signal independently calibrated.
        self.heads = nn.ModuleList(nn.Linear(trunk_dim, 1) for _ in range(num_targets))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                return torch.as_tensor(self.backbone(x))
        return torch.as_tensor(self.backbone(x))

    def forward_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.trunk(feats)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)  # (N, num_targets), raw units (~0..100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_features(self.forward_features(x))

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.trunk.parameters()) + list(self.heads.parameters())

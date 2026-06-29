"""Load and run the distilled compact quality model.

The checkpoint saved by ``python -m video_benchmark.distill`` is self-contained:
backbone + trunk + heads weights (fp16) plus the backbone name and target order.
This module reconstructs the network and predicts the per-frame quality vector in
one forward pass.

CLI:
    uv run python -m video_benchmark.distill.infer <image-or-video> [--ckpt PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from video_benchmark.distill.data import build_backbone_transform
from video_benchmark.distill.model import CompactQualityNet


def _device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CompactQualityModel:
    """Runnable wrapper around a distilled checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = _device(device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.targets: list[str] = list(payload["targets"])
        self.backbone_name: str = payload["backbone_name"]
        net = CompactQualityNet(
            backbone_name=self.backbone_name,
            num_targets=len(self.targets),
            pretrained=False,
            freeze_backbone=True,
        )
        state = {k: v.float() for k, v in payload["state_dict"].items()}
        net.load_state_dict(state, strict=True)
        net.eval().to(self.device)
        self.net = net
        self._transform = build_backbone_transform(net.backbone)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> dict[str, float]:
        """Predict the per-target quality scores (0..100) for one BGR frame."""
        return self.predict_batch([frame_bgr])[0]

    @torch.no_grad()
    def predict_batch(self, frames_bgr: list[np.ndarray]) -> list[dict[str, float]]:
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        x = torch.stack([self._transform(p) for p in pil]).to(self.device)
        out = self.net(x).clamp(0.0, 100.0).cpu().numpy()
        return [{name: float(row[i]) for i, name in enumerate(self.targets)} for row in out]


def load(
    checkpoint: str | Path = "models/compact_quality.pt", device: str = "auto"
) -> CompactQualityModel:
    return CompactQualityModel(checkpoint, device=device)


def _first_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read a frame from {path}")
        return frame
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"could not read image {path}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the distilled compact quality model.")
    ap.add_argument("input", type=Path, help="image or video file")
    ap.add_argument("--ckpt", type=Path, default=Path("models/compact_quality.pt"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model = load(args.ckpt, device=args.device)
    scores = model.predict(_first_frame(args.input))
    width = max(len(k) for k in scores)
    print(f"compact_quality  ·  {model.backbone_name}  ·  {model.device}")
    for name, value in scores.items():
        bar = "█" * int(value / 5)
        print(f"  {name:<{width}}  {value:6.1f}  {bar}")


if __name__ == "__main__":
    main()

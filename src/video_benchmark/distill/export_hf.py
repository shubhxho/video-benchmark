"""Package the distilled checkpoint as a Hugging Face model repo and push it.

Builds a self-contained bundle (safetensors weights + config.json + model card)
and optionally uploads it to the Hub. Pushing needs a token (``hf auth login`` or
``HF_TOKEN``); building the bundle does not.

    uv run python -m video_benchmark.distill.export_hf \
        --repo shubhxho/video-benchmark-compact-quality --push
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from video_benchmark.distill.model import CompactQualityNet
from video_benchmark.distill.teacher import DEEP

_REPO_URL = "https://github.com/shubhxho/video-benchmark"


def _model_card(repo: str, cfg: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    deep = ", ".join(DEEP)
    targets = cast("list[str]", cfg["targets"])
    speed = ""
    if metrics:
        s = metrics.get("speed", {})
        speed = (
            f"- **Throughput:** {s.get('speedup_throughput', 0):.1f}× the teacher "
            f"stack ({s.get('student_fps', 0):.0f} vs {s.get('teacher_fps', 0):.0f} fps)\n"
        )
    return f"""---
license: other
license_name: apple-amlr
license_link: https://github.com/apple/ml-mobileclip/blob/main/LICENSE
library_name: pytorch
tags:
- video-quality-assessment
- image-quality-assessment
- knowledge-distillation
- mobileclip
- robotics
- first-person-video
---

# Compact Operator-Video Quality Model

A single **{cfg['params_millions']:.1f}M-param** multi-task model that scores
first-person / operator video frame quality in **one forward pass**, distilled
from a much heavier teacher stack (learned IQA + a CLIP scene model + classical
OpenCV metrics). It is ~**{cfg['fp16_mb']:.0f} MB** (fp16) / **{cfg['int8_mb']:.0f} MB** (int8).

Backbone: **MobileCLIP-S0** (`{cfg['backbone']}`). Heads: small per-target linear
regressors. Outputs are on a 0–100, higher-is-better scale.

## Targets

`{', '.join(targets)}` — the **deep** signals ({deep}) are the ones the
student genuinely reproduces (a learned model is needed for them); the rest are
exact OpenCV stats included for a unified read-out and better computed directly.

## Performance (distillation fidelity vs. the teacher)

{speed}- Learned IQA reproduced at high correlation; see the repo for the full report.

## Usage

```python
import cv2
from huggingface_hub import hf_hub_download
# clone {_REPO_URL} for the model code (video_benchmark.distill)
from video_benchmark.distill.infer import CompactQualityModel

ckpt = hf_hub_download("{repo}", "compact_quality.pt")
model = CompactQualityModel(ckpt)
frame = cv2.imread("frame.jpg")          # or a decoded video frame (BGR)
print(model.predict(frame))              # {{'brightness': ..., 'iqa': ..., 'scene': ...}}
```

## How it was trained

Self-distillation: the production pipeline at {_REPO_URL} is run over an
operator-video corpus to produce per-frame teacher labels, and this compact
student is trained to regress them. Train it on your own footage with
`python -m video_benchmark.distill`.

## License

The trained heads are MIT (this project). The bundled backbone weights derive
from Apple **MobileCLIP-S0**, released under the **apple-amlr** research license —
review it before commercial use.
"""


def build_bundle(
    checkpoint: Path,
    out_dir: Path,
    metrics_json: Path | None = None,
) -> dict[str, object]:
    """Write safetensors weights, config.json, README.md into ``out_dir``."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targets = list(payload["targets"])
    backbone = payload["backbone_name"]

    # safetensors weights (contiguous fp16)
    state = {k: v.half().contiguous() for k, v in payload["state_dict"].items()}
    save_file(state, str(out_dir / "model.safetensors"))
    # keep the torch-native checkpoint too (what infer.py loads directly)
    torch.save(payload, out_dir / "compact_quality.pt")

    n_params = sum(v.numel() for v in payload["state_dict"].values())
    net = CompactQualityNet(backbone, num_targets=len(targets), pretrained=False)
    import timm

    data_cfg = timm.data.resolve_model_data_config(  # type: ignore[attr-defined,no-untyped-call]
        net.backbone
    )
    cfg: dict[str, object] = {
        "model_type": "compact_quality_net",
        "architecture": "MobileCLIP-S0 backbone + multi-task linear heads",
        "backbone": backbone,
        "targets": targets,
        "deep_targets": DEEP,
        "task": "multi-task frame-quality regression (0-100, higher is better)",
        "framework": "pytorch",
        "input_size": list(data_cfg.get("input_size", [3, 256, 256])),
        "mean": list(data_cfg.get("mean", [0, 0, 0])),
        "std": list(data_cfg.get("std", [1, 1, 1])),
        "params_millions": n_params / 1e6,
        "fp16_mb": n_params * 2 / 1e6,
        "int8_mb": n_params / 1e6,
        "distilled_from": ["pyiqa topiq_nr", "MobileCLIP2-S0 zero-shot scene", "OpenCV metrics"],
        "source": _REPO_URL,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    metrics = None
    if metrics_json and metrics_json.exists():
        metrics = json.loads(metrics_json.read_text())
        (out_dir / "results.json").write_text(json.dumps(metrics, indent=2))

    (out_dir / "README.md").write_text(_model_card("REPO_PLACEHOLDER", cfg, metrics))
    return cfg


def push(out_dir: Path, repo_id: str, private: bool = False) -> str:
    """Create the repo (if needed) and upload the bundle. Requires a token."""
    from huggingface_hub import HfApi

    # finalize the model card with the real repo id
    card = out_dir / "README.md"
    card.write_text(card.read_text().replace("REPO_PLACEHOLDER", repo_id))

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Package + push the compact model to the HF Hub.")
    ap.add_argument("--ckpt", type=Path, default=Path("models/compact_quality.pt"))
    ap.add_argument("--out", type=Path, default=Path("hf_export"))
    ap.add_argument("--metrics", type=Path, default=None, help="results.json from --emit-json")
    ap.add_argument("--repo", default="shubhxho/video-benchmark-compact-quality")
    ap.add_argument("--push", action="store_true", help="upload to the Hub (needs a token)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    cfg = build_bundle(args.ckpt, args.out, args.metrics)
    print(f"bundle ready in {args.out}/  ({cfg['params_millions']:.1f}M params, "
          f"{cfg['fp16_mb']:.0f} MB fp16)")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")

    if args.push:
        url = push(args.out, args.repo, private=args.private)
        print(f"\npushed → {url}")
    else:
        print(f"\nnot pushed (use --push). target repo: {args.repo}")


if __name__ == "__main__":
    main()

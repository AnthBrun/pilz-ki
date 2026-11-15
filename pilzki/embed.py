"""Step 3: compute frozen backbone features for every packed photo.

    python -m pilzki embed --backbone fungi-effv2b3

Writes work/emb/<backbone>/{features.npy, ids.json, meta.json}. Features are
float16, one row per id. Run it for a few backbones and compare them with
`train-heads` + `evaluate` (linear probe) before fine-tuning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import DATA, read_json, read_jsonl, work_dir, write_json


def emb_dir(backbone: str) -> Path:
    from .backbones import safe_name

    return work_dir() / "emb" / safe_name(backbone)


def load_embeddings(backbone: str):
    d = emb_dir(backbone)
    if not (d / "features.npy").exists():
        raise SystemExit(f"no embeddings in {d}; run `pilzki embed --backbone {backbone}` first")
    return np.load(d / "features.npy"), read_json(d / "ids.json"), read_json(d / "meta.json")


def compute(backbone_spec: str, rows: list[dict], batch_size: int, workers: int, tta: bool):
    import torch

    from .backbones import Backbone, device, eval_transform
    from .model import predict

    dev = device()
    backbone = Backbone(backbone_spec).to(dev).eval()
    transform = eval_transform(backbone.preprocess)
    feats = predict(backbone, None, rows, transform, dev, batch_size, workers)
    if tta:
        import torchvision.transforms as T

        flip = T.Compose([T.RandomHorizontalFlip(p=1.0), transform])
        feats = (feats + predict(backbone, None, rows, flip, dev, batch_size, workers)) / 2
    return feats, backbone


def main(args) -> None:
    from .backbones import resolve
    from .model import existing

    rows = existing(list(read_jsonl(DATA / "manifest.jsonl")))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no packed images found; run `pilzki pack` or set PILZKI_IMAGES")
    print(f"embedding {len(rows)} images with {resolve(args.backbone)}")
    feats, backbone = compute(args.backbone, rows, args.batch_size, args.workers, args.tta)
    out = emb_dir(args.backbone)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", feats.astype(np.float16))
    write_json(out / "ids.json", [r["id"] for r in rows])
    write_json(out / "meta.json", {"backbone": backbone.spec, "feat_dim": int(feats.shape[1]), "preprocess": backbone.preprocess.to_json(), "tta": args.tta})
    print(f"wrote {feats.shape} to {out}")

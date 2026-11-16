"""Step 6: train the "is this a fungus photo?" gate for a run.

    python -m pilzki train-gate --run work/runs/ft-fungi-effv2b3-final --negatives work/neg/coco

Without a gate the model prefills tags on anything: on a uniform species
distribution the species marginal equals the tags' base rates (hutpilz ~0.7),
so cakes, grey walls or spore prints get `hutpilz`, `auf_boden`, `lamellen`.
Thresholding the top species probability does not help: real photos are often
unsure about the species (median top probability ~0.3) while their tags are
still right.

The gate is one logistic unit on the run's backbone features, trained on
  positives  fruit-body photos of the crawl (train split),
  negatives  the crawl's microscopy and graphic photos, any image folders given
             with --negatives (e.g. COCO val2017 in work/neg/coco), and
             synthetic images (solid colours, noise, gradients, blurred photos).
The threshold keeps --keep (default 99 %) of the val fruit-body photos.
Writes <run>/gate.pt (weight, bias, threshold) and <run>/gate_report.md;
`export` adds the output `fungus_prob` and meta.json["gate"].
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .common import DATA, image_path, read_jsonl
from .labels import TRAIN_PHOTO_TYPES

NEGATIVE_PHOTO_TYPES = ("microscopy", "graphic")  # spore_print is too mixed (lichens, moulds, fruit bodies)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def synthetic_images(n: int, seed: int, blur_sources: list[Path] = ()):
    """Images without any photo content: solid colours, noise, gradients, heavily blurred photos."""
    from PIL import Image, ImageFilter, ImageOps

    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        kind = k % 4 if blur_sources else k % 3
        if kind == 0:
            out.append(Image.new("RGB", (640, 480), tuple(int(v) for v in rng.integers(0, 256, 3))))
        elif kind == 1:
            out.append(Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)))
        elif kind == 2:
            a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            w = np.linspace(0, 1, 640)[None, :, None]
            out.append(Image.fromarray((a * (1 - w) + b * w).repeat(480, 0).astype(np.uint8)))
        else:
            src = blur_sources[int(rng.integers(0, len(blur_sources)))]
            with Image.open(src) as im:
                out.append(ImageOps.exif_transpose(im).convert("RGB").filter(ImageFilter.GaussianBlur(25)))
    return out


def held_out(path: Path, share: float = 0.2) -> bool:
    """Stable split of extra negative images by file name."""
    return int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def fit_logistic(x_pos: np.ndarray, x_neg: np.ndarray, weight_decay: float = 1e-3, steps: int = 300):
    """Class-balanced logistic regression; standardisation folded into (weight, bias) of raw features."""
    import torch

    x = torch.as_tensor(np.concatenate([x_pos, x_neg]), dtype=torch.float32)
    y = torch.cat([torch.ones(len(x_pos)), torch.zeros(len(x_neg))])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    sample_w = torch.where(y == 1, 0.5 / len(x_pos), 0.5 / len(x_neg))
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y, reduction="none") * sample_w).sum()
        loss = loss + weight_decay * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w, b = w.detach(), b.detach()
    return (w / sd).numpy(), float(b - (w * mu / sd).sum())


def gate_scores(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(features @ weight + bias)))


def main(args) -> None:
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset

    from .backbones import autocast, device, eval_transform
    from .export import load_classifier

    run = Path(args.run)
    model, config, _, _ = load_classifier(run)
    dev = device()
    model = model.to(dev).eval()
    tf = eval_transform(model.backbone.preprocess)
    rng = np.random.default_rng(args.seed)

    class Items(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            if isinstance(it, Image.Image):
                return tf(it)
            with Image.open(it) as im:
                return tf(ImageOps.exif_transpose(im).convert("RGB"))

    @torch.no_grad()
    def features(items) -> np.ndarray:
        if not items:
            return np.zeros((0, model.backbone.feat_dim), np.float32)
        workers = 0 if isinstance(items[0], Image.Image) else args.workers
        out = []
        for x in DataLoader(Items(items), batch_size=args.batch_size, num_workers=workers):
            with autocast(dev):
                out.append(model.backbone(x.to(dev)).float().cpu())
        return torch.cat(out).numpy()

    def sample(xs, n):
        return [xs[i] for i in rng.choice(len(xs), min(n, len(xs)), replace=False)] if n else list(xs)

    ptype = {r["id"]: r["type"] for r in read_jsonl(DATA / "phototype.jsonl")}
    manifest = [r for r in read_jsonl(DATA / "manifest.jsonl") if image_path(r["id"]).exists()]
    train_splits = ("train", "holdout") if config.get("final") else ("train",)

    def crawl(types, splits, n):
        return sample([image_path(r["id"]) for r in manifest if ptype.get(r["id"]) in types and r["split"] in splits], n)

    extra = sorted(p for d in args.negatives for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    extra_train = sample([p for p in extra if not held_out(p)], args.max_extra)
    extra_test = [p for p in extra if held_out(p)]
    print(f"positives: crawl fruit bodies | negatives: crawl {'/'.join(NEGATIVE_PHOTO_TYPES)}, "
          f"{len(extra)} images from {len(args.negatives)} folder(s), synthetic", flush=True)

    sets = {
        "pos_train": crawl(TRAIN_PHOTO_TYPES, train_splits, args.max_positives),
        "pos_val": crawl(TRAIN_PHOTO_TYPES, ("val",), 3000),
        "pos_test": crawl(TRAIN_PHOTO_TYPES, ("test",), 3000),
        "crawl_neg_train": crawl(NEGATIVE_PHOTO_TYPES, train_splits, 0),
        "crawl_neg_test": crawl(NEGATIVE_PHOTO_TYPES, ("val", "test"), 0),
        "extra_train": extra_train,
        "extra_test": extra_test,
        "synthetic_train": synthetic_images(400, args.seed + 1, extra_train),
        "synthetic_test": synthetic_images(200, args.seed + 2, extra_test),
    }
    F = {}
    for name, items in sets.items():
        print(f"features {name}: {len(items)}", flush=True)
        F[name] = features(items)

    neg = np.concatenate([F["crawl_neg_train"], F["extra_train"], F["synthetic_train"]])
    weight, bias = fit_logistic(F["pos_train"], neg, weight_decay=args.weight_decay)
    val_scores = gate_scores(F["pos_val"], weight, bias)
    threshold = float(np.quantile(val_scores, 1 - args.keep))

    rows = [
        ("val fruit bodies (calibration)", "pos_val"),
        ("test fruit bodies", "pos_test"),
        ("crawl microscopy/graphic (val+test)", "crawl_neg_test"),
        ("extra negatives, held out", "extra_test"),
        ("synthetic, other seed", "synthetic_test"),
    ]
    lines = [f"# Gate: {run.name}", "", f"- threshold {threshold:.4f} (keeps {args.keep:.0%} of val fruit bodies)",
             f"- trained on {len(F['pos_train'])} positives, {len(neg)} negatives", "", "| set | photos | pass |", "|---|---|---|"]
    for label, key in rows:
        if len(F[key]):
            lines.append(f"| {label} | {len(F[key])} | {np.mean(gate_scores(F[key], weight, bias) >= threshold):.1%} |")
    report = "\n".join(lines) + "\n"
    torch.save({"weight": torch.as_tensor(weight), "bias": bias, "threshold": threshold, "keep": args.keep,
                "negatives": [str(d) for d in args.negatives]}, run / "gate.pt")
    (run / "gate_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"gate saved to {run / 'gate.pt'}; rerun `export` to include it")


def add_arguments(p) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--negatives", nargs="*", default=[], help="folders with non-fungus images (e.g. work/neg/coco)")
    p.add_argument("--keep", type=float, default=0.99, help="share of val fruit-body photos that must pass")
    p.add_argument("--max-positives", type=int, default=10000)
    p.add_argument("--max-extra", type=int, default=20000, help="max. extra negative images used for training")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

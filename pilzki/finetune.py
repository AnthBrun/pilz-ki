"""Step 6: fine-tune backbone + heads end to end on the photos (GPU, hours).

    python -m pilzki finetune --backbone fungi-effv2b3 --init-heads work/runs/heads-fungi-effv2b3
    python -m pilzki evaluate --run work/runs/ft-fungi-effv2b3

Writes model.pt, config.json and preds_<split>.npz like `train-heads`, so
`evaluate` and `export` work the same on both run types.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np

from .common import read_json, work_dir, write_json
from .labels import load_labels, load_manifest, splits_for
from .train_heads import class_weights


def main(args) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm

    from .backbones import Backbone, autocast, device, eval_transform, safe_name, train_transform
    from .model import Heads, ImageDataset, existing, predict, save_predictions

    torch.manual_seed(args.seed)
    dev = device()
    labels = load_labels()
    rows = existing(load_manifest())
    if args.limit:
        rows = rows[: args.limit]
    splits = splits_for(args.final)
    train_rows = [r for r in rows if r["split"] in splits["train"]]
    eval_rows = {s: [r for r in rows if r["split"] == s] for s in splits["eval"]}
    val_rows = eval_rows["val"][: args.val_limit] if args.val_limit else eval_rows["val"]
    print(f"train {len(train_rows)} | " + " | ".join(f"{s} {len(v)}" for s, v in eval_rows.items()))

    backbone = Backbone(args.backbone).to(dev)
    heads = Heads(backbone.feat_dim, labels.n_tags, labels.n_species, hidden=args.hidden, dropout=args.dropout).to(dev)
    if args.init_heads:
        init = Path(args.init_heads)
        cfg = read_json(init / "config.json")
        if cfg["heads"]["feat_dim"] != backbone.feat_dim or cfg["tags"] != labels.tags or cfg["slugs"] != labels.slugs:
            raise SystemExit("--init-heads run does not match this backbone/data")
        heads = Heads(**cfg["heads"]).to(dev)
        heads.load_state_dict(torch.load(init / "heads.pt", map_location=dev))
    M = torch.as_tensor(labels.species_tags, device=dev)

    y_train = np.array([labels.species_index[r["slug"]] for r in train_rows])
    sampler = WeightedRandomSampler(class_weights(y_train, labels.n_species, args.balance).tolist(), num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(
        ImageDataset(train_rows, train_transform(backbone.preprocess), labels.species_index),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        ImageDataset(val_rows, eval_transform(backbone.preprocess), labels.species_index),
        batch_size=args.batch_size * 2, num_workers=args.workers, pin_memory=True,
    )

    opt = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": args.lr_backbone},
        {"params": heads.parameters(), "lr": args.lr_heads},
    ], weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    use_scaler = dev.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def loss_fn(tl, sl, y):
        return args.tag_weight * F.binary_cross_entropy_with_logits(tl.float(), M[y]) + F.cross_entropy(sl.float(), y, label_smoothing=args.label_smoothing)

    run = Path(args.run) if args.run else work_dir() / "runs" / f"ft-{safe_name(args.backbone)}{'-final' if args.final else ''}"
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "type": "finetune",
        "backbone": backbone.spec,
        "preprocess": backbone.preprocess.to_json(),
        "heads": heads.config,
        "tags": labels.tags,
        "slugs": labels.slugs,
        "final": args.final,
        "args": vars(args),
    }
    write_json(run / "config.json", config)

    best_loss, history = float("inf"), []
    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        for p in backbone.parameters():
            p.requires_grad_(not frozen)
        backbone.train()
        heads.train()
        start, total = time.time(), 0.0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast(dev):
                tl, sl = heads(backbone(x))
            loss = loss_fn(tl, sl, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(heads.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        backbone.eval()
        heads.eval()
        val_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with autocast(dev):
                    tl, sl = heads(backbone(x))
                val_loss += loss_fn(tl, sl, y).item() * len(y)
                correct += (sl.argmax(-1) == y).sum().item()
                n += len(y)
        val_loss /= max(1, n)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, len(train_loader)), "val_loss": val_loss, "species_top1": correct / max(1, n), "minutes": (time.time() - start) / 60})
        print(history[-1])
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"backbone": backbone.state_dict(), "heads": heads.state_dict()}, run / "model.pt")
        write_json(run / "history.json", history)

    ckpt = torch.load(run / "model.pt", map_location=dev)
    backbone.load_state_dict(ckpt["backbone"])
    heads.load_state_dict(ckpt["heads"])
    for split, rs in eval_rows.items():
        if rs:
            out = predict(backbone, heads, rs, eval_transform(backbone.preprocess), dev, args.batch_size * 2, args.workers)
            save_predictions(run / f"preds_{split}.npz", [r["id"] for r in rs], [r["slug"] for r in rs], out[:, : labels.n_tags], out[:, labels.n_tags:])
    print(f"run saved to {run}")


def add_arguments(p) -> None:
    p.add_argument("--backbone", required=True)
    p.add_argument("--init-heads", help="run dir of `train-heads` for the same backbone")
    p.add_argument("--run", help="output dir (default work/runs/ft-<backbone>)")
    p.add_argument("--final", action="store_true", help="also train on holdout species (for the deployed model)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--freeze-epochs", type=int, default=1, help="epochs with frozen backbone (heads warm-up)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-backbone", type=float, default=5e-5)
    p.add_argument("--lr-heads", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--tag-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--balance", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-limit", type=int, default=0, help="validate on at most N photos per epoch (0 = all)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N photos (smoke tests)")
    p.add_argument("--seed", type=int, default=0)

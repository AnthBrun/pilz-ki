"""Step 4: train the two heads on frozen embeddings (minutes on a GPU).

    python -m pilzki train-heads --backbone fungi-effv2b3
    python -m pilzki evaluate --run work/runs/heads-fungi-effv2b3

Used to compare backbones cheaply (linear probe) and as initialisation for
fine-tuning. Writes heads.pt, config.json and preds_<split>.npz to the run dir.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .common import work_dir, write_json
from .labels import load_labels, load_manifest, splits_for


def class_weights(species_idx: np.ndarray, n_species: int, power: float) -> np.ndarray:
    """Per-sample sampling weight ~ count(species)^-power (0 = plain, 0.5 = sqrt-balanced)."""
    counts = np.bincount(species_idx, minlength=n_species).astype(np.float64)
    return (counts[species_idx] ** -power) if power else np.ones(len(species_idx))


def main(args) -> None:
    import torch
    import torch.nn.functional as F

    from .backbones import device, safe_name
    from .embed import load_embeddings
    from .model import Heads, save_predictions

    torch.manual_seed(args.seed)
    dev = device()
    labels = load_labels()
    feats, ids, meta = load_embeddings(args.backbone)
    pos = {i: k for k, i in enumerate(ids)}
    rows = [r for r in load_manifest() if r["id"] in pos]
    splits = splits_for(args.final)
    train_rows = [r for r in rows if r["split"] in splits["train"]]
    eval_rows = {s: [r for r in rows if r["split"] == s] for s in splits["eval"]}
    print(f"train {len(train_rows)} | " + " | ".join(f"{s} {len(v)}" for s, v in eval_rows.items()))

    M = torch.as_tensor(labels.species_tags, device=dev)

    def tensors(rs):
        x = torch.as_tensor(feats[[pos[r["id"]] for r in rs]].astype(np.float32), device=dev)
        y = torch.as_tensor([labels.species_index[r["slug"]] for r in rs], device=dev)
        return x, y

    x_train, y_train = tensors(train_rows)
    if not eval_rows.get("val"):
        raise SystemExit("no val photos with embeddings; embed more images")
    x_val, y_val = tensors(eval_rows["val"])

    heads = Heads(int(meta["feat_dim"]), labels.n_tags, labels.n_species, hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_rows) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    weights = torch.as_tensor(class_weights(y_train.cpu().numpy(), labels.n_species, args.balance), device=dev)

    def loss_fn(tl, sl, y):
        return args.tag_weight * F.binary_cross_entropy_with_logits(tl, M[y]) + F.cross_entropy(sl, y, label_smoothing=args.label_smoothing)

    best, best_loss = None, float("inf")
    for epoch in range(args.epochs):
        heads.train()
        order = torch.multinomial(weights, len(train_rows), replacement=True) if args.balance else torch.randperm(len(train_rows), device=dev)
        total = 0.0
        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            x = x_train[idx]
            if args.feature_noise:
                x = x + args.feature_noise * x.std() * torch.randn_like(x)
            tl, sl = heads(x)
            loss = loss_fn(tl, sl, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        heads.eval()
        with torch.no_grad():
            tl, sl = heads(x_val)
            val_loss = loss_fn(tl, sl, y_val).item()
            top1 = (sl.argmax(-1) == y_val).float().mean().item()
        if val_loss < best_loss:
            best, best_loss = copy.deepcopy(heads.state_dict()), val_loss
        print(f"epoch {epoch + 1:3d} train {total / steps_per_epoch:.4f} val {val_loss:.4f} species@1 {top1:.3f}")

    heads.load_state_dict(best)
    run = Path(args.run) if args.run else work_dir() / "runs" / f"heads-{safe_name(args.backbone)}{'-final' if args.final else ''}"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), run / "heads.pt")
    write_json(run / "config.json", {
        "type": "heads",
        "backbone": meta["backbone"],
        "preprocess": meta["preprocess"],
        "heads": heads.config,
        "tags": labels.tags,
        "slugs": labels.slugs,
        "final": args.final,
        "args": vars(args),
    })
    heads.eval()
    with torch.no_grad():
        for split, rs in eval_rows.items():
            if not rs:
                continue
            x, _ = tensors(rs)
            tl, sl = heads(x)
            save_predictions(run / f"preds_{split}.npz", [r["id"] for r in rs], [r["slug"] for r in rs], tl.cpu().numpy(), sl.cpu().numpy())
    print(f"run saved to {run}")


def add_arguments(p) -> None:
    p.add_argument("--backbone", required=True)
    p.add_argument("--run", help="output dir (default work/runs/heads-<backbone>)")
    p.add_argument("--final", action="store_true", help="also train on holdout species (for the deployed model)")
    p.add_argument("--hidden", type=int, default=0, help="hidden layer size, 0 = linear heads")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tag-weight", type=float, default=1.0, help="weight of the tag loss vs. the species loss")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--balance", type=float, default=0.5, help="species sampling: count^-balance (0 = off)")
    p.add_argument("--feature-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

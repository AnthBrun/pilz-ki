"""Step 5: calibrate per-tag thresholds and measure what prefill would do.

    python -m pilzki evaluate --run work/runs/heads-fungi-effv2b3 [--target-precision 0.9]

The web filter ANDs tags. One wrong prefilled tag removes the right species
from the results, so the key metric is **survival**: the share of photos for
which every prefilled tag is a tag of the true species. Thresholds are tuned
on `val` for a per-tag precision target; survival is reported on the other
splits (test = seen species, new photos; holdout = species never trained on).

Writes thresholds.json (consumed by `export`) and report.md to the run dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .labels import load_labels

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def tag_probs(tag_logits, species_logits, species_tags, alpha):
    """Numpy twin of model.combine."""
    p_species = softmax(species_logits.astype(np.float64))
    return alpha * sigmoid(tag_logits.astype(np.float64)) + (1 - alpha) * p_species @ species_tags, p_species


def average_precision(scores, y) -> float | None:
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-scores, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def threshold_for_precision(scores, y, target: float, min_support: int):
    """Lowest threshold whose top-k precision is >= target (max recall), or None."""
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], y[order]
    precision = np.cumsum(t) / np.arange(1, len(t) + 1)
    ok = np.nonzero(precision >= target)[0]
    ok = ok[ok + 1 >= min_support]
    if len(ok) == 0:
        return None
    k = ok.max()
    return float(s[k])


def prefill(p, groups, tag_index, thresholds, require_p) -> list[str]:
    """Tags the web app would preselect for one probability vector."""
    chosen = []
    for g in groups:
        if g["requires"] and p[tag_index[g["requires"]]] < require_p:
            continue
        if g["exclusive"]:
            best = max(g["tags"], key=lambda t: p[tag_index[t]])
            thr = thresholds.get(best)
            if thr is not None and p[tag_index[best]] >= thr:
                chosen.append(best)
        else:
            chosen += [t for t in g["tags"] if thresholds.get(t) is not None and p[tag_index[t]] >= thresholds[t]]
    return chosen


def load_preds(run: Path, split: str):
    path = run / f"preds_{split}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def split_metrics(name, probs, p_species, y_species, labels, thresholds, rng, photos_per_query=3):
    tag_index = {t: i for i, t in enumerate(labels.tags)}
    Y = labels.species_tags[y_species]
    out = {"split": name, "photos": int(len(y_species))}
    rank = (p_species > p_species[np.arange(len(y_species)), y_species][:, None]).sum(1)
    out["species_top1"] = float((rank == 0).mean())
    out["species_top5"] = float((rank < 5).mean())
    aps = [average_precision(probs[:, j], Y[:, j]) for j in range(labels.n_tags)]
    out["tag_map"] = float(np.mean([a for a in aps if a is not None]))

    def survival(prob_rows, species_rows):
        n_tags, alive, remaining = [], [], []
        for p, sp in zip(prob_rows, species_rows):
            tags = prefill(p, labels.groups, tag_index, thresholds, labels.require_p)
            idx = [tag_index[t] for t in tags]
            n_tags.append(len(tags))
            alive.append(bool(labels.species_tags[sp, idx].all()) if idx else True)
            remaining.append(int(labels.species_tags[:, idx].all(axis=1).sum()) if idx else labels.n_species)
        return float(np.mean(n_tags)), float(np.mean(alive)), float(np.median(remaining))

    out["prefill_tags"], out["survival"], out["remaining_species_median"] = survival(probs, y_species)

    # Several photos of one find (e.g. top, underside, stem): average the probabilities.
    groups = {}
    for i, sp in enumerate(y_species):
        groups.setdefault(int(sp), []).append(i)
    multi_p, multi_sp = [], []
    for sp, idx in groups.items():
        if len(idx) >= 2:
            pick = rng.choice(idx, size=min(photos_per_query, len(idx)), replace=False)
            multi_p.append(probs[pick].mean(0))
            multi_sp.append(sp)
    if multi_p:
        n, s, r = survival(np.array(multi_p), np.array(multi_sp))
        out.update({"multi_queries": len(multi_p), "multi_prefill_tags": n, "multi_survival": s, "multi_remaining_species_median": r})
    return out


def main(args) -> None:
    run = Path(args.run)
    labels = load_labels()
    config = read_json(run / "config.json")
    if config["tags"] != labels.tags or config["slugs"] != labels.slugs:
        raise SystemExit("run was trained with a different data/ (tags or species changed); retrain")
    preds = {s: load_preds(run, s) for s in ("val", "test", "holdout")}
    val = preds["val"]
    if val is None:
        raise SystemExit(f"{run}/preds_val.npz missing")
    sidx = labels.species_index
    y_val = np.array([sidx[s] for s in val["slugs"]])
    Y_val = labels.species_tags[y_val]

    # alpha: mix of attribute head and species-head marginal, chosen on val.
    if args.alpha is None:
        scores = {}
        for a in ALPHAS:
            pv, _ = tag_probs(val["tag_logits"], val["species_logits"], labels.species_tags, a)
            scores[a] = np.mean([x for x in (average_precision(pv[:, j], Y_val[:, j]) for j in range(labels.n_tags)) if x is not None])
        alpha = max(scores, key=scores.get)
        print("tag mAP on val by alpha: " + ", ".join(f"{a}: {s:.3f}" for a, s in scores.items()))
    else:
        alpha = args.alpha
    p_val, _ = tag_probs(val["tag_logits"], val["species_logits"], labels.species_tags, alpha)

    thresholds, tag_rows = {}, []
    for j, tag in enumerate(labels.tags):
        thr = None if tag in args.disable else threshold_for_precision(p_val[:, j], Y_val[:, j], args.target_precision, args.min_support)
        thresholds[tag] = thr
        sel = p_val[:, j] >= thr if thr is not None else np.zeros(len(p_val), bool)
        tag_rows.append({
            "tag": tag,
            "ap": average_precision(p_val[:, j], Y_val[:, j]),
            "threshold": thr,
            "val_precision": float(Y_val[sel, j].mean()) if sel.any() else None,
            "val_recall": float(Y_val[sel, j].sum() / max(1, Y_val[:, j].sum())) if thr is not None else 0.0,
            "positives": int(Y_val[:, j].sum()),
        })

    rng = np.random.default_rng(0)
    results = []
    for split, d in preds.items():
        if d is None:
            continue
        y = np.array([sidx[s] for s in d["slugs"]])
        p, ps = tag_probs(d["tag_logits"], d["species_logits"], labels.species_tags, alpha)
        results.append(split_metrics(split, p, ps, y, labels, thresholds, rng))

    write_json(run / "thresholds.json", {
        "alpha": alpha,
        "target_precision": args.target_precision,
        "min_support": args.min_support,
        "require_p": labels.require_p,
        "thresholds": thresholds,
        "tags": tag_rows,
        "results": results,
    })
    report = render_report(run, config, alpha, args, tag_rows, results, labels)
    (run / "report.md").write_text(report, encoding="utf-8")
    print(report)


def fmt(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def render_report(run, config, alpha, args, tag_rows, results, labels) -> str:
    lines = [
        f"# Evaluation: {run.name}",
        "",
        f"- backbone: `{config['backbone']}` ({config['type']})",
        f"- alpha (attribute head share): {alpha}",
        f"- target precision per tag: {args.target_precision}, min support {args.min_support}",
        f"- enabled tags: {sum(r['threshold'] is not None for r in tag_rows)} / {len(tag_rows)}",
        "",
        "## Splits",
        "",
        "survival = share of queries where all prefilled tags belong to the true species (higher is better; this is the safety metric).",
        "",
        "| split | photos | species@1 | species@5 | tag mAP | prefilled tags | survival | remaining species (median) | 3-photo tags | 3-photo survival |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['split']} | {r['photos']} | {fmt(r['species_top1'])} | {fmt(r['species_top5'])} | {fmt(r['tag_map'])} | "
            f"{fmt(r['prefill_tags'], 2)} | {fmt(r['survival'])} | {fmt(r['remaining_species_median'], 0)} | "
            f"{fmt(r.get('multi_prefill_tags'), 2)} | {fmt(r.get('multi_survival'))} |"
        )
    lines += ["", "holdout species are never trained on, so species@k is meaningless there; tag metrics show generalisation.", "",
              "## Tags (val)", "", "| group | tag | AP | threshold | precision | recall | positives |", "|---|---|---|---|---|---|---|"]
    group_of = {t: g["name"] for g in labels.groups for t in g["tags"]}
    for r in tag_rows:
        lines.append(f"| {group_of[r['tag']]} | {r['tag']} | {fmt(r['ap'])} | {fmt(r['threshold'])} | {fmt(r['val_precision'])} | {fmt(r['val_recall'])} | {r['positives']} |")
    return "\n".join(lines) + "\n"


def add_arguments(p) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--target-precision", type=float, default=0.9)
    p.add_argument("--min-support", type=int, default=20, help="min. predicted positives on val to enable a tag")
    p.add_argument("--alpha", type=float, help="fix the attribute/species mix instead of tuning it")
    p.add_argument("--disable", nargs="*", default=[], help="tags never to prefill")

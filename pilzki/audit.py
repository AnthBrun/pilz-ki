"""Find suspicious species labels in pilzdaten (to fix there, not here).

    python -m pilzki audit-labels --run work/runs/ft-fungi-effv2b3

Labels are per species, so a missing or wrong tag hurts every photo of that
species: training punishes correct predictions and evaluation counts correct
prefills as errors (e.g. Jodwasserkopf had neither `hutpilz` nor `lamellen`).

Two kinds of checks:
- rules on the label matrix alone (no gruppe tag, cap mushroom without
  underside type, no substrate ...);
- model evidence: the attribute head alone (sigmoid of tag logits, alpha = 1,
  so the species marginal cannot echo the labels back) on photos the run did
  not train on (val, test, holdout). A tag the head sees on most photos of a
  species that lacks it is "probably missing"; a structural tag (body type,
  underside, substrate, stem details) the head never sees is "probably wrong".
  Use a run trained without --final so holdout species are truly unseen.

Writes work/label_audit.md (readable) and work/label_audit.json. Precision
checked on photos (Nov 2026): about 2 in 3 "probably wrong" and 1 in 2
structural "probably missing" findings are real label errors, so this is a
review list for a human, not an automatic fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import read_json, work_dir
from .labels import load_labels

# Groups where a species-level tag should be visible on (nearly) every photo.
STRUCTURAL = ("gruppe", "unterseite", "untergrund", "stiel_details", "wuchs", "hut_oberflaeche")


def rule_findings(labels) -> list[dict]:
    ti = {t: i for i, t in enumerate(labels.tags)}
    group_tags = {g["name"]: [ti[t] for t in g["tags"]] for g in labels.groups}
    M = labels.species_tags
    out = []

    def add(i, rule):
        s = labels.species[i]
        out.append({"slug": s["slug"], "name": s.get("name", ""), "wissenschaftlich": s.get("wissenschaftlich", ""), "rule": rule})

    for i in range(labels.n_species):
        has = lambda g: M[i, group_tags[g]].any()  # noqa: E731
        hut = M[i, ti["hutpilz"]] > 0
        if not has("gruppe"):
            add(i, "no body type (gruppe) tag")
        if hut and not has("unterseite"):
            add(i, "hutpilz without underside type (lamellen/roehren/stachel/leisten)")
        if not has("untergrund"):
            add(i, "no substrate (auf_boden/auf_wiese/holz)")
        if not has("hut_farbe"):
            add(i, "no cap/fruit body colour")
    return out


def model_findings(run: Path, labels, min_photos: int, missing_p: float, wrong_p: float) -> list[dict]:
    from .evaluate import load_preds, sigmoid

    ti = {t: i for i, t in enumerate(labels.tags)}
    group_of = {t: g["name"] for g in labels.groups for t in g["tags"]}
    per_species: dict[int, list[np.ndarray]] = {}
    split_of: dict[int, str] = {}
    for split in ("val", "test", "holdout"):
        d = load_preds(run, split)
        if d is None:
            continue
        p = sigmoid(d["tag_logits"].astype(np.float64))
        for slug, row in zip(d["slugs"], p):
            i = labels.species_index[str(slug)]
            per_species.setdefault(i, []).append(row)
            split_of[i] = "holdout" if split == "holdout" else "seen"
    out = []
    for i, rows in per_species.items():
        if len(rows) < min_photos:
            continue
        P = np.stack(rows)
        mean, share = P.mean(0), (P >= 0.5).mean(0)
        s = labels.species[i]
        for t, j in ti.items():
            label = bool(labels.species_tags[i, j])
            if not label and mean[j] >= missing_p and share[j] >= 0.75:
                kind = "probably missing"
            elif label and group_of[t] in STRUCTURAL and mean[j] <= wrong_p and len(rows) >= max(3, min_photos):
                kind = "probably wrong"
            else:
                continue
            out.append({
                "slug": s["slug"], "name": s.get("name", ""), "wissenschaftlich": s.get("wissenschaftlich", ""),
                "tag": t, "group": group_of[t], "kind": kind, "labelled": label, "mean_p": round(float(mean[j]), 3),
                "share_photos": round(float(share[j]), 2), "photos": len(rows), "species_split": split_of[i],
                "score": round(float(len(rows) ** 0.5 * (mean[j] if not label else 1 - mean[j])), 3),
            })
    return sorted(out, key=lambda r: -r["score"])


def main(args) -> None:
    run = Path(args.run)
    config = read_json(run / "config.json")
    if config.get("final"):
        print("warning: --final run; holdout species were trained on, evidence for them is biased")
    labels = load_labels()
    rules = rule_findings(labels)
    model = model_findings(run, labels, args.min_photos, args.missing_p, args.wrong_p)
    out = Path(args.out) if args.out else work_dir() / "label_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"rules": rules, "model": model}, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# Label audit ({run.name})", "",
             "Species-level tags in pilzdaten that look wrong. Fix them in pilzdaten, then rerun `prepare` and retrain.", "",
             f"## Rule checks ({len(rules)})", "", "| species | scientific | problem |", "|---|---|---|"]
    lines += [f"| {r['name'] or r['slug']} (`{r['slug']}`) | {r['wissenschaftlich']} | {r['rule']} |" for r in rules]
    sections = (
        ("structural tag probably wrong", lambda r: r["kind"] == "probably wrong"),
        ("structural tag probably missing", lambda r: r["kind"] == "probably missing" and r["group"] in STRUCTURAL),
        ("colour or cap shape probably missing (lower priority: species often vary)",
         lambda r: r["kind"] == "probably missing" and r["group"] not in STRUCTURAL),
    )
    for title, keep in sections:
        rows = [r for r in model if keep(r)]
        lines += ["", f"## Model evidence: {title} ({len(rows)})", "",
                  "mean p = attribute head on unseen photos of the species; share = photos with p >= 0.5.", "",
                  "| species | scientific | tag | mean p | share | photos | species |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['name'] or r['slug']} (`{r['slug']}`) | {r['wissenschaftlich']} | `{r['tag']}` | {r['mean_p']:.2f} | "
                  f"{r['share_photos']:.0%} | {r['photos']} | {r['species_split']} |" for r in rows[: args.top]]
        if len(rows) > args.top:
            lines.append(f"| … {len(rows) - args.top} more in {out.with_suffix('.json').name} | | | | | | |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_missing = sum(r["kind"] == "probably missing" for r in model)
    print(f"{len(rules)} rule findings, {n_missing} probably missing, {len(model) - n_missing} probably wrong -> {out}")


def add_arguments(p) -> None:
    p.add_argument("--run", required=True, help="run with preds_{val,test,holdout}.npz (ideally not --final)")
    p.add_argument("--out", help="markdown report (default work/label_audit.md; a .json is written next to it)")
    p.add_argument("--min-photos", type=int, default=2)
    p.add_argument("--missing-p", type=float, default=0.7, help="mean attribute probability to flag a missing tag")
    p.add_argument("--wrong-p", type=float, default=0.03, help="mean attribute probability to flag a wrong structural tag")
    p.add_argument("--top", type=int, default=150, help="rows per section in the markdown")

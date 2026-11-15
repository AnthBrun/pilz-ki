"""Step 0 (runs on the web server, CPU only): build the training manifest.

Reads the app data of the pilzdaten repo plus the crawl's image hashes and
writes small, committed files to data/:

  species.json    one entry per species: slug, name, all tags (incl. derived)
  manifest.jsonl  one entry per usable photo: id, slug, source paths, split
  vocab.json      trait groups the model predicts (see vocab.py) with counts
  prepare_report.json  what was dropped and why

Labels are per species, not per photo: every photo inherits the tags of its
species. That is exactly what the web filter matches against, so the model
learns "does the species on this photo carry tag X".
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from . import vocab
from .common import DATA, read_json, read_jsonl, stable_fraction, write_json, write_jsonl

SPLIT_SALT = "pilzki-split-v1"
HOLDOUT_SALT = "pilzki-holdout-v1"
MIN_PREVIEW_BYTES = 4000  # previews below this without an original are icons/markers


def has_leisten_category(merkmale: dict) -> bool:
    """Mirror of hasLeistenCategory in pilzdaten/internal/catalog/load.go."""
    return any(key.strip().lower() in ("leisten", "lamellen/leisten") for key in merkmale)


def species_tags(raw_tags: list[str], merkmale: dict) -> list[str]:
    tags = set(raw_tags)
    if has_leisten_category(merkmale):
        tags.add("leisten")
    if tags & vocab.HAT_TAGS:
        tags.add("hutpilz")
    return sorted(tags)


def load_sha_file(path: Path | None) -> dict[str, str]:
    """`sha256  ./daten/details/...jpg` lines (sha256sum output) -> {rel path: hash}."""
    if not path or not path.exists():
        return {}
    hashes = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) == 2:
                hashes[parts[1].strip().removeprefix("./")] = parts[0]
    return hashes


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    pilzdaten_data: Path,
    image_root: Path | None,
    sha_file: Path | None,
    out_dir: Path = DATA,
    holdout_frac: float = 0.05,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> dict:
    index = {row["slug"]: row for row in read_jsonl(pilzdaten_data / "index_arts.jsonl")}
    filter_tags = {row["slug"]: row for row in read_jsonl(pilzdaten_data / "filter_tags.jsonl")}
    originals_path = pilzdaten_data / "originals.json"
    originals = read_json(originals_path) if originals_path.exists() else {}
    hashes = load_sha_file(sha_file)
    check_disk = image_root is not None and image_root.exists()

    dropped = Counter()
    species_out = []
    candidates = []  # (slug, preview, original, sha)
    for sp in read_jsonl(pilzdaten_data / "species.jsonl"):
        slug = sp["slug"]
        if slug not in index:
            continue
        entry = index[slug]
        tags = species_tags(filter_tags.get(slug, {}).get("tags", []), sp.get("merkmale") or {})
        species_out.append({
            "slug": slug,
            "name": entry.get("name") or slug,
            "wissenschaftlich": entry.get("wissenschaftlich", ""),
            "gattung": entry.get("gattung", ""),
            "tags": tags,
        })
        seen = set()
        for preview in sp.get("bilddateien", []):
            if not preview.lower().endswith(".jpg") or preview in seen:
                continue
            seen.add(preview)
            original = originals.get(preview)
            if check_disk:
                preview_file = image_root / preview
                if original and not (image_root / original).exists():
                    original = None
                if not original:
                    if not preview_file.exists():
                        dropped["missing_on_disk"] += 1
                        continue
                    if preview_file.stat().st_size < MIN_PREVIEW_BYTES:
                        dropped["tiny_preview_without_original"] += 1
                        continue
            sha = hashes.get(preview)
            if not sha and check_disk and (image_root / preview).exists():
                sha = file_sha(image_root / preview)
            candidates.append((slug, preview, original, sha or "path:" + preview))

    # Identical photos filed under several species are markers/icons or
    # ambiguous; drop all copies. Duplicates within one species keep the first.
    slugs_by_sha = defaultdict(set)
    for slug, _, _, sha in candidates:
        slugs_by_sha[sha].add(slug)
    rows, kept_sha = [], set()
    for slug, preview, original, sha in candidates:
        if len(slugs_by_sha[sha]) > 1:
            dropped["shared_between_species"] += 1
            continue
        if sha in kept_sha:
            dropped["duplicate_within_species"] += 1
            continue
        kept_sha.add(sha)
        image_id = hashlib.sha256(sha.encode()).hexdigest()[:16] if sha.startswith("path:") else sha[:16]
        rows.append({"id": image_id, "slug": slug, "preview": preview, "original": original})

    # Splits: a few species are held out completely to measure generalisation
    # to unseen species; the rest is split per photo.
    n_images = Counter(row["slug"] for row in rows)
    holdout = {s["slug"] for s in species_out if n_images[s["slug"]] and stable_fraction(s["slug"], HOLDOUT_SALT) < holdout_frac}
    has_train = set()
    for row in rows:
        if row["slug"] in holdout:
            row["split"] = "holdout"
            continue
        f = stable_fraction(row["id"], SPLIT_SALT)
        row["split"] = "test" if f < test_frac else "val" if f < test_frac + val_frac else "train"
        if row["split"] == "train":
            has_train.add(row["slug"])
    for row in rows:  # every seen species needs at least one training photo
        if row["split"] in ("val", "test") and row["slug"] not in has_train:
            row["split"] = "train"
            has_train.add(row["slug"])

    for s in species_out:
        s["n_images"] = n_images[s["slug"]]
        s["holdout"] = s["slug"] in holdout

    tags_by_slug = {s["slug"]: s["tags"] for s in species_out}
    tag_species = Counter(t for s in species_out for t in s["tags"])
    tag_images = Counter(t for row in rows for t in tags_by_slug[row["slug"]])
    vocab_json = {
        "groups": vocab.as_json(),
        "tags": [{"tag": t, "group": vocab.TAG_GROUP[t], "species": tag_species[t], "images": tag_images[t]} for t in vocab.TAGS],
        "require_p": vocab.REQUIRE_P,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "species.json", species_out)
    write_jsonl(out_dir / "manifest.jsonl", rows)
    write_json(out_dir / "vocab.json", vocab_json)
    report = {
        "species": len(species_out),
        "species_with_images": sum(1 for s in species_out if s["n_images"]),
        "holdout_species": len(holdout),
        "images": len(rows),
        "images_with_original": sum(1 for r in rows if r["original"]),
        "splits": dict(Counter(r["split"] for r in rows)),
        "dropped": dict(dropped),
        "disk_checked": check_disk,
        "tags_without_training_images": [t for t in vocab.TAGS if tag_images[t] == 0],
    }
    write_json(out_dir / "prepare_report.json", report)
    return report


def main(args) -> None:
    report = build(
        pilzdaten_data=Path(args.pilzdaten_data),
        image_root=Path(args.image_root) if args.image_root else None,
        sha_file=Path(args.sha_file) if args.sha_file else None,
        out_dir=Path(args.out),
        holdout_frac=args.holdout,
    )
    for key, value in report.items():
        print(f"{key}: {value}")

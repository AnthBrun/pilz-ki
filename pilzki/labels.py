"""Label matrices and manifest filtering (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import DATA, read_json, read_jsonl

TRAIN_PHOTO_TYPES = ("fruitbody", "cut")


@dataclass
class Labels:
    species: list[dict]
    slugs: list[str]
    species_index: dict[str, int]
    tags: list[str]
    groups: list[dict]
    require_p: float
    species_tags: np.ndarray  # (n_species, n_tags) float32 0/1

    @property
    def n_species(self) -> int:
        return len(self.slugs)

    @property
    def n_tags(self) -> int:
        return len(self.tags)


def load_labels(data_dir: Path = DATA) -> Labels:
    species = read_json(data_dir / "species.json")
    vocab = read_json(data_dir / "vocab.json")
    tags = [t["tag"] for t in vocab["tags"]]
    tag_index = {t: i for i, t in enumerate(tags)}
    matrix = np.zeros((len(species), len(tags)), dtype=np.float32)
    for i, s in enumerate(species):
        for t in s["tags"]:
            if t in tag_index:
                matrix[i, tag_index[t]] = 1.0
    slugs = [s["slug"] for s in species]
    return Labels(species, slugs, {s: i for i, s in enumerate(slugs)}, tags, vocab["groups"], vocab.get("require_p", 0.5), matrix)


def load_phototypes(data_dir: Path = DATA) -> dict[str, str]:
    path = data_dir / "phototype.jsonl"
    if not path.exists():
        return {}
    return {row["id"]: row["type"] for row in read_jsonl(path)}


def load_manifest(data_dir: Path = DATA, photo_types: tuple[str, ...] | None = TRAIN_PHOTO_TYPES, warn=print) -> list[dict]:
    """Manifest rows, optionally restricted to photo types (needs data/phototype.jsonl)."""
    rows = list(read_jsonl(data_dir / "manifest.jsonl"))
    if photo_types is None:
        return rows
    types = load_phototypes(data_dir)
    if not types:
        warn("warning: data/phototype.jsonl missing, training on all photo types (run `pilzki phototype`)")
        return rows
    return [r for r in rows if types.get(r["id"], "fruitbody") in photo_types]


def splits_for(final: bool) -> dict[str, tuple[str, ...]]:
    """Which manifest splits train/calibrate/report.

    Experiments keep holdout species out of training to measure generalisation
    to unseen species. The final model also learns the holdout species.
    """
    if final:
        return {"train": ("train", "holdout"), "eval": ("val", "test")}
    return {"train": ("train",), "eval": ("val", "test", "holdout")}

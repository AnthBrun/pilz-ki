"""Paths and small helpers shared by all steps (no torch imports here)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"  # small, committed outputs (manifest, species, vocab, phototype)


def work_dir() -> Path:
    """Large, uncommitted outputs: images, embeddings, runs, exports."""
    return Path(os.environ.get("PILZKI_WORK", REPO / "work"))


def images_dir() -> Path:
    return Path(os.environ.get("PILZKI_IMAGES", work_dir() / "images"))


def image_path(image_id: str) -> Path:
    return images_dir() / f"{image_id}.jpg"


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def stable_fraction(key: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0, 1) for splits."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64

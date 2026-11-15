"""Contact sheets to eyeball photo types or a sample of the data."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from .common import DATA, image_path, read_jsonl
from .labels import load_phototypes


def main(args) -> None:
    rows = [r for r in read_jsonl(DATA / "manifest.jsonl") if image_path(r["id"]).exists()]
    types = load_phototypes()
    if args.type:
        rows = [r for r in rows if types.get(r["id"]) == args.type]
    if args.slug:
        rows = [r for r in rows if r["slug"] == args.slug]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    cols, cell = 8, args.cell
    sheet = Image.new("RGB", (cols * cell, max(1, -(-len(rows) // cols)) * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        with Image.open(image_path(r["id"])) as im:
            im = im.convert("RGB")
            im.thumbnail((cell, cell - 14))
            x, y = (i % cols) * cell, (i // cols) * cell
            sheet.paste(im, (x, y))
            draw.text((x + 2, y + cell - 13), f"{r['slug'][:22]} {types.get(r['id'], '')}", fill="black")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"{len(rows)} images -> {out}")

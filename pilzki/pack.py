"""Step 1 (GPU machine, after copying the crawl's photos): compact training images.

Uses the large original if the crawl has one, else the ~400 px preview.
Images are resized so the short side is at most --short (default 512) and
written as <id>.jpg, either into a directory (resumable) or a tar stream:

    python -m pilzki pack --image-root ~/pilzsuche-images --out work/images --workers 8

Do not run this on the small web server VM (CPU-heavy for ~40 min).
"""

from __future__ import annotations

import io
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

from .common import DATA, read_jsonl

Image.MAX_IMAGE_PIXELS = 60_000_000


def resize_bytes(src: Path, short: int, max_long: int, quality: int) -> bytes:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        scale = min(1.0, short / min(w, h), max_long / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def _job(task):
    image_id, sources, short, max_long, quality = task
    for src in sources:
        try:
            return image_id, resize_bytes(Path(src), short, max_long, quality), None
        except Exception as exc:  # broken original -> try the preview
            err = f"{src}: {exc}"
    return image_id, None, err


def main(args) -> None:
    root = Path(args.image_root)
    rows = list(read_jsonl(Path(args.manifest)))
    if args.split:
        rows = [r for r in rows if r["split"] in args.split.split(",")]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [r for r in rows if not (out_dir / f"{r['id']}.jpg").exists()]
    tasks = []
    for r in rows:
        sources = [root / p for p in (r["original"], r["preview"]) if p]
        tasks.append((r["id"], [str(s) for s in sources], args.short, args.max_long, args.quality))

    log = sys.stderr
    tar = None
    if args.tar:
        tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|") if args.tar == "-" else tarfile.open(args.tar, "w")
    elif not out_dir:
        raise SystemExit("pack: give --out DIR or --tar FILE|-")

    done = failed = 0
    total_bytes = 0
    start = time.time()
    with Pool(args.workers) as pool:
        for image_id, data, err in pool.imap_unordered(_job, tasks, chunksize=16):
            if data is None:
                failed += 1
                print(f"skip {image_id}: {err}", file=log)
                continue
            total_bytes += len(data)
            if tar:
                info = tarfile.TarInfo(f"{image_id}.jpg")
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(data))
            else:
                (out_dir / f"{image_id}.jpg").write_bytes(data)
            done += 1
            if done % 1000 == 0:
                rate = done / (time.time() - start)
                print(f"{done}/{len(tasks)} images, {total_bytes / 1e9:.2f} GB, {rate:.0f}/s", file=log)
    if tar:
        tar.close()
    print(f"packed {done} images ({total_bytes / 1e9:.2f} GB), failed {failed}", file=log)


def add_arguments(p) -> None:
    p.add_argument("--image-root", default="../pilzsuche/raw/images", help="folder that contains daten/details/")
    p.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    p.add_argument("--out", help="output directory (resumable)")
    p.add_argument("--tar", help="write a tar file instead; '-' streams to stdout")
    p.add_argument("--short", type=int, default=512, help="max length of the short side")
    p.add_argument("--max-long", type=int, default=1024, help="max length of the long side")
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--split", help="only these splits, comma separated")
    p.add_argument("--limit", type=int, help="only the first N images (smoke tests)")
    p.add_argument("--workers", type=int, default=2)

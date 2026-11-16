"""Step 2: sort photos into types with zero-shot CLIP.

The crawl mixes field photos with microscopy, spore prints, icons and scans.
Only fruit-body photos (incl. cut specimens, which still show cap, gills and
stem) are used for training and evaluation.

    python -m pilzki phototype            # writes data/phototype.jsonl (commit it)
    python -m pilzki sheet --type microscopy   # eyeball a class

Reuses work/emb/<backbone> if `embed` already ran for the same CLIP backbone.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .common import DATA, read_jsonl, write_jsonl

PROMPTS = {
    "fruitbody": [
        "a photo of a mushroom in the forest",
        "a photo of mushrooms growing on wood",
        "a close-up photo of a mushroom cap",
        "a photo of the gills under a mushroom cap",
        "a photo of a mushroom held in a hand",
        "a photo of a mushroom on a table",
        "a photo of a bracket fungus on a tree",
        "a photo of a lichen on a rock",
        "a photo of a slime mold",
        "a photo of a crust fungus on bark",
        "a photo of a small cup fungus",
        "a photo of a lichen on bark",
        "a photo of tiny fungi on a twig",
        "a macro photo of tiny fungi on dead wood",
        "a photo of mold growing on a plant",
        "a photo of a crust lichen on stone",
    ],
    "cut": [
        "a photo of a mushroom cut in half showing the flesh",
        "a cross-section of a mushroom",
    ],
    "microscopy": [
        "a microscope image of fungal spores",
        "a microscopy image of cells",
        "a microscope image of hyphae stained red",
        "a light micrograph with a scale bar",
        "a fungal culture in a petri dish",
        "a microscope image of cells stained blue",
    ],
    "spore_print": [
        "a mushroom spore print, coloured powder on white paper",
        "a spore print on paper",
        "a spore print on glass",
    ],
    "graphic": ["a drawing", "an icon", "a warning sign", "a diagram with text", "a scanned page of text", "a logo"],
}


def main(args) -> None:
    import torch

    from .backbones import Backbone, device
    from .embed import compute, emb_dir, load_embeddings
    from .model import existing

    rows = existing(list(read_jsonl(DATA / "manifest.jsonl")))
    if args.limit:
        rows = rows[: args.limit]
    dev = device()
    if (emb_dir(args.backbone) / "features.npy").exists():
        feats, ids, _ = load_embeddings(args.backbone)
        by_id = dict(zip(ids, range(len(ids))))
        rows = [r for r in rows if r["id"] in by_id]
        feats = feats[[by_id[r["id"]] for r in rows]]
        backbone = Backbone(args.backbone).to(dev).eval()
    else:
        feats, backbone = compute(args.backbone, rows, args.batch_size, args.workers, tta=False)

    names = list(PROMPTS)
    with torch.no_grad():
        text = []
        for name in names:
            t = backbone.encode_text(PROMPTS[name], dev).float()
            t = t / t.norm(dim=-1, keepdim=True)
            text.append(t.mean(0))
        text = torch.stack(text)
        text = text / text.norm(dim=-1, keepdim=True)
        img = torch.as_tensor(np.asarray(feats, dtype=np.float32), device=dev)
        img = img / img.norm(dim=-1, keepdim=True)
        probs = (100.0 * img @ text.T).softmax(-1).cpu().numpy()

    out = []
    for r, p in zip(rows, probs):
        k = int(p.argmax())
        out.append({"id": r["id"], "type": names[k], "p": round(float(p[k]), 3)})
    write_jsonl(DATA / "phototype.jsonl", out)
    print(dict(Counter(o["type"] for o in out)))

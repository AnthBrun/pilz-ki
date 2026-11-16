"""Run an exported model on arbitrary photos, exactly like the web app will.

    python -m pilzki predict --export work/export/ft-fungi-effv2b3-final photo1.jpg photo2.jpg
    python -m pilzki predict --export ... --find a.jpg b.jpg c.jpg   # one find, averaged

Photos whose `fungus_prob` is below meta.json["gate"]["threshold"] get no
prefill ("fungus": false); in --find mode they are left out of the average.

Reference implementation for the JS port (phase 6): it only uses meta.json and
the ONNX file, no torch and no training code. Preprocessing follows
meta.json["input"]: resize the short side, center crop, RGB in [0, 1], NCHW.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


def load_image(path_or_image, inp: dict, resample: str | None = None) -> np.ndarray:
    """-> float32 [3, S, S] in [0, 1] (meta.json["input"] conventions)."""
    im = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    im = ImageOps.exif_transpose(im).convert("RGB")
    short, crop = inp["resize_short_side"], inp["center_crop"]
    w, h = im.size
    scale = short / min(w, h)
    im = im.resize((max(crop, round(w * scale)), max(crop, round(h * scale))), RESAMPLE[resample or inp["interpolation"]])
    w, h = im.size
    left, top = (w - crop) // 2, (h - crop) // 2
    im = im.crop((left, top, left + crop, top + crop))
    return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def prefill(p: np.ndarray, meta: dict) -> list[str]:
    """Same rule as evaluate.prefill, driven by meta.json only."""
    idx = {t["tag"]: i for i, t in enumerate(meta["tags"])}
    thr = {t["tag"]: t["threshold"] for t in meta["tags"]}
    chosen = []
    for g in meta["groups"]:
        if g["requires"] and p[idx[g["requires"]]] < meta["require_p"]:
            continue
        if g["exclusive"]:
            best = max(g["tags"], key=lambda t: p[idx[t]])
            if thr[best] is not None and p[idx[best]] >= thr[best]:
                chosen.append(best)
        else:
            chosen += [t for t in g["tags"] if thr[t] is not None and p[idx[t]] >= thr[t]]
    return chosen


class Predictor:
    def __init__(self, export_dir: Path, model_file: str = "model.int8.onnx"):
        import onnxruntime as ort

        self.meta = json.loads((Path(export_dir) / "meta.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(Path(export_dir) / model_file), providers=["CPUExecutionProvider"])

    @property
    def gate(self) -> dict | None:
        return self.meta.get("gate")

    def run(self, images: np.ndarray, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """[N, 3, S, S] -> (tag_probs [N, n_tags], species_probs [N, n_species], fungus_prob [N]).

        fungus_prob is all ones for models exported without a gate."""
        names = ["tag_probs", "species_probs"] + (["fungus_prob"] if self.gate else [])
        out = [[] for _ in names]
        for i in range(0, len(images), batch_size):
            for acc, arr in zip(out, self.session.run(names, {"image": images[i:i + batch_size]})):
                acc.append(arr)
        tags, species = np.concatenate(out[0]), np.concatenate(out[1])
        fungus = np.concatenate(out[2]) if self.gate else np.ones(len(tags), np.float32)
        return tags, species, fungus

    def is_fungus(self, fungus_prob) -> np.ndarray:
        return np.asarray(fungus_prob) >= (self.gate["threshold"] if self.gate else 0.0)

    def describe(self, tag_probs: np.ndarray, species_probs: np.ndarray, fungus_prob: float = 1.0, top: int = 5) -> dict:
        if not self.is_fungus(fungus_prob):
            return {"fungus": False, "fungus_prob": round(float(fungus_prob), 3), "prefill": [], "species": []}
        order = np.argsort(-species_probs)[:top]
        return {
            "fungus": True,
            "fungus_prob": round(float(fungus_prob), 3),
            "prefill": prefill(tag_probs, self.meta),
            "species": [(self.meta["species"][i], round(float(species_probs[i]), 3)) for i in order],
        }

    def describe_find(self, tags: np.ndarray, species: np.ndarray, fungus: np.ndarray, top: int = 5) -> dict:
        """Several photos of one find: average the photos that pass the gate."""
        ok = self.is_fungus(fungus)
        if not ok.any():
            return {"fungus": False, "photos_used": 0, "prefill": [], "species": []}
        d = self.describe(tags[ok].mean(0), species[ok].mean(0), float(fungus[ok].mean()), top)
        return {**d, "photos_used": int(ok.sum())}


def main(args) -> None:
    pred = Predictor(Path(args.export), args.model)
    paths = [Path(p) for p in args.images]
    x = np.stack([load_image(p, pred.meta["input"]) for p in paths])
    tags, species, fungus = pred.run(x)
    if args.find:
        print(json.dumps({"find": [str(p) for p in paths], **pred.describe_find(tags, species, fungus)}, ensure_ascii=False))
        return
    for p, t, s, f in zip(paths, tags, species, fungus):
        print(json.dumps({"image": str(p), **pred.describe(t, s, f)}, ensure_ascii=False))


def add_arguments(p) -> None:
    p.add_argument("--export", required=True, help="export dir with meta.json and the ONNX model")
    p.add_argument("--model", default="model.int8.onnx")
    p.add_argument("--find", action="store_true", help="treat all images as one find (average the outputs)")
    p.add_argument("images", nargs="+")

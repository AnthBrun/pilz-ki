"""CLI: python -m pilzki <step> ... (see AGENTS.md for the full pipeline)."""

from __future__ import annotations

import argparse
import importlib
import sys

from .common import DATA


def _prepare_args(p):
    p.add_argument("--pilzdaten-data", default="../pilzdaten/data", help="data/ folder of the pilzdaten repo")
    p.add_argument("--image-root", default="../pilzsuche/raw/images", help="checks that photos exist (optional)")
    p.add_argument("--sha-file", default="../pilzsuche/image-sha256.txt", help="sha256sum of the preview photos")
    p.add_argument("--out", default=str(DATA))
    p.add_argument("--holdout", type=float, default=0.05, help="share of species held out completely")


def _embed_args(p):
    p.add_argument("--backbone", required=True, help="alias (see backbones.ALIASES) or timm:/openclip: spec")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tta", action="store_true", help="average with horizontally flipped images")
    p.add_argument("--limit", type=int, default=0)


def _phototype_args(p):
    p.add_argument("--backbone", default="clip-b32", help="an openclip backbone")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)


def _sheet_args(p):
    p.add_argument("--type", help="photo type from data/phototype.jsonl")
    p.add_argument("--slug", help="only this species")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--cell", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="work/sheet.jpg")


def _from_module(name):
    return lambda p: importlib.import_module(f"pilzki.{name}").add_arguments(p)


STEPS = {
    "prepare": ("prepare", _prepare_args, "build data/ manifest, species and vocab from pilzdaten (CPU)"),
    "pack": ("pack", _from_module("pack"), "resize photos into work/images or a tar stream (CPU)"),
    "phototype": ("phototype", _phototype_args, "zero-shot sort photos (field/cut/microscopy/...)"),
    "sheet": ("sheet", _sheet_args, "contact sheet for eyeballing photos"),
    "embed": ("embed", _embed_args, "frozen backbone features for all photos"),
    "train-heads": ("train_heads", _from_module("train_heads"), "train tag + species heads on embeddings"),
    "evaluate": ("evaluate", _from_module("evaluate"), "thresholds, survival metrics, report.md"),
    "finetune": ("finetune", _from_module("finetune"), "end-to-end fine-tuning on photos"),
    "export": ("export", _from_module("export"), "ONNX + meta.json for the web app"),
}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="pilzki", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    chosen = argv[0] if argv else None
    for step, (_, add_args, help_text) in STEPS.items():
        step_parser = sub.add_parser(step, help=help_text, description=help_text)
        if step == chosen:  # import only the chosen step: `prepare` must run without numpy/PIL/torch
            add_args(step_parser)
    args = parser.parse_args(argv)
    importlib.import_module(f"pilzki.{STEPS[args.step][0]}").main(args)


if __name__ == "__main__":
    main()

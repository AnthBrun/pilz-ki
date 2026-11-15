"""Step 7: export a run to ONNX for the browser (onnxruntime-web).

    python -m pilzki export --run work/runs/ft-fungi-effv2b3-final

Writes work/export/<run>/ with
  model.onnx       fp32, input `image` float32 [N,3,S,S] RGB in [0,1]
  model.int8.onnx  weights of MatMul/Gemm quantised (smaller download)
  meta.json        input size, tag order + thresholds + groups, species slugs

Outputs: `tag_probs` [N, n_tags] (already mixed with the species head, see
model.combine) and `species_probs` [N, n_species]. Several photos of one find
are combined by averaging both outputs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import read_json, work_dir, write_json
from .labels import load_labels, load_manifest


def load_classifier(run: Path):
    import torch

    from .backbones import Backbone
    from .model import Classifier, Heads

    config = read_json(run / "config.json")
    labels = load_labels()
    if config["tags"] != labels.tags or config["slugs"] != labels.slugs:
        raise SystemExit("run was trained with a different data/; retrain")
    thresholds = read_json(run / "thresholds.json")
    backbone = Backbone(config["backbone"], pretrained=config["type"] == "heads")
    heads = Heads(**config["heads"])
    if config["type"] == "heads":
        heads.load_state_dict(torch.load(run / "heads.pt", map_location="cpu"))
    else:
        ckpt = torch.load(run / "model.pt", map_location="cpu")
        backbone.load_state_dict(ckpt["backbone"])
        heads.load_state_dict(ckpt["heads"])
    model = Classifier(backbone, heads, labels.species_tags, thresholds["alpha"]).eval()
    return model, config, thresholds, labels


def main(args) -> None:
    import torch
    from torchvision import transforms as T

    from .evaluate import prefill
    from .model import ImageDataset, existing

    run = Path(args.run)
    model, config, thresholds, labels = load_classifier(run)
    pp = model.backbone.preprocess
    out = Path(args.out) if args.out else work_dir() / "export" / run.name
    out.mkdir(parents=True, exist_ok=True)

    onnx_path = out / "model.onnx"
    dummy = torch.rand(1, 3, pp.size, pp.size)
    torch.onnx.export(
        model, (dummy,), str(onnx_path), input_names=["image"], output_names=["tag_probs", "species_probs"],
        dynamic_axes={"image": {0: "n"}, "tag_probs": {0: "n"}, "species_probs": {0: "n"}},
        opset_version=args.opset, dynamo=False,
    )
    paths = [onnx_path]
    if not args.no_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"])
        paths.append(int8_path)

    # Compare torch vs. ONNX (and int8) on real validation photos.
    import onnxruntime as ort

    interp = T.InterpolationMode.BICUBIC if pp.interpolation == "bicubic" else T.InterpolationMode.BILINEAR
    to01 = T.Compose([T.Resize(pp.resize, interpolation=interp), T.CenterCrop(pp.size), T.ToTensor()])
    rows = [r for r in existing(load_manifest()) if r["split"] == "val"][: args.check]
    tag_index = {t: i for i, t in enumerate(labels.tags)}
    thr = thresholds["thresholds"]
    if rows:
        x = torch.stack([ImageDataset(rows, to01)[i][0] for i in range(len(rows))])
        with torch.no_grad():
            ref = model(x)[0].numpy()
        ref_fill = [prefill(p, labels.groups, tag_index, thr, labels.require_p) for p in ref]
        for path in paths:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            got = sess.run(["tag_probs"], {"image": x.numpy()})[0]
            fill = [prefill(p, labels.groups, tag_index, thr, labels.require_p) for p in got]
            agree = np.mean([a == b for a, b in zip(ref_fill, fill)])
            print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, max |Δp| {np.abs(got - ref).max():.4f}, same prefill {agree:.0%} of {len(rows)} photos")

    group_of = {t: g["name"] for g in labels.groups for t in g["tags"]}
    write_json(out / "meta.json", {
        "version": time.strftime("%Y%m%d-%H%M%S"),
        "run": run.name,
        "backbone": config["backbone"],
        "input": {
            "name": "image", "layout": "NCHW", "color": "RGB", "range": [0, 1],
            "resize_short_side": pp.resize, "center_crop": pp.size, "interpolation": pp.interpolation,
            "note": "mean/std normalisation happens inside the model",
        },
        "outputs": {"tag_probs": "n_tags, order of `tags`", "species_probs": "n_species, order of `species`"},
        "alpha": thresholds["alpha"],
        "require_p": labels.require_p,
        "groups": labels.groups,
        "tags": [{"tag": t, "group": group_of[t], "threshold": thr.get(t)} for t in labels.tags],
        "species": labels.slugs,
        "aggregate": "mean of tag_probs and species_probs over all photos of one find",
    })
    print(f"exported to {out}")


def add_arguments(p) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--out", help="output dir (default work/export/<run name>)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-int8", action="store_true")
    p.add_argument("--check", type=int, default=32, help="validation photos for the torch/ONNX comparison")

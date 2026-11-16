"""Step 7: export a run to ONNX for the browser (onnxruntime-web).

    python -m pilzki export --run work/runs/ft-fungi-effv2b3-final

Writes work/export/<run>/ with
  model.onnx       fp32, input `image` float32 [N,3,S,S] RGB in [0,1]
  model.int8.onnx  same graph with compressed weights (smaller download, see compress_weights)
  meta.json        input size, tag order + thresholds + groups, species slugs, gate threshold

With <run>/gate.pt (from `train-gate`) the model has a third output
`fungus_prob` [N]; photos below meta.json["gate"]["threshold"] are ignored.

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


def compress_weights(src: Path, dst: Path, int8_min: int = 300_000, fp16_min: int = 1024) -> None:
    """Weight-only compression; all arithmetic stays fp32.

    Large Conv/Gemm weights are stored as per-output-channel symmetric int8
    behind DequantizeLinear, mid-sized ones as fp16 behind Cast, tiny ones stay
    fp32. onnxruntime folds both back to fp32 constants when the session loads,
    so only the download shrinks. Real int8 kernels (quantize_dynamic/static)
    were tried and break EfficientNet: 5-27 % prefill agreement; quantising the
    species-marginal MatMul alone already drops agreement to 42 %. Keeping the
    smaller (depthwise, SE, stem) convs out of int8 is what keeps agreement
    high (~99 % vs ~91 % with every conv in int8).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    weights = {n.input[1] for n in m.graph.node if n.op_type in ("Conv", "Gemm") and len(n.input) > 1 and n.input[1] in inits}
    decode = []
    for name in sorted(weights):
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        if w.size < fp16_min:
            continue
        m.graph.initializer.remove(inits[name])
        if w.size >= int8_min:
            scale = np.abs(w.reshape(w.shape[0], -1)).max(1) / 127.0
            scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
            q = np.clip(np.round(w / scale.reshape(-1, *[1] * (w.ndim - 1))), -127, 127).astype(np.int8)
            m.graph.initializer.extend([
                numpy_helper.from_array(q, f"{name}_q"),
                numpy_helper.from_array(scale, f"{name}_scale"),
                numpy_helper.from_array(np.zeros(len(scale), np.int8), f"{name}_zero"),
            ])
            decode.append(helper.make_node("DequantizeLinear", [f"{name}_q", f"{name}_scale", f"{name}_zero"], [name], axis=0, name=f"{name}_dequant"))
        else:
            m.graph.initializer.append(numpy_helper.from_array(w.astype(np.float16), f"{name}_fp16"))
            decode.append(helper.make_node("Cast", [f"{name}_fp16"], [name], to=TensorProto.FLOAT, name=f"{name}_cast"))
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(decode + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


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
    gate = None
    if (run / "gate.pt").exists():
        g = torch.load(run / "gate.pt", map_location="cpu")
        gate = torch.nn.Linear(g["weight"].numel(), 1)
        with torch.no_grad():
            gate.weight.copy_(g["weight"].float().view(1, -1))
            gate.bias.fill_(float(g["bias"]))
        thresholds = {**thresholds, "gate_threshold": float(g["threshold"])}
    model = Classifier(backbone, heads, labels.species_tags, thresholds["alpha"], gate).eval()
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
    outputs = ["tag_probs", "species_probs"] + (["fungus_prob"] if model.gate is not None else [])
    if model.gate is None:
        print("warning: no gate.pt in the run; the model will prefill tags on non-fungus photos (run `pilzki train-gate`)")
    torch.onnx.export(
        model, (dummy,), str(onnx_path), input_names=["image"], output_names=outputs,
        dynamic_axes={"image": {0: "n"}, **{o: {0: "n"} for o in outputs}},
        opset_version=args.opset, dynamo=False,
    )
    paths = [onnx_path]
    if not args.no_int8:
        int8_path = out / "model.int8.onnx"
        compress_weights(onnx_path, int8_path)
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
            ref_out = model(x)
        ref = ref_out[0].numpy()
        ref_fill = [prefill(p, labels.groups, tag_index, thr, labels.require_p) for p in ref]
        for path in paths:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            got_out = sess.run(outputs, {"image": x.numpy()})
            got = got_out[0]
            fill = [prefill(p, labels.groups, tag_index, thr, labels.require_p) for p in got]
            agree = np.mean([a == b for a, b in zip(ref_fill, fill)])
            msg = f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, max |Δp| {np.abs(got - ref).max():.4f}, same prefill {agree:.0%} of {len(rows)} photos"
            if model.gate is not None:
                gt = thresholds["gate_threshold"]
                same_gate = np.mean((got_out[2] >= gt) == (ref_out[2].numpy() >= gt))
                msg += f", same gate decision {same_gate:.0%}"
            print(msg)

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
        "outputs": {"tag_probs": "n_tags, order of `tags`", "species_probs": "n_species, order of `species`",
                    **({"fungus_prob": "probability that the photo shows a fungus fruit body"} if model.gate is not None else {})},
        "gate": {"output": "fungus_prob", "threshold": thresholds["gate_threshold"],
                 "rule": "photos with fungus_prob < threshold are ignored; if no photo of a find passes, prefill nothing"}
        if model.gate is not None else None,
        "alpha": thresholds["alpha"],
        "require_p": labels.require_p,
        "groups": labels.groups,
        "tags": [{"tag": t, "group": group_of[t], "threshold": thr.get(t)} for t in labels.tags],
        "species": labels.slugs,
        "aggregate": "mean of tag_probs and species_probs over the photos of one find that pass the gate",
    })
    print(f"exported to {out}")


def add_arguments(p) -> None:
    p.add_argument("--run", required=True)
    p.add_argument("--out", help="output dir (default work/export/<run name>)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-int8", action="store_true")
    p.add_argument("--check", type=int, default=32, help="validation photos for the torch/ONNX comparison")

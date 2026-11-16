"""Pretrained image backbones behind one interface.

A spec is `timm:<model name>` or `openclip:<model name>[@<pretrained tag>]`,
optionally followed by `?key=value&...` keyword arguments for the model
constructor (e.g. `?img_size=224`). Short aliases are listed in ALIASES.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision import transforms as T

ALIASES = {
    # Fungi-specific (BVRA, trained on FungiTastic / Danish Fungi 2020 photos).
    "fungi-effv2b3": "timm:hf-hub:BVRA/tf_efficientnetv2_b3.in1k_ft_fungitastic_384",
    "fungi-convnext": "timm:hf-hub:BVRA/convnext_base.in1k_ft_fungitastic_224",
    "fungi-vitb": "timm:hf-hub:BVRA/vit_base_patch16_224.in1k_ft_fungitastic_224",
    "fungi-effb0": "timm:hf-hub:BVRA/tf_efficientnet_b0.in1k_ft_df20m_224",
    # General biology / general vision.
    "bioclip": "openclip:hf-hub:imageomics/bioclip",
    "bioclip2": "openclip:hf-hub:imageomics/bioclip-2",
    "dinov2-b": "timm:vit_base_patch14_dinov2.lvd142m?img_size=224",
    # Generic CLIP for zero-shot photo-type sorting (good at "microscope image" vs "photo").
    "clip-b32": "openclip:ViT-B-32@laion2b_s34b_b79k",
}

# Preprocessing fixes for checkpoints whose hub config is wrong. The BVRA
# TF-EfficientNets declare ImageNet mean/std but were trained on inputs in
# [-1, 1]; with the declared values the classifier's mean top-1 confidence on
# our photos drops from 0.70 to 0.42 (effv2b3), and linear probes collapse.
_TF_NORM = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
PREPROCESS_OVERRIDES = {
    ALIASES["fungi-effv2b3"]: _TF_NORM,
    ALIASES["fungi-effb0"]: _TF_NORM,
}


@dataclass
class Preprocess:
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_pct: float = 1.0
    interpolation: str = "bicubic"

    @property
    def resize(self) -> int:
        return int(round(self.size / self.crop_pct))

    def to_json(self) -> dict:
        return {**asdict(self), "resize": self.resize}


def resolve(spec: str) -> str:
    return ALIASES.get(spec, spec)


def safe_name(spec: str) -> str:
    """Directory-friendly name for a backbone spec."""
    if spec in ALIASES:
        return spec
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in resolve(spec))


def _parse(spec: str):
    full = resolve(spec)
    kind, _, rest = full.partition(":")
    name, _, query = rest.partition("?")
    kwargs = {}
    for part in filter(None, query.split("&")):
        key, _, value = part.partition("=")
        kwargs[key] = int(value) if value.isdigit() else value
    return kind, name, kwargs


class Backbone(nn.Module):
    """Image -> feature vector. Input is already normalised (see Preprocess)."""

    def __init__(self, spec: str, pretrained: bool = True):
        super().__init__()
        self.spec = resolve(spec)
        # PILZKI_RANDOM_WEIGHTS=1: skip weight downloads (offline smoke tests only).
        pretrained = pretrained and os.environ.get("PILZKI_RANDOM_WEIGHTS") != "1"
        kind, name, kwargs = _parse(spec)
        self.kind = kind
        if kind == "timm":
            import timm

            # Load with the checkpoint's own classifier, then drop it: passing
            # num_classes=0 directly fails for hf-hub checkpoints whose config
            # does not name the classifier (e.g. the BVRA fungi models).
            self.model = timm.create_model(name, pretrained=pretrained, **kwargs)
            self.model.reset_classifier(0)
            cfg = timm.data.resolve_model_data_config(self.model)
            self.feat_dim = self.model.num_features
            self.preprocess = Preprocess(
                size=int(cfg["input_size"][-1]),
                mean=tuple(cfg["mean"]),
                std=tuple(cfg["std"]),
                crop_pct=float(cfg.get("crop_pct", 1.0)),
                interpolation=cfg.get("interpolation", "bicubic"),
            )
            for key, value in PREPROCESS_OVERRIDES.get(self.spec, {}).items():
                setattr(self.preprocess, key, value)
        elif kind == "openclip":
            import open_clip

            model_name, _, tag = name.partition("@")
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=(tag or None) if pretrained else None, **kwargs)
            self.model = model
            self.tokenizer = open_clip.get_tokenizer(model_name)
            size = model.visual.image_size
            size = size[0] if isinstance(size, (tuple, list)) else size
            mean = getattr(model.visual, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN
            std = getattr(model.visual, "image_std", None) or open_clip.OPENAI_DATASET_STD
            self.feat_dim = model.visual.output_dim
            self.preprocess = Preprocess(size=int(size), mean=tuple(mean), std=tuple(std), crop_pct=1.0, interpolation="bicubic")
        else:
            raise ValueError(f"unknown backbone kind {kind!r} in {spec!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "openclip":
            return self.model.encode_image(x)
        return self.model(x)

    @torch.no_grad()
    def encode_text(self, texts: list[str], device) -> torch.Tensor:
        if self.kind != "openclip":
            raise ValueError("text prompts need an openclip backbone")
        tokens = self.tokenizer(texts).to(device)
        return self.model.encode_text(tokens)


def eval_transform(pp: Preprocess):
    interp = T.InterpolationMode.BICUBIC if pp.interpolation == "bicubic" else T.InterpolationMode.BILINEAR
    return T.Compose([
        T.Resize(pp.resize, interpolation=interp),
        T.CenterCrop(pp.size),
        T.ToTensor(),
        T.Normalize(pp.mean, pp.std),
    ])


def train_transform(pp: Preprocess):
    """Augmentations that keep colours intact: colour is a label, so no hue jitter."""
    interp = T.InterpolationMode.BICUBIC if pp.interpolation == "bicubic" else T.InterpolationMode.BILINEAR
    return T.Compose([
        T.RandomResizedCrop(pp.size, scale=(0.3, 1.0), interpolation=interp),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        T.ToTensor(),
        T.Normalize(pp.mean, pp.std),
    ])


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(dev: torch.device):
    if dev.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast(dev.type, enabled=False)

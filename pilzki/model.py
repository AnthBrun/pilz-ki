"""Heads, the full classifier and image datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .backbones import Backbone, Preprocess, autocast
from .common import image_path


class Heads(nn.Module):
    """Outputs on one feature vector: trait tags (multi-label), species, and
    optionally genus (an auxiliary training target, see finetune --genus-weight)."""

    def __init__(self, feat_dim: int, n_tags: int, n_species: int, hidden: int = 0, dropout: float = 0.2, n_genera: int = 0):
        super().__init__()
        self.config = {"feat_dim": feat_dim, "n_tags": n_tags, "n_species": n_species, "hidden": hidden, "dropout": dropout, "n_genera": n_genera}
        self.norm = nn.LayerNorm(feat_dim)
        if hidden:
            self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU())
            dim = hidden
        else:
            self.trunk = nn.Identity()
            dim = feat_dim
        self.dropout = nn.Dropout(dropout)
        self.tags = nn.Linear(dim, n_tags)
        self.species = nn.Linear(dim, n_species)
        self.genus = nn.Linear(dim, n_genera) if n_genera else None

    def forward(self, features: torch.Tensor, with_genus: bool = False):
        h = self.dropout(self.trunk(self.norm(features.float())))
        if with_genus:
            return self.tags(h), self.species(h), self.genus(h) if self.genus is not None else None
        return self.tags(h), self.species(h)


def combine(tag_logits: torch.Tensor, species_logits: torch.Tensor, species_tags: torch.Tensor, alpha: float):
    """Tag probability = alpha * attribute head + (1 - alpha) * species head marginal.

    The species marginal sums the probabilities of all species that carry a
    tag, which handles species with several cap colours naturally.
    """
    p_species = species_logits.float().softmax(-1)
    p_attr = tag_logits.float().sigmoid()
    return alpha * p_attr + (1 - alpha) * (p_species @ species_tags), p_species


class Classifier(nn.Module):
    """Deployable model: RGB in [0, 1], NCHW, already resized/cropped to `size`.

    Returns (tag_probs, species_probs), plus fungus_prob [N] when a gate
    (see gate.py) is attached.
    """

    def __init__(self, backbone: Backbone, heads: Heads, species_tags: np.ndarray, alpha: float = 0.5, gate: nn.Linear | None = None):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.gate = gate
        pp = backbone.preprocess
        self.register_buffer("mean", torch.tensor(pp.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(pp.std).view(1, 3, 1, 1))
        self.register_buffer("species_tags", torch.as_tensor(species_tags, dtype=torch.float32))
        self.alpha = float(alpha)

    def forward(self, image: torch.Tensor):
        x = (image - self.mean) / self.std
        features = self.backbone(x)
        tag_logits, species_logits = self.heads(features)
        tag_probs, species_probs = combine(tag_logits, species_logits, self.species_tags, self.alpha)
        if self.gate is None:
            return tag_probs, species_probs
        return tag_probs, species_probs, self.gate(features.float()).squeeze(-1).sigmoid()


class ImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform, species_index: dict[str, int] | None = None):
        self.rows = rows
        self.transform = transform
        self.species_index = species_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(image_path(row["id"])) as im:
            x = self.transform(im.convert("RGB"))
        target = self.species_index[row["slug"]] if self.species_index else -1
        return x, target, i


def existing(rows: list[dict]) -> list[dict]:
    return [r for r in rows if image_path(r["id"]).exists()]


@torch.no_grad()
def predict(backbone: nn.Module, heads: Heads | None, rows: list[dict], transform, dev, batch_size=64, workers=4):
    """Features (heads=None) or [tag_logits | species_logits | genus_logits if any] for rows, in order."""
    loader = DataLoader(ImageDataset(rows, transform), batch_size=batch_size, num_workers=workers, pin_memory=dev.type == "cuda")
    backbone.eval()
    if heads is not None:
        heads.eval()
    outs = []
    from tqdm import tqdm

    for x, _, _ in tqdm(loader, desc="predict", leave=False):
        with autocast(dev):
            f = backbone(x.to(dev, non_blocking=True))
        if heads is None:
            outs.append(f.float().cpu())
        else:
            tl, sl, gl = heads(f.float(), with_genus=True)
            outs.append(torch.cat([tl, sl] + ([gl] if gl is not None else []), 1).float().cpu())
    return torch.cat(outs).numpy() if outs else np.zeros((0, 0), np.float32)


def save_predictions(path: Path, ids, slugs, tag_logits: np.ndarray, species_logits: np.ndarray, genus_logits: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"genus_logits": genus_logits.astype(np.float16)} if genus_logits is not None and genus_logits.size else {}
    np.savez_compressed(path, ids=np.array(ids), slugs=np.array(slugs), tag_logits=tag_logits.astype(np.float16),
                        species_logits=species_logits.astype(np.float16), **extra)


def preprocess_from_json(d: dict) -> Preprocess:
    return Preprocess(size=d["size"], mean=tuple(d["mean"]), std=tuple(d["std"]), crop_pct=d["crop_pct"], interpolation=d["interpolation"])

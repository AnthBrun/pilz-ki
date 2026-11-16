# pilz-ki: photo → mushroom traits

Goal: on <https://pilze.anthonybrunner.de> (repo `pilzdaten`), a user takes one or
more photos of a find and the trait filter (cap colour, gills, ring, substrate, …)
is **prefilled automatically**. As a by-product the app can show the most similar
species. This repo holds the data preparation, training, evaluation and export
pipeline. Integration into the web app happens in the `pilzdaten` repo (phase 6).

## Ground rules for agents

- **The web server VM (2 cores, 3 GB RAM, no GPU, little disk) only runs `prepare`.**
  Everything heavy (`pack`, `phototype`, `embed`, training, export) runs on the
  GPU machine. Do not start downloads of model weights, `pip install torch` or
  image processing on the VM.
- `data/` is small and committed (manifest, species, vocab, photo types).
  `work/` is large and ignored (images, embeddings, runs, exports).
- Code, comments and docs in English. UI strings in `pilzdaten` stay German.
- Tag names, groups and derivation rules mirror `pilzdaten`
  (`internal/web/static/js/config.js`, `internal/catalog/{load,field}.go`).
  If those change, update `pilzki/vocab.py` / `pilzki/prepare.py` and rerun `prepare`.
- Never commit model weights or photos to git.

## Sibling repos and data

| Path | What |
|---|---|
| `../pilzdaten/data/` | App data: `species.jsonl`, `index_arts.jsonl`, `filter_tags.jsonl` (tags per species), `originals.json` (preview → original photo) |
| `../pilzsuche/raw/images/daten/details/` | Crawl of 123pilzsuche.de: ~95k previews (~150–400 px, `<Slug>-Dateien/imageNNN.jpg`) and ~67k originals (up to 1200 px, ~20 GB) |
| `../pilzsuche/image-sha256.txt` | sha256 of all previews (dedup) |

Facts that drive the design (measured, Nov 2026):

- 4,179 species, 66,763 usable photos after `prepare` (99.9 % have a large original).
  Many species have only a handful of photos (long tail).
- **Labels exist per species, not per photo.** Every photo inherits the species'
  tags. That is exactly what the web filter matches against, so the target
  "does the species on this photo carry tag X" is the right one, not a compromise.
- Tags are multi-valued: on average a species has **3.2 cap colours** and 2.2 gill colours.
- The web filter **ANDs** tags. A single wrong prefilled tag removes the true
  species from the result list. → Prefill only high-precision tags; the key
  metric is *survival* (see Evaluation).
- In the UI several groups are **exclusive** (one cap colour, one cap shape,
  one gill type …) → prefill at most one tag per exclusive group.
- Roughly 10–15 % of photos are not fruit bodies: microscopy, spore prints,
  icons/warning signs, scanned text. They must be filtered out.
- The app targets people **in the forest**. Only traits a forager can see on an
  ordinary photo and confirm by looking are predicted (53 of ~110 tags). Not
  predicted: invisible traits (smell, taste, edibility, spore print, microscopy,
  colour change, milk, season, rarity), traits that need touching, cutting,
  digging or waiting (slimy, hard, flesh colour, hollow/rooting stem,
  hygrophanous) and details too fine for a typical photo (gill attachment,
  spacing, forking, edge; snakeskin stem). The species head could "guess" such
  tags via the species, but the user could not verify them, so they are never prefilled.

## Approach

One backbone, two heads (`pilzki/model.py`):

1. **Tag head** – multi-label sigmoid over the visible tags in `pilzki/vocab.py`
   (53 tags in 10 groups: body type, cap colour, cap shape, cap surface (warts, striate margin),
   underside type, underside colour, stem colour, stem (ring, bulb, net), substrate, clustered growth).
2. **Species head** – softmax over all 4,179 species.

Tag probability used for prefill =
`alpha * sigmoid(tag head) + (1 - alpha) * (softmax(species head) @ species_tag_matrix)`.
The species marginal handles multi-coloured species naturally; `alpha` is tuned on val: the value that prefills the most tags at the target precision (tag mAP always picks alpha = 0, which hurts unseen species).

Backbones start from fungi-pretrained weights (BVRA FungiTastic / Danish Fungi
models on Hugging Face) or BioCLIP; see `ALIASES` in `pilzki/backbones.py`.
Strategy: compare cheaply with frozen embeddings + linear heads, then fine-tune the
best one end to end. For the browser we prefer a small model
(EfficientNetV2-B3 ~14M params ≈ 26 MB compressed incl. species head).

Prefill rule (`evaluate.prefill`, reimplemented in JS later):

- per tag a threshold calibrated on val for precision ≥ target (default 0.995; at 0.9 ~30 % of queries lose the true species);
  tags that never reach it are disabled;
- exclusive group: take the argmax tag of the group, prefill only if it passes its threshold;
- groups with `requires: hutpilz` (cap shape, gills, stem, …) only if p(hutpilz) ≥ 0.5;
- several photos of one find: average the probabilities first.

## Splits (`prepare`)

Deterministic, hash based, stored in `data/manifest.jsonl`:

- `holdout`: 5 % of species (224) completely excluded from training → generalisation to unseen species.
- remaining species: per photo 80 % `train`, 10 % `val` (threshold calibration, model selection), 10 % `test`.
- every non-holdout species keeps ≥ 1 training photo.
- photos identical across species are dropped, as are tiny previews without an original.

Final model (`--final`): also trains on holdout species, still calibrates on val and reports on test.

## Pipeline

```sh
# --- on the web server VM (already done, outputs committed in data/) ---
python3 -m pilzki prepare                        # needs only numpy, seconds

# --- on the GPU machine ---
git clone https://github.com/AnthBrun/pilz-ki.git && cd pilz-ki
nvidia-smi                                        # driver must work (EndeavourOS/Arch: nvidia or nvidia-open package)
uv sync --extra train --extra export --extra dev  # uses Python 3.12 from .python-version (uv downloads it); PyPI torch ships CUDA

# 0. smoke test first (the training code has not run end to end yet; expect small fixes)
uv run pytest
uv run pilzki pack --image-root ~/pilzsuche-images --out work/images --limit 500   # previews suffice here
PILZKI_RANDOM_WEIGHTS=1 uv run pilzki embed --backbone fungi-effb0 --limit 500
PILZKI_RANDOM_WEIGHTS=1 uv run pilzki train-heads --backbone fungi-effb0 --epochs 2
uv run pilzki evaluate --run work/runs/heads-fungi-effb0 --min-support 2
PILZKI_RANDOM_WEIGHTS=1 uv run pilzki finetune --backbone fungi-effb0 --epochs 1 --limit 500 --workers 2
uv run pilzki evaluate --run work/runs/ft-fungi-effb0 --min-support 2   # export needs thresholds.json
uv run pilzki export --run work/runs/ft-fungi-effb0 --check 8
rm -rf work/emb work/runs work/export   # then run the real pipeline

# 1. photos: copy the crawl's originals over once (~20 GB) and pack them locally
rsync -a server:projects/pilzsuche/raw/images/daten/ ~/pilzsuche-images/daten/
uv run pilzki pack --image-root ~/pilzsuche-images --out work/images --workers 8   # ~7 GB, short side 512 px
#    (alternative without the 20 GB copy: run `pack --tar -` on the server and pipe it
#     over ssh – but that keeps the small VM busy for ~40 min, so avoid it)

# 2. photo types (zero-shot CLIP), then eyeball each class and commit the result
uv run pilzki phototype                          # -> data/phototype.jsonl
uv run pilzki sheet --type microscopy --out work/sheets/microscopy.jpg
uv run pilzki sheet --type graphic    --out work/sheets/graphic.jpg
uv run pilzki sheet --type cut        --out work/sheets/cut.jpg
git add data/phototype.jsonl && git commit -m "data: photo types"

# 3. linear-probe benchmark of backbones
for b in fungi-effv2b3 fungi-convnext fungi-vitb bioclip; do
  uv run pilzki embed --backbone $b
  uv run pilzki train-heads --backbone $b
  uv run pilzki evaluate --run work/runs/heads-$b
done

# 4. fine-tune the winner (and, if it is large, also fungi-effv2b3 for the browser)
uv run pilzki finetune --backbone fungi-effv2b3 --init-heads work/runs/heads-fungi-effv2b3
uv run pilzki evaluate --run work/runs/ft-fungi-effv2b3

# 5. final model + export
uv run pilzki finetune --backbone fungi-effv2b3 --init-heads work/runs/heads-fungi-effv2b3 --final
uv run pilzki evaluate --run work/runs/ft-fungi-effv2b3-final
# 6. gate against non-fungus photos (negatives: COCO val2017, 5k everyday photos, ~0.8 GB)
mkdir -p work/neg/coco && curl -o work/neg/val2017.zip http://images.cocodataset.org/zips/val2017.zip \
  && unzip -q work/neg/val2017.zip -d work/neg/coco && rm work/neg/val2017.zip
uv run pilzki train-gate --run work/runs/ft-fungi-effv2b3-final --negatives work/neg/coco   # -> <run>/gate.pt, gate_report.md
uv run pilzki export   --run work/runs/ft-fungi-effv2b3-final    # -> work/export/<run>/{model.onnx, model.int8.onnx, meta.json}
uv run pilzki predict  --export work/export/ft-fungi-effv2b3-final photo.jpg   # reference for the JS port
```

Training machine: EndeavourOS (Arch), RTX 3060, 12 GB VRAM (Ampere, bf16 autocast is used).
Python is pinned to 3.12 in `.python-version` because Arch's system Python is often newer than the ML wheels support.
`finetune` defaults to batch 32; for 384 px models lower `--batch-size` to 24 if
CUDA runs out of memory, and adjust `--workers` to the CPU cores.
Rough budget: `pack` ~3 min, `embed` a few minutes per backbone,
`train-heads` < 5 min, `finetune` ~6 min per epoch for fungi-effv2b3 (15 epochs ≈ 1.5 h).

`uv run pilzki <step> -h` lists all options. `PILZKI_WORK` relocates `work/`,
`PILZKI_IMAGES` the packed photos. `PILZKI_RANDOM_WEIGHTS=1` skips weight
downloads (code smoke tests only, results are meaningless).

Hugging Face: some files are served from `cas-bridge.xethub.hf.co`; if that host
is blocked, set `HF_HUB_DISABLE_XET=1` or allow it.

## Evaluation (`evaluate` → `report.md`, `thresholds.json`)

Per split (`val`, `test`, `holdout`):

- **survival**: share of queries where every prefilled tag belongs to the true
  species (= the right species is still in the result list). Main safety metric.
- prefilled tags per query, median number of species still matching.
- the same for 3 averaged photos of one species (simulates top/underside/stem shots).
- species top-1 / top-5 (not meaningful on holdout), tag mAP.
- per tag: AP, threshold, precision, recall on val.

Acceptance targets for a first release (to be confirmed with real numbers):

- test survival ≥ 0.95 with ≥ 3 prefilled tags on average,
- holdout survival ≥ 0.9 (unseen species must not be filtered away),
- int8 ONNX gives the same prefill as the torch model on ≥ 95 % of check photos.
- non-fungus photos (everyday photos, microscopy, blank images) get no prefill: the gate keeps ≥ 98 % of fruit-body photos and rejects ≥ 95 % of COCO.

### Results so far (Nov 2026, RTX 3060)

Linear probe on frozen features (originals, 512 px), test split:

| backbone | species@1 | species@5 | tag mAP |
|---|---|---|---|
| fungi-effv2b3 | 0.634 | 0.821 | 0.915 |
| fungi-convnext | 0.616 | 0.808 | 0.908 |
| bioclip | 0.583 | 0.789 | 0.899 |
| fungi-vitb | 0.565 | 0.787 | 0.892 |

Fine-tuned fungi-effv2b3 (15 epochs, ~6 min each), target precision 0.995, alpha 0.5:

| split | species@1 | prefilled tags | survival | 3-photo survival |
|---|---|---|---|---|
| test | 0.750 | 4.33 | 0.983 | 0.998 |
| holdout | – | 2.60 | 0.938 | 0.986 |

Final model (`--final`, also trained on the holdout species): test species@1 0.743,
4.39 prefilled tags, survival 0.981 (3 photos: 0.998), alpha 0.4.
`work/export/ft-fungi-effv2b3-final/model.int8.onnx` is 26 MB and gives the same
prefill as torch on 98 % of 500 val photos.

Gate (`train-gate`, threshold keeps 99 % of val fruit bodies): passes 98.7 % of test
and 96.6 % of 262 Wikimedia Commons fungus photos (rejects are mostly b/w, microscopy
or market photos), 3 % of held-out COCO, 6 % of crawl microscopy/graphic, 1 % of
synthetic images. Weak spot: nature photos without fungi (forest floor, leaves,
berries, snails) still pass 60-100 %, but get only 0-1 tags (`auf_boden`, `holz`).
Without the gate 21 % of COCO photos got tags (a wedding cake became a Parasol with
7 tags); with it 1 %. External Commons photos (54 species, 258 photos): species@1 0.62,
@5 0.78, 4.3 tags, survival 0.984, i.e. as reliable as the test split.

Lessons:

- The BVRA TF-EfficientNets need inputs normalised to [-1, 1]; their hub config
  says ImageNet mean/std (fixed in `backbones.PREPROCESS_OVERRIDES`; with the
  wrong values species@1 drops from 0.49 to 0.14 on previews).
- Fine-tuning makes the model more confident on unseen species: at the same
  target precision (0.99) holdout survival drops from 0.91 (heads) to 0.85.
  0.995 restores it while still prefilling more tags than the heads model.
- int8 kernels (onnxruntime `quantize_dynamic`/`quantize_static`) break
  EfficientNet; `export.compress_weights` only compresses the stored weights.
- Tried without gain (Nov 2026, non-final fungi-effv2b3, target precision 0.995):
  horizontal-flip TTA (identical numbers), species temperature (T > 1 fewer tags,
  T 0.85 +0.17 tags but holdout survival -0.01), ensemble with convnext heads
  (more tags, lower survival), auxiliary genus head (`finetune --genus-weight 0.5`:
  holdout genus@1 0.35 -> 0.39, but test/holdout tags and survival unchanged within
  noise). Mixing a genus marginal into the tag probability trades ~0.3 tags for
  +0.01 holdout survival. The prefill is limited by the labels more than by the model.
- Label quality: `pilzki audit-labels --run work/runs/ft-fungi-effv2b3` lists
  suspicious species tags for pilzdaten (e.g. Jodwasserkopf without lamellen,
  Orangefüßiger Stacheling with lamellen instead of spines; the meta pages
  "Kuriositäten" and "Teepilz" are not field species).

If survival is too low: raise `--target-precision`, `--disable` noisy tags, or
prefill only the most reliable groups (body type, underside type, substrate).
Adding a tag back: put it into `pilzki/vocab.py`, rerun `prepare`, retrain.

## Phase 6: integration into `pilzdaten` (done, branch `feat/photo-prefill` there)

- `pilzdaten/scripts/model-bundle.sh <export dir> data/model` copies `meta.json` and
  `model.int8.onnx`, adds a checksum-pinned onnxruntime-web (WASM) and `.gz` variants.
  `MODEL_DIR` (default `DATA_DIR/model`) is served under `/model/`, not embedded in the
  Go binary and not in git; without it the "Foto" button is hidden.
- `static/js/photo.js` mirrors `pilzki/predict.py`: resize the short side, then centre crop
  1:1, RGB in [0, 1]; gate; mean over the photos of a find that pass it; prefill rule.
  Change both together.
- Suggestions are dashed chips with a camera icon (`state.suggested`), removable like any
  filter; a new photo replaces old suggestions but never the user's own choices in an
  exclusive group. Up to three similar species (>= 1 %) are listed for comparison.
- The service worker caches model and runtime (versioned URLs), so it works offline.
- Checked with Playwright on 262 Commons photos: browser and Python give the same quality
  (4.25 vs 4.30 tags, survival 0.981 vs 0.985; identical tag sets on 77 %, differences are
  tags at the threshold, from JPEG decoding and resampling); ~0.3 s per photo on a desktop CPU.
- Photos never leave the device.

## Open questions

- ~~Which backbone wins?~~ fungi-effv2b3: best accuracy and the smallest download (see Results).
- Hut colour labels are coarse (species-level, several colours). Is per-photo relabelling of a small set worth it later?
- Use the photo's EXIF month for the season filter in the app (no model needed)?

## Layout

```
pilzki/
  vocab.py        predicted tag groups (mirror of config.js)
  prepare.py      data/ manifest, species, vocab, splits          (CPU, numpy only)
  pack.py         resize photos into work/images or a tar stream  (CPU, Pillow)
  phototype.py    zero-shot photo types -> data/phototype.jsonl
  sheet.py        contact sheets for manual checks
  backbones.py    timm / open_clip backbones, preprocessing, aliases
  model.py        heads, deployable Classifier, dataset, predict
  embed.py        frozen features -> work/emb/<backbone>/
  train_heads.py  heads on embeddings -> work/runs/heads-<backbone>/
  finetune.py     end-to-end training  -> work/runs/ft-<backbone>/
  evaluate.py     thresholds, prefill simulation, report.md
  gate.py         fungus/non-fungus gate on a run's features -> <run>/gate.pt
  audit.py        suspicious species labels for pilzdaten -> work/label_audit.md
  export.py       ONNX (+ int8) and meta.json -> work/export/<run>/
  predict.py      run an export on photos (reference for the JS port)
data/             committed small outputs of prepare / phototype
work/             ignored: images, emb, runs, export, sheets, neg (gate negatives)
```

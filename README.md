# Frozen Foundation Models for Mammogram Classification

Code accompanying the paper *"Frozen Foundation Models Rival Fine-Tuned Networks for
Mammogram Classification at a Fraction of the Cost"* (submitted to IEEE Access).

Four frozen, pre-trained foundation-model encoders feed a lightweight multi-view
linear classifier for breast-cancer classification on mammograms. No backbone is
trained or fine-tuned; only a small set of logistic-regression heads over the
concatenated frozen features is fitted.

## Data

Experiments use the public CBIS-DDSM (official patient-disjoint split) and
Mini-DDSM datasets. Neither dataset is redistributed in this repository; see the
original sources:

- CBIS-DDSM: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- Mini-DDSM: https://www.kaggle.com/datasets/cheddad/miniddsm2

## Key scripts

- `config.py` — central configuration (paths, experiment grid, seeds)
- `embedders.py` — frozen foundation-model encoders
- `multiview_probe.py`, `linear_probe.py` — the multi-view linear classifier
- `data_scaling*.py` — database-growth / learning-curve experiments
- `finetune_eval*.py`, `finetune_vit.py`, `finetune_mammo_clip.py` — the fine-tuned-backbone comparison
- `scanner_confound_check.py` — site-shortcut check on Mini-DDSM
- `paper_figures.py`, `paper_numbers.py` — figure and numbers generation for the paper

## Reproducibility

### Regenerating every number in the paper

```
python make_paper_numbers.py --list   # the steps and which table/figure each produces
python make_paper_numbers.py          # all CPU steps (about 20 minutes)
python make_paper_numbers.py --gpu    # also the occlusion maps (needs a GPU)
```

It runs the analysis scripts in order on the cached embeddings and writes every result to
`cache/` and the figures. All operating-point metrics use a fixed, untuned decision
threshold of 0.5 (`DECISION_THRESHOLD` in `config.py`); AUC confidence intervals are a
case-level bootstrap with 5,000 resamples (`bootstrap_ci.py`).

All experiments use a fixed seed (`SEED = 42` in `config.py`). The official
CBIS-DDSM patient-disjoint train/test split is used throughout; for Mini-DDSM, a
patient-stratified 20% test partition is redrawn with a different random seed for
each of the four evaluation runs reported in the paper (class balance preserved, no
patient ever placed in both partitions).

### Dataset versions

- CBIS-DDSM: the official train/test split distributed via The Cancer Imaging Archive
  (mass subset; 1,231 training / 361 test images; case-level counts in the paper).
- Mini-DDSM: the JPEG-8 ("light") release, `MINI-DDSM-Complete-JPEG-8`, with its
  accompanying `DataWMask.xlsx` metadata (lesion contours, per-breast `Status`).

### Pre-trained checkpoints (frozen, used as-is; no fine-tuning except the explicit
fine-tuning control described in the paper)

- DINOv2 ViT-L/14 — `facebookresearch/dinov2` via `torch.hub`, `dinov2_vitl14`
  (used at the 518x518 high-resolution setting).
- RAD-DINO — Hugging Face `microsoft/rad-dino`.
- Mammo-CLIP (EfficientNet-B5) — Hugging Face `shawn24/Mammo-CLIP` checkpoint, loaded
  via `mammo_clip.py`.
- MammoScreen — Hugging Face `ianpan/mammoscreen` (three-model EfficientNetV2-S
  ensemble pre-trained on CBIS-DDSM + the RSNA screening challenge), classification
  head removed and read as a feature extractor via `mammoscreen.py`.
- Breast-region cropping (Mini-DDSM only) — Hugging Face `ianpan/mammo-crop`.

### Environment

Python 3.12, PyTorch 2.10, scikit-learn 1.8, FAISS 1.13, Hugging Face Transformers 5.4.
All four frozen extractors load through their released interfaces with no modification;
only the multi-view logistic-regression classifier (`multiview_probe.py`) is fitted.

### Split-generation procedure

CBIS-DDSM uses TCIA's fixed official split as released; no split is generated here.
For Mini-DDSM, `prep_miniddsm.py` ingests lesion-bearing breasts (keeping only the
breast side with an annotated tumour contour, to avoid the healthy contralateral
breast inheriting its folder's `Status` label) and normal breasts separately; the
patient-stratified test partition is then drawn at evaluation time (see
`data_scaling*.py` / `miniddsm_B.py`) using `SEED` plus a per-run offset for the four
repeated draws.

### What "reproducible" means here

Re-running the pipeline end-to-end from the raw datasets (not attempted by an
independent party as part of this work) requires the checkpoints and dataset versions
above, plus the breast-region cropping for Mini-DDSM (`mammo_crop.py`); no contrast
enhancement is applied for the reported results.
All embeddings, configurations, and per-case predictions used to produce the paper's
numbers are cached in a single relational database (`db.py`), so every reported number
traces back to a stored input; this is reproducibility of the analysis pipeline, not
an independent third-party replication.

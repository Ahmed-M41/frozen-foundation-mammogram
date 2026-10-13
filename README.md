# Frozen Foundation Models for Mammogram Classification

Code accompanying the paper *"Frozen Foundation Models Rival Fine-Tuned Networks for
Mammogram Classification at a Fraction of the Cost"* (IEEE Access, submitted).

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

All experiments use a fixed seed (`SEED = 42` in `config.py`). The official
CBIS-DDSM patient-disjoint train/test split is used throughout.

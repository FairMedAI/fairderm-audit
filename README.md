# FairDerm

Fixing skin-tone bias in melanoma detection using synthetic data augmentation for underrepresented dark-skin images.

## Project Overview

FairDerm is a three-stage pipeline that:

1. **Trains** an EfficientNet-B0 melanoma classifier on HAM10000 (predominantly light-skin)
2. **Fine-tunes** on DDI (Diverse Dermatology Images) to adapt to diverse skin tones
3. **Augments** with synthetic dark-skin melanoma images to improve fairness — **train-only, SHA256 verified 0 overlap**

An ablation study sweeps synthetic multipliers (0x/2x/5x/10x) to find the optimal trade-off between overall accuracy and dark-skin sensitivity.

**Key finding:** Standard fine-tuning can *amplify* the AUROC gap (from **-0.131 to -0.209**), while adding train-only synthetic dark melanomas does not improve Dark AUROC (**gap -0.209 → -0.291**, paired delta **-0.066**, **p=0.962 NS**). This suggests we need GANs or diffusion models for real improvement.

## Repository Structure

```
FairDerm/
├── fairderm.py              # Complete pipeline (all stages)
├── configs/config.yaml      # Single source of truth for hyperparams
├── requirements.txt         # Pinned Python dependencies
├── scripts/check_leakage.py # SHA256 byte-level leakage verification
├── scripts/generate_clean_artifacts.py
├── splits/                  # Saved train/val/test splits + leakage report
│   ├── ddi_split_seed42.json
│   └── leakage_report.json  # SHA256 0 overlap proof
├── .gitignore
├── FINAL_RESULTS.md         # Frozen results — source of truth for the paper
├── website_update.md        # Plain-language site update
├── README.md                # This file
│
├── paper_latex/             # Full LaTeX manuscript
│   ├── main.tex
│   ├── generate_tables.py   # Auto-generates .tex tables from metrics_clean.json
│   ├── generate_figures.py  # Auto-generates PDF figures from metrics_clean.json
│   ├── sections/            # Paper sections (.tex)
│   ├── tables/              # Generated LaTeX tables
│   └── figures/             # Generated PDF figures
│
├── models/                  # Final checkpoints
│   ├── fairderm_ham_baseline.pth  # partial epoch-5 best val 0.9650
│   ├── fairderm_ddi_finetuned.pth
│   └── fairderm_final.pth
│
├── results/                 # Per-stage outputs
│   ├── metrics_clean.json   # Canonical results (12:17) — all metrics + CIs
│   ├── MORNING_VERIFY.md    # Light run verification
│   ├── baseline/
│   ├── finetune/
│   ├── augment/
│   ├── evaluate/
│   └── ablation/
│
├── data/
│   ├── ham10000/
│   │   ├── HAM10000_metadata.csv
│   │   └── images/          # 10,015 .jpg files
│   └── synthetic_train_only/ # 290 generated images (29 train dark mel × 10) — train-only
│
└── ddidiversedermatologyimages/
    ├── ddi_metadata.csv
    └── *.png                # 656 DDI images
```

## Environment Setup

```bash
# 1. Create virtual environment (Python 3.9+)
python3 -m venv fairderm_env
source fairderm_env/bin/activate

# 2. Install dependencies (exact versions pinned)
pip install -r requirements.txt
```

## Data Placement

### HAM10000

Download from [ISIC Archive](https://www.isic-archive.com/) or [Harvard Dataverse](https://dataverse.harvard.edu/). Place files so the structure is:

```
data/ham10000/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0026046.jpg
    ├── ISIC_0026047.jpg
    └── ...  (10,015 .jpg files)
```

### DDI (Diverse Dermatology Images)

Download from the [DDI repository](https://github.com/mattgroh/fitzpatrick-scale). Place the extracted folder in the project root:

```
ddidiversedermatologyimages/
├── ddi_metadata.csv
├── 000001.png
├── 000002.png
└── ...  (656 .png files)
```

## Running the Pipeline

Each stage runs independently via CLI. Run stages in order for a full experiment, or individually for debugging.

```bash
# 0. Sanity check — verify imports, data, device, forward pass
python fairderm.py --stage sanity

# 1. HAM10000 baseline training (~40 min on MPS, batch 32, 15 epochs)
python fairderm.py --stage baseline

# 2. DDI fine-tuning (~5 min)
python fairderm.py --stage finetune

# 3. Synthetic augmentation + retraining (~5 min)
#    - Generates 290 synthetic images from TRAIN dark melanomas only (29 × 10)
#    - Never touches val/test data — SHA256 verified
python fairderm.py --stage augment

# 4. Fairness audit + plots + metrics_clean.json (~2 min)
python fairderm.py --stage evaluate

# 5. Ablation study — 0x/2x/5x/10x multipliers (~20 min total)
python fairderm.py --stage ablation

# 6. Verify no leakage (SHA256 byte-level)
python scripts/check_leakage.py

# 7. Auto-generate LaTeX tables and figures from metrics_clean.json
python paper_latex/generate_tables.py
python paper_latex/generate_figures.py
```

### Light clean run (25 min, crash-recovered, what paper currently uses)
```bash
# Uses existing baseline checkpoint (partial epoch-5, best val 0.9650)
nice -n 19 python fairderm.py --stage finetune --batch-size 16 --epochs 10
nice -n 19 python fairderm.py --stage augment --batch-size 16 --epochs 10
python fairderm.py --stage evaluate
python scripts/generate_clean_artifacts.py
python scripts/check_leakage.py  # must say PASSED overlap 0
```

## Reproducibility Checklist

- **Seed:** 42 (all stages)
- **Device:** MPS (Apple Silicon) preferred; falls back to CUDA then CPU
- **Key versions:** PyTorch 2.8.0, torchvision 0.23.0, numpy 2.0.2, Python 3.9.6
- **Split saved:** `splits/ddi_split_seed42.json` with filenames per partition
- **Leakage check:** `splits/leakage_report.json` — SHA256 byte-level, synthetic_vs_test=0, synthetic_vs_val=0, synthetic_vs_any_ddi=0
- **Metrics saved:** `results/metrics_clean.json` (Sep 20 controlled seed-42 run) with all metrics, CIs, p-values + `MORNING_VERIFY.md`
- **Tables auto-generated:** `python paper_latex/generate_tables.py` reads from metrics_clean.json
- **Figures auto-generated:** `python paper_latex/generate_figures.py` reads from metrics_clean.json
- **Config as code:** `configs/config.yaml` is single source of truth for hyperparams
- **Checkpoint caveat:** Current baseline is partial (best val epoch 5, AUROC 0.9650) — full 15-epoch run pending overnight for final paper

## MPS Non-Determinism Warning

All experiments use Apple Silicon (MPS backend). The MPS backend introduces floating-point non-determinism in certain operations. Exact numerical reproduction on different hardware (CUDA, CPU) is not guaranteed. Results should be replicated on at least one other hardware platform for confirmation.

## Results — Controlled Seed-42 Run (leakage 0, n=132 test: 42 Light / 42 Dark, 10 mel each)

See [`FINAL_RESULTS.md`](FINAL_RESULTS.md) for the frozen source of truth. See [`results/metrics_clean.json`](results/metrics_clean.json) for machine-readable results with bootstrap CIs and p-values.

**Current controlled run (Sep 20, seed 42):**

| Stage | Light AUROC | Dark AUROC | Gap | Overall |
|-------|-------------|------------|-----|---------|
| Baseline (partial) | 0.7188 | 0.5875 | -0.1313 | 0.6622 |
| Fine-tuned | 0.7563 | 0.5469 [0.341–0.750] | -0.2094 | 0.6336 |
| +Synthetic (290) | 0.7719 [0.582–0.925] | 0.4813 [0.276–0.688] | -0.2906 | 0.6395 |

Paired Dark AUROC delta finetuned→augment: **-0.066 (95% bootstrap CI [-0.151, 0.008]), p=0.962 NS**

Key findings:
- **Fine-tuning amplifies bias:** AUROC gap widens from **-0.131 to -0.209**
- **Synthetic augmentation does not improve Dark AUROC:** gap **-0.209 → -0.291**, Dark 0.547→0.481 (paired delta **-0.066**, p=0.962 NS)
- **Dark AUROC CIs huge:** finetuned 0.547 [0.341–0.750] → final 0.481 [0.276–0.688] with only 10 melanomas per subgroup
- **Ablation:** Dark AUROC peaks at 2x (0.575), degrades at 10x (0.481), non-monotonic single-seed pattern

## Citation

```bibtex
@article{fairderm2026,
  title={FairDerm: Evaluating and Mitigating Skin-Tone Bias in Melanoma Detection via Synthetic Augmentation},
  author={Gottimukkala, Shanmuka},
  year={2026},
  note={Working Paper #1, FairMed AI — leakage-free audit, SHA256 verified, gap -0.131→-0.209→-0.291, Dark 0.547→0.481 p=0.962}
}
```

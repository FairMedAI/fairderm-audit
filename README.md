# FairDerm

Fixing skin-tone bias in melanoma detection using synthetic data augmentation for underrepresented dark-skin images.

## Project Overview

FairDerm is a three-stage pipeline that:

1. **Trains** an EfficientNet-B0 melanoma classifier on HAM10000 (predominantly light-skin)
2. **Fine-tunes** on DDI (Diverse Dermatology Images) to adapt to diverse skin tones
3. **Augments** with synthetic dark-skin melanoma images to improve fairness

An ablation study sweeps synthetic multipliers (0x/2x/5x/10x) to find the optimal trade-off between overall accuracy and dark-skin sensitivity.

**Key finding:** Standard fine-tuning can *amplify* the AUROC gap (from -0.050 to -0.328), while simple photometric augmentation does not significantly improve Dark AUROC (p=0.524). This suggests we'd need GANs or diffusion models to get real improvement.

## Repository Structure

```
FairDerm/
├── fairderm.py              # Complete pipeline (all stages)
├── configs/config.yaml      # Single source of truth for hyperparams
├── requirements.txt         # Pinned Python dependencies
├── scripts/check_leakage.py # Verifies no synthetic-test overlap
├── splits/                  # Saved train/val/test splits
│   └── ddi_split_seed42.json
├── .gitignore
├── FINAL_RESULTS.md         # Frozen results — source of truth for the paper
├── README.md                # This file
│
├── paper_latex/             # Full LaTeX manuscript
│   ├── main.tex
│   ├── generate_tables.py   # Auto-generates .tex tables from metrics JSON
│   ├── generate_figures.py  # Auto-generates PDF figures from metrics JSON
│   ├── sections/            # Paper sections (.tex)
│   ├── tables/              # Generated LaTeX tables
│   └── figures/             # Generated PDF figures
│
├── models/                  # Final checkpoints
│   ├── fairderm_ham_baseline.pth
│   ├── fairderm_ddi_finetuned.pth
│   └── fairderm_final.pth
│
├── results/                 # Per-stage outputs
│   ├── metrics_seed42.json  # Canonical results (all metrics + CIs)
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
│   └── synthetic_train_only/ # 290 generated images (train dark mel × 10)
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

# 1. HAM10000 baseline training (~40 min on MPS)
python fairderm.py --stage baseline

# 2. DDI fine-tuning (~5 min)
python fairderm.py --stage finetune

# 3. Synthetic augmentation + retraining (~5 min)
#    - Generates 290 synthetic images from TRAIN dark melanomas only
#    - Never touches val/test data
python fairderm.py --stage augment

# 4. Fairness audit + plots + metrics_seed42.json (~2 min)
python fairderm.py --stage evaluate

# 5. Ablation study — 0x/2x/5x/10x multipliers (~20 min total)
python fairderm.py --stage ablation

# 6. Verify no leakage (synthetic vs test)
python scripts/check_leakage.py

# 7. Auto-generate LaTeX tables and figures
python paper_latex/generate_tables.py
python paper_latex/generate_figures.py
```

### Additional Flags

| Flag | Effect |
|------|--------|
| `--dry-run` | Load data + model, run 1 forward pass, skip training |
| `--epochs N` | Override max epochs (baseline/finetune/augment only) |
| `--batch-size N` | Override batch size |
| `--resume` | Resume training from last checkpoint |
| `--extract` | Extract HAM10000 from `dataverse_files.zip` (if present) |

## Reproducibility Checklist

- **Seed:** 42 (all stages)
- **Device:** MPS (Apple Silicon) preferred; falls back to CUDA then CPU
- **Key versions:** PyTorch 2.8.0, torchvision 0.23.0, numpy 2.0.2, Python 3.9.6
- **Split saved:** `splits/ddi_split_seed42.json` with filenames per partition
- **Leakage check:** `python scripts/check_leakage.py` verifies no synthetic-test overlap
- **Metrics saved:** `results/metrics_seed42.json` with all metrics, CIs, p-values
- **Tables auto-generated:** `python paper_latex/generate_tables.py` reads from JSON
- **Figures auto-generated:** `python paper_latex/generate_figures.py` reads from JSON
- **Config as code:** `configs/config.yaml` is single source of truth for hyperparams
- **Checkpoint hash:** Verify checkpoint integrity with `sha256sum models/*.pth`

## MPS Non-Determinism Warning

All experiments use Apple Silicon (MPS backend). The MPS backend introduces floating-point non-determinism in certain operations. Exact numerical reproduction on different hardware (CUDA, CPU) is not guaranteed. Results should be replicated on at least one other hardware platform for confirmation.

## Results

See [`FINAL_RESULTS.md`](FINAL_RESULTS.md) for the frozen source of truth. See [`results/metrics_seed42.json`](results/metrics_seed42.json) for machine-readable results with bootstrap CIs and p-values.

Key findings:
- **Fine-tuning amplifies bias:** AUROC gap widens from -0.050 to -0.328
- **Synthetic augmentation doesn't meaningfully help:** gap goes from -0.328 to -0.303 (p=0.524, not significant)
- **Dark AUROC stays wide:** CIs are huge with only 10 melanomas per subgroup
- **Ablation peaks at 5x:** Dark AUROC best at 5x (0.547), degrades at 10x (0.525)

## Citation

```bibtex
@article{fairderm2026,
  title={FairDerm: Evaluating and Mitigating Skin-Tone Bias in Melanoma Detection via Synthetic Augmentation},
  author={Gottimukkala, Shanmuka},
  year={2026}
}
```

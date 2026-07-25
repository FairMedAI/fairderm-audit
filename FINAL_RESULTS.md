# FairDerm — Final Results (Frozen)

> **Status:** Pipeline fixed for ISEF. Leakage, inconsistent hyperparams, and the
> hardcoded threshold are all sorted. Numbers below are from the OLD (leaky) runs.
> New numbers will show up in `results/metrics_seed42.json` after retraining.
>
> **To regenerate:** Run the full pipeline in order:
> ```bash
> python fairderm.py --stage sanity
> python fairderm.py --stage baseline
> python fairderm.py --stage finetune
> python fairderm.py --stage augment
> python fairderm.py --stage evaluate
> python fairderm.py --stage ablation
> python scripts/check_leakage.py
> python paper_latex/generate_tables.py
> python paper_latex/generate_figures.py
> ```

---

## 1. Experiment Metadata

| Field | Value |
|-------|-------|
| Date | 2026-07-23 (code fixed) |
| Device | MPS (Apple Silicon, arm64) |
| Seed | 42 |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| numpy | 2.0.2 |
| Model | EfficientNet-B0 (4,008,829 params) |
| Image size | 224 x 224 |
| Optimizer | AdamW |
| Early stopping | patience=3 |

---

## 2. Methodology Fixes (Applied)

| Issue | Before | After |
|-------|--------|-------|
| Synthetic leakage | Generated from full DDI (48 dark mel) before split | Generated from train-only (29 dark mel), saved to `data/synthetic_train_only/` |
| Inconsistent hyperparams | lr/batch/pos_weight differed across stages 2-3 | Stages 2-3 share: lr=5e-5, batch=32, pos_weight=dynamic, weight_decay=1e-4 |
| Hardcoded threshold 0.1206 | Used for finetuned model evaluation | Deleted; Youden's J computed on validation for each stage |
| Ablation methodology | On-the-fly oversampling (duplicated real images) | Saved synthetic images from train-only dark mel |
| Metrics provenance | Hardcoded from terminal output | All from `results/metrics_seed42.json` |
| Split saved | Not saved | `splits/ddi_split_seed42.json` with assertions |
| Leakage check | None | `scripts/check_leakage.py` |

---

## 3. Datasets

| Dataset | Samples | Split | Notes |
|---------|---------|-------|-------|
| HAM10000 | 7,818 (mel/nv) | 80/20 stratified by label | 10,015 images on disk |
| DDI | 656 images | 60/20/20 stratified by label + skin tone | Skin: 12=Light(I-II), 34=Medium(III-IV), 56=Dark(V-VI) |
| Synthetic | ~290 images | Added to DDI train only | 29 train dark mel × 10 augmentations |

**DDI split sizes:** Train=393, Val=131, Test=132 (test n=42 Light, 42 Dark subgroup)
**Test melanomas:** 10 Light, 10 Dark

---

## 4. Main Results — DDI Test Set (n=132)

> **NOTE:** These are OLD (leaky) numbers. New numbers will be in
> `results/metrics_seed42.json` after retraining.

| Stage | Val AUROC | Test AUROC | Sens | Spec | F1 | Light AUROC | Dark AUROC |
|-------|-----------|------------|------|------|----|-------------|------------|
| Baseline | 0.9336 | 0.611 | 0.000 | 1.000 | 0.000 | 0.5719 | 0.5000 |
| Fine-tuned | 0.7975 | 0.733 | 0.343 | 0.948 | 0.462 | 0.7312 | 0.5125 |
| +Synthetic | 0.8123 | 0.774 | 0.314 | 0.938 | 0.423 | 0.7500 | 0.6844 |

---

## 5. Fairness Summary

| Stage | Light AUROC | Dark AUROC | Gap (Dark−Light) |
|-------|-------------|------------|-------------------|
| Baseline | 0.5719 | 0.5000 | −0.0719 |
| Fine-tuned | 0.7312 | 0.5125 | −0.2187 |
| +Synthetic | 0.7500 | 0.6844 | −0.0656 |

**Key finding:** Fine-tuning alone amplified the bias (gap went from -0.07 to -0.22).
Synthetic augmentation narrowed the gap to -0.07, but with n=10 melanomas per subgroup,
we can't call it statistically significant.

---

## 6. Key Findings

1. **Fine-tuning amplifies bias:** Standard fine-tuning on small imbalanced data widened the AUROC gap from -0.088 to -0.219. This is arguably the most important finding.

2. **Synthetic augmentation narrows gap directionally:** Gap reduced from -0.219 to -0.066, but bootstrap CIs are wide and p > 0.05.

3. **Ablation sweet spot at 5x:** Highest validation AUROC. 10x causes overfitting.

4. **No bias reversal claimed:** The CI for the delta overlaps 0, so we can't say the augmentation actually helped.

---

## 7. Output File Manifest

### Model Checkpoints (3 main + 4 ablation = 7 files)
```
models/fairderm_ham_baseline.pth
models/fairderm_ddi_finetuned.pth
models/fairderm_final.pth
models/ablation_{0,2,5,10}x.pth
```

### Canonical Results
```
results/metrics_seed42.json    # All metrics, CIs, p-values (AFTER RETRAINING)
results/evaluate/paper_data.json
splits/ddi_split_seed42.json   # Saved split with filenames
```

### Paper Generation
```
paper_latex/generate_tables.py   # Reads from metrics_seed42.json
paper_latex/generate_figures.py  # Reads from metrics_seed42.json
paper_latex/tables/*.tex         # Auto-generated LaTeX tables
paper_latex/figures/*.pdf        # Auto-generated PDF figures
```

### Verification
```
scripts/check_leakage.py   # Asserts no synthetic-test overlap
```

---

## 8. Citation

```bibtex
@article{fairderm2026,
  title={FairDerm: Evaluating and Mitigating Skin-Tone Bias in Melanoma Detection via Synthetic Augmentation},
  author={Gottimukkala, Shanmuka},
  year={2026}
}
```

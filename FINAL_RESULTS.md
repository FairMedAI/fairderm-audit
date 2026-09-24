# FairDerm — Final Results (Controlled Seed-42 Rerun)

> **Status:** Current controlled single-seed results. The machine-readable
> artifacts in `results/metrics_seed42.json`, `results/metrics_clean.json`, and
> the regenerated LaTeX tables are the numerical source of truth. Earlier
> leaky-run values remain in the repository only as historical provenance.
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
| Date | 2026-09-20 |
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
| Batch size | 32 |

---

## 2. Methodology Fixes (Applied)

| Issue | Before | After |
|-------|--------|-------|
| Synthetic leakage | Generated from full DDI (48 dark mel) before split | Generated from train-only (29 dark mel), saved to `data/synthetic_train_only/` |
| Inconsistent hyperparams | lr/batch/pos_weight differed across stages 2-3 | Stages 2-3 share: lr=5e-5, batch=32, pos_weight=dynamic, weight_decay=1e-4 |
| Hardcoded threshold | Used for finetuned model evaluation | Deleted; Youden's J computed on validation for each stage |
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

**DDI split sizes:** Train=393, Val=131, Test=132 (test n=42 Light, 48 Medium, 42 Dark)
**Test melanomas:** 10 Light, 10 Dark

---

## 4. Controlled Results — DDI Test Set

> Thresholds for sensitivity, specificity, and F1 were selected separately
> using Youden's J on each validation set. AUROC is the primary threshold-free
> comparison metric.

| Stage | Val AUROC | Test AUROC | Sens | Spec | F1 | Light AUROC | Dark AUROC |
|-------|-----------|------------|------|------|----|-------------|------------|
| Baseline | 0.9650 | 0.6622 | 0.6000 | 0.6598 | 47.19% | 0.7188 | 0.5875 |
| Fine-tuned | 0.7787 | 0.6336 | 0.5714 | 0.5876 | 0.4211 | 0.7563 | 0.5469 |
| +Synthetic | 0.7596 | 0.6395 | 0.5429 | 0.6186 | 0.4176 | 0.7719 | 0.4813 |

---

## 5. Fairness Summary

| Stage | Light AUROC | Dark AUROC | Gap (Dark−Light) |
|-------|-------------|------------|-------------------|
| Baseline | 0.7188 | 0.5875 | −0.1313 |
| Fine-tuned | 0.7563 | 0.5469 | −0.2094 |
| +Synthetic | 0.7719 | 0.4813 | −0.2906 |

**Key finding:** Fine-tuning and synthetic augmentation did not reduce the
Light--Dark AUROC gap in the controlled seed-42 rerun. The augmented model's
Dark AUROC was lower than the fine-tuned model's (paired delta -0.0663,
95% bootstrap interval [-0.1508, 0.0080], p=0.9620). With only 10 melanomas
per subgroup, the result is preliminary.

---

## 6. Key Findings

1. **Fine-tuning did not remove the disparity:** The Light--Dark AUROC gap changed from -0.131 at baseline to -0.209 after fine-tuning.

2. **Synthetic augmentation did not improve Dark AUROC:** Dark AUROC changed from 0.5469 to 0.4813, with p=0.9620 for the paired comparison.

3. **The ablation was non-monotonic:** Dark AUROC was 0.5438 at 0x, 0.5750 at 2x, 0.5344 at 5x, and 0.4812 at 10x.

4. **Claims remain preliminary:** The experiment uses one seed and a test subgroup containing only 10 melanoma cases.

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

# FairDerm — Research Audit

> Historical pre-submission check. Current code and clean artifacts supersede
> findings explicitly marked as fixed below.

---

## 1. Experiment Metadata Verification

Every claim in `FINAL_RESULTS.md` Section 1 checked against saved files:

| Claim | Source File | Verified |
|-------|-------------|----------|
| Seed = 42 | `results/baseline/config.json` → `reproducibility.seed` | Yes |
| PyTorch 2.8.0 | `results/baseline/config.json` → `reproducibility.pytorch_version` | Yes |
| EfficientNet-B0 | `fairderm.py:251-258` `create_model()` | Yes |
| DDI n = 656 | `ddidiversedermatologyimages/ddi_metadata.csv` row count | Yes |
| DDI test n = 132 | `results/finetune/config.json` → `test_samples` | Yes |
| Dark test n = 42 | `paper_data.json` → `Dark.n_total` = 42 | Yes |
| Synthetic 290 images | `results/augment/config.json` → `synthetic_images` = 290 | Yes |
| 29 train dark melanomas | `results/augment/train.log` → "Found 29 TRAIN dark-skin melanomas" | Yes |
| HAM10000 7,818 mel/nv | `results/baseline/train.log` → "loaded 7818 samples" | Yes |
| Val AUROC 0.9650 | `results/baseline/metrics.csv` → best `val_auroc` = 0.9650 | Yes |

---

## 2. Claim-by-Claim Verification

### Current result: baseline Dark AUROC = 0.5875

- **Source:** `results/metrics_seed42.json` → `stages.baseline.Dark`
- **Status:** Current controlled rerun
- **Caveat:** The baseline sensitivity is 0.400 at the validation-selected threshold. AUROC remains the primary metric.

### Current result: synthetic augmentation does not improve Dark AUROC

- **Source:** `results/metrics_seed42.json` → paired Dark AUROC delta
- **Status:** Current controlled rerun
- **Caveat:** Dark AUROC changes from 0.5469 to 0.4813; paired bootstrap p=0.9620. Sensitivity is 0.500 at the augmented model's validation-selected threshold.

### Current result: AUROC gap widens after augmentation

- **Source:** `paper_data.json`
  - Finetune: Dark 0.5469 − Light 0.7563 = −0.2094
  - Final: Dark 0.4813 − Light 0.7719 = −0.2906
- **Status:** Current controlled rerun.

### Claim 4: Threshold-dependent sensitivity

- **Source:** `results/metrics_seed42.json` stage-specific subgroup metrics
- **Status:** Sensitivity is 0.700/0.400 for baseline, 0.600/0.600 for fine-tuned, and 0.600/0.500 for augmented (Light/Dark).
- **Caveat:** These values depend on validation-selected thresholds and only 10 melanoma cases per subgroup.

### Current result: 2× has the highest Dark AUROC in the ablation

- **Source:** `results/ablation/ablation_report.csv`
- **Status:** 2× Dark AUROC = 0.5750; 0× = 0.5438, 5× = 0.5344, 10× = 0.4812.

### Claim 6: 10× degrades observed generalization

- **Source:** `results/ablation/ablation_report.csv`
- **Status:** 10× has the lowest Dark AUROC (0.4812) and overall test AUROC (0.6433) in this single-seed run; mechanism is not established.

---

## 3. Metrics Provenance Audit

| Metric | Source | Method | Reproducible from Code? |
|--------|--------|--------|------------------------|
| Subgroup AUROC/Sens/Spec/PPV | `results/evaluate/paper_data.json` | Programmatic (`get_metrics()`) | Yes |
| Ablation overall metrics | `results/ablation/ablation_report.csv` | Programmatic (`compute_metrics()`) | Yes |
| Val AUROCs | `results/*/metrics.csv` | Programmatic (`train_model()`) | Yes |
| Overall test AUROC/Sens/Spec/F1 | `results/metrics_seed42.json` | Programmatic (`stage_evaluate()`) | Yes |
| Bootstrap CIs | `results/metrics_seed42.json` | Programmatic bootstrap | Yes |
| Validation thresholds | `results/metrics_seed42.json` | Youden's J on validation | Yes |

**Implication:** The current overall test metrics, bootstrap CIs, and
validation thresholds are generated and persisted by `stage_evaluate`. MPS
non-determinism can still produce small numerical differences across hardware.

---

## 4. Overstated Wording Audit

| Current Wording (FINAL_RESULTS.md) | Issue | Recommended Revision |
|-------------------------------------|-------|----------------------|
| "reverses the sensitivity gap" | Light sensitivity decreased; gap reversal is partly degradation | "narrows the sensitivity gap from −0.40 to +0.10" |
| "dramatically" (in "AUROC gap narrows dramatically") | Subjective editorial | "substantially" or remove |
| "completely fails on dark skin" | Defensible (AUROC=0.50 = chance) but Sens depends on threshold | "achieves no discriminative ability on dark skin (AUROC=0.50)" |
| "causes overfitting" (re: 10x) | Interpretation, not proven mechanism | "degrades generalization" |
| "sweet spot" (re: 5x) | Informal language | "optimal multiplier of 5×" |

---

## 5. Methodology Strengths

1. **Clean train/val/test separation** — Stratified by the cross-product of {benign, malignant} × {Light, Medium, Dark}. Explicit disjointness assertions at `fairderm.py:747-749`. No data leakage between splits.

2. **Threshold-independent primary metric** — AUROC is the main comparison metric, appropriate for comparing models at different operating points.

3. **Bootstrap 95% CIs reported** — 1000-iteration percentile bootstrap for uncertainty quantification. Degenerate single-class samples skipped. Appropriate for small subgroups.

4. **Ablation study included** — Tests the oversampling multiplier variable, showing the trade-off curve from 0× to 10×.

5. **Full reproducibility infrastructure** — Seed=42, `config.json` per stage with environment metadata, all checkpoints saved, TeeLogger captures complete stdout.

6. **Standard architecture** — EfficientNet-B0 is well-validated for medical imaging. Progressive unfreezing (features.6-7 + classifier) is standard transfer learning practice.

7. **Appropriate class imbalance handling** — BCEWithLogitsLoss with pos_weight addresses the malignant:benign imbalance in both DDI (26:74) and HAM10000 (14:86).

8. **Leakage prevention** — Assertions enforce disjoint filenames across splits. Synthetic images are added only to the training DataLoader, never to val/test.

---

## 6. Methodology Weaknesses

### 6.1 [FIXED] Synthetic Images Generated from Full DDI Before Splitting

`fairderm.py:1031` selects dark-skin melanomas from `ddi_df` (all 656 samples) before `_make_splits()` partitions into train/val/test. The 480 synthetic images are therefore derived from source images that may include val/test examples.

**To be fair:** The augmentation pipeline (brightness, contrast, noise, elastic transform, rotation, flip) produces pixel-level modifications, not exact copies. The model can't memorize a test image through these transforms. But the lesion shape and texture are preserved, so it's a gray area.

**Resolution:** Current `fairderm.py` generates synthetic images from `train_df`
dark-skin melanomas after splitting. `scripts/check_leakage.py` and
`scripts/generate_clean_artifacts.py` provide filename and SHA256 checks.

> **Current disclosure:** The reported rerun generates all synthetic samples
> exclusively from the DDI training split. Earlier pre-clean runs used a
> different source procedure and are not used for the current results.

### 6.2 [HIGH] Small DDI Test Subgroup Size

n=42 per subgroup in the test set, with only 10 melanomas per subgroup. Sensitivity estimates have ±10 percentage point uncertainty per single detection. Bootstrap CIs are correspondingly wide:
- Dark AUROC (fine-tuned): 0.269–0.757
- Dark AUROC (+Synthetic): 0.448–0.902

The results are directional, not definitive. Larger multi-center validation is required.

### 6.3 [FIXED IN CODE; HISTORICAL ARTIFACTS REMAIN] Inconsistent Hyperparameters Across Stages

| Parameter | Baseline | Fine-tune | Augment | Ablation |
|-----------|----------|-----------|---------|----------|
| pos_weight | None | 3.0 | 2.0 | None |
| weight_decay | 1e-4 | 1e-4 | 0.01 (default) | 0.01 (default) |
| batch_size | 8 | 8 | 16 | 8 |
| epochs | 10 | 10 | 5 | 5 |
| lr | 3e-4 | 5e-5 | 2e-5 | 1e-4 |

The current code fixes the Stage 2/3 training settings, including a shared
10-epoch budget, batch size 32, learning rate, weight decay, and dynamic
pos_weight. Existing `results/*/config.json` files record older runs and must
be regenerated before quoting controlled results.

### 6.4 [RESOLVED] Different Threshold Selection Strategies

- Baseline: Youden's J on validation set (0.074603)
- Fine-tuned: Youden's J on validation set (0.043219)
- Final (+Synthetic): Youden's J on validation set (0.017312)

The strategy is now implemented in `fairderm.py` and saved in
`results/metrics_seed42.json`. Sensitivity, specificity, and PPV remain
threshold-dependent; AUROC is the primary threshold-independent comparison.

### 6.5 [RESOLVED] Ablation Tests Oversampling, Not Augmentation

The ablation duplicates real dark-skin images with on-the-fly transforms (`train_transform`). The main pipeline generates and saves new synthetic images via Albumentations, then loads them with a simpler transform (resize + normalize only). These are different interventions:
- Ablation: more copies of existing images, diverse on-the-fly augmentation
- Main pipeline: new synthetic images, limited on-the-fly augmentation

The ablation now uses saved synthetic images from the same train-only source
directory as the main augmentation pipeline.

### 6.6 [MEDIUM] No Synthetic Image Quality Validation

No FID (Frechet Inception Distance), no human expert evaluation, no diversity metrics. The augmentation is purely low-level pixel transforms, which limits semantic diversity compared to GAN-based or style-transfer approaches.

### 6.7 [MEDIUM] Ablation Tradeoff Figure — Mislabeled Y-Axis (Corrected)

The original figure (`results/ablation/ablation_tradeoff.png`, generated by `fairderm.py:1282`) labeled the right Y-axis "Dark Skin Sensitivity" but plotted overall test sensitivity (values ~0.2, not dark-skin-specific). **This was a visualization labeling error, corrected during paper preparation** by regenerating the figure with `paper/fix_ablation_figure.py`. The corrected figure (`paper/ablation_tradeoff.png`) labels the axis "Overall Sensitivity".

### 6.8 [RESOLVED] Hardcoded Threshold

The old hardcoded threshold was removed. Thresholds are recomputed by
`compute_youden_threshold()` from validation predictions and persisted in
`results/metrics_seed42.json`.

### 6.9 [LOW] Medium Skin Tone Subgroup Excluded

DDI has three skin tone groups (12=Light, 34=Medium, 56=Dark). Only Light and Dark are reported. Medium (n≈16 in test) was excluded because the subset is too small for reliable metric estimation.

### 6.10 [LOW] MPS Non-Determinism

PyTorch MPS backend can produce slightly different floating-point results across runs due to operation ordering. The results reported are from a single run. All configs and checkpoints are saved to enable reproduction on the same hardware, but exact numerical reproduction on different hardware is not guaranteed.

---

## 7. Synthetic Source Issue (Resolved in Current Run)

The earlier pipeline generated synthetic images before the DDI split. The current
pipeline calls `_make_splits()` first, selects the 29 dark-skin melanoma cases
from `train_df`, and writes 290 train-only images. `scripts/check_leakage.py`
and `scripts/generate_clean_artifacts.py` verify split disjointness and
SHA-256 overlap; the current report records zero overlap with train, validation,
or test DDI images. The earlier issue remains documented for provenance only.

---

## 8. Submission Readiness Assessment

| Category | Assessment | Notes |
|----------|------------|-------|
| Problem significance | **Strong** | Skin-tone bias in melanoma detection is a documented, important health equity problem |
| Technical implementation | **Strong** | Clean pipeline, proper splits, standard architecture, full reproducibility infrastructure |
| Reproducibility | **Moderate** | Seed/configs/checkpoints preserved; hardcoded threshold and manual metric entry are gaps |
| Statistical rigor | **Moderate** | Bootstrap CIs reported; small subgroup sizes limit confidence; overlapping CIs between stages |
| Fairness claims | **Preliminary** | The controlled rerun did not improve Dark AUROC; p=0.962 and the subgroup contains 10 melanomas |
| Publication readiness | **Ready with limitations** | Current artifacts and narrative are reconciled; multi-seed and external validation remain future work |

---

## 9. Required Actions Before Submission

| # | Action | Status |
|---|--------|--------|
| 1 | Fix ablation tradeoff figure Y-axis label | ✅ Done — `paper/fix_ablation_figure.py` regenerated `paper/ablation_tradeoff.png` |
| 2 | Document the former hardcoded threshold or re-derive in code | ✅ Done — old constant removed; validation Youden thresholds are persisted |
| 3 | Revise "reverses the sensitivity gap" wording to "narrows the sensitivity gap" | ✅ Done |
| 4 | Add prominent sample-size caveat to subgroup results | ✅ Done |
| 5 | Disclose synthetic source issue (full DDI before splitting) | ✅ Done — current pipeline uses train-only sources; historical issue retained as provenance |
| 6 | Note MPS non-determinism in reproducibility section | ✅ Done |
| 7 | Clarify that ablation tests oversampling, not synthetic augmentation | ✅ Done — ablation now uses saved train-only synthetics |
| 8 | Decide on threshold presentation strategy (common vs. model-specific) | ✅ Done — model-specific validation Youden thresholds; AUROC is primary |

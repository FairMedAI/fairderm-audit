# FairDerm — Research Audit

> Pre-submission check.
> Self-review to catch problems before reviewers do.

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
| Synthetic 480 images | `results/augment/config.json` → `synthetic_images` = 480 | Yes |
| 48 dark melanomas | `results/augment/train.log` → "Found 48 real dark-skin melanomas" | Yes |
| HAM10000 7,818 mel/nv | `results/baseline/train.log` → "loaded 7818 samples" | Yes |
| Val AUROC 0.9336 | `results/baseline/metrics.csv` → `val_auroc` = 0.933551 | Yes (rounds to 0.9336) |

---

## 2. Claim-by-Claim Verification

### Claim 1: "Baseline completely fails on dark skin (Sens=0.0)"

- **Source:** `paper_data.json` → `baseline.Dark.sensitivity` = 0.0
- **Status:** Supported
- **Caveat:** Baseline uses threshold=0.5 (default, not optimized for DDI). At a lower threshold, sensitivity would be non-zero. However, AUROC=0.50 for Dark confirms genuinely poor discrimination regardless of threshold — the model has no ability to distinguish dark-skin melanoma from benign.

### Claim 2: "Synthetic augmentation raises Dark Sens to 0.600"

- **Source:** `paper_data.json` → `final_synthetic.Dark.sensitivity` = 0.6
- **Status:** Supported
- **Caveat:** This is at threshold=0.2157 (Youden's J on validation set). Different threshold would give different value. The improvement is threshold-dependent.

### Claim 3: "AUROC gap narrows from −0.22 to −0.07"

- **Source:** `paper_data.json`
  - Finetune: Dark 0.5125 − Light 0.7312 = −0.2187
  - Final: Dark 0.6844 − Light 0.7500 = −0.0656
- **Status:** Supported. Arithmetic is correct.

### Claim 4: "Reverses the sensitivity gap"

- **Source:** Finetune Light Sens=0.80, Dark Sens=0.40 (gap −0.40); Final Light Sens=0.50, Dark Sens=0.60 (gap +0.10)
- **Status:** **Partially overstated.** The gap reversal is partly because Light sensitivity *decreased* (0.80→0.50), not solely because Dark improved. Recommended revision: "narrows the sensitivity gap from −0.40 to +0.10, achieving more balanced performance across skin tones."

### Claim 5: "5× oversampling achieves best val AUROC (0.8417)"

- **Source:** `results/ablation/5x/config.json` → `final_best_val_auroc` = 0.8417
- **Status:** Supported.

### Claim 6: "10× causes overfitting"

- **Source:** `ablation_report.csv` → 10x AUROC = 0.6622 (lowest of all multipliers)
- **Status:** Supported as observation. The interpretation "overfitting" is one possible explanation; "excessive duplication degrades generalization" is more precise.

---

## 3. Metrics Provenance Audit

| Metric | Source | Method | Reproducible from Code? |
|--------|--------|--------|------------------------|
| Subgroup AUROC/Sens/Spec/PPV | `results/evaluate/paper_data.json` | Programmatic (`get_metrics()`) | Yes |
| Ablation overall metrics | `results/ablation/ablation_report.csv` | Programmatic (`compute_metrics()`) | Yes |
| Val AUROCs | `results/*/metrics.csv` | Programmatic (`train_model()`) | Yes |
| Overall test AUROC/Sens/Spec/F1 | `paper/generate_paper_assets.py` hardcoded dict | **Manual entry** from stdout | No — not saved to file by pipeline |
| Bootstrap CIs | `paper/generate_paper_assets.py` hardcoded dict | **Manual entry** from stdout | No — not saved to file by pipeline |
| Threshold 0.1206 | `fairderm.py:1197` hardcoded constant | **Manual entry**, origin unknown | No — derivation not in codebase |

**Implication:** The overall test metrics and bootstrap CIs are correct (they match the evaluation stdout) but are fragile — a re-run of `stage_evaluate` could produce slightly different values on non-deterministic hardware. The threshold 0.1206 cannot be independently verified.

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

### 6.1 [HIGH] Synthetic Images Generated from Full DDI Before Splitting

`fairderm.py:1031` selects dark-skin melanomas from `ddi_df` (all 656 samples) before `_make_splits()` partitions into train/val/test. The 480 synthetic images are therefore derived from source images that may include val/test examples.

**To be fair:** The augmentation pipeline (brightness, contrast, noise, elastic transform, rotation, flip) produces pixel-level modifications, not exact copies. The model can't memorize a test image through these transforms. But the lesion shape and texture are preserved, so it's a gray area.

**Required future fix:** Generate synthetic images only from `train_df` dark-skin melanomas after splitting.

> **Disclose this in the paper:**
> Synthetic samples were generated from available dark-skin melanoma examples before final dataset splitting. Although the transformations create modified images rather than exact copies, future work should generate synthetic samples only from training-set images to eliminate any possibility of source-image overlap.

### 6.2 [HIGH] Small DDI Test Subgroup Size

n=42 per subgroup in the test set, with only 10 melanomas per subgroup. Sensitivity estimates have ±10 percentage point uncertainty per single detection. Bootstrap CIs are correspondingly wide:
- Dark AUROC (fine-tuned): 0.269–0.757
- Dark AUROC (+Synthetic): 0.448–0.902

The results are directional, not definitive. Larger multi-center validation is required.

### 6.3 [MEDIUM] Inconsistent Hyperparameters Across Stages

| Parameter | Baseline | Fine-tune | Augment | Ablation |
|-----------|----------|-----------|---------|----------|
| pos_weight | None | 3.0 | 2.0 | None |
| weight_decay | 1e-4 | 1e-4 | 0.01 (default) | 0.01 (default) |
| batch_size | 8 | 8 | 16 | 8 |
| epochs | 10 | 10 | 5 | 5 |
| lr | 3e-4 | 5e-5 | 2e-5 | 1e-4 |

These confounds make it impossible to attribute performance changes solely to synthetic augmentation. A controlled experiment would fix all hyperparameters except the augmentation.

### 6.4 [MEDIUM] Different Threshold Selection Strategies

- Baseline: 0.5 (default)
- Fine-tuned: 0.1206 (hardcoded, origin not documented in codebase)
- Final (+Synthetic): Youden's J (max TPR−FPR) on validation set

Sensitivity, specificity, and PPV are threshold-dependent metrics. Evaluating models at different operating points makes these metrics incomparable across stages. AUROC remains valid as a threshold-independent comparison.

### 6.5 [MEDIUM] Ablation Tests Oversampling, Not Augmentation

The ablation duplicates real dark-skin images with on-the-fly transforms (`train_transform`). The main pipeline generates and saves new synthetic images via Albumentations, then loads them with a simpler transform (resize + normalize only). These are different interventions:
- Ablation: more copies of existing images, diverse on-the-fly augmentation
- Main pipeline: new synthetic images, limited on-the-fly augmentation

The ablation does not test the same thing as the main augmentation pipeline.

### 6.6 [MEDIUM] No Synthetic Image Quality Validation

No FID (Frechet Inception Distance), no human expert evaluation, no diversity metrics. The augmentation is purely low-level pixel transforms, which limits semantic diversity compared to GAN-based or style-transfer approaches.

### 6.7 [MEDIUM] Ablation Tradeoff Figure — Mislabeled Y-Axis (Corrected)

The original figure (`results/ablation/ablation_tradeoff.png`, generated by `fairderm.py:1282`) labeled the right Y-axis "Dark Skin Sensitivity" but plotted overall test sensitivity (values ~0.2, not dark-skin-specific). **This was a visualization labeling error, corrected during paper preparation** by regenerating the figure with `paper/fix_ablation_figure.py`. The corrected figure (`paper/ablation_tradeoff.png`) labels the axis "Overall Sensitivity".

### 6.8 [LOW] Hardcoded Threshold 0.1206

The fine-tuned model threshold (0.1206) was used during final evaluation, but the exact derivation procedure was not preserved in the repository. It is documented as a reproducibility limitation.

### 6.9 [LOW] Medium Skin Tone Subgroup Excluded

DDI has three skin tone groups (12=Light, 34=Medium, 56=Dark). Only Light and Dark are reported. Medium (n≈16 in test) was excluded because the subset is too small for reliable metric estimation.

### 6.10 [LOW] MPS Non-Determinism

PyTorch MPS backend can produce slightly different floating-point results across runs due to operation ordering. The results reported are from a single run. All configs and checkpoints are saved to enable reproduction on the same hardware, but exact numerical reproduction on different hardware is not guaranteed.

---

## 7. Synthetic Source Issue (Elevated)

> **Synthetic samples were generated from available dark-skin melanoma examples before final dataset splitting.** Although the transformations create modified images rather than exact copies, future work should generate synthetic samples only from training-set images to eliminate any possibility of source-image overlap.

**Technical details:**
- `fairderm.py:1031`: `dark_mel = ddi_df[(ddi_df["skin_tone"] == 56) & (ddi_df["malignant"] == 1)]` selects from the full DDI DataFrame
- `_make_splits()` is called at line 1026, but the synthetic generation at line 1031 uses `ddi_df` (pre-split), not `train_df`
- The 48 source images may include examples that end up in val/test splits
- Augmentation transforms (brightness, contrast, noise, elastic, rotation, flip) preserve semantic content while modifying pixel values

**Impact assessment:** The degree of concern depends on whether the augmented images are "too similar" to their sources. For low-level transforms, the model sees meaningfully different pixel patterns. For high-level features (lesion shape, border irregularity), the source content is preserved. This is a gray area that should be disclosed.

---

## 8. Submission Readiness Assessment

| Category | Assessment | Notes |
|----------|------------|-------|
| Problem significance | **Strong** | Skin-tone bias in melanoma detection is a documented, important health equity problem |
| Technical implementation | **Strong** | Clean pipeline, proper splits, standard architecture, full reproducibility infrastructure |
| Reproducibility | **Moderate** | Seed/configs/checkpoints preserved; hardcoded threshold and manual metric entry are gaps |
| Statistical rigor | **Moderate** | Bootstrap CIs reported; small subgroup sizes limit confidence; overlapping CIs between stages |
| Fairness claims | **Promising but preliminary** | Directional improvement is clear; statistical significance not established due to sample size |
| Publication readiness | **Needs clarification** | Threshold derivation, figure label correction (done), synthetic source disclosure, and wording revisions required before submission |

---

## 9. Required Actions Before Submission

| # | Action | Status |
|---|--------|--------|
| 1 | Fix ablation tradeoff figure Y-axis label | ✅ Done — `paper/fix_ablation_figure.py` regenerated `paper/ablation_tradeoff.png` |
| 2 | Document origin of threshold 0.1206 or re-derive in code | ⬜ Pending |
| 3 | Revise "reverses the sensitivity gap" wording to "narrows the sensitivity gap" | ⬜ Pending |
| 4 | Add prominent sample-size caveat to subgroup results | ⬜ Pending |
| 5 | Disclose synthetic source issue (full DDI before splitting) | ⬜ Pending |
| 6 | Note MPS non-determinism in reproducibility section | ⬜ Pending |
| 7 | Clarify that ablation tests oversampling, not synthetic augmentation | ⬜ Pending |
| 8 | Decide on threshold presentation strategy (common vs. model-specific) | ⬜ Pending |

# FairDerm: Evaluating the Limits of Photometric Augmentation for Skin-Tone Bias

Simple photometric augmentation from 29 dark melanomas (290 images) does **not** close the skin-tone performance gap — this is a **negative-result methodology study**, not evidence of an AI improvement.

---

## Abstract

Dermatological deep learning models exhibit performance disparities across skin tones, with diagnostic accuracy often degrading for darker presentations. FairDerm audits this bias on the **Dermatology Data Initiative (DDI)** dataset (**656 images**, split **393 / 131 / 132**) and tests whether heuristic augmentation can mitigate it. The test partition is small and imbalanced — **42 Light (10 melanoma), 48 Medium (8 melanoma), 42 Dark (10 melanoma)** — so subgroup estimates are **low-power** (Dark AUROC bootstrap SD ~0.15) and **fragile**; results are directional, not definitive.

An **EfficientNet-B0** model is pretrained on **HAM10000 (7,818 mel/nv samples)** and fine-tuned on DDI, then augmented with **290 images generated from 29 training-split dark melanoma lesions (10× photometric/geometric transforms; no generative model)**. Provenance is verified per image: byte-level SHA256 confirms zero overlap with held-out partitions, and a perceptual check matches all 290 images to a training source (**median similarity 0.9985**, i.e. near-duplicates), with **no FID score and no clinician review** of realism.

Seed-42 point estimates (Dark AUROC 0.547 → 0.481, paired delta −0.066, p=0.962) are **one draw from a high-variance distribution** and are confounded by class composition. Generating only malignant dark images while keeping the static `pos_weight` **2.8529** changes the effective training ratio from **102/291 to 392/291** (balanced = **0.7423**), over-weighting malignancy by **≈3.8×**. An arm study from a fixed checkpoint isolates this:

| Arm | Composition | pos_weight | Dark AUROC | Light–Dark gap |
|-----|-------------|-----------|-----------|----------------|
| static (headline) | 290 dark melanomas | 2.8529 | **0.484** | −0.222 |
| recomputed | same images | 0.7423 | **0.478** | −0.222 |
| symmetric | +101 dark benigns | 1.0 | **0.550** | **−0.141** |

Recomputing the weight changes nothing; **balancing composition recovers Dark AUROC to 0.550** (at the fine-tuned level, 0.547). The apparent degradation is largely a **class-composition artifact**, not an augmentation effect.

Because single-split estimates proved unstable, the full pipeline was repeated across **5 seeds (42–46)**, re-deriving the split, generated images, and class ratio per seed: mean Dark AUROC **0.575 ± 0.145** (fine-tuned) → **0.604 ± 0.152** (+generated), paired delta **+0.028 (95% seed-level CI [−0.001, 0.057], not significant)**; the Light–Dark gap is **unchanged (−0.081 → −0.083)**. Thresholds are selected by Youden's J per stage from validation predictions, are **not directly comparable across stages**, and should not be over-interpreted; **AUROC is the primary metric**. All experiments are **DDI-only — no external validation** was performed, so generalizability is not demonstrated.

---

## Key Findings

- **Low-Power.** The Dark test subgroup holds only 10 melanomas; bootstrap SD ~0.15 AUROC with wide CIs. Subgroup conclusions are preliminary and directional.
- **No External Validation.** Every result is DDI-only (n=656). No independent dataset (e.g. Fitzpatrick17k, held-out HAM10000, clinical cohorts) was used; generalizability is untested.
- **Composition Confound.** Augmenting only dark malignancies under a static `pos_weight` (2.8529) shifts the combined ratio 102/291 → 392/291 and over-weights malignancy ≈3.8×. The balanced arm (0.7423; +101 dark benigns at weight 1.0) recovers Dark AUROC 0.484 → 0.550 and narrows the gap −0.222 → −0.141.
- **Limited Diversity.** 290 images are transforms of just 29 lesions (median similarity 0.9985 — near-duplicates); no FID, no LPIPS, no clinician review, no GAN/diffusion comparison. Semantic diversity is bounded by the source pool.
- **Multi-Seed Primary.** Across 5 seeds the Dark delta is +0.028 (CI includes 0, NS) and the gap is unchanged; between-seed variability exceeds every between-stage difference and the gap sign flips across seeds. Seed-42 (p=0.962) is one unfavorable draw.
- **Threshold Note.** Per-stage Youden-J thresholds are not comparable across stages; sensitivity/specificity/PPV are threshold-dependent. AUROC is the threshold-free primary comparison.

---

## Repository Structure

```
FairDerm/
├── fairderm.py                  # Full pipeline: sanity/baseline/finetune/augment/evaluate/ablation
├── configs/config.yaml          # Single source of truth for hyperparameters
├── requirements.txt             # Pinned dependencies
├── scripts/
│   ├── verify_paper_numbers.py  # Asserts manuscript numbers match frozen JSON/CSV
│   ├── check_leakage.py         # SHA256 byte-level leakage check
│   ├── generate_clean_artifacts.py
│   ├── verify_synthetic_provenance.py
│   ├── run_multiseed.py         # 5-seed (42–46) robustness driver
│   ├── run_pos_weight_arms.py   # static / recomputed / symmetric arm study
│   ├── compare_mps_repeat.py
│   └── validate_synthetic.py
├── paper/
│   ├── FAIRDERM.pdf             # Canonical paper (Rev 2, 22 pp)
│   └── FairDerm_v2_Rev2.pdf     # Same build, versioned name
├── FairDerm_Final_Paper.pdf     # Newest paper build (identical to paper/ copies)
├── paper_latex/                 # LaTeX source for the Rev 2 paper
│   ├── main.tex
│   ├── references.bib
│   ├── generate_tables.py       # Frozen JSON/CSV -> tables/*.tex
│   ├── generate_figures.py      # Frozen JSON/CSV -> figures/*.pdf
│   ├── sections/                # abstract, introduction, methods, experiments,
│   │                            # results, discussion, limitations, appendix
│   ├── tables/                  # tab1..tab8 + tab_multiseed
│   └── figures/                 # fig1..fig6 + supplementary
├── models/                      # Checkpoints (baseline, finetuned, final, ablation_*)
├── results/                     # Frozen metrics, CIs, p-values, per-seed runs
│   ├── metrics_clean.json       # Canonical machine-readable results
│   ├── metrics_seed42.json
│   ├── multiseed_summary.json
│   ├── ablation/
│   └── pos_weight_arms/
├── splits/                      # Train/val/test splits + leakage + provenance reports
│   ├── ddi_split_seed42.json … ddi_split_seed46.json
│   ├── leakage_report.json
│   └── synthetic_provenance_report.json
├── FINAL_RESULTS.md             # Frozen results + experiment metadata
├── RESEARCH_AUDIT.md            # Methodology strengths/weaknesses audit
├── SUBMISSION_GUIDE.md          # Compile/format instructions per venue
├── PRE_SUBMISSION_CHECKLIST.md
└── LICENSE                      # Apache License 2.0
```

**Data is not committed** (large files, see `.gitignore`). To reproduce, place:
- `data/ham10000/` — `HAM10000_metadata.csv` + `images/` (10,015 `.jpg`)
- `ddidiversedermatologyimages/` — `ddi_metadata.csv` + 656 `.png`

---

## How to Reproduce

```bash
# 1. Install pinned dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. Verify every number quoted in the manuscript against the frozen artifacts
python scripts/verify_paper_numbers.py   # must print "All checked manuscript numbers match"

# 3. (optional) Re-verify no synthetic leakage
python scripts/check_leakage.py          # must report overlap 0
```

Full pipeline (requires the datasets above; **no retraining needed** to inspect the shipped results):

```bash
python fairderm.py --stage sanity
python fairderm.py --stage baseline
python fairderm.py --stage finetune
python fairderm.py --stage augment
python fairderm.py --stage evaluate
python fairderm.py --stage ablation
```

Regenerate paper assets from frozen data (no inference):

```bash
python paper_latex/generate_tables.py    # -> paper_latex/tables/*.tex
python paper_latex/generate_figures.py   # -> paper_latex/figures/*.pdf
```

Compile the paper (needs LaTeX, or use Overleaf with `paper_latex/main.tex` as the root):

```bash
cd paper_latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Reproducibility

- **Seed:** 42 (primary single-split); **multi-seed** 42–46
- **Device:** MPS (Apple Silicon); falls back to CUDA then CPU
- **Key versions:** PyTorch 2.8.0, torchvision 0.23.0, numpy 2.0.2, Python 3.9.6
- **Splits:** `splits/ddi_split_seed{42..46}.json`
- **Leakage:** `splits/leakage_report.json` — byte-level SHA256, zero overlap
- **Provenance:** `splits/synthetic_provenance_report.json` — per-image, all 290 match a training source
- **Config as code:** `configs/config.yaml`

> **MPS non-determinism:** the MPS backend can produce small floating-point differences across runs. Exact reproduction on CUDA/CPU is not guaranteed.

---

## Limitations

- **Statistical power.** Dark test n=42 with ~10 melanomas; bootstrap SD ~0.15 and gap SD ~0.19. Single-split comparisons are fragile by construction.
- **No external validation.** DDI-only (n=656); no cross-dataset or multi-centre evaluation.
- **Confounded headline contrast.** The Stage 2→3 comparison does not isolate augmentation — it changes class composition and weighting (quantified in the arm study).
- **Augmentation realism & diversity.** 290 near-duplicate transforms of 29 lesions; no FID/LPIPS, no clinician review, no generative baseline.
- **Threshold dependence.** Sensitivity/specificity/PPV depend on per-stage validation Youden-J thresholds; AUROC is primary.

---

## Changelog

### v2 — October 12, 2026
- Reframed as a **negative/limits study**; multi-seed (42–46) aggregation is now the primary result.
- Added the **class-composition confound** analysis and the **static / recomputed / symmetric** `pos_weight` arm study.
- Re-analyzed the **ablation** (matched frozen layers and lesion coverage): generated-image quantity has no measurable Dark AUROC effect; Light AUROC declines.
- Added **per-image synthetic provenance** verification (perceptual + SHA256).
- Addressed 6 reviewer points in the manuscript text (power, external validation, composition confound, diversity/realism, multi-seed variability, thresholding) and corrected the ablation figure.

---

## Citation

```bibtex
@article{fairderm2026,
  title={FairDerm: Evaluating the Limits of Photometric Augmentation for Skin-Tone Bias in Melanoma Detection},
  author={Gottimukkala, Shanmuka},
  year={2026},
  note={Working Paper #1, FairMed AI — leakage-free audit, SHA256 verified; negative-result study}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

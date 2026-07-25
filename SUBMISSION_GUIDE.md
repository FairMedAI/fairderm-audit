# FairDerm — Submission Guide

> How to compile, format, and submit the manuscript.

---

## Compiling the LaTeX Paper

LaTeX is **not installed** on this system. To compile:

```bash
# On a system with LaTeX installed:
cd paper_latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use [Overleaf](https://www.overleaf.com):
1. Upload the entire `paper_latex/` directory
2. Set `main.tex` as the main document
3. Compile — all `\input{}`, `\cite{}`, and `\ref{}` should resolve

---

## Regenerating Tables and Figures

Tables and figures are generated from frozen experiment data. To regenerate:

```bash
# From the FairDerm root directory:
python paper_latex/generate_tables.py    # → paper_latex/tables/*.tex
python paper_latex/generate_figures.py   # → paper_latex/figures/*.pdf + copied PNGs
```

These scripts read from `results/` and write to `paper_latex/`. No model inference is performed.

---

## Source Data Files

| File | Content | Used By |
|------|---------|---------|
| `results/evaluate/paper_data.json` | Subgroup AUROC/Sens/Spec/PPV for Light/Dark | Tables 1–3, Figures 2–4 |
| `results/ablation/ablation_report.csv` | Ablation overall metrics | Table 4, Figure 5 |
| `results/*/config.json` | Hyperparams, val AUROCs | Tables 5–6, Figure 1 |
| `results/*/training_history.csv` | Epoch-by-epoch training logs | Figure 6 |
| Terminal output (not saved) | Overall test metrics, bootstrap CIs | Tables 1–2, Figure 4 |

---

## Hardcoded Values (Terminal Output)

These values appear in `generate_tables.py` and `generate_figures.py` as hardcoded dicts. They are sourced from terminal output and are **not** in any saved file:

| Value | Source | Confidence |
|-------|--------|------------|
| Overall test AUROC/Sens/Spec/F1 | `fairderm.py --stage evaluate` stdout | High (matches FINAL_RESULTS.md) |
| Bootstrap 95% CIs | `fairderm.py --stage evaluate` stdout | High |
| Overall bootstrap CIs | `fairderm.py --stage evaluate` stdout | High |

To update these if you re-run the pipeline, search the terminal output for `stage_evaluate` results and update the `OVERALL_TEST`, `BOOTSTRAP_CIS`, and `OVERALL_BOOTSTRAP` dicts in `generate_tables.py` and `generate_figures.py`.

---

## Venue-Specific Formatting

### ISEF (International Science & Engineering Fair)
- Use the IMRad structure as-is
- Add student name, school, and mentor on the title page
- Maximum 25 pages including references and appendix
- No IRB needed — we only used published datasets.

### Student Journals (JSHS, JSE, etc.)
- May require abstract ≤250 words (check current limit)
- May require specific author formatting
- May require a "Future Work" section (add to Discussion)
- Check page limits — current paper is ~12 pages body + references + appendix

### ML Workshops (FAccT, CHIL, etc.)
- May require extended abstract format (4–8 pages)
- Condense Methods and Experiments sections
- Move detailed limitations to supplementary
- Add algorithmic fairness terminology (demographic parity, equalized odds)

### Peer-Reviewed Journals (Nature Medicine, Scientific Reports, etc.)
- May require structured abstract with specific section headers
- May require clinical significance statement
- May require data availability statement and code availability statement
- May require author contributions in CRediT format
- Check word count limits — current paper is ~5,000 words body

---

## Citation Verification Log

All 12 references were verified via web search on 2026-07-22:

| Key | Title | Journal | Verified Via |
|-----|-------|---------|-------------|
| Tschandl2018 | HAM10000 dataset | Scientific Data | DOI 10.1038/sdata.2018.161 |
| Groh2021 | FITD dataset evaluation | Scientific Reports | DOI 10.1038/s41598-021-01192-8 |
| Tan2019 | Dermoscopy taxonomy | Clin Exp Dermatol | DOI 10.1111/ced.13755 |
| Buslaev2020 | Albumentations | Information (MDPI) | DOI 10.3390/info11020125 |
| Mehrabi2021 | Bias and fairness survey | ACM Comput Surv | DOI 10.1145/3457607 |
| Adamson2018 | ML and health disparities | JAMA Dermatol | DOI 10.1001/jamadermatol.2018.2348 |
| Hanley1982 | ROC curve interpretation | Radiology | DOI 10.1148/radiology.143.1.7063747 |
| Efron1986 | Bootstrap methods | Ann Statist | DOI 10.1214/aos/1176344552 (year=1979) |
| Esteva2017 | Dermatologist-level CNN | Nature | DOI 10.1038/nature21056 |
| Daneshjou2022 | Dermatology AI disparities | Nat Med | DOI 10.1038/s41591-022-01905-x |
| Raghu2019 | Transfer learning for medical imaging | NeurIPS 2019 | proceedings |
| Fitzpatrick1988 | Skin types I–VI | Arch Dermatol | DOI 10.1001/archderm.1988.01620060093029 |

---

## Key Wording Corrections

The following wording changes from `FINAL_RESULTS.md` have been applied:

| Original | Corrected | Reason |
|----------|-----------|--------|
| "reverses the sensitivity gap" | "narrows the sensitivity gap from −0.40 to +0.10" | Light sensitivity decreased |
| "dramatically" | Removed or replaced with "substantially" | Subjective editorial |
| "completely fails on dark skin" | "achieves no discriminative ability on dark skin (AUROC=0.50)" | More precise |
| "causes overfitting" (re: 10x) | "degrades generalization" | Interpretation, not proven mechanism |
| "sweet spot" (re: 5x) | "optimal multiplier of 5×" | Informal language |

---

## File Manifest

```
paper_latex/
├── main.tex                          # Root document
├── references.bib                    # 12 verified citations
├── generate_tables.py                # Frozen data → LaTeX tables
├── generate_figures.py               # Frozen data → PDF figures
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── methods.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── limitations.tex
│   └── appendix.tex
├── tables/
│   ├── tab1_main_results.tex
│   ├── tab2_subgroup_auroc.tex
│   ├── tab3_sensitivity.tex
│   ├── tab4_ablation.tex
│   ├── tab5_training_config.tex
│   ├── tab6_dataset.tex
│   └── tab7_provenance.tex
└── figures/
    ├── fig1_val_auroc.pdf
    ├── fig2_subgroup_auroc.pdf
    ├── fig3_sensitivity_gap.pdf
    ├── fig4_bootstrap_ci.pdf
    ├── fig5_ablation_tradeoff.pdf
    ├── fig6_training_curves.pdf
    ├── supp_roc_progression.png
    ├── supp_ablation_tradeoff_original.png
    ├── supp_roc_progression_paper.png
    └── supp_ablation_tradeoff_corrected.png

PRE_SUBMISSION_CHECKLIST.md           # 42-item checklist
SUBMISSION_GUIDE.md                  # This file
```

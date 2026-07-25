# FairDerm — Pre-Submission Checklist

> Complete this checklist before submitting to any venue.

---

## Manuscript Files

| # | Item | Status |
|---|------|--------|
| 1 | `main.tex` compiles without errors (run `pdflatex` + `bibtex` + `pdflatex` ×2) | ☐ |
| 2 | All 8 section files present in `paper_latex/sections/` | ☐ |
| 3 | `references.bib` contains all 12 cited works | ☐ |
| 4 | No `TODO` placeholders left in `main.tex` or section files | ☐ |
| 5 | Author names and affiliations filled in | ☐ |

## Figures

| # | Item | Status |
|---|------|--------|
| 6 | All 6 PDF figures present in `paper_latex/figures/` | ☐ |
| 7 | Figures render correctly in compiled PDF | ☐ |
| 8 | Figure captions are descriptive (standalone understandable) | ☐ |
| 9 | Supplementary PNGs included (ROC progression, original ablation) | ☐ |

## Tables

| # | Item | Status |
|---|------|--------|
| 10 | All 7 table `.tex` files present in `paper_latex/tables/` | ☐ |
| 11 | Tables compile and render correctly | ☐ |
| 12 | Table numbers match `\ref{}` references in text | ☐ |

## Data Integrity

| # | Item | Status |
|---|------|--------|
| 13 | All numbers in tables match source JSON/CSV files | ☐ |
| 14 | Bootstrap CIs labeled as "hardcoded from terminal output" with `% SOURCE:` comments | ☐ |
| 15 | Overall test metrics (AUROC, Sens, Spec, F1) match `generate_paper_assets.py` values | ☐ |
| 16 | Threshold 0.1206 documented as origin-unknown in limitations section | ☐ |

## Language and Tone

| # | Item | Status |
|---|------|--------|
| 17 | Tone check: no hype words ("proves", "dramatic") | ☐ |
| 18 | Hedged language used ("suggests", "indicates", "directional") | ☐ |
| 19 | Sample-size caveats appear in results and limitations | ☐ |
| 20 | "Narrows the sensitivity gap" used (NOT "reverses") | ☐ |

## Limitations Coverage

| # | Item | Status |
|---|------|--------|
| 21 | Limitations section covers all the issues from the audit | ☐ |
| 22 | High-severity items (synthetic source, small n) prominently discussed | ☐ |
| 23 | Medium-severity items (hyperparameter confounds, threshold strategies, ablation distinction) covered | ☐ |
| 24 | Low-severity items (threshold undocumented, medium excluded, MPS non-determinism) noted | ☐ |

## Citations

| # | Item | Status |
|---|------|--------|
| 25 | All 12 references verified via web search | ☐ |
| 26 | All references actually exist | ☐ |
| 27 | DOIs present for all journal articles | ☐ |
| 28 | Groh2021 correctly cited as Scientific Reports (not JAMA Dermatology) | ☐ |

## Submission Materials

| # | Item | Status |
|---|------|--------|
| 29 | `PRE_SUBMISSION_CHECKLIST.md` completed | ☐ |
| 30 | `SUBMISSION_GUIDE.md` reviewed for venue-specific formatting | ☐ |
| 31 | Cover letter prepared (if required) | ☐ |
| 32 | Author contributions statement prepared (if required) | ☐ |
| 33 | Data availability statement prepared | ☐ |

## Reproducibility

| # | Item | Status |
|---|------|--------|
| 34 | Seed=42 documented | ☐ |
| 35 | Software versions documented (PyTorch 2.8.0, Python 3.9.6) | ☐ |
| 36 | All config.json files saved in `results/` | ☐ |
| 37 | All checkpoints saved in `models/` | ☐ |
| 38 | `generate_tables.py` and `generate_figures.py` re-runnable from frozen data | ☐ |

## Final Review

| # | Item | Status |
|---|------|--------|
| 39 | Read full paper aloud for flow and clarity | ☐ |
| 40 | Have a non-expert read the abstract for accessibility | ☐ |
| 41 | Spell-check completed | ☐ |
| 42 | All `\ref{}` and `\cite{}` resolve without warnings | ☐ |

---

**Completion date:** _______________
**Reviewed by:** _______________

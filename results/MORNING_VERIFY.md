# FairDerm — Morning Verify (Light Run)

**Timestamp:** 2026-08-09 12:18:16 EDT
**Mode:** LIGHT RUN — baseline skipped (used existing checkpoint), low priority (`nice -n 19`), batch 16

## Run Summary

| Step | Command | Status |
|------|---------|--------|
| Baseline | SKIPPED — reused `models/fairderm_ham_baseline.pth` (exists) | OK |
| Finetune | `--stage finetune --batch-size 16 --epochs 10` | OK (best val AUROC 0.7538, early stop ep 8) |
| Augment | `--stage augment --batch-size 16 --epochs 10` | OK (best val AUROC 0.7653, early stop ep 4) |
| Evaluate | `--stage evaluate` | OK |
| Clean artifacts | `scripts/generate_clean_artifacts.py` | OK |
| Leakage check | `scripts/check_leakage.py` | PASSED (overlap 0) |

Environment: Python 3.9.6 · PyTorch 2.8.0 · torchvision 0.23.0 · numpy 2.0.2 · MPS (Apple Silicon) · seed 42

## Results — DDI Test Set (n=132; 42 Light / 42 Dark, 10 melanomas each)

| Stage | Light AUROC | Dark AUROC | Gap (Dark − Light) |
|-------|-------------|------------|--------------------|
| Baseline | 0.7188 | 0.5875 | −0.1313 |
| Fine-tuned | 0.7844 | 0.5000 | −0.2844 |
| +Synthetic | 0.7156 | 0.5250 | −0.1906 |

Overall test AUROC: baseline 0.6622 → finetuned 0.6704 → final 0.6842

Dark AUROC bootstrap 95% CIs: finetuned 0.499 [0.279–0.733] → final 0.528 [0.287–0.754]
Dark AUROC improvement p-value (finetuned → augment): **0.429** (not significant)

Thresholds: baseline 0.5 (fixed) · finetuned 0.0322 (Youden-J) · final 0.0505 (Youden-J)

## Leakage

- `splits/leakage_report.json`: `passed=true`, `synthetic_vs_test=0`, `synthetic_vs_val=0`, `synthetic_vs_any_ddi=0` (SHA256 byte-level dedup)
- `check_leakage.py`: `LEAKAGE CHECK PASSED` — 290 synthetic vs 132 test, overlap 0; all splits disjoint
- `results/evaluate/paper_data.json` present

## Caveats (light run)

- **Baseline checkpoint is partial** — from the interrupted 15-epoch run (best val epoch 5, AUROC 0.9650), not a completed 15-epoch model.
- Synthetic regeneration skipped (290 train-only images already present) — no new leakage risk.
- Ablation study and paper table/figure regeneration NOT run in this light pass.
- Bootstrap Dark CIs are wide (n=42, 10 melanomas) — subgroup claims are directional only.

**Reproducible: YES** — seed 42, saved splits, per-stage `config.json`, checkpoints, and fixed hyperparams (finetune/augment lr=5e-5, batch=16, pos_weight=dynamic). Exact floats may vary slightly across runs due to MPS backend non-determinism.

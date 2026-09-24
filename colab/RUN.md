# FairDerm 5-seed experiment runbook (Colab)

This reproduces the research plan on Google Colab after the full repo was backed
up to `github.com/FairMedAI/fairderm-audit`.

## What Colab needs (upload once to Google Drive)

Under `MyDrive/fairderm_assets/`:

```
fairderm_assets/
  data/ddi/                 # ddi_metadata.csv + DDI images (from the local repo)
  models/fairderm_ham_baseline.pth   # the Stage-1 HAM10000 baseline checkpoint
```

The DDI metadata + images are small (~656 images). Only the baseline is needed:
the fine-tuned and augmented checkpoints are regenerated per seed by the script.

## Steps

1. Open Colab, mount Drive, create `MyDrive/fairderm_assets` and upload the two
   folders above.
2. In the first cell run: `!bash colab/colab_setup.sh`
3. Run the preflight gate:
   `!python scripts/run_multiseed.py --dry-run`
4. Run the experiments (each writes its own summary file):

   As-written (Original, seeds 42-46)  — 290 syn/seed (29 dark mel x10):
   ```
   !python scripts/run_multiseed.py --seeds 42 43 44 45 46 --design original
   ```

   Design A (symmetric-corrected, seeds 42-46) — 1240 syn/seed
   (29 dark mel x10 + 95 dark benign x10):
   ```
   !python scripts/run_multiseed.py --seeds 42 43 44 45 46 --design designA
   ```

5. Download `results/multiseed_summary.csv`, `multiseed_summary_designA.csv`
   (and their `.json`), plus `results/multiseed/seedN_metrics.json`.

## Outputs / reporting

Per seed: `results/multiseed/seed{seed}.log`, `seed{seed}_metrics.json`.

Summaries give mean +/- SD and seed-level 95% CIs for: overall AUROC
(baseline/finetuned/synthetic), Light and Dark AUROC, gap (Dark-Light),
paired Dark AUROC delta (synthetic - finetuned) with one-tailed p.

Also capture `results/multiseed/seed42_*` comparison: seed 42 MUST match the
paper's numbers within the MPS +/-0.01 run-to-run tolerance (or exactly, since
this is the anchor run). Report which seeds completed, all mean/SD/CI, any seed
failures, whether Design A changes the Dark-gap interpretation, and which
manuscript tables/numbers need updating. Do NOT edit manuscript seed-42 values
until runs verified.

## Notes

- Batch 32, lr 5e-5, max 10 epochs with patience 3 + ReduceLROnPlateau (early stop).
- MPS/CUDA non-determinism: per-seed AUROCs vary ~+/-0.01 run-to-run; the 5-seed
  mean +/- SD absorbs this.
- Results dir is git-ignored; push only the CSV/JSON summaries you want tracked.
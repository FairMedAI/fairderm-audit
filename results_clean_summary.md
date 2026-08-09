# Clean Audit Result — 1-Paragraph Summary

Regenerating the audit with a leakage-proof, train-only synthetic pipeline (290 images generated
solely from the 29 training-split dark-skin melanomas, SHA256-verified to share zero bytes with
the 132 test images in `splits/leakage_report.json`) corrects the earlier optimistic estimate of
the bias gap. The old run, which generated synthetics from the full DDI set before splitting and
used a hardcoded threshold, reported a Dark-minus-Light AUROC gap of only **-0.066** after
synthetic augmentation (Light 0.750 vs Dark 0.684); the clean, Youden-J-consistent run shows that
fine-tuning alone leaves a gap of **-0.328** (Light 0.800 vs Dark 0.472), which synthetic
augmentation narrows only marginally to **-0.303** (Light 0.766 vs Dark 0.463), with a bootstrap
p-value of **0.524** indicating the improvement is not statistically significant. In short, simple
photometric synthetic augmentation does not meaningfully close the skin-tone gap, and the wide
bootstrap CIs (Dark test n=42, only 10 melanomas) underscore how underpowered the test set is for
subgroup-level claims.

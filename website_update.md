# FairDerm Update

FairDerm investigates skin-tone bias in melanoma detection and whether synthetic augmentation can close the gap. This update reflects our clean, leakage-free run.

**Leakage-proof pipeline.** All 290 synthetic dark-skin melanoma images were generated exclusively from the 29 training-split dark melanomas. Byte-level SHA256 verification confirms zero overlap with the 132 test images.

**Results (DDI test, n=132; 42 Light / 48 Medium / 42 Dark, 10 melanomas per reported subgroup).** The baseline achieves Light AUROC 0.7188 vs. Dark AUROC 0.5875 (gap −0.1313). Fine-tuning changes these to 0.7563 vs. 0.5469 (gap −0.2094). Adding synthetic dark melanomas changes them to 0.7719 vs. 0.4813 (gap −0.2906); the paired Dark AUROC delta is −0.0663 (95% CI [−0.1508, 0.0080], p = 0.9620).

**Takeaway.** Simple photometric augmentation cannot meaningfully close the skin-tone gap. The next step is richer generative models, such as GANs or diffusion models.

*Caveat:* the baseline checkpoint is partial (best validation epoch 5); a full 15-epoch run is pending. Bootstrap CIs are wide — results are directional, not definitive.

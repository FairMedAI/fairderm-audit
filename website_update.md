# FairDerm Update

FairDerm investigates skin-tone bias in melanoma detection and whether synthetic augmentation can close the gap. This update reflects our clean, leakage-free run.

**Leakage-proof pipeline.** All 290 synthetic dark-skin melanoma images were generated exclusively from the 29 training-split dark melanomas. Byte-level SHA256 verification confirms zero overlap with the 132 test images.

**Results (DDI test, n=132; 42 Light / 42 Dark, 10 melanomas each).** The HAM10000 baseline achieves Light AUROC 0.719 vs. Dark AUROC 0.588 (gap −0.131). Fine-tuning on DDI amplifies the bias: Light AUROC 0.784 vs. Dark AUROC 0.500 (gap −0.284). Adding synthetic dark melanomas narrows the gap to −0.191 (Light 0.716, Dark 0.525), but the change is not statistically significant (p = 0.429).

**Takeaway.** Simple photometric augmentation cannot meaningfully close the skin-tone gap. The next step is richer generative models, such as GANs or diffusion models.

*Caveat:* the baseline checkpoint is partial (best validation epoch 5); a full 15-epoch run is pending. Bootstrap CIs are wide — results are directional, not definitive.

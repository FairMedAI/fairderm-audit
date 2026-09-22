# Clean Audit Result — 1-Paragraph Summary

Regenerating the audit with a leakage-proof, train-only synthetic pipeline (290 images generated
solely from the 29 training-split dark-skin melanomas, SHA256-verified to share zero bytes with
the 132 test images in `splits/leakage_report.json`) corrects the earlier optimistic estimate of
the bias gap. The controlled, Youden-J-consistent run shows that baseline, fine-tuned, and
synthetic test AUROC are **0.6622**, **0.6336**, and **0.6395**, respectively. The corresponding
Light--Dark gaps are **-0.1313**, **-0.2094**, and **-0.2906**; synthetic augmentation widens
the gap rather than closing it. The paired Dark AUROC delta was **-0.0663**, with 95% bootstrap
CI **[-0.1508, 0.0080]** and **p=0.9620**. The wide intervals and only 10 melanomas per
subgroup underscore that these single-seed results are preliminary.

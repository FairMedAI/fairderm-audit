#!/usr/bin/env python3
"""Compare two independent full-pipeline runs of the same seed.

Both runs use seed 42, so any difference is attributable to run-to-run
non-determinism (on this hardware, MPS) rather than to the seed. The paper
previously asserted an approximate tolerance; this measures it instead.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = {
    "run_a": os.path.join(ROOT, "results", "multiseed", "metrics_seed42.json"),
    "run_b": os.path.join(ROOT, "results", "repeat_seed42", "multiseed", "metrics_seed42.json"),
}

FIELDS = [
    ("baseline", "overall_auroc"),
    ("baseline", "dark_auroc"),
    ("finetuned", "overall_auroc"),
    ("finetuned", "dark_auroc"),
    ("augmented", "overall_auroc"),
    ("augmented", "light_auroc"),
    ("augmented", "medium_auroc"),
    ("augmented", "dark_auroc"),
    ("augmented", "gap_dark_minus_light"),
]


def main():
    loaded = {}
    for name, path in RUNS.items():
        if not os.path.exists(path):
            print(f"MISSING {name}: {path}")
            return 2
        with open(path) as f:
            loaded[name] = json.load(f)

    a, b = loaded["run_a"], loaded["run_b"]
    if a["seed"] != b["seed"]:
        print(f"ERROR: seeds differ ({a['seed']} vs {b['seed']}); comparison is meaningless.")
        return 1

    out = {
        "seed": a["seed"],
        "compared": {k: os.path.relpath(v, ROOT) for k, v in RUNS.items()},
        "note": (
            "Both runs use the same seed and the same code. Differences are run-to-run "
            "non-determinism on the MPS backend, not seed variation."
        ),
        "deltas": {},
    }
    print(f"Same-seed repeatability (seed {a['seed']}, MPS)")
    print(f"{'metric':38s} {'run A':>9s} {'run B':>9s} {'|diff|':>9s}")
    print("-" * 70)
    worst = 0.0
    worst_key = None
    for stage, field in FIELDS:
        va, vb = a["stages"][stage].get(field), b["stages"][stage].get(field)
        if va is None or vb is None:
            print(f"{stage + '.' + field:38s} {'-':>9s} {'-':>9s} {'-':>9s}")
            continue
        d = abs(va - vb)
        out["deltas"][f"{stage}.{field}"] = {
            "run_a": round(va, 4), "run_b": round(vb, 4), "abs_diff": round(d, 4)
        }
        if d > worst:
            worst, worst_key = d, f"{stage}.{field}"
        print(f"{stage + '.' + field:38s} {va:9.4f} {vb:9.4f} {d:9.4f}")

    da = a.get("paired_dark_auroc_delta_augment_minus_finetuned", {})
    db = b.get("paired_dark_auroc_delta_augment_minus_finetuned", {})
    if da and db:
        d = abs(da["mean"] - db["mean"])
        out["deltas"]["paired_dark_delta"] = {
            "run_a": round(da["mean"], 4), "run_b": round(db["mean"], 4),
            "abs_diff": round(d, 4),
        }
        if d > worst:
            worst, worst_key = d, "paired_dark_delta"
        print(f"{'paired_dark_delta':38s} {da['mean']:9.4f} {db['mean']:9.4f} {d:9.4f}")

    out["max_abs_diff"] = round(worst, 4)
    out["max_abs_diff_metric"] = worst_key
    print("-" * 70)
    print(f"max |difference| = {worst:.4f} ({worst_key})")

    dest = os.path.join(ROOT, "results", "repeat_seed42", "mps_repeatability.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {os.path.relpath(dest, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

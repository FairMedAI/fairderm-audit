#!/usr/bin/env python3
"""Verify that every headline number in the manuscript matches results/metrics_clean.json.

The paper states that its tables and figures are generated from the metrics JSON
rather than transcribed. This script checks the reverse direction: that the
values written into the prose match the JSON. Run it after regenerating metrics.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "results", "metrics_clean.json")
SECTIONS = os.path.join(ROOT, "paper_latex", "sections")
TOL = 5e-4

# Values as written in the manuscript (abstract, results, discussion, methods).
EXPECTED = {
    "overall": {"baseline": 0.6622, "finetuned": 0.6336, "final": 0.6395},
    "Light": {"baseline": 0.7188, "finetuned": 0.7563, "final": 0.7719},
    "Medium": {"baseline": 0.6808, "finetuned": 0.6485, "final": 0.6828},
    "Dark": {"baseline": 0.5875, "finetuned": 0.5469, "final": 0.4813},
    "paired_delta": -0.0663,
    "ci_lo": -0.1508,
    "ci_hi": 0.0080,
    "p_value": 0.962,
}
STAGE_KEY = {"baseline": "baseline", "finetuned": "finetune", "final": "final_synthetic"}

# Arm-study values, verified against results/pos_weight_arms/pos_weight_arms_summary.json.
EXPECTED_ARMS = {
    "static": {"pos_weight": 2.8529, "dark_auroc": 0.4844, "gap": -0.2219, "overall_auroc": 0.6404},
    "recomputed": {"pos_weight": 0.7423, "dark_auroc": 0.4781, "gap": -0.2219, "overall_auroc": 0.6371},
    "symmetric": {"pos_weight": 1.0, "dark_auroc": 0.5500, "gap": -0.1406, "overall_auroc": 0.6680},
}


# Ablation values, verified against results/ablation/ablation_report.csv.
EXPECTED_ABLATION = {
    "0": {"Dark_auroc": 0.5094, "AUROC": 0.6669, "Light_auroc": 0.7844, "n_distinct_sources": 0},
    "2": {"Dark_auroc": 0.5125, "AUROC": 0.6677, "Light_auroc": 0.7406, "n_distinct_sources": 29},
    "5": {"Dark_auroc": 0.5031, "AUROC": 0.6580, "Light_auroc": 0.7469, "n_distinct_sources": 29},
    "10": {"Dark_auroc": 0.4969, "AUROC": 0.6610, "Light_auroc": 0.6969, "n_distinct_sources": 29},
}
ABLATION_TRAINABLE = 2744861

# Multi-seed values, verified against results/multiseed_summary.json aggregates.
# Manuscript quotes these at three decimals, so tolerance is half-a-ulp at that
# precision (6e-4) rather than the four-decimal TOL used for metrics_clean.json.
TOL_MS = 6e-4
EXPECTED_MULTISEED = {
    "overall_finetuned": 0.644,
    "overall_synthetic": 0.658,
    "light_finetuned": 0.656,
    "light_synthetic": 0.688,
    "medium_finetuned": 0.698,
    "medium_synthetic": 0.677,
    "dark_finetuned": 0.575,
    "dark_synthetic": 0.604,
    "gap_finetuned": -0.081,
    "gap_synthetic": -0.083,
    "dark_delta": 0.028,
    "overall_sens_finetuned": 0.623,
    "overall_sens_synthetic": 0.709,
    "overall_spec_finetuned": 0.602,
    "overall_spec_synthetic": 0.540,
}
EXPECTED_MULTISEED_SD = {
    "dark_finetuned": 0.145,
    "dark_synthetic": 0.152,
    "gap_finetuned": 0.193,
    "gap_synthetic": 0.161,
    "dark_delta": 0.038,
}
# Per-seed dark_delta span quoted in Results/Discussion; seed-42 row value.
EXPECTED_SEED42_DELTA = 0.0131


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    failures = []

    def check(ok, label, got, want):
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures.append(label)
        print(f"{status} {label:44s} json={got} paper={want}")

    m = load(METRICS)
    stages, overall = m["stages"], m["overall_test"]

    for stage, skey in STAGE_KEY.items():
        got = overall[stage]["auroc"]
        check(abs(got - EXPECTED["overall"][stage]) < TOL, f"{stage} overall AUROC",
              got, EXPECTED["overall"][stage])
        for group in ("Light", "Medium", "Dark"):
            got = stages[skey][group]["auroc"]
            check(abs(got - EXPECTED[group][stage]) < TOL, f"{stage} {group} AUROC",
                  got, EXPECTED[group][stage])

    d = m["paired_dark_auroc_delta"]
    check(abs(d["mean"] - EXPECTED["paired_delta"]) < TOL, "paired Dark AUROC delta",
          round(d["mean"], 4), EXPECTED["paired_delta"])
    check(abs(d["lo"] - EXPECTED["ci_lo"]) < 6e-4, "paired delta CI lower",
          round(d["lo"], 4), EXPECTED["ci_lo"])
    check(abs(d["hi"] - EXPECTED["ci_hi"]) < 6e-4, "paired delta CI upper",
          round(d["hi"], 4), EXPECTED["ci_hi"])
    check(abs(m["p_value_dark_improvement"] - EXPECTED["p_value"]) < 6e-4,
          "p-value (Dark improvement)", m["p_value_dark_improvement"], EXPECTED["p_value"])

    arms_path = os.path.join(ROOT, "results", "pos_weight_arms", "pos_weight_arms_summary.json")
    if os.path.exists(arms_path):
        arms = load(arms_path)["arms"]
        for name, exp in EXPECTED_ARMS.items():
            if name not in arms:
                print(f"SKIP arm {name}: not present in summary")
                continue
            a = arms[name]
            check(abs(a["pos_weight"] - exp["pos_weight"]) < 1e-3, f"arm {name} pos_weight",
                  a["pos_weight"], exp["pos_weight"])
            check(abs(a["dark_auroc"] - exp["dark_auroc"]) < TOL, f"arm {name} Dark AUROC",
                  a["dark_auroc"], exp["dark_auroc"])
            check(abs(a["gap_dark_minus_light"] - exp["gap"]) < TOL, f"arm {name} gap",
                  a["gap_dark_minus_light"], exp["gap"])
            check(abs(a["overall_auroc"] - exp["overall_auroc"]) < TOL,
                  f"arm {name} overall AUROC", a["overall_auroc"], exp["overall_auroc"])
    else:
        print("SKIP arm study: pos_weight_arms_summary.json not found")

    abl_path = os.path.join(ROOT, "results", "ablation", "ablation_report.csv")
    if os.path.exists(abl_path):
        import csv
        with open(abl_path) as f:
            rows = {str(int(float(r["multiplier"]))): r for r in csv.DictReader(f)}
        for mult, exp in EXPECTED_ABLATION.items():
            if mult not in rows:
                print(f"SKIP ablation {mult}x: not in report")
                continue
            r = rows[mult]
            for field, want in exp.items():
                got = float(r[field])
                check(abs(got - want) < TOL, f"ablation {mult}x {field}", got, want)
            check(abs(float(r["n_trainable_params"]) - ABLATION_TRAINABLE) < 1,
                  f"ablation {mult}x frozen trainable params",
                  int(float(r["n_trainable_params"])), ABLATION_TRAINABLE)
    else:
        print("SKIP ablation: ablation_report.csv not found")

    ms_path = os.path.join(ROOT, "results", "multiseed_summary.json")
    if os.path.exists(ms_path):
        ms = load(ms_path)
        agg = ms.get("aggregates", {})
        for key, want in EXPECTED_MULTISEED.items():
            got = agg.get(key, {}).get("mean")
            if got is None:
                print(f"SKIP multiseed {key}: not in aggregates")
                continue
            check(abs(got - want) < TOL_MS, f"multiseed mean {key}", round(got, 4), want)
        for key, want in EXPECTED_MULTISEED_SD.items():
            got = agg.get(key, {}).get("sd")
            if got is None:
                print(f"SKIP multiseed SD {key}: not in aggregates")
                continue
            check(abs(got - want) < TOL_MS, f"multiseed SD {key}", round(got, 4), want)
        rows = {r["seed"]: r for r in ms.get("rows", [])}
        if 42 in rows:
            check(abs(rows[42]["dark_delta"] - EXPECTED_SEED42_DELTA) < TOL,
                  "seed-42 multiseed dark_delta", rows[42]["dark_delta"], EXPECTED_SEED42_DELTA)
        deltas = sorted(r["dark_delta"] for r in ms.get("rows", []))
        if deltas:
            check(deltas[0] > -0.010 - TOL and deltas[-1] < 0.073 + TOL,
                  "per-seed dark_delta span within [-0.010, 0.073]",
                  f"[{deltas[0]}, {deltas[-1]}]", "[-0.010, 0.073]")
    else:
        print("SKIP multiseed: multiseed_summary.json not found")

    # No unfilled placeholders may remain in the manuscript.
    for name in sorted(os.listdir(SECTIONS)):
        if name.endswith(".tex"):
            with open(os.path.join(SECTIONS, name)) as f:
                text = f.read()
            for bad in ("PLACEHOLDER_", "PLACEHOLDER\\_"):
                if bad in text:
                    failures.append(f"unfilled placeholder in {name}")
                    print(f"FAIL unfilled placeholder in {name}")

    # Prose numbers must not contain stale four-decimal values for the headline
    # metrics, which would indicate a prose/JSON drift the checks above miss.
    prose = ""
    for name in os.listdir(SECTIONS):
        if name.endswith(".tex"):
            with open(os.path.join(SECTIONS, name)) as f:
                prose += f.read()
    for stale in ("0.6620", "0.6340", "0.6400", "0.5900"):
        if stale in prose:
            failures.append(f"stale rounded value {stale} in prose")
            print(f"FAIL stale value {stale} appears in the manuscript prose")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("All checked manuscript numbers match the metrics JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

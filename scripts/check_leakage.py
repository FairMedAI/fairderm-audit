#!/usr/bin/env python3

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_leakage():
    split_path = os.path.join(PROJECT_ROOT, "splits", "ddi_split_seed42.json")
    syn_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_train_only")

    errors = []

    if not os.path.exists(split_path):
        errors.append(f"Split file not found: {split_path}")
        print("LEAKAGE CHECK FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    with open(split_path) as f:
        split = json.load(f)

    test_files = set(split["test"])
    train_files = set(split["train"])
    val_files = set(split["val"])

    if not train_files.isdisjoint(val_files):
        errors.append("Train and val sets overlap!")
    if not train_files.isdisjoint(test_files):
        errors.append("Train and test sets overlap!")
    if not val_files.isdisjoint(test_files):
        errors.append("Val and test sets overlap!")

    if os.path.exists(syn_dir):
        syn_files = [f for f in os.listdir(syn_dir)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))
                     and not f.startswith("_")]
        syn_stems = set(os.path.splitext(f)[0] for f in syn_files)
        test_stems = set(os.path.splitext(f)[0] for f in test_files)
        train_stems = set(os.path.splitext(f)[0] for f in train_files)

        overlap_with_test = syn_stems & test_stems
        overlap_with_train = syn_stems & train_stems

        if overlap_with_test:
            errors.append(
                f"SYNTHETIC LEAKAGE: {len(overlap_with_test)} synthetic stems match test stems: "
                f"{sorted(overlap_with_test)[:5]}..."
            )
        # Synthetics matching train is expected (they're generated from train)
        if overlap_with_train:
            print(f"  INFO: {len(overlap_with_train)} synthetic stems match train stems (expected)")

        print(f"  Synthetic images: {len(syn_files)}")
        print(f"  Test images: {len(test_files)}")
        print(f"  Overlap with test: {len(overlap_with_test)}")
    else:
        print(f"  WARNING: Synthetic directory not found: {syn_dir}")
        print("  (OK if synthetics not yet generated)")

    counts = split.get("counts", {})
    expected = {"train": 393, "val": 131, "test": 132}
    for key, expected_val in expected.items():
        actual = counts.get(key)
        if actual != expected_val:
            errors.append(f"Count mismatch for {key}: expected {expected_val}, got {actual}")

    train_dark_mel = split.get("train_dark_mel", [])
    if len(train_dark_mel) != 29:
        errors.append(f"Train dark mel count: expected 29, got {len(train_dark_mel)}")

    test_files_list = split["test"]
    import pandas as pd
    meta_path = os.path.join(PROJECT_ROOT, "ddidiversedermatologyimages", "ddi_metadata.csv")
    if os.path.exists(meta_path):
        ddi_meta = pd.read_csv(meta_path)
        test_meta = ddi_meta[ddi_meta["DDI_file"].isin(test_files_list)]
        light_test = test_meta[test_meta["skin_tone"] == 12]
        dark_test = test_meta[test_meta["skin_tone"] == 56]
        light_mel = light_test[light_test["malignant"] == True]
        dark_mel = dark_test[dark_test["malignant"] == True]
        print(f"  Test Light: {len(light_test)} ({len(light_mel)} mel)")
        print(f"  Test Dark: {len(dark_test)} ({len(dark_mel)} mel)")
        if len(light_test) != 42:
            errors.append(f"Test Light count: expected 42, got {len(light_test)}")
        if len(dark_test) != 42:
            errors.append(f"Test Dark count: expected 42, got {len(dark_test)}")
        if len(light_mel) != 10:
            errors.append(f"Test Light melanoma count: expected 10, got {len(light_mel)}")
        if len(dark_mel) != 10:
            errors.append(f"Test Dark melanoma count: expected 10, got {len(dark_mel)}")

    print()
    if errors:
        print("LEAKAGE CHECK FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    else:
        print("LEAKAGE CHECK PASSED")
        print(f"  Train: {counts.get('train')}, Val: {counts.get('val')}, Test: {counts.get('test')}")
        print(f"  Train dark mel: {len(train_dark_mel)}")
        print(f"  All sets disjoint: YES")
        sys.exit(0)


if __name__ == "__main__":
    check_leakage()

#!/usr/bin/env python3
"""Strong provenance check for the train-only synthetic images.

The byte-level SHA256 check in generate_clean_artifacts.py can only detect
verbatim copies. It cannot detect a re-encoded, resized, colour-jittered
derivative of a held-out image, which is exactly the failure mode that matters
here: the synthetics are 224x224 JPEGs derived from native-resolution PNGs, so
their bytes can never match their sources even if a test image were used.

This script instead establishes provenance perceptually. Each synthetic image is
matched against every candidate source under a dihedral-invariant descriptor,
and we verify that the best match is always one of the training-split dark
melanomas and never a validation or test image.

Descriptor: 32x32 grayscale, per-image z-score normalised, compared by cosine
similarity across all 8 dihedral transforms (4 rotations x optional mirror).
This is invariant to the augmentation stack actually used to build the
synthetics: brightness/contrast/HSV jitter (removed by z-scoring), Gaussian
noise (averaged out by downsampling), RandomRotate90 and HorizontalFlip (taken
as the max over the dihedral group), and JPEG re-encoding.

Exit code 0 if every synthetic resolves to a training-split source with a
margin over the best held-out decoy, 1 otherwise.
"""

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDI_DIR = os.path.join(PROJECT_ROOT, "ddidiversedermatologyimages")
SYN_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_train_only")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "splits", "ddi_split_seed42.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "splits", "synthetic_provenance_report.json")

DESCRIPTOR_SIZE = 32
# The generator ends with A.Resize(224, 224), so every synthetic is a squashed
# 224x224 image. Sources must be put through the identical resize before
# descriptor extraction, otherwise the sampling grids do not line up and the
# comparison collapses to noise.
SYNTHETIC_SIZE = 224
N_TRAIN_SOURCES = 29
# A synthetic's true training source must beat the closest held-out decoy.
# The decoy pool contains other dark melanomas that are genuinely similar
# lesions, so the margin is modest by construction; what matters is that it is
# strictly positive for every image, i.e. no synthetic is ever better explained
# by a held-out image than by a training source.
MARGIN = 0.02
# Minimum absolute similarity to be accepted as a confident match.
MIN_SIM = 0.80

SPLIT_GROUPS = ["train", "val", "test"]


def descriptor(img_bgr):
    """Dihedral-invariant descriptor bank for one image (8 variants)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    if norm < 1e-8:
        return np.zeros((8, DESCRIPTOR_SIZE * DESCRIPTOR_SIZE), dtype=np.float32)
    small /= norm
    flat = small.reshape(-1)
    variants = []
    for k in range(4):
        rot = np.rot90(small.reshape(DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), k)
        variants.append(rot.reshape(-1))
        variants.append(np.fliplr(rot).reshape(-1))
    return np.stack(variants)


def load_bgr(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if size is not None and (img.shape[1] != size or img.shape[0] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def main():
    if not os.path.exists(SPLIT_PATH):
        sys.exit(f"Split file not found: {SPLIT_PATH}")

    split = json.load(open(SPLIT_PATH))
    meta = {}
    import csv
    with open(os.path.join(DDI_DIR, "ddi_metadata.csv")) as fh:
        for row in csv.DictReader(fh):
            meta[row["DDI_file"]] = row

    train_sources = list(split["train_dark_mel"])
    if len(train_sources) != N_TRAIN_SOURCES:
        print(f"  WARNING: expected {N_TRAIN_SOURCES} train dark-mel sources, "
              f"got {len(train_sources)}")

    # Decoys: every val and test image, all skin tones.
    decoys = list(split["val"]) + list(split["test"])

    # Sources are native-resolution PNGs; put them through the same 224x224
    # squash the generator applies so the sampling grids line up.
    src_bank, src_names = [], []
    for fn in train_sources:
        img = load_bgr(os.path.join(DDI_DIR, fn), size=SYNTHETIC_SIZE)
        if img is None:
            print(f"  WARNING: missing train source {fn}")
            continue
        src_bank.append(descriptor(img))
        src_names.append(("train_dark_mel", fn))
    for fn in decoys:
        img = load_bgr(os.path.join(DDI_DIR, fn), size=SYNTHETIC_SIZE)
        if img is None:
            print(f"  WARNING: missing decoy {fn}")
            continue
        src_bank.append(descriptor(img))
        src_names.append(("held_out", fn))
    src_bank = np.stack(src_bank)  # (N, 8, D)

    syn_files = sorted(f for f in os.listdir(SYN_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not syn_files:
        sys.exit(f"No synthetic images found in {SYN_DIR}")

    n_train = len(train_sources)
    results = []
    failures = []
    margin_min = 1.0
    sim_min = 1.0

    print(f"  Matching {len(syn_files)} synthetics against "
          f"{n_train} train sources + {len(decoys)} held-out decoys ...")

    for fn in syn_files:
        img = load_bgr(os.path.join(SYN_DIR, fn))
        if img is None:
            failures.append({"synthetic": fn, "reason": "unreadable"})
            continue
        desc = descriptor(img)  # (8, D)
        n_src = src_bank.shape[0]
        # (D, N*8) with column index = source*8 + variant. The transpose must
        # put D first so that reshape preserves that column ordering; getting
        # this wrong pairs each synthetic variant with the wrong source variant
        # and silently degrades every match to noise.
        flat_bank = src_bank.transpose(2, 0, 1).reshape(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE, n_src * 8)
        sims = desc @ flat_bank                                    # (8, N*8)
        best_per_source = sims.reshape(8, n_src, 8).max(axis=(0, 2))  # (N,)
        best_idx = int(np.argmax(best_per_source))
        best_sim = float(best_per_source[best_idx])
        group, matched = src_names[best_idx]

        train_slice = best_per_source[:n_train]
        held_slice = best_per_source[n_train:]
        best_train_sim = float(train_slice.max())
        best_held_sim = float(held_slice.max()) if held_slice.size else -1.0
        margin = best_train_sim - best_held_sim
        margin_min = min(margin_min, margin)
        sim_min = min(sim_min, best_train_sim)

        rec = {
            "synthetic": fn,
            "best_match_group": group,
            "best_match_file": matched,
            "best_match_similarity": round(best_sim, 4),
            "best_train_source_similarity": round(best_train_sim, 4),
            "best_heldout_similarity": round(best_held_sim, 4),
            "margin": round(margin, 4),
        }
        results.append(rec)
        if group != "train_dark_mel" or margin < MARGIN or best_train_sim < MIN_SIM:
            failures.append(rec)

    n_ok = len(results) - len([r for r in results if r in failures])
    passed = not failures
    distinct_sources = len({r["best_match_file"] for r in results})

    report = {
        "method": "dihedral-invariant 32x32 grayscale cosine matching",
        "purpose": ("Establishes that every synthetic image derives from a "
                    "training-split dark melanoma and not from a held-out image. "
                    "Complements the byte-level SHA256 check, which cannot detect "
                    "re-encoded derivatives."),
        "descriptor_size": DESCRIPTOR_SIZE,
        "dihedral_variants": 8,
        "margin_threshold": MARGIN,
        "min_similarity_threshold": MIN_SIM,
        "counts": {
            "synthetic": len(syn_files),
            "train_dark_mel_sources": n_train,
            "heldout_decoys": len(decoys),
            "matched_to_train": len(results) - len([r for r in results if r["best_match_group"] != "train_dark_mel"]),
            "matched_to_heldout": len([r for r in results if r["best_match_group"] == "held_out"]),
            "distinct_source_images_used": distinct_sources,
        },
        "min_margin_observed": round(margin_min, 4),
        "min_train_similarity_observed": round(sim_min, 4),
        "passed": passed,
        "per_image": results,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"  synthetics matched to a TRAIN source : "
          f"{report['counts']['matched_to_train']}/{len(syn_files)}")
    print(f"  synthetics matched to a HELD-OUT image: "
          f"{report['counts']['matched_to_heldout']}/{len(syn_files)}")
    print(f"  distinct source images exercised      : {distinct_sources}/{n_train}")
    print(f"  worst margin over held-out decoy      : {margin_min:+.4f} "
          f"(threshold {MARGIN})")
    print(f"  lowest train-source similarity         : {sim_min:.4f} "
          f"(threshold {MIN_SIM})")
    print(f"  Saved {OUT_PATH}")
    print(f"\n  PROVENANCE CHECK: {'PASSED' if passed else 'FAILED'}")
    if failures:
        for rec in failures[:10]:
            print(f"    FAIL {rec['synthetic']}: {rec.get('reason', '')} "
                  f"best={rec.get('best_match_group')} "
                  f"({rec.get('best_match_file')}) margin={rec.get('margin')}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

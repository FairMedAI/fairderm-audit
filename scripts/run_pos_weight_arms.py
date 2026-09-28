#!/usr/bin/env python3
"""Stage-3 pos_weight arm study for FairDerm.

Stage 3 adds 290 synthetic dark-skin melanomas to a training set that holds
102 malignant and 291 benign images, but the loss keeps the *pre-synthesis*
class ratio of 291/102 = 2.8529. The combined set is actually 392 malignant
against 291 benign, so the intended ratio is 291/392 = 0.7423. The gap is a
class-imbalance confounder riding along with the augmentation effect.

This script runs three arms from the same seed-42 fine-tuned checkpoint, so
only Stage 3 differs between them:

  static     pos_weight = 2.8529, dark-melanoma synthetics only
             (the current headline configuration)
  recomputed pos_weight = 0.7423, dark-melanoma synthetics only
  symmetric  pos_weight = 0.7423, dark-melanoma AND dark-benign synthetics,
             so the combined class ratio stays near 1:1

Usage:
  python scripts/run_pos_weight_arms.py --stage static recomputed symmetric
  python scripts/run_pos_weight_arms.py --dry-run
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fairderm import (
    CONFIG,
    DDI_GROUP_MAP,
    TeeLogger,
    device,
    set_seed,
    DermDataset,
    SyntheticDataset,
    bootstrap_auroc,
    compute_metrics,
    compute_youden_threshold,
    create_model,
    freeze_for_ddi_finetune,
    get_preds,
    paired_bootstrap_auroc_delta,
    setup_ddi_data,
    train_model,
    train_transform,
    val_test_transform,
)
from fairderm import _make_splits
import albumentations as A
import cv2
import pandas as pd

SYN_ROOT = os.path.join(PROJECT_ROOT, "results", "pos_weight_arms")
SEED = 42
VARIANTS_PER_SOURCE = 10

ARMS = {
    "static": {
        "pos_weight": 2.8529,
        "synthetic": "dark_malignant_only",
        "description": (
            "Pre-synthesis class ratio 291/102 = 2.8529 applied to the "
            "post-synthesis set. This is the current headline configuration."
        ),
    },
    "recomputed": {
        "pos_weight": 0.7423,
        "synthetic": "dark_malignant_only",
        "description": (
            "Class ratio recomputed on the actual combined set 291/392 = 0.7423, "
            "isolating the augmentation effect from the imbalance change."
        ),
    },
    "symmetric": {
        "pos_weight": 1.0,
        "synthetic": "dark_malignant_and_benign",
        "description": (
            "Benign synthetics added so the combined set is class-balanced, and "
            "pos_weight set to 1.0 because no reweighting is then needed. Augmentation "
            "therefore changes neither the class ratio nor the loss weighting."
        ),
    },
}


def benign_variants_needed(n_base_pos, n_base_neg, n_syn_mel, n_dark_ben):
    """Benign synthetic count that makes the combined set exactly balanced.

    The static arm leaves 392 malignant against 291 benign, so 101 extra benign
    images equalise the classes. They are spread round-robin over the available
    dark benign lesions rather than taken 10-per-lesion, which would overshoot
    badly (95 dark benign x 10 = 950).
    """
    return max(0, (n_base_pos + n_syn_mel) - n_base_neg)


def _sha256_file(path, chunk_size=65536):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_synthetic_transform(seed):
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.7),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(alpha=0.5, sigma=5, p=0.1),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(224, 224),
        ],
        seed=seed,
    )


def distribute_variants(total, n_sources):
    """Split `total` synthetic images across `n_sources` lesions as evenly as possible."""
    if n_sources <= 0:
        return []
    base, extra = divmod(total, n_sources)
    return [base + (1 if i < extra else 0) for i in range(n_sources)]


def generate_variants(source_df, ddi_img_dir, out_dir, tag, seed, variants):
    """Generate `variants` photometric variants per source lesion.

    `variants` may be an int (same count for every lesion) or a list of
    per-lesion counts. Filenames are `syn_{tag}_{index:04d}.jpg` with the
    variants for one lesion written consecutively, so the source ordinal is
    recoverable as index // variants_per_source.
    """
    """Generate `variants` photometric variants per source lesion.

    Filenames are `syn_{tag}_{index:04d}.jpg` with `variants` consecutive files
    per source, so the source ordinal is recoverable as index // variants.
    Returns (files, provenance) where provenance maps filename -> DDI_file.
    """
    import shutil

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    alb = build_synthetic_transform(seed)
    count = 0
    provenance = {}
    rows = list(source_df.iterrows())
    for i, (_, row) in enumerate(rows):
        n_var = variants[i] if isinstance(variants, (list, tuple)) else variants
        if n_var <= 0:
            continue
        img_path = os.path.join(ddi_img_dir, row["DDI_file"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: unreadable source {img_path} - skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for _ in range(n_var):
            aug = alb(image=img)["image"]
            fn = f"syn_{tag}_{count:04d}.jpg"
            cv2.imwrite(
                os.path.join(out_dir, fn), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            )
            provenance[fn] = row["DDI_file"]
            count += 1
        if n_var:
            print(f"  [{tag}] {row['DDI_file']} -> {n_var} variants")
    files = sorted(f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return files, provenance


def check_leakage(syn_dir, heldout_df, ddi_img_dir):
    held = {}
    for fn in heldout_df["DDI_file"]:
        p = os.path.join(ddi_img_dir, fn)
        if os.path.exists(p):
            held[_sha256_file(p)] = fn
    syn = []
    for root, _dirs, files in os.walk(syn_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                syn.append(os.path.relpath(os.path.join(root, f), syn_dir))
    if not syn:
        raise ValueError(f"No generated images found under {syn_dir}; leakage check would "
                         f"pass vacuously.")
    syn_h = {}
    for f in sorted(syn):
        syn_h[_sha256_file(os.path.join(syn_dir, f))] = f
    overlap = sorted(set(syn_h) & set(held))
    return {
        "check": "sha256_exact_bytes",
        "n_heldout_hashed": len(held),
        "n_syn": len(syn),
        "n_syn_unique": len(syn_h),
        "n_duplicate_syn": len(syn) - len(syn_h),
        "n_overlap": len(overlap),
        "overlap_files": [syn_h[h] for h in overlap][:20],
        "passed": len(overlap) == 0,
    }


def subgroup_auroc(labels, probs, mask):
    y, p = np.asarray(labels)[mask], np.asarray(probs)[mask]
    if len(np.unique(y)) < 2:
        return None
    return float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, p))


def sens_spec(labels, probs, mask, thr):
    y, p = np.asarray(labels)[mask], np.asarray(probs)[mask]
    if len(y) == 0:
        return None, None
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    s = tp / (tp + fn) if (tp + fn) else None
    sp = tn / (tn + fp) if (tn + fp) else None
    return s, sp


def safe_boot(y, p, n):
    if len(np.unique(y)) < 2:
        return None
    med, lo, hi = bootstrap_auroc(y, p, n=n, seed=0)
    return {"median": round(float(med), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def evaluate(model, val_loader, test_loader, n_boot):
    v_p, v_y, _ = get_preds(model, val_loader, device)
    thr = compute_youden_threshold(v_y, v_p)
    t_p, t_y, t_f = get_preds(model, test_loader, device)
    ov = compute_metrics(t_y, t_p, threshold=thr)
    masks = {name: (t_f == code) for code, name in DDI_GROUP_MAP.items()}
    m = {
        "val_threshold": float(thr),
        "overall_auroc": round(float(ov["AUROC"]), 4),
        "overall_sens": round(float(ov["Sens"]), 4),
        "overall_spec": round(float(ov["Spec"]), 4),
        "overall_f1": round(float(ov["F1"]), 4),
        "overall_auroc_ci": safe_boot(t_y, t_p, n_boot),
        "subgroup_n": {k: int(v.sum()) for k, v in masks.items()},
    }
    aurocs = {}
    for name, mask in masks.items():
        a = subgroup_auroc(t_y, t_p, mask)
        aurocs[name] = a
        s, sp = sens_spec(t_y, t_p, mask, thr)
        m[f"{name.lower()}_auroc"] = round(a, 4) if a is not None else None
        m[f"{name.lower()}_sens"] = round(s, 4) if s is not None else None
        m[f"{name.lower()}_spec"] = round(sp, 4) if sp is not None else None
        m[f"{name.lower()}_auroc_ci"] = safe_boot(t_y[mask], t_p[mask], n_boot)
    if aurocs["Light"] is not None and aurocs["Dark"] is not None:
        m["gap_dark_minus_light"] = round(aurocs["Dark"] - aurocs["Light"], 4)
    return m


def run_arm(name, train_df, val_df, test_df, test_loader, ddi_img_dir,
            finetuned_path, args):
    spec = ARMS[name]
    t0 = time.time()
    set_seed(SEED)
    CONFIG["SEED"] = SEED

    arm_dir = os.path.join(SYN_ROOT, f"arm_{name}")
    os.makedirs(arm_dir, exist_ok=True)
    log_path = os.path.join(arm_dir, "train.log")

    with TeeLogger(log_path):
        print("=" * 70)
        print(f"ARM: {name}  |  pos_weight={spec['pos_weight']}  |  synthetic={spec['synthetic']}")
        print("=" * 70)
        print(f"  {spec['description']}")

        dark_mel = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 1)]
        dark_ben = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 0)]
        n_dark_mel, n_dark_ben = len(dark_mel), len(dark_ben)
        print(f"  Train dark: {n_dark_mel} malignant, {n_dark_ben} benign")

        syn_dir = os.path.join(arm_dir, "synthetics")
        provenance = {}
        if spec["synthetic"] == "dark_malignant_only":
            files, provenance = generate_variants(
                dark_mel, ddi_img_dir, syn_dir, "darkmel", SEED, VARIANTS_PER_SOURCE
            )
        else:
            n_base_pos_0 = int(train_df["label"].sum())
            n_base_neg_0 = len(train_df) - n_base_pos_0
            n_ben_needed = benign_variants_needed(
                n_base_pos_0, n_base_neg_0, n_dark_mel * VARIANTS_PER_SOURCE, n_dark_ben
            )
            # Spread the benign budget round-robin over the dark benign lesions.
            per_source = distribute_variants(n_ben_needed, n_dark_ben)
            nz = sorted({v for v in per_source if v})
            span = f"{nz[0]}" if len(nz) == 1 else f"{nz[0]}-{nz[-1]}"
            print(f"  Benign synthetics needed for balance: {n_ben_needed} "
                  f"({span} per lesion across {n_dark_ben} dark benign)")

            mel_dir = os.path.join(syn_dir, "mel")
            ben_dir = os.path.join(syn_dir, "ben")
            mel_files, mel_prov = generate_variants(
                dark_mel, ddi_img_dir, mel_dir, "darkmel", SEED, VARIANTS_PER_SOURCE
            )
            ben_files, ben_prov = generate_variants(
                dark_ben, ddi_img_dir, ben_dir, "darkben", SEED, per_source
            )
            for f in mel_files:
                provenance[os.path.join("mel", f)] = mel_prov[f]
            for f in ben_files:
                provenance[os.path.join("ben", f)] = ben_prov[f]
            files = (
                [os.path.join("mel", f) for f in mel_files]
                + [os.path.join("ben", f) for f in ben_files]
            )
        print(f"  Generated {len(files)} synthetic images")

        leak = check_leakage(syn_dir, test_df, ddi_img_dir)
        print(f"  [LEAKAGE vs test] overlap={leak['n_overlap']} passed={leak['passed']}")
        if not leak["passed"]:
            raise RuntimeError(f"arm {name}: SHA256 overlap with test")

        leak_val = check_leakage(syn_dir, val_df, ddi_img_dir)
        print(f"  [LEAKAGE vs val]   overlap={leak_val['n_overlap']} passed={leak_val['passed']}")
        if not leak_val["passed"]:
            raise RuntimeError(f"arm {name}: SHA256 overlap with val")

        n_base_pos = int(train_df["label"].sum())
        n_base_neg = len(train_df) - n_base_pos
        n_syn_mel = sum(1 for f in files if "darkmel" in f)
        n_syn_ben = len(files) - n_syn_mel
        comb_pos = n_base_pos + n_syn_mel
        comb_neg = n_base_neg + n_syn_ben
        print(f"  Base: {n_base_pos} pos / {n_base_neg} neg")
        print(f"  Combined: {comb_pos} pos / {comb_neg} neg "
              f"(true ratio {comb_neg / comb_pos:.4f})")
        print(f"  Loss pos_weight: {spec['pos_weight']:.4f}")

        model = create_model(pretrained=False)
        model.load_state_dict(torch.load(finetuned_path, weights_only=True))
        freeze_for_ddi_finetune(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_trainable:,}")

        base_ds = DermDataset(train_df, ddi_img_dir, transform=train_transform, is_ddi=True)
        if spec["synthetic"] == "dark_malignant_only":
            syn_ds = SyntheticDataset(syn_dir, label=1.0, fst=56)
        else:
            syn_ds = ConcatDataset(
                [
                    SyntheticDataset(mel_dir, label=1.0, fst=56),
                    SyntheticDataset(ben_dir, label=0.0, fst=56),
                ]
            )

        combined = ConcatDataset([base_ds, syn_ds])
        print(f"  Combined dataset: {len(combined)} samples")

        train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(
            DermDataset(val_df, ddi_img_dir, val_test_transform, True),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([spec["pos_weight"]]).to(device)
        )

        save_path = os.path.join(arm_dir, f"arm_{name}.pth")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=args.epochs, patience=args.patience, save_path=save_path,
            experiment_dir=arm_dir,
        )

        metrics = evaluate(model, val_loader, test_loader, args.bootstrap)
        metrics.update(
            {
                "arm": name,
                "seed": SEED,
                "generated_by": "scripts/run_pos_weight_arms.py",
                "finetuned_checkpoint": os.path.relpath(finetuned_path, PROJECT_ROOT),
                "pos_weight": spec["pos_weight"],
                "pos_weight_basis": (
                    "pre-synthesis 291/102 = 2.8529" if name == "static"
                    else ("post-synthesis 291/392 = 0.7423" if name == "recomputed"
                          else "classes balanced by construction, so 1.0")
                ),
                "synthetic_mode": spec["synthetic"],
                "description": spec["description"],
                "class_counts": {
                    "base_pos": n_base_pos,
                    "base_neg": n_base_neg,
                    "syn_pos": n_syn_mel,
                    "syn_neg": n_syn_ben,
                    "combined_pos": comb_pos,
                    "combined_neg": comb_neg,
                    "combined_true_ratio": round(comb_neg / comb_pos, 4),
                },
                "n_trainable_params": n_trainable,
                "epochs_used": len(history),
                "hyperparams": {
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "weight_decay": 1e-4,
                    "freeze": "features.6,features.7,classifier",
                },
                "leakage_vs_test": leak,
                "leakage_vs_val": leak_val,
                "synthetic_provenance": provenance,
                "elapsed_seconds": round(time.time() - t0, 1),
            }
        )

        out = os.path.join(arm_dir, f"arm_{name}_metrics.json")
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved -> {out}")
        print(f"  AUROC={metrics['overall_auroc']:.4f} "
              f"Light={metrics['light_auroc']} Medium={metrics['medium_auroc']} "
              f"Dark={metrics['dark_auroc']} gap={metrics.get('gap_dark_minus_light')}")
        return metrics


def write_summary(results, failed):
    summary = {
        "generated_by": "scripts/run_pos_weight_arms.py",
        "seed": SEED,
        "device": str(device),
        "arms_completed": list(results.keys()),
        "arms_failed": failed,
        "question": (
            "Does the Stage-3 Dark-skin AUROC change when the class-imbalance "
            "confounder is removed, holding the fine-tuned checkpoint fixed?"
        ),
        "design": (
            "All three arms start from the same seed-42 fine-tuned checkpoint and "
            "differ only in Stage 3: loss pos_weight and/or synthetic composition. "
            "This isolates the imbalance confounder from the augmentation effect."
        ),
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "synthetic_provenance"}
                 for k, v in results.items()},
    }
    path = os.path.join(SYN_ROOT, "pos_weight_arms_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {path}")

    print("\n" + "=" * 78)
    hdr = f"{'arm':<12}{'pos_wt':>8}{'comb_ratio':>12}{'overall':>10}{'Light':>9}{'Medium':>9}{'Dark':>9}{'gap':>9}"
    print(hdr)
    print("-" * 78)
    for k, v in results.items():
        cr = v["class_counts"]["combined_true_ratio"]
        gap = v.get("gap_dark_minus_light")
        print(
            f"{k:<12}{v['pos_weight']:>8.4f}{cr:>12.4f}{v['overall_auroc']:>10.4f}"
            f"{(v['light_auroc'] or 0):>9.4f}{(v['medium_auroc'] or 0):>9.4f}"
            f"{(v['dark_auroc'] or 0):>9.4f}{(gap if gap is not None else 0):>9.4f}"
        )
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description="FairDerm Stage-3 pos_weight arm study")
    p.add_argument("--stage", nargs="+", default=list(ARMS.keys()), choices=list(ARMS.keys()))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(SYN_ROOT, exist_ok=True)
    set_seed(SEED)

    ddi_df, ddi_img_dir = setup_ddi_data()
    train_df, val_df, test_df = _make_splits(ddi_df)
    finetuned_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_ddi_finetuned.pth")

    if args.dry_run:
        print("DRY RUN")
        print(f"  arms: {args.stage}")
        print(f"  finetuned ckpt exists: {os.path.exists(finetuned_path)} ({finetuned_path})")
        print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        dm = train_df[(train_df.skin_tone == 56) & (train_df.malignant == 1)]
        db = train_df[(train_df.skin_tone == 56) & (train_df.malignant == 0)]
        npos = int(train_df.label.sum())
        nneg = len(train_df) - npos
        print(f"  dark mel={len(dm)} dark ben={len(db)}")
        for a in args.stage:
            s = ARMS[a]
            syn_mel = len(dm) * VARIANTS_PER_SOURCE
            if s["synthetic"] == "dark_malignant_only":
                syn_ben = 0
            else:
                syn_ben = benign_variants_needed(npos, nneg, syn_mel, len(db))
            cp, cn = npos + syn_mel, nneg + syn_ben
            print(f"  {a:<11} pw={s['pos_weight']:.4f} syn={syn_mel}+{syn_ben} "
                  f"combined={cp}pos/{cn}neg true_ratio={cn/cp:.4f}")
        return

    if not os.path.exists(finetuned_path):
        sys.exit(f"Missing {finetuned_path}; run: python fairderm.py --stage finetune")

    test_loader = DataLoader(
        DermDataset(test_df, ddi_img_dir, val_test_transform, True),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    results, failed = {}, []
    for name in args.stage:
        try:
            results[name] = run_arm(
                name, train_df, val_df, test_df, test_loader, ddi_img_dir,
                finetuned_path, args,
            )
        except Exception as e:
            failed.append(name)
            print(f"\n[ARM {name}] FAILED: {e}")
            traceback.print_exc()

    if not results:
        sys.exit("All arms failed - no summary written.")
    write_summary(results, failed)
    if failed:
        sys.exit(f"Arms failed: {failed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-seed robustness study for FairDerm.

Replaces the single-seed (42) claim with a 5-seed mean +/- SD for the
Dark AUROC delta, mirroring fairderm.py stages 2/3 exactly.

For each seed:
  1. New 60/20/20 DDI split stratified by label x skin_tone
     (seed 42 reuses splits/ddi_split_seed42.json verbatim; new seed files
     are saved as splits/ddi_split_seed{seed}.json).
  2. The Stage-1 HAM10000 checkpoint (models/fairderm_ham_baseline.pth) is
     REUSED without retraining; it is only re-evaluated on each seed's test.
  3. Stage-2 fine-tune on that seed's train split (freeze features.0-5).
  4. Generate 290 synthetic dark-skin melanomas (29 x 10 variants) into
     results/multiseed/seed{seed}_synthetics/ and prove SHA256 byte-level
     zero overlap vs that seed's val + test.
  5. Stage-3: train from the fine-tuned checkpoint on train + synthetics.
  6. Evaluate on that seed's test set: overall/Light/Dark AUROC, gap,
     Light/Dark sensitivity/specificity, bootstrap 95% CIs, and the paired
     bootstrap Dark AUROC delta (augment - finetuned, one-tailed p).

Per-seed outputs: results/multiseed/seed{seed}.log, seed{seed}_metrics.json,
seed{seed}_finetuned.pt, seed{seed}_augmented.pt, seed{seed}_synthetics/
Summary: results/multiseed_summary.csv and results/multiseed_summary.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import albumentations as A
from sklearn.metrics import roc_auc_score, confusion_matrix

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
    get_preds,
    paired_bootstrap_auroc_delta,
    setup_ddi_data,
    train_model,
    train_transform,
    val_test_transform,
)

from sklearn.model_selection import train_test_split

import cv2

COLS = [
    "overall_baseline",
    "overall_finetuned",
    "overall_synthetic",
    "dark_finetuned",
    "dark_synthetic",
    "dark_delta",
    "gap_finetuned",
    "gap_synthetic",
    "light_finetuned",
    "light_synthetic",
]

SEED42 = 42


def sha256_file(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_seeded_split(seed, ddi_df):
    ddi_df = ddi_df.copy()
    ddi_df["stratify_key"] = (
        ddi_df["label"].astype(str) + "_" + ddi_df["skin_tone"].astype(str)
    )
    train_val_df, test_df = train_test_split(
        ddi_df, test_size=0.2, stratify=ddi_df["stratify_key"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25,
        stratify=train_val_df["stratify_key"], random_state=seed,
    )
    assert set(train_df["DDI_file"]).isdisjoint(set(val_df["DDI_file"]))
    assert set(train_df["DDI_file"]).isdisjoint(set(test_df["DDI_file"]))
    assert set(val_df["DDI_file"]).isdisjoint(set(test_df["DDI_file"]))
    train_dark_mel = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 1)]
    split_data = {
        "seed": seed,
        "stratify": "label x skin_tone",
        "train": sorted(train_df["DDI_file"].tolist()),
        "val": sorted(val_df["DDI_file"].tolist()),
        "test": sorted(test_df["DDI_file"].tolist()),
        "train_dark_mel": sorted(train_dark_mel["DDI_file"].tolist()),
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "train_dark_mel": len(train_dark_mel),
        },
    }
    return train_df, val_df, test_df, split_data


def get_or_make_split(seed, ddi_df):
    splits_dir = os.path.join(PROJECT_ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    if seed == SEED42:
        path = os.path.join(splits_dir, "ddi_split_seed42.json")
        if os.path.exists(path):
            with open(path) as f:
                split = json.load(f)
            train_df = ddi_df[ddi_df["DDI_file"].isin(split["train"])].copy()
            val_df = ddi_df[ddi_df["DDI_file"].isin(split["val"])].copy()
            test_df = ddi_df[ddi_df["DDI_file"].isin(split["test"])].copy()
            assert len(train_df) == len(split["train"])
            assert len(val_df) == len(split["val"])
            assert len(test_df) == len(split["test"])
            print(f"  Reused existing {path} (seed 42 test set identical to paper)")
            return train_df, val_df, test_df, split
    train_df, val_df, test_df, split_data = make_seeded_split(seed, ddi_df)
    path = os.path.join(splits_dir, f"ddi_split_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"  Saved split -> {path}")
    return train_df, val_df, test_df, split_data


def build_albumentations():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.7),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(alpha=0.5, sigma=5, p=0.1),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Resize(224, 224),
    ])


def generate_synthetics(dark_mel_df, ddi_img_dir, syn_dir, variants=10):
    if os.path.isdir(syn_dir):
        shutil.rmtree(syn_dir)
    os.makedirs(syn_dir, exist_ok=True)
    alb = build_albumentations()
    count = 0
    for i, (_, row) in enumerate(dark_mel_df.iterrows()):
        img_path = os.path.join(ddi_img_dir, row["DDI_file"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: unreadable source {img_path} — skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for _ in range(variants):
            aug = alb(image=img)["image"]
            cv2.imwrite(os.path.join(syn_dir, f"syn_dark_{count:04d}.jpg"),
                        cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
            count += 1
        print(f"  [{i + 1}/{len(dark_mel_df)}] {row['DDI_file']} -> {variants} variants")
    files = sorted(
        f for f in os.listdir(syn_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    print(f"  Generated {len(files)} synthetic images in {syn_dir}")
    return files


def sha256_leak_check(syn_dir, val_files, test_files, ddi_img_dir):
    target = set()
    for fn in val_files + test_files:
        target.add(sha256_file(os.path.join(ddi_img_dir, fn)))
    syn = sorted(
        f for f in os.listdir(syn_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    overlaps = [f for f in syn if sha256_file(os.path.join(syn_dir, f)) in target]
    return {
        "n_val": len(val_files),
        "n_test": len(test_files),
        "n_syn": len(syn),
        "synthetic_vs_val_or_test": len(overlaps),
        "passed": len(overlaps) == 0,
    }


def load_checkpoint(path):
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def subgroup_auroc(labels, probs, mask):
    y, p = np.asarray(labels)[mask], np.asarray(probs)[mask]
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def subgroup_sens_spec(labels, probs, mask, threshold):
    y, p = np.asarray(labels)[mask], np.asarray(probs)[mask]
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens), float(spec)


def safe_bootstrap_auroc(y, p, n):
    if len(np.unique(y)) < 2:
        return None
    med, lo, hi = bootstrap_auroc(y, p, n=n, seed=0)
    return {"median": round(float(med), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def stage_eval(model, val_loader, test_loader, n_bootstrap):
    v_probs, v_labels, _ = get_preds(model, val_loader, device)
    v_thr = compute_youden_threshold(v_labels, v_probs)
    t_probs, t_labels, t_fsts = get_preds(model, test_loader, device)

    overall = compute_metrics(t_labels, t_probs, threshold=v_thr)
    light_mask = t_fsts == 12
    dark_mask = t_fsts == 56
    dark_auroc = subgroup_auroc(t_labels, t_probs, dark_mask)
    light_auroc = subgroup_auroc(t_labels, t_probs, light_mask)

    m = {
        "val_threshold": float(v_thr),
        "overall_auroc": float(overall["AUROC"]),
        "overall_sens": float(overall["Sens"]),
        "overall_spec": float(overall["Spec"]),
        "overall_f1": float(overall["F1"]),
        "light_auroc": light_auroc,
        "dark_auroc": dark_auroc,
        "gap_dark_minus_light": round(dark_auroc - light_auroc, 4),
        "light_sens": subgroup_sens_spec(t_labels, t_probs, light_mask, v_thr)[0],
        "light_spec": subgroup_sens_spec(t_labels, t_probs, light_mask, v_thr)[1],
        "dark_sens": subgroup_sens_spec(t_labels, t_probs, dark_mask, v_thr)[0],
        "dark_spec": subgroup_sens_spec(t_labels, t_probs, dark_mask, v_thr)[1],
    }
    m["overall_auroc_ci"] = safe_bootstrap_auroc(t_labels, t_probs, n_bootstrap)
    m["light_auroc_ci"] = safe_bootstrap_auroc(t_labels[light_mask], t_probs[light_mask], n_bootstrap)
    m["dark_auroc_ci"] = safe_bootstrap_auroc(t_labels[dark_mask], t_probs[dark_mask], n_bootstrap)
    return m, t_probs, t_labels, t_fsts


def freeze_for_finetune(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "features.6" in name or "features.7" in name or "classifier" in name:
            param.requires_grad = True


def paired_dark_delta(labels, p_ft, p_aug, dark_mask, n):
    y = np.asarray(labels)[dark_mask]
    if len(np.unique(y)) < 2:
        return None
    mean, lo, hi, p_val = paired_bootstrap_auroc_delta(
        y,
        np.asarray(p_ft)[dark_mask],
        np.asarray(p_aug)[dark_mask],
        n=n,
        seed=0,
    )
    return {
        "mean": float(mean),
        "lo": float(lo),
        "hi": float(hi),
        "p_one_tail": float(p_val),
    }


def run_seed(seed, ddi_df, ddi_img_dir, args):
    results_dir = os.path.join(PROJECT_ROOT, "results", "multiseed")
    os.makedirs(results_dir, exist_ok=True)
    with TeeLogger(os.path.join(results_dir, f"seed{seed}.log")):
        return _run_seed(seed, ddi_df, ddi_img_dir, results_dir, args)


def _run_seed(seed, ddi_df, ddi_img_dir, results_dir, args):
    t0 = time.time()
    set_seed(seed)
    CONFIG["SEED"] = seed
    print("=" * 70)
    print(f"SEED {seed} — started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    train_df, val_df, test_df, split_data = get_or_make_split(seed, ddi_df)
    counts = split_data["counts"]
    print(f"Split: train={counts['train']} val={counts['val']} test={counts['test']} "
          f"train_dark_mel={counts['train_dark_mel']}")

    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Dynamic pos_weight (train): {pos_weight_val:.4f} (n_neg={n_neg}, n_pos={n_pos})")

    train_loader = DataLoader(
        DermDataset(train_df, ddi_img_dir, train_transform, True),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        DermDataset(val_df, ddi_img_dir, val_test_transform, True),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        DermDataset(test_df, ddi_img_dir, val_test_transform, True),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    baseline_path = os.path.join(PROJECT_ROOT, "models", "fairderm_ham_baseline.pth")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint missing: {baseline_path}")

    print(f"\n[Stage: eval baseline (REUSED, not retrained)] {baseline_path}")
    base_metrics, _, _, _ = stage_eval(
        load_checkpoint(baseline_path), val_loader, test_loader, args.bootstrap
    )
    print(f"  baseline overall_auroc={base_metrics['overall_auroc']:.4f} "
          f"dark_auroc={base_metrics['dark_auroc']:.4f} "
          f"light_auroc={base_metrics['light_auroc']:.4f}")

    print(f"\n[Stage: finetune] lr={args.lr} bs={args.batch_size} epochs={args.epochs} "
          f"patience={args.patience} freeze=features.0-5")
    model = create_model(pretrained=True)
    model.load_state_dict(torch.load(baseline_path, weights_only=True))
    print("  Loaded baseline weights into finetune model")
    freeze_for_finetune(model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    ft_path = os.path.join(results_dir, f"seed{seed}_finetuned.pt")
    model, ft_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.epochs, patience=args.patience, save_path=ft_path,
        experiment_dir=results_dir,
    )
    print(f"  Finetuned checkpoint -> {ft_path} (epochs used: {len(ft_history)})")

    ft_metrics, ft_probs, ft_labels, ft_fsts = stage_eval(
        model, val_loader, test_loader, args.bootstrap
    )
    print(f"  finetuned: overall={ft_metrics['overall_auroc']:.4f} "
          f"dark={ft_metrics['dark_auroc']:.4f} "
          f"light={ft_metrics['light_auroc']:.4f} "
          f"gap={ft_metrics['gap_dark_minus_light']:.4f}")

    print("\n[Stage: synthetics]")
    dark_mel = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 1)]
    if args.design == "original":
        syn_source = dark_mel
        syn_desc = "train_dark_mel_only (29 dark melanomas x 10 variants)"
    else:
        assert args.design == "designA"
        dark_ben = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 0)]
        syn_source = pd.concat([dark_mel, dark_ben], ignore_index=True)
        syn_desc = (f"train_dark_mel_and_dark_benign symmetric "
                    f"({len(dark_mel)} dark mel x 10 + {len(dark_ben)} dark benign x 10)")
    syn_dir = os.path.join(results_dir, f"seed{seed}_synthetics")
    syn_files = generate_synthetics(syn_source, ddi_img_dir, syn_dir, variants=10)
    leak = sha256_leak_check(
        syn_dir,
        val_df["DDI_file"].tolist(),
        test_df["DDI_file"].tolist(),
        ddi_img_dir,
    )
    print(f"  [LEAKAGE CHECK] synthetic_vs_val_or_test={leak['synthetic_vs_val_or_test']} "
          f"PASSED={leak['passed']}")
    if not leak["passed"]:
        raise RuntimeError("SHA256 overlap detected between synthetics and val/test")

    print(f"\n[Stage: augment] train({len(train_df)}) + synthetic({len(syn_files)})")
    model2 = create_model(pretrained=True)
    model2.load_state_dict(torch.load(ft_path, weights_only=True))
    print("  Loaded finetuned weights into augment model")
    freeze_for_finetune(model2)
    base_ds = DermDataset(train_df, ddi_img_dir, transform=train_transform, is_ddi=True)
    syn_ds = SyntheticDataset(syn_dir)
    combined_train = ConcatDataset([base_ds, syn_ds])
    combined_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model2.parameters()), lr=args.lr, weight_decay=1e-4
    )
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="max", factor=0.5, patience=2
    )
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    aug_path = os.path.join(results_dir, f"seed{seed}_augmented.pt")
    model2, aug_history = train_model(
        model2, combined_loader, val_loader, criterion2, optimizer2, scheduler2,
        num_epochs=args.epochs, patience=args.patience, save_path=aug_path,
        experiment_dir=results_dir,
    )
    print(f"  Augmented checkpoint -> {aug_path} (epochs used: {len(aug_history)})")

    aug_metrics, aug_probs, aug_labels, aug_fsts = stage_eval(
        model2, val_loader, test_loader, args.bootstrap
    )
    print(f"  augmented: overall={aug_metrics['overall_auroc']:.4f} "
          f"dark={aug_metrics['dark_auroc']:.4f} "
          f"light={aug_metrics['light_auroc']:.4f} "
          f"gap={aug_metrics['gap_dark_minus_light']:.4f}")

    dark_mask = ft_fsts == 56
    delta = paired_dark_delta(ft_labels, ft_probs, aug_probs, dark_mask, args.bootstrap)
    if delta:
        print(f"  Paired Dark AUROC delta (augment - finetuned): "
              f"mean={delta['mean']:.4f} [{delta['lo']:.4f}, {delta['hi']:.4f}] "
              f"p_one_tail={delta['p_one_tail']:.4f}")

    mps_note = ("MPS is non-deterministic: run-to-run AUROC for the same seed may vary by "
                "approximately +/-0.01; the 5-seed mean +/- SD absorbs this noise.")
    metrics = {
        "seed": seed,
        "generated_by": "scripts/run_multiseed.py",
        "split_file": f"splits/ddi_split_seed{seed}.json",
        "split_sizes": counts,
        "test_subgroup_sizes": {
            "light": int((test_df["skin_tone"] == 12).sum()),
            "dark": int((test_df["skin_tone"] == 56).sum()),
            "light_melanoma": int(((test_df["skin_tone"] == 12) & (test_df["label"] == 1)).sum()),
            "dark_melanoma": int(((test_df["skin_tone"] == 56) & (test_df["label"] == 1)).sum()),
        },
        "pos_weight_dynamic": round(pos_weight_val, 4),
        "hyperparams_shared": {
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": 1e-4,
            "patience": args.patience,
            "freeze_layers": "features.6,features.7,classifier",
            "synthetic_source": syn_desc,
        },
        "epochs_used": {"finetuned": len(ft_history), "augmented": len(aug_history)},
        "synthetic": {
            "n_images": len(syn_files),
            "leakage_sha256_vs_val_or_test": leak,
        },
        "val_thresholds": {
            "baseline": base_metrics["val_threshold"],
            "finetuned": ft_metrics["val_threshold"],
            "augmented": aug_metrics["val_threshold"],
        },
        "stages": {"baseline": base_metrics, "finetuned": ft_metrics, "augmented": aug_metrics},
        "paired_dark_auroc_delta_augment_minus_finetuned": delta,
        "mps_nondeterminism_note": mps_note,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    metrics_path = os.path.join(results_dir, f"seed{seed}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics -> {metrics_path}")

    row = {
        "seed": seed,
        "overall_baseline": round(base_metrics["overall_auroc"], 4),
        "overall_finetuned": round(ft_metrics["overall_auroc"], 4),
        "overall_synthetic": round(aug_metrics["overall_auroc"], 4),
        "dark_finetuned": round(ft_metrics["dark_auroc"], 4),
        "dark_synthetic": round(aug_metrics["dark_auroc"], 4),
        "dark_delta": round(delta["mean"], 4) if delta else None,
        "gap_finetuned": round(ft_metrics["gap_dark_minus_light"], 4),
        "gap_synthetic": round(aug_metrics["gap_dark_minus_light"], 4),
        "light_finetuned": round(ft_metrics["light_auroc"], 4),
        "light_synthetic": round(aug_metrics["light_auroc"], 4),
    }
    print(f"\nSEED {seed} complete in {time.time() - t0:.1f}s")
    return row


def ci_lo_hi(vals):
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 2:
        return None, None
    arr = np.asarray(vals)
    rng = np.random.default_rng(0)
    dist = []
    for _ in range(1000):
        boot = arr[rng.integers(0, len(arr), len(arr))]
        dist.append(float(np.mean(boot)))
    return round(float(np.percentile(dist, 2.5)), 4), round(float(np.percentile(dist, 97.5)), 4)


def build_summary_df(rows):
    df = pd.DataFrame(rows)
    ci_targets = [
        ("dark_delta", "dark_delta_ci95_lo", "dark_delta_ci95_hi"),
        ("gap_finetuned", "gap_finetuned_ci95_lo", "gap_finetuned_ci95_hi"),
        ("gap_synthetic", "gap_synthetic_ci95_lo", "gap_synthetic_ci95_hi"),
    ]
    ci_cols = [col for _, lo, hi in ci_targets for col in (lo, hi)]
    out = df[["seed"] + COLS].copy()
    for col in ci_cols:
        out[col] = np.nan
    mean_row = {"seed": "mean"}
    sd_row = {"seed": "sd"}
    for c in COLS:
        vals = df[c].dropna().astype(float)
        m = float(np.mean(vals)) if len(vals) else np.nan
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else float(np.std(vals)) if len(vals) else np.nan
        mean_row[c] = round(m, 4) if not np.isnan(m) else np.nan
        sd_row[c] = round(s, 4) if not np.isnan(s) else np.nan
    for target, lo, hi in ci_targets:
        vals = df[target].dropna().astype(float).tolist()
        lo_v, hi_v = ci_lo_hi(vals)
        mean_row[lo] = lo_v
        mean_row[hi] = hi_v
    out = pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)
    out = pd.concat([out, pd.DataFrame([sd_row])], ignore_index=True)
    return out


def write_summary(rows, seeds, failed, args, design_suffix=""):
    df = build_summary_df(rows)
    csv_path = os.path.join(PROJECT_ROOT, "results", f"multiseed_summary{design_suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV -> {csv_path}")

    aggregates = {}
    for c in COLS:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if not vals:
            aggregates[c] = None
            continue
        lo, hi = ci_lo_hi(vals)
        aggregates[c] = {
            "mean": round(float(np.mean(vals)), 4),
            "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "ci95": [lo, hi] if lo is not None else None,
        }
    summary = {
        "generated_by": "scripts/run_multiseed.py",
        "device": str(device),
        "seeds": seeds,
        "seeds_failed": failed,
        "method": ("Reuse models/fairderm_ham_baseline.pth (SEED-independent, not retrained); "
                   "per-seed 60/20/20 split; Stage-2 finetune + Stage-3 augment; evaluate on that "
                   "seed's test; paired bootstrap Dark AUROC delta (augment - finetuned) n=1000 "
                   "per seed; mean/SD/95% CI (seed-level percentile bootstrap, n=1000) across seeds."),
        "rows": rows,
        "aggregates": aggregates,
        "mps_nondeterminism_note": ("MPS is non-deterministic; per-seed AUROC may vary ~+/-0.01 "
                                    "run-to-run. 5-seed mean +/- SD absorbs this."),
    }
    json_path = os.path.join(PROJECT_ROOT, "results", f"multiseed_summary{design_suffix}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON -> {json_path}")


def preflight(args):
    print(f"Device: {device}  PyTorch: {torch.__version__}")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_ok}")
    if not mps_ok:
        print("WARNING: MPS not available; training will be slow.")
    baseline_path = os.path.join(PROJECT_ROOT, "models", "fairderm_ham_baseline.pth")
    if not os.path.exists(baseline_path):
        print(f"FAIL: baseline checkpoint missing: {baseline_path}")
        return False
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(baseline_path, weights_only=True))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Baseline checkpoint loads OK; forward pass shape={tuple(out.shape)}")
    print("(preflight uses pretrained=False so no download; weights only from baseline ckpt)")

    ddi_df, ddi_img_dir = setup_ddi_data()
    for seed in args.seeds:
        train_df, val_df, test_df, split_data = make_seeded_split(seed, ddi_df)
        dark = train_df[(train_df["skin_tone"] == 56) & (train_df["label"] == 1)]
        print(f"  preflight seed={seed}: train={len(train_df)} val={len(val_df)} "
              f"test={len(test_df)} train_dark_mel={len(dark)}")
    print("PREFLIGHT OK")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="FairDerm multi-seed robustness (Dark AUROC delta, 5 seeds mean +/- SD)."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
        help="Seeds to run (research plan: 42 43 44 45 46; seed 42 anchors the paper)",
    )
    parser.add_argument(
        "--design", choices=["original", "designA"], default="original",
        help="original = as-written (29 dark mel x 10; 290 syn). "
             "designA = symmetric-corrected (29 dark mel + 95 dark benign, x10 both; 1240 syn)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pre-flight gate: verify imports, device, baseline ckpt, splits. Writes nothing.",
    )
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    print(f"Device: {device}")
    print(f"Seeds: {seeds} (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, "
          f"patience={args.patience}, bootstrap={args.bootstrap})")

    if args.dry_run:
        sys.exit(0 if preflight(args) else 1)

    results_dir = os.path.join(PROJECT_ROOT, "results", "multiseed")
    os.makedirs(results_dir, exist_ok=True)

    set_seed(seeds[0])
    ddi_df, ddi_img_dir = setup_ddi_data()
    print("Note: Stage-1 baseline checkpoint is reused for all seeds; it is NOT retrained.\n")

    rows = []
    failed = []
    for seed in seeds:
        try:
            rows.append(run_seed(seed, ddi_df, ddi_img_dir, args))
        except Exception as e:
            failed.append(seed)
            print(f"\n[SEED {seed}] FAILED: {e}")
            traceback.print_exc()
            log_path = os.path.join(results_dir, f"seed{seed}.log")
            with open(log_path, "a") as f:
                f.write(f"\n[SEED {seed}] FAILED: {e}\n")
                f.write(traceback.format_exc())

    if not rows:
        sys.exit("All seeds failed — no summary written.")

    write_summary(rows, seeds, failed, args, design_suffix=(
        "" if args.design == "original" else f"_{args.design}"
    ))

    print("\n" + "=" * 70)
    for c in COLS:
        agg = None
        print(f"  {c}: completed seeds must be checked in multiseed_summary.json")
    print("=" * 70)

    if failed:
        sys.exit(f"Seeds failed: {failed}")


if __name__ == "__main__":
    main()
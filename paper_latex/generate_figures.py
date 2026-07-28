#!/usr/bin/env python3
import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def load_metrics():
    path = os.path.join(ROOT, "results", "metrics_seed42.json")
    with open(path) as f:
        return json.load(f)


def load_config(stage):
    path = os.path.join(ROOT, "results", stage, "config.json")
    with open(path) as f:
        return json.load(f)


def load_val_aurocs():
    result = {}
    for stage_cfg, key in [("baseline", "baseline"), ("finetune", "finetune"), ("augment", "+synthetic")]:
        try:
            result[key] = load_config(stage_cfg)["final_best_val_auroc"]
        except (FileNotFoundError, KeyError):
            result[key] = 0.0
    return result


def load_ablation_report():
    path = os.path.join(ROOT, "results", "ablation", "ablation_report.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_training_history(stage):
    path = os.path.join(ROOT, "results", stage, "training_history.csv")
    if not os.path.exists(path):
        return None
    epochs, train_loss, val_auroc = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_auroc.append(float(row["val_auroc"]))
    return {"epochs": epochs, "train_loss": train_loss, "val_auroc": val_auroc}


def fig1_val_auroc_comparison(metrics):
    val_aurocs = load_val_aurocs()
    labels = ["Baseline", "Fine-tuned", "+Synthetic"]
    keys = ["baseline", "finetune", "+synthetic"]
    vals = [val_aurocs[k] for k in keys]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Validation AUROC", fontsize=11)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Validation AUROC by Pipeline Stage", fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_val_auroc.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig2_subgroup_auroc(metrics):
    stages = metrics["stages"]
    labels = ["Baseline", "Fine-tuned", "+Synthetic"]
    keys = ["baseline", "finetune", "final_synthetic"]
    light_vals = [stages[k]["Light"]["auroc"] for k in keys]
    dark_vals = [stages[k]["Dark"]["auroc"] for k in keys]

    # Bootstrap CIs for error bars
    bootstrap = metrics.get("bootstrap_ci", {})

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width / 2, light_vals, width, label="Light (Fitzpatrick I\u2013II)",
           color="#5B9BD5", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, dark_vals, width, label="Dark (Fitzpatrick V\u2013VI)",
           color="#ED7D31", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.3, 0.9)
    ax.legend(fontsize=9)
    ax.set_title("Subgroup AUROC Across Pipeline Stages", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_subgroup_auroc.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig3_sensitivity_gap(metrics):
    stages = metrics["stages"]
    labels = ["Baseline", "Fine-tuned", "+Synthetic"]
    keys = ["baseline", "finetune", "final_synthetic"]
    light_sens = [stages[k]["Light"]["sensitivity"] for k in keys]
    dark_sens = [stages[k]["Dark"]["sensitivity"] for k in keys]
    gaps = [d - l for d, l in zip(dark_sens, light_sens)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(labels))
    ax.plot(x, light_sens, "o-", color="#5B9BD5", label="Light Sens.", linewidth=2, markersize=8)
    ax.plot(x, dark_sens, "s-", color="#ED7D31", label="Dark Sens.", linewidth=2, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for i, g in enumerate(gaps):
        ax.annotate(f"{g:+.2f}", (i, dark_sens[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=9)
    ax.set_title("Sensitivity by Skin Tone Across Pipeline Stages", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_sensitivity_gap.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig4_bootstrap_ci(metrics):
    bootstrap = metrics.get("bootstrap_ci", {})
    fig, ax = plt.subplots(figsize=(7, 3.5))

    y_positions = []
    y_labels = []
    y = 0
    stage_info = [
        ("Fine-tuned", "finetuned", "Light"),
        ("Fine-tuned", "finetuned", "Dark"),
        ("+Synthetic", "final", "Light"),
        ("+Synthetic", "final", "Dark"),
    ]
    colors = {"Light": "#5B9BD5", "Dark": "#ED7D31"}

    for stage_label, stage_key, tone in stage_info:
        ci = bootstrap.get(stage_key, {}).get(tone, {})
        point = ci.get("median", 0.5)
        lo = ci.get("lo", 0.3)
        hi = ci.get("hi", 0.7)
        ax.plot([lo, hi], [y, y], color=colors[tone], linewidth=2, solid_capstyle="round")
        ax.plot(point, y, "o", color=colors[tone], markersize=6, zorder=5)
        y_positions.append(y)
        y_labels.append(f"{stage_label} \u2014 {tone}")
        y += 1

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("AUROC (95% Bootstrap CI)", fontsize=10)
    ax.set_title("Bootstrap Confidence Intervals", fontsize=12)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0.2, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_bootstrap_ci.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig5_ablation(ablation_rows):
    if not ablation_rows:
        print("  WARNING: No ablation data, skipping fig5")
        return

    multipliers = []
    test_aurocs = []
    val_aurocs = []
    dark_aurocs = []
    for row in ablation_rows:
        multipliers.append(int(row["multiplier"]))
        test_aurocs.append(float(row["AUROC"]))
        val_aurocs.append(float(row.get("val_auroc", row["AUROC"])))
        dark_aurocs.append(float(row.get("Dark_auroc", row["AUROC"])))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(multipliers))
    width = 0.25

    ax1.bar(x - width, val_aurocs, width, label="Validation AUROC",
            color="#55A868", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.bar(x, test_aurocs, width, label="Test AUROC",
            color="#4C72B0", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.bar(x + width, dark_aurocs, width, label="Dark Test AUROC",
            color="#ED7D31", alpha=0.7, edgecolor="black", linewidth=0.5)

    ax1.set_xlabel("Synthetic Multiplier (x)", fontsize=10)
    ax1.set_ylabel("AUROC", fontsize=10)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m}x" for m in multipliers])
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("Ablation: Saved Synthetic Multiplier", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_ablation_tradeoff.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig6_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)
    stage_configs = [
        ("Baseline", "baseline", "#4C72B0"),
        ("Fine-tuned", "finetune", "#55A868"),
        ("+Synthetic", "augment", "#C44E52"),
    ]
    for ax, (label, stage, color) in zip(axes, stage_configs):
        hist = load_training_history(stage)
        if hist is None:
            continue
        ax.plot(hist["epochs"], hist["train_loss"], "-", color=color, alpha=0.7, label="Train loss")
        ax.plot(hist["epochs"], hist["val_auroc"], "--", color=color, linewidth=1.5, label="Val AUROC")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="center right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Loss / Val AUROC", fontsize=10)
    plt.suptitle("Training Curves by Pipeline Stage", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_training_curves.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def fig_real_vs_synthetic():
    import cv2

    ddi_dir = os.path.join(ROOT, "ddidiversedermatologyimages")
    syn_dir = os.path.join(ROOT, "data", "synthetic_train_only")
    split_path = os.path.join(ROOT, "splits", "ddi_split_seed42.json")

    if not os.path.exists(split_path) or not os.path.exists(syn_dir):
        print("  WARNING: Split or synthetic dir not found, skipping real vs synthetic figure")
        return

    with open(split_path) as f:
        split = json.load(f)

    import pandas as pd
    meta = pd.read_csv(os.path.join(ddi_dir, "ddi_metadata.csv"))
    train_dark_mel = meta[meta["DDI_file"].isin(split["train_dark_mel"])]
    sample_files = train_dark_mel["DDI_file"].tolist()[:3]

    syn_files = sorted([f for f in os.listdir(syn_dir) if f.endswith((".jpg", ".png"))])[:3]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, f in enumerate(sample_files):
        img = cv2.imread(os.path.join(ddi_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Real: {f}", fontsize=9)
        axes[0, i].axis("off")

    for i, f in enumerate(syn_files):
        img = cv2.imread(os.path.join(syn_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Synthetic: {f}", fontsize=9)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real (Train)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Synthetic", fontsize=11, fontweight="bold")
    plt.suptitle("Dark-Skin Melanoma: Real vs. Synthetic", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_real_vs_synthetic.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    metrics = load_metrics()
    ablation_rows = load_ablation_report()

    print("Generating PDF figures ...")
    fig1_val_auroc_comparison(metrics)
    fig2_subgroup_auroc(metrics)
    fig3_sensitivity_gap(metrics)
    fig4_bootstrap_ci(metrics)
    fig5_ablation(ablation_rows)
    fig6_training_curves()
    fig_real_vs_synthetic()
    print("Done.")


if __name__ == "__main__":
    main()

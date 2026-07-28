#!/usr/bin/env python3
"""
FairDerm: Evaluating and Mitigating Skin-Tone Bias in Melanoma Detection

Local Mac-compatible version. Run individual stages via CLI:
    python fairderm.py --stage sanity       # verify imports, data, device, forward pass
    python fairderm.py --stage baseline     # HAM10000 baseline training
    python fairderm.py --stage finetune     # DDI fine-tuning
    python fairderm.py --stage augment      # synthetic generation + augmented retraining
    python fairderm.py --stage evaluate     # fairness audit + plots + paper_data.json
    python fairderm.py --stage ablation     # 0x / 2x / 5x / 10x ablation study
"""

import os
import sys
import time
import random
import json
import glob
import shutil
import zipfile
import argparse
import platform
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import albumentations as A
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "IMG_SIZE": (224, 224),
    "BATCH_SIZE": 8,
    "LR": 1e-4,
    "EPOCHS": 15,
    "SEED": 42,
    "DATA_DIR": os.path.join(PROJECT_ROOT, "data"),
    "CHECKPOINT_DIR": os.path.join(PROJECT_ROOT, "models"),
    "RESULT_DIR": os.path.join(PROJECT_ROOT, "results"),
}


def load_yaml_config():
    """Load configs/config.yaml."""
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    print(f"  WARNING: {cfg_path} not found, using hardcoded defaults")
    return None

YAML_CFG = load_yaml_config()


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeeLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._file = None
        self._original_stdout = None

    def __enter__(self):
        self._file = open(self.log_path, "w", encoding="utf-8")
        self._original_stdout = sys.stdout
        sys.stdout = self
        print(f"  Log file: {self.log_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        if self._file:
            self._file.close()
            self._file = None
        return False

    def write(self, msg):
        if self._original_stdout:
            self._original_stdout.write(msg)
        if self._file:
            self._file.write(msg)
            self._file.flush()

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()
        if self._file:
            self._file.flush()


def get_reproducibility_metadata():
    meta = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "python_version": sys.version,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "pytorch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "seed": CONFIG["SEED"],
        "command_line_args": sys.argv,
        "working_directory": os.getcwd(),
        "git": None,
    }
    return meta


train_transform = transforms.Compose([
    transforms.Resize(CONFIG["IMG_SIZE"]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize(CONFIG["IMG_SIZE"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DermDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_ddi=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_ddi = is_ddi

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["DDI_file"] if self.is_ddi else f"{row['image_id']}.jpg"
        img_path = os.path.join(self.img_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        fst = int(row.get("skin_tone", -1))
        return img, label, fst


class SyntheticDataset(Dataset):
    def __init__(self, syn_dir):
        self.files = sorted([
            os.path.join(syn_dir, f)
            for f in os.listdir(syn_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(CONFIG["IMG_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tfm(img), torch.tensor(1.0, dtype=torch.float32), 56


def setup_dataloaders(df, img_dir, transform, is_ddi=False, batch_size=8, shuffle=True):
    ds = DermDataset(df, img_dir, transform=transform, is_ddi=is_ddi)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def create_model(model_name="efficientnet_b0", pretrained=True):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model.to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=15, patience=4, save_path="model.pth",
                dry_run=False, experiment_dir=None, resume=False):
    best_val_auc = 0.0
    early_stop_count = 0
    history = []
    start_epoch = 0

    if dry_run:
        print("  [DRY RUN] Running 1 forward pass only (no training) ...")
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"  [DRY RUN] Model params: {total_params:,} total | "
              f"{trainable_params:,} trainable | {frozen_params:,} frozen")

        print(f"  [DRY RUN] Optimizer: {optimizer.__class__.__name__}")
        for i, pg in enumerate(optimizer.param_groups):
            print(f"  [DRY RUN]   param_group[{i}]: lr={pg['lr']:.2e}, "
                  f"weight_decay={pg.get('weight_decay', 'default')}")

        x, y, _ = next(iter(train_loader))
        x, y = x.to(device), y.to(device).unsqueeze(1)
        with torch.no_grad():
            loss = criterion(model(x), y)
        print(f"  [DRY RUN] Sample batch loss: {loss.item():.4f}")
        print(f"  [DRY RUN] Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)}")
        print("  [DRY RUN] Skipping training loop.")
        return model, history

    if resume and experiment_dir:
        last_ckpt_path = os.path.join(experiment_dir, "last_checkpoint.pth")
        if os.path.exists(last_ckpt_path):
            ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_val_auc = ckpt["best_val_auc"]
            history = ckpt["history"]
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Best validation AUROC: {best_val_auc:.4f}")
        else:
            print(f"  No checkpoint found at {last_ckpt_path} — starting fresh.")

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            model.train()
            train_losses = []
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_probs, val_labels, _ = get_preds(model, val_loader, device)
            val_auc = roc_auc_score(val_labels, val_probs)
            avg_train_loss = float(np.mean(train_losses))
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start
            remaining = (num_epochs - epoch - 1) * epoch_time
            r_min, r_sec = divmod(int(remaining), 60)
            print(f"  Epoch {epoch + 1}/{num_epochs}: Loss {avg_train_loss:.4f} | "
                  f"Val AUC {val_auc:.4f}")
            print(f"    Time: {epoch_time:.1f}s | ETA: {r_min}m {r_sec:02d}s")

            history.append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_auroc": round(float(val_auc), 6),
                "lr": current_lr,
                "early_stop_count": early_stop_count,
            })

            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), save_path)
                early_stop_count = 0
                print(f"    -> Saved best model (AUROC={best_val_auc:.4f})")
            else:
                early_stop_count += 1

            if experiment_dir:
                last_ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_auc": best_val_auc,
                    "history": history,
                    "config": CONFIG,
                    "seed": CONFIG["SEED"],
                }
                torch.save(last_ckpt, os.path.join(experiment_dir, "last_checkpoint.pth"))

            if early_stop_count >= patience:
                print("  Early stopping triggered.")
                break

    except Exception as e:
        print(f"\n  [CRASH] Training interrupted at epoch {epoch + 1}")
        print(f"  [CRASH] Error: {e}")
        traceback.print_exc()
        if experiment_dir:
            crash_ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_auc": best_val_auc,
                "history": history,
                "config": CONFIG,
                "seed": CONFIG["SEED"],
            }
            crash_path = os.path.join(experiment_dir, "crash_checkpoint.pth")
            torch.save(crash_ckpt, crash_path)
            print(f"  [CRASH] Saved checkpoint to {crash_path}")
        raise

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, history


def get_preds(model, loader, dev):
    model.eval()
    all_probs, all_labels, all_fsts = [], [], []
    with torch.no_grad():
        for x, y, fst in loader:
            logits = model(x.to(dev))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
            all_fsts.extend(fst.numpy())
    return np.array(all_probs).flatten(), np.array(all_labels), np.array(all_fsts)


def compute_medical_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan
    return {
        "AUROC": auroc,
        "Sens": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Spec": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "Brier": brier_score_loss(y_true, y_prob),
    }


def compute_metrics(y_true, y_prob, threshold=0.5):
    metrics = compute_medical_metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["F1"] = f1
    return metrics


def run_bootstrap_audit(y_true, y_prob, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        stats.append(compute_medical_metrics(y_true[indices], y_prob[indices]))
    df_boot = pd.DataFrame(stats)
    summary = {}
    for col in df_boot.columns:
        mean = df_boot[col].mean()
        lo = np.percentile(df_boot[col], 2.5)
        hi = np.percentile(df_boot[col], 97.5)
        summary[col] = f"{mean:.3f} ({lo:.3f}-{hi:.3f})"
    return summary


def bootstrap_metrics(y_true, y_prob, n_bootstrap=1000):
    all_stats = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        y_t, y_p = y_true[idx], y_prob[idx]
        y_pred = (y_p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        all_stats.append({
            "AUROC": roc_auc_score(y_t, y_p),
            "Sens": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Spec": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1": precision_recall_fscore_support(
                y_t, y_pred, average="binary", zero_division=0
            )[2],
            "Brier": brier_score_loss(y_t, y_p),
        })
    df_bs = pd.DataFrame(all_stats)
    return {
        col: f"{df_bs[col].mean():.3f} ({np.percentile(df_bs[col], 2.5):.3f}-{np.percentile(df_bs[col], 97.5):.3f})"
        for col in df_bs.columns
    }


def bootstrap_ci(y_true, y_score):
    rng = np.random.default_rng(0)
    scores = []
    arr_true, arr_score = np.array(y_true), np.array(y_score)
    for _ in range(1000):
        idx = rng.choice(len(arr_true), len(arr_true), replace=True)
        if len(set(arr_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(arr_true[idx], arr_score[idx]))
    lo, med, hi = np.percentile(scores, [2.5, 50, 97.5])
    return f"{med:.3f} [{lo:.3f}-{hi:.3f}]"


def bootstrap_auroc(y_true, y_score, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    arr_true, arr_score = np.array(y_true), np.array(y_score)
    for _ in range(n):
        idx = rng.choice(len(arr_true), len(arr_true), replace=True)
        if len(np.unique(arr_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(arr_true[idx], arr_score[idx]))
    lo, med, hi = np.percentile(aucs, [2.5, 50, 97.5])
    return float(med), float(lo), float(hi)


def compute_youden_threshold(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def perform_fairness_analysis(y_true, y_prob, fsts, group_map=None):
    if group_map is None:
        groups = {12: "Light", 34: "Medium", 56: "Dark"}
    else:
        groups = group_map
    results = {}
    for code, label in groups.items():
        mask = np.isin(fsts, code)
        if mask.any():
            results[label] = run_bootstrap_audit(y_true[mask], y_prob[mask])
    return pd.DataFrame(results).T


fairness_audit = perform_fairness_analysis


def plot_reliability_curve(y_true, y_prob, label):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability)")
    plt.legend()


def plot_final_results(results_df):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for group in results_df["group"].unique():
        sub = results_df[results_df["group"] == group]
        fpr, tpr, _ = roc_curve(sub["label"], sub["pred"])
        auroc_val = roc_auc_score(sub["label"], sub["pred"])
        plt.plot(fpr, tpr, label=f"{group} (AUC={auroc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve by Skin Tone Group")
    plt.legend()

    plt.subplot(1, 2, 2)
    for group in results_df["group"].unique():
        sub = results_df[results_df["group"] == group]
        plot_reliability_curve(sub["label"], sub["pred"], group)
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["RESULT_DIR"], "roc_calibration.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved {CONFIG['RESULT_DIR']}/roc_calibration.png")


def get_metrics(df, group_name, thresh):
    sub = df[df["group"] == group_name]
    if len(sub) == 0:
        return {}
    fpr, tpr, _ = roc_curve(sub["label"], sub["pred"])
    auroc_val = auc(fpr, tpr)
    pred_bin = (sub["pred"] > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(sub["label"], pred_bin, labels=[0, 1]).ravel()
    return {
        "n_total": int(len(sub)),
        "n_melanoma": int(sub["label"].sum()),
        "auroc": round(float(auroc_val), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0,
        "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
        "ppv": round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0,
        "thresh": float(thresh),
    }


def evaluate_checkpoint(path, val_loader, group_map=None):
    if group_map is None:
        group_map = {12: "Light", 34: "Medium", 56: "Dark"}
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    p, l, f = get_preds(model, val_loader, device)
    df = pd.DataFrame({"pred": p, "label": l, "fst": f})
    df["group"] = df["fst"].map(group_map)
    return df


def setup_ddi_data():
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "ddi"),
        os.path.join(PROJECT_ROOT, "ddidiversedermatologyimages"),
    ]
    for ddi_dir in candidates:
        meta_path = os.path.join(ddi_dir, "ddi_metadata.csv")
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path)
            df["label"] = df["malignant"].astype(int)
            print(f"  DDI: loaded {len(df)} samples from {ddi_dir}")
            return df, ddi_dir
    raise FileNotFoundError(
        "ddi_metadata.csv not found. Check data/ddi/ or ddidiversedermatologyimages/."
    )


def _extract_ham10000(zip_path, target_dir):
    print(f"  Extracting {zip_path} ...")
    temp_dir = os.path.join(target_dir, "_temp_extract")
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    images_dir = os.path.join(target_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for part in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]:
        part_path = os.path.join(temp_dir, part)
        if os.path.exists(part_path):
            print(f"  Extracting {part} ...")
            with zipfile.ZipFile(part_path, "r") as z:
                z.extractall(images_dir)

    meta_src = os.path.join(temp_dir, "HAM10000_metadata")
    if not os.path.exists(meta_src):
        found = glob.glob(os.path.join(temp_dir, "**", "HAM10000_metadata*"), recursive=True)
        if found:
            meta_src = found[0]
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(target_dir, "HAM10000_metadata.csv"))

    shutil.rmtree(temp_dir)
    print(f"  HAM10000 extracted to {target_dir}")


def setup_ham10000_data():
    ham_dir = os.path.join(PROJECT_ROOT, "data", "ham10000")
    meta_path = os.path.join(ham_dir, "HAM10000_metadata.csv")
    img_dir = os.path.join(ham_dir, "images")

    if not os.path.exists(meta_path) or not os.path.isdir(img_dir):
        raise FileNotFoundError(
            "HAM10000 data not found locally.\n"
            "  Expected: data/ham10000/HAM10000_metadata.csv\n"
            "            data/ham10000/images/"
        )

    df = pd.read_csv(meta_path)
    ham_df = df[df["dx"].isin(["mel", "nv"])].copy()
    ham_df["label"] = (ham_df["dx"] == "mel").astype(int)
    print(f"  HAM10000: loaded {len(ham_df)} samples (mel/nv) from {ham_dir}")
    return ham_df, img_dir


def run_ablation_study(base_model_path, train_df, val_df, test_df, test_loader,
                       ddi_img_dir, syn_dir, multipliers=(0, 2, 5, 10),
                       dry_run=False, resume=False):
    results_log = []

    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0

    # Load all synthetic images available
    all_syn_files = sorted([
        f for f in os.listdir(syn_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]) if os.path.exists(syn_dir) else []
    total_synth = len(all_syn_files)
    print(f"  Total synthetic images available: {total_synth}")

    for m in multipliers:
        print(f"\n--- Training Multiplier: {m}x ---")
        model = create_model(pretrained=False)
        model.load_state_dict(torch.load(base_model_path, weights_only=True))

        # Calculate how many synthetic images to use for this multiplier
        # 0x: 0 synth, 2x: ~58, 5x: ~145, 10x: ~290 (all available)
        n_synth = min(int(m * 29), total_synth) if m > 0 else 0  # 29 = train dark mel count
        syn_subset_dir = None

        if n_synth > 0 and total_synth > 0:
            # Select first n_synth synthetic images
            selected_syn = all_syn_files[:n_synth]
            syn_subset_dir = os.path.join(syn_dir, f"_ablation_{m}x")
            os.makedirs(syn_subset_dir, exist_ok=True)
            for sf in selected_syn:
                src = os.path.join(syn_dir, sf)
                dst = os.path.join(syn_subset_dir, sf)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

            base_ds = DermDataset(train_df, ddi_img_dir, train_transform, True)
            syn_ds = SyntheticDataset(syn_subset_dir)
            combined = ConcatDataset([base_ds, syn_ds])
        else:
            combined = DermDataset(train_df, ddi_img_dir, train_transform, True)

        train_loader = DataLoader(
            combined, batch_size=32, shuffle=True, num_workers=0,
        )
        val_loader_local = DataLoader(
            DermDataset(val_df, ddi_img_dir, val_test_transform, True),
            batch_size=32, shuffle=False, num_workers=0,
        )

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5, weight_decay=1e-4,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        m_dir = os.path.join(CONFIG["RESULT_DIR"], "ablation", f"{m}x")
        os.makedirs(m_dir, exist_ok=True)

        save_path = os.path.join(CONFIG["CHECKPOINT_DIR"], f"ablation_{m}x.pth")
        with TeeLogger(os.path.join(m_dir, "train.log")):
            model, history = train_model(
                model, train_loader, val_loader_local, criterion, optimizer, scheduler,
                num_epochs=5, patience=3, save_path=save_path, dry_run=dry_run,
                experiment_dir=m_dir, resume=resume,
            )

        probs, labels, fsts = get_preds(model, test_loader, device)
        overall = compute_metrics(labels, probs)
        overall["multiplier"] = m

        test_df_copy = test_df.copy()
        test_df_copy["pred"] = probs
        test_df_copy["group"] = test_df_copy["skin_tone"].map(DDI_GROUP_MAP)
        for grp in ["Light", "Dark"]:
            sub = test_df_copy[test_df_copy["group"] == grp]
            if len(sub) > 0:
                grp_auroc = roc_auc_score(sub["label"], sub["pred"]) if len(sub["label"].unique()) > 1 else 0.5
                overall[f"{grp}_auroc"] = round(grp_auroc, 4)
            else:
                overall[f"{grp}_auroc"] = None

        val_probs, val_labels, _ = get_preds(model, val_loader_local, device)
        val_auroc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        overall["val_auroc"] = round(val_auroc, 4)

        results_log.append(overall)

        cfg = build_experiment_config(
            f"ablation_{m}x", train_df, val_df,
            lr=5e-5, epochs=5, batch_size=32,
            pos_weight=round(pos_weight_val, 4), weight_decay=1e-4,
            multiplier=m, n_synth_used=n_synth,
            combined_size=len(combined),
        )
        save_experiment(m_dir, cfg, history, checkpoint_path=save_path, dry_run=dry_run)

        if syn_subset_dir and os.path.exists(syn_subset_dir):
            shutil.rmtree(syn_subset_dir, ignore_errors=True)

    return pd.DataFrame(results_log).set_index("multiplier")


DDI_GROUP_MAP = {12: "Light", 34: "Medium", 56: "Dark"}


def _make_splits(ddi_df):
    ddi_df = ddi_df.copy()
    ddi_df["stratify_key"] = ddi_df["label"].astype(str) + "_" + ddi_df["skin_tone"].astype(str)
    train_val_df, test_df = train_test_split(
        ddi_df, test_size=0.2, stratify=ddi_df["stratify_key"], random_state=CONFIG["SEED"],
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, stratify=train_val_df["stratify_key"],
        random_state=CONFIG["SEED"],
    )
    assert set(train_df["DDI_file"]).isdisjoint(set(val_df["DDI_file"]))
    assert set(train_df["DDI_file"]).isdisjoint(set(test_df["DDI_file"]))
    assert set(val_df["DDI_file"]).isdisjoint(set(test_df["DDI_file"]))

    # Save split to JSON
    splits_dir = os.path.join(PROJECT_ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    split_path = os.path.join(splits_dir, "ddi_split_seed42.json")
    if not os.path.exists(split_path):
        train_dark_mel = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 1)]
        split_data = {
            "seed": CONFIG["SEED"],
            "stratify": "label x skin_tone",
            "train": sorted(train_df["DDI_file"].tolist()),
            "val": sorted(val_df["DDI_file"].tolist()),
            "test": sorted(test_df["DDI_file"].tolist()),
            "train_dark_mel": sorted(train_dark_mel["DDI_file"].tolist()),
            "counts": {
                "train": len(train_df), "val": len(val_df), "test": len(test_df),
                "train_dark_mel": len(train_dark_mel),
            },
        }
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved split to {split_path}")

    print(f"  Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df



def print_training_summary(train_df, val_df, test_df=None, batch_size=8, lr=1e-4,
                           is_ddi=True, extra_info=None):
    print("\n  Dataset:")
    print(f"    Train:      {len(train_df)} samples")
    print(f"    Validation: {len(val_df)} samples")
    if test_df is not None:
        print(f"    Test:       {len(test_df)} samples")

    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    print(f"\n  Class balance (train):")
    print(f"    Malignant: {n_pos} ({100 * n_pos / len(train_df):.1f}%)")
    print(f"    Benign:    {n_neg} ({100 * n_neg / len(train_df):.1f}%)")

    if is_ddi and "skin_tone" in train_df.columns:
        print(f"\n  Skin tone distribution (train):")
        for code, name in DDI_GROUP_MAP.items():
            n = int((train_df["skin_tone"] == code).sum())
            print(f"    {name:8s}: {n} ({100 * n / len(train_df):.1f}%)")

    if extra_info:
        for k, v in extra_info.items():
            print(f"    {k}: {v}")

    print(f"\n  Device:     {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print()


def build_experiment_config(stage_name, train_df, val_df, test_df=None, **kwargs):
    cfg = {
        "stage": stage_name,
        "seed": CONFIG["SEED"],
        "device": str(device),
        "img_size": CONFIG["IMG_SIZE"],
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "reproducibility": get_reproducibility_metadata(),
    }
    if test_df is not None:
        cfg["test_samples"] = len(test_df)
    cfg.update(kwargs)
    return cfg


def save_experiment(experiment_dir, config_dict, history_rows, checkpoint_path=None, dry_run=False):
    os.makedirs(experiment_dir, exist_ok=True)

    if history_rows:
        best = max(history_rows, key=lambda r: r.get("val_auroc", -1))
        config_dict["final_best_val_auroc"] = best.get("val_auroc")
        config_dict["total_epochs_trained"] = len(history_rows)

    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    if dry_run:
        config_dict["dry_run"] = True
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"  [DRY RUN] Config saved to {experiment_dir}/ (no training data)")
        return

    if history_rows:
        hist_df = pd.DataFrame(history_rows)
        hist_df.to_csv(os.path.join(experiment_dir, "training_history.csv"), index=False)

        best = max(history_rows, key=lambda r: r.get("val_auroc", -1))
        metrics_df = pd.DataFrame([best])
        metrics_df.to_csv(os.path.join(experiment_dir, "metrics.csv"), index=False)

    if checkpoint_path and os.path.exists(checkpoint_path):
        dst = os.path.join(experiment_dir, os.path.basename(checkpoint_path))
        shutil.copy2(checkpoint_path, dst)

    print(f"  Experiment saved to {experiment_dir}/")


def stage_sanity():
    print("=" * 60)
    print("FAIRDERM — SANITY CHECK")
    print("=" * 60)

    print(f"\n[1] Device: {device}")

    print("\n[2] Checking data ...")
    ddi_loaded = False
    ham_loaded = False
    try:
        ddi_df, ddi_dir = setup_ddi_data()
        print(f"    DDI OK — {len(ddi_df)} samples, images in {ddi_dir}")
        ddi_loaded = True
    except FileNotFoundError as exc:
        print(f"    DDI SKIPPED — {exc}")

    try:
        ham_df, ham_dir = setup_ham10000_data()
        print(f"    HAM10000 OK — {len(ham_df)} samples, images in {ham_dir}")
        ham_loaded = True
    except FileNotFoundError as exc:
        print(f"    HAM10000 SKIPPED — {exc}")

    print("\n[3] Verifying image loading & transforms ...")
    if ddi_loaded:
        sample_row = ddi_df.iloc[0]
        img_path = os.path.join(ddi_dir, sample_row["DDI_file"])
        img = Image.open(img_path).convert("RGB")
        tensor = val_test_transform(img)
        print(f"    DDI image: {sample_row['DDI_file']} -> tensor {tensor.shape}, "
              f"range [{tensor.min():.3f}, {tensor.max():.3f}]")
    if ham_loaded:
        sample_row = ham_df.iloc[0]
        img_path = os.path.join(ham_dir, "images", f"{sample_row['image_id']}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            tensor = val_test_transform(img)
            print(f"    HAM image: {sample_row['image_id']}.jpg -> tensor {tensor.shape}, "
                  f"range [{tensor.min():.3f}, {tensor.max():.3f}]")
        else:
            print(f"    HAM image: {sample_row['image_id']}.jpg — FILE NOT FOUND")

    print("\n[4] Initializing EfficientNet-B0 ...")
    model = create_model(pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model created on {device} — {total_params:,} parameters")

    print("\n[5] Forward pass with dummy input ...")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"    Output shape: {out.shape}  value: {out.item():.4f}")

    print("\n[6] Checking output directories ...")
    for d in ["models", "results"]:
        path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(path, exist_ok=True)
        print(f"    {d}/ OK")

    print("\n" + "=" * 60)
    print("SANITY CHECK PASSED")
    print("=" * 60)


def stage_baseline(dry_run=False, num_epochs=10, batch_size=8, resume=False):
    print("=" * 60)
    print("STAGE: HAM10000 BASELINE TRAINING")
    print("=" * 60)

    exp_dir = os.path.join(CONFIG["RESULT_DIR"], "baseline")
    os.makedirs(exp_dir, exist_ok=True)

    with TeeLogger(os.path.join(exp_dir, "train.log")):
        set_seed(CONFIG["SEED"])
        ham_df, img_dir = setup_ham10000_data()

        existing = {f.replace(".jpg", "") for f in os.listdir(img_dir) if f.endswith(".jpg")}
        ham_df = ham_df[ham_df["image_id"].isin(existing)].copy()
        print(f"  {len(ham_df)} images found on disk")

        ham_train_df, ham_val_df = train_test_split(
            ham_df, test_size=0.2, stratify=ham_df["label"], random_state=CONFIG["SEED"],
        )

        print_training_summary(
            ham_train_df, ham_val_df, batch_size=batch_size, lr=3e-4, is_ddi=False,
        )

        train_loader = setup_dataloaders(ham_train_df, img_dir, train_transform, batch_size=batch_size)
        val_loader = setup_dataloaders(
            ham_val_df, img_dir, val_test_transform, batch_size=batch_size, shuffle=False,
        )

        model = create_model(pretrained=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

        save_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_ham_baseline.pth")
        print(f"\n  Training {num_epochs} epochs -> {save_path}")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, patience=3, save_path=save_path,
            dry_run=dry_run, experiment_dir=exp_dir, resume=resume,
        )

        cfg = build_experiment_config(
            "baseline", ham_train_df, ham_val_df,
            lr=3e-4, epochs=num_epochs, batch_size=batch_size, dataset="HAM10000",
        )
        save_experiment(exp_dir, cfg, history, checkpoint_path=save_path, dry_run=dry_run)
        print("\nHAM10000 baseline complete.")


def stage_finetune(dry_run=False, num_epochs=10, batch_size=32, resume=False):
    print("=" * 60)
    print("STAGE: DDI FINE-TUNING")
    print("=" * 60)

    exp_dir = os.path.join(CONFIG["RESULT_DIR"], "finetune")
    os.makedirs(exp_dir, exist_ok=True)

    with TeeLogger(os.path.join(exp_dir, "train.log")):
        set_seed(CONFIG["SEED"])
        ddi_df, ddi_img_dir = setup_ddi_data()
        train_df, val_df, test_df = _make_splits(ddi_df)

        n_pos = int(train_df["label"].sum())
        n_neg = len(train_df) - n_pos
        pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  Dynamic pos_weight: {pos_weight_val:.2f} (n_neg={n_neg}, n_pos={n_pos})")

        print_training_summary(
            train_df, val_df, test_df, batch_size=batch_size, lr=5e-5, is_ddi=True,
        )

        train_loader = setup_dataloaders(train_df, ddi_img_dir, train_transform, is_ddi=True, batch_size=batch_size)
        val_loader = setup_dataloaders(
            val_df, ddi_img_dir, val_test_transform, is_ddi=True, batch_size=batch_size, shuffle=False,
        )

        model = create_model(pretrained=True)
        baseline_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_ham_baseline.pth")
        if os.path.exists(baseline_path):
            model.load_state_dict(torch.load(baseline_path, weights_only=True))
            print(f"  Loaded baseline weights from {baseline_path}")
        else:
            print(f"  WARNING: {baseline_path} not found — training from ImageNet weights.")

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "features.6" in name or "features.7" in name or "classifier" in name:
                param.requires_grad = True

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

        save_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_ddi_finetuned.pth")
        print(f"\n  Fine-tuning {num_epochs} epochs -> {save_path}")
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, patience=3, save_path=save_path,
            dry_run=dry_run, experiment_dir=exp_dir, resume=resume,
        )

        cfg = build_experiment_config(
            "finetune", train_df, val_df, test_df,
            lr=5e-5, epochs=num_epochs, batch_size=batch_size,
            pos_weight=round(pos_weight_val, 4), weight_decay=1e-4,
            freeze_layers="features.6,features.7,classifier",
        )
        save_experiment(exp_dir, cfg, history, checkpoint_path=save_path, dry_run=dry_run)
        print("\nDDI fine-tuning complete.")


def stage_augment(dry_run=False, num_epochs=5, batch_size=32, resume=False):
    print("=" * 60)
    print("STAGE: SYNTHETIC AUGMENTATION")
    print("=" * 60)

    exp_dir = os.path.join(CONFIG["RESULT_DIR"], "augment")
    os.makedirs(exp_dir, exist_ok=True)

    with TeeLogger(os.path.join(exp_dir, "train.log")):
        set_seed(CONFIG["SEED"])
        ddi_df, ddi_img_dir = setup_ddi_data()
        train_df, val_df, _ = _make_splits(ddi_df)

        syn_dir = os.path.join(CONFIG["DATA_DIR"], "synthetic_train_only")
        os.makedirs(syn_dir, exist_ok=True)

        dark_mel = train_df[(train_df["skin_tone"] == 56) & (train_df["malignant"] == 1)]
        print(f"  Found {len(dark_mel)} TRAIN dark-skin melanomas (source for synthetics)")

        existing_syn = [f for f in os.listdir(syn_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if len(existing_syn) == 0 and not dry_run:
            alb_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.7),
                A.GaussNoise(p=0.2),
                A.ElasticTransform(alpha=0.5, sigma=5, p=0.1),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(224, 224),
            ])
            count = 0
            for _, row in tqdm(dark_mel.iterrows(), total=len(dark_mel), desc="Generating"):
                img_path = os.path.join(ddi_img_dir, row["DDI_file"])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for _ in range(10):
                    aug = alb_transform(image=img)["image"]
                    out_path = os.path.join(syn_dir, f"syn_dark_{count:04d}.jpg")
                    cv2.imwrite(out_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
                    count += 1
            print(f"  Generated {count} synthetic dark-skin melanomas in {syn_dir}")
        else:
            print(f"  Synthetic images already exist ({len(existing_syn)} files)")

        n_pos = int(train_df["label"].sum())
        n_neg = len(train_df) - n_pos
        pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  Dynamic pos_weight: {pos_weight_val:.2f} (n_neg={n_neg}, n_pos={n_pos})")

        model = create_model(pretrained=True)
        finetuned_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_ddi_finetuned.pth")
        if os.path.exists(finetuned_path):
            model.load_state_dict(torch.load(finetuned_path, weights_only=True))
            print(f"  Loaded finetuned weights from {finetuned_path}")
        else:
            print(f"  WARNING: {finetuned_path} not found — training from ImageNet weights.")

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "features.6" in name or "features.7" in name or "classifier" in name:
                param.requires_grad = True

        base_ds = DermDataset(train_df, ddi_img_dir, transform=train_transform, is_ddi=True)
        syn_ds = SyntheticDataset(syn_dir)
        combined_train = ConcatDataset([base_ds, syn_ds])
        combined_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = setup_dataloaders(
            val_df, ddi_img_dir, val_test_transform, is_ddi=True, batch_size=batch_size, shuffle=False,
        )

        print_training_summary(
            train_df, val_df, batch_size=batch_size, lr=5e-5, is_ddi=True,
            extra_info={"Synthetic images": len(syn_ds), "Combined train": len(combined_train)},
        )

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))

        save_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_final.pth")
        print(f"\n  Training {num_epochs} epochs on combined data ({len(combined_train)} samples) ...")
        model, history = train_model(
            model, combined_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, patience=3, save_path=save_path,
            dry_run=dry_run, experiment_dir=exp_dir, resume=resume,
        )

        cfg = build_experiment_config(
            "augment", train_df, val_df,
            lr=5e-5, epochs=num_epochs, batch_size=batch_size,
            pos_weight=round(pos_weight_val, 4), weight_decay=1e-4,
            synthetic_images=len(syn_ds), synthetic_source="train_dark_mel_only",
        )
        save_experiment(exp_dir, cfg, history, checkpoint_path=save_path, dry_run=dry_run)
        print("Synthetic augmentation complete.")


def stage_evaluate():
    print("=" * 60)
    print("STAGE: EVALUATION & FAIRNESS AUDIT")
    print("=" * 60)

    set_seed(CONFIG["SEED"])
    ddi_df, ddi_img_dir = setup_ddi_data()
    train_df, val_df, test_df = _make_splits(ddi_df)

    val_loader = setup_dataloaders(
        val_df, ddi_img_dir, val_test_transform, is_ddi=True, batch_size=32, shuffle=False,
    )
    test_loader = setup_dataloaders(
        test_df, ddi_img_dir, val_test_transform, is_ddi=True, batch_size=32, shuffle=False,
    )

    ckpt_dir = CONFIG["CHECKPOINT_DIR"]
    result_dir = os.path.join(CONFIG["RESULT_DIR"], "evaluate")
    os.makedirs(result_dir, exist_ok=True)

    checkpoints = {
        "baseline": os.path.join(ckpt_dir, "fairderm_ham_baseline.pth"),
        "finetuned": os.path.join(ckpt_dir, "fairderm_ddi_finetuned.pth"),
        "final": os.path.join(ckpt_dir, "fairderm_final.pth"),
    }

    val_results = {}
    test_results = {}
    for stage_name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"\n  SKIP {stage_name} — {path} not found")
            continue
        print(f"\n  Evaluating {stage_name} ...")
        model = create_model(pretrained=False)
        model.load_state_dict(torch.load(path, weights_only=True))

        v_probs, v_labels, v_fsts = get_preds(model, val_loader, device)
        t_probs, t_labels, t_fsts = get_preds(model, test_loader, device)

        t_metrics = compute_metrics(t_labels, t_probs)
        print(f"    [test] AUROC={t_metrics['AUROC']:.3f}  Sens={t_metrics['Sens']:.3f}  "
              f"Spec={t_metrics['Spec']:.3f}  F1={t_metrics['F1']:.3f}")

        val_results[stage_name] = pd.DataFrame(
            {"pred": v_probs, "label": v_labels, "fst": v_fsts}
        )
        test_results[stage_name] = pd.DataFrame(
            {"pred": t_probs, "label": t_labels, "fst": t_fsts}
        )

    thresholds = {}
    for name in ["baseline", "finetuned", "final"]:
        if name not in val_results:
            continue
        vr = val_results[name]
        if name == "baseline":
            thresholds[name] = 0.5
            print(f"  {name}: threshold = 0.5 (fixed)")
        else:
            thr = compute_youden_threshold(vr["label"].values, vr["pred"].values)
            thresholds[name] = thr
            print(f"  {name}: threshold = {thr:.4f} (Youden's J on validation)")

    all_bootstrap = {}
    for name in ["finetuned", "final"]:
        if name not in test_results:
            continue
        df = test_results[name].copy()
        df["group"] = df["fst"].map(DDI_GROUP_MAP)
        all_bootstrap[name] = {}
        print(f"\n  {name.title()} — Bootstrap CIs (test):")
        for grp in ["Light", "Dark"]:
            sub = df[df["group"] == grp]
            if len(sub) > 0:
                med, lo, hi = bootstrap_auroc(sub["label"].values, sub["pred"].values, n=1000, seed=0)
                all_bootstrap[name][grp] = {"median": med, "lo": lo, "hi": hi}
                print(f"    {grp} n={len(sub)} -> {med:.3f} [{lo:.3f}-{hi:.3f}]")

    p_value_dark = None
    if "finetuned" in all_bootstrap and "final" in all_bootstrap:
        dark_ft = test_results["finetuned"].copy()
        dark_ft["group"] = dark_ft["fst"].map(DDI_GROUP_MAP)
        dark_ft = dark_ft[dark_ft["group"] == "Dark"]
        dark_fn = test_results["final"].copy()
        dark_fn["group"] = dark_fn["fst"].map(DDI_GROUP_MAP)
        dark_fn = dark_fn[dark_fn["group"] == "Dark"]

        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(1000):
            idx_ft = rng.choice(len(dark_ft), len(dark_ft), replace=True)
            idx_fn = rng.choice(len(dark_fn), len(dark_fn), replace=True)
            if len(np.unique(dark_ft["label"].values[idx_ft])) < 2:
                continue
            if len(np.unique(dark_fn["label"].values[idx_fn])) < 2:
                continue
            auc_ft = roc_auc_score(dark_ft["label"].values[idx_ft], dark_ft["pred"].values[idx_ft])
            auc_fn = roc_auc_score(dark_fn["label"].values[idx_fn], dark_fn["pred"].values[idx_fn])
            diffs.append(auc_fn - auc_ft)
        p_value_dark = float(np.mean(np.array(diffs) <= 0))
        print(f"\n  p-value for Dark AUROC improvement (finetuned -> augment): {p_value_dark:.4f}")

    if len(test_results) >= 2:
        plt.figure(figsize=(8, 6))
        palette = {"baseline": "#d62728", "finetuned": "#2ca02c", "final": "#1f77b4"}
        labels_map = {"baseline": "Baseline", "finetuned": "Fine-tuned", "final": "+Synthetic"}
        for sn, df in test_results.items():
            df = df.copy()
            df["group"] = df["fst"].map(DDI_GROUP_MAP)
            sub = df[df["group"] == "Dark"]
            if len(sub) > 0:
                fpr, tpr, _ = roc_curve(sub["label"], sub["pred"])
                plt.plot(fpr, tpr, color=palette.get(sn, "gray"), linewidth=3,
                         label=f"{labels_map.get(sn, sn)} (AUC={auc(fpr, tpr):.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("FairDerm: Dark Skin Performance Across Stages", fontsize=16, fontweight="bold")
        plt.legend(fontsize=12, loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(result_dir, "fig4_roc_progression.png")
        plt.savefig(roc_path, dpi=600)
        plt.close()
        print(f"\n  Saved ROC progression -> {roc_path}")

    if "finetuned" in val_results and "final" in val_results:
        paper_data = {"stages": {}}
        stage_mapping = [
            ("baseline", "baseline", 0.5),
            ("finetune", "finetuned", thresholds.get("finetuned", 0.5)),
            ("final_synthetic", "final", thresholds.get("final", 0.5)),
        ]
        for stage_label, key, thresh in stage_mapping:
            if key not in test_results:
                continue
            test_results[key] = test_results[key].copy()
            test_results[key]["group"] = test_results[key]["fst"].map(DDI_GROUP_MAP)
            paper_data["stages"][stage_label] = {}
            for grp in ["Light", "Dark"]:
                paper_data["stages"][stage_label][grp] = get_metrics(
                    test_results[key], grp, thresh,
                )

        json_path = os.path.join(result_dir, "paper_data.json")
        with open(json_path, "w") as f:
            json.dump(paper_data, f, indent=2)
        print(f"\n  Saved paper data -> {json_path}")

        n_pos = int(train_df["label"].sum())
        n_neg = len(train_df) - n_pos
        metrics_data = {
            "seed": CONFIG["SEED"],
            "split_file": "splits/ddi_split_seed42.json",
            "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "test_subgroup_sizes": {
                "light": int((test_df["skin_tone"] == 12).sum()),
                "dark": int((test_df["skin_tone"] == 56).sum()),
                "light_melanoma": int(test_df[(test_df["skin_tone"] == 12) & (test_df["label"] == 1)].shape[0]),
                "dark_melanoma": int(test_df[(test_df["skin_tone"] == 56) & (test_df["label"] == 1)].shape[0]),
            },
            "synthetic_source": "train_dark_mel_only",
            "pos_weight_dynamic": round(n_neg / n_pos, 4) if n_pos > 0 else None,
            "hyperparams_shared": {"lr": 5e-5, "batch_size": 32, "weight_decay": 1e-4},
            "thresholds": {k: round(v, 6) for k, v in thresholds.items()},
            "threshold_strategy": {
                "baseline": "fixed_0.5",
                "finetuned": "youden_j_on_validation",
                "final": "youden_j_on_validation",
            },
            "stages": paper_data["stages"],
            "overall_test": {},
            "bootstrap_ci": all_bootstrap,
            "p_value_dark_improvement": p_value_dark,
            "note": "AUROC is primary threshold-free metric. Sens/Spec reported at per-stage validation thresholds.",
        }

        for key in ["baseline", "finetuned", "final"]:
            if key in test_results:
                df = test_results[key]
                overall = compute_metrics(df["label"].values, df["pred"].values)
                metrics_data["overall_test"][key] = {
                    "auroc": round(overall["AUROC"], 4),
                    "sensitivity": round(overall["Sens"], 4),
                    "specificity": round(overall["Spec"], 4),
                    "f1": round(overall["F1"], 4),
                }
                med, lo, hi = bootstrap_auroc(df["label"].values, df["pred"].values, n=1000, seed=0)
                metrics_data["overall_test"][key]["auroc_ci"] = {"median": round(med, 4), "lo": round(lo, 4), "hi": round(hi, 4)}

        metrics_path = os.path.join(os.path.dirname(result_dir), "metrics_seed42.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"  Saved metrics -> {metrics_path}")

    print("\nEvaluation complete.")


def stage_ablation(dry_run=False, batch_size=32, resume=False):
    print("=" * 60)
    print("STAGE: ABLATION STUDY")
    print("=" * 60)

    set_seed(CONFIG["SEED"])
    ddi_df, ddi_img_dir = setup_ddi_data()
    train_df, val_df, test_df = _make_splits(ddi_df)

    syn_dir = os.path.join(CONFIG["DATA_DIR"], "synthetic_train_only")

    print_training_summary(
        train_df, val_df, test_df, batch_size=batch_size, lr=5e-5, is_ddi=True,
    )

    test_loader = setup_dataloaders(
        test_df, ddi_img_dir, val_test_transform, is_ddi=True, batch_size=batch_size, shuffle=False,
    )

    base_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fairderm_ddi_finetuned.pth")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {base_path}")

    print(f"\n  Using baseline: {base_path}")
    print(f"  Synthetic source: {syn_dir}")
    ablation_df = run_ablation_study(
        base_model_path=base_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        test_loader=test_loader,
        ddi_img_dir=ddi_img_dir,
        syn_dir=syn_dir,
        multipliers=(0, 2, 5, 10),
        dry_run=dry_run,
        resume=resume,
    )

    result_dir = os.path.join(CONFIG["RESULT_DIR"], "ablation")
    os.makedirs(result_dir, exist_ok=True)
    ablation_df.to_csv(os.path.join(result_dir, "ablation_report.csv"))

    print("\n--- Ablation Study Results ---")
    print(ablation_df.to_string())

    if "AUROC" in ablation_df.columns:
        plot_df = ablation_df.reset_index()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        x = np.arange(len(plot_df))
        width = 0.35
        ax1.bar(x - width/2, plot_df["AUROC"], width, label="Test AUROC",
                color="#4C72B0", alpha=0.7, edgecolor="black", linewidth=0.5)
        if "val_auroc" in plot_df.columns:
            ax1.bar(x + width/2, plot_df["val_auroc"], width, label="Validation AUROC",
                    color="#55A868", alpha=0.7, edgecolor="black", linewidth=0.5)

        ax1.set_ylabel("AUROC", fontsize=12)
        ax1.set_xlabel("Synthetic Multiplier (x)", fontsize=12)
        ax1.set_ylim(0.5, 1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{m}x" for m in plot_df["multiplier"]])
        ax1.legend(loc="lower left", fontsize=10)

        if "Dark_auroc" in plot_df.columns:
            ax2 = ax1.twinx()
            ax2.plot(x, plot_df["Dark_auroc"], "s--", color="#ED7D31", linewidth=2,
                     markersize=8, label="Dark AUROC")
            ax2.set_ylabel("Dark Subgroup AUROC", fontsize=12, color="#ED7D31")
            ax2.tick_params(axis="y", labelcolor="#ED7D31")
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines2, labels2, loc="upper right", fontsize=10)

        plt.title("Ablation: Oversampling Multiplier (Saved Synthetics)", fontsize=14)
        plt.savefig(
            os.path.join(result_dir, "ablation_tradeoff.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"\n  Saved ablation plot -> {result_dir}/ablation_tradeoff.png")

    print("\nAblation study complete.")


STAGES = {
    "sanity": stage_sanity,
    "baseline": stage_baseline,
    "finetune": stage_finetune,
    "augment": stage_augment,
    "evaluate": stage_evaluate,
    "ablation": stage_ablation,
}

STAGE_KWARGS = {
    "baseline":  {"dry_run", "num_epochs", "batch_size", "resume"},
    "finetune":  {"dry_run", "num_epochs", "batch_size", "resume"},
    "augment":   {"dry_run", "num_epochs", "batch_size", "resume"},
    "ablation":  {"dry_run", "batch_size", "resume"},
    "evaluate":  set(),
}


def main():
    parser = argparse.ArgumentParser(
        description="FairDerm: Melanoma fairness analysis pipeline",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=False,
        choices=list(STAGES.keys()),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract HAM10000 from dataverse_files.zip to data/ham10000/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and model but skip full training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint (if available)",
    )
    args = parser.parse_args()

    if args.extract:
        zip_path = os.path.join(PROJECT_ROOT, "dataverse_files.zip")
        ham_dir = os.path.join(PROJECT_ROOT, "data", "ham10000")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"dataverse_files.zip not found at {zip_path}")
        _extract_ham10000(zip_path, ham_dir)
        print("Extraction complete. You can now run any stage.")

    if not args.stage:
        parser.print_help()
        return

    kwargs = {}
    allowed = STAGE_KWARGS.get(args.stage, set())
    if args.dry_run and "dry_run" in allowed:
        kwargs["dry_run"] = True
    if args.epochs is not None and "num_epochs" in allowed:
        kwargs["num_epochs"] = args.epochs
    if args.batch_size is not None and "batch_size" in allowed:
        kwargs["batch_size"] = args.batch_size
    if args.resume and "resume" in allowed:
        kwargs["resume"] = True

    STAGES[args.stage](**kwargs)


if __name__ == "__main__":
    main()

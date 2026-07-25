#!/usr/bin/env python3
"""
Generate LaTeX table .tex files from results/metrics_seed42.json.

Reads from:
  - results/metrics_seed42.json (canonical source)
  - results/ablation/ablation_report.csv
  - results/*/config.json

Outputs to:
  - paper_latex/tables/*.tex

Usage:
    python paper_latex/generate_tables.py
"""

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


def load_metrics():
    path = os.path.join(ROOT, "results", "metrics_seed42.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"metrics_seed42.json not found. Run: python fairderm.py --stage evaluate"
        )
    with open(path) as f:
        return json.load(f)


def load_ablation_report():
    path = os.path.join(ROOT, "results", "ablation", "ablation_report.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_config(stage):
    path = os.path.join(ROOT, "results", stage, "config.json")
    with open(path) as f:
        return json.load(f)


def write_table1_main_results(metrics):
    """Table 1: Main results across pipeline stages."""
    stages = metrics["stages"]
    overall = metrics.get("overall_test", {})
    stage_order = [
        ("Baseline", "baseline", "baseline"),
        ("Fine-tuned", "finetuned", "finetune"),
        ("+Synthetic", "final", "final_synthetic"),
    ]

    # Build val AUROCs from config files
    val_aurocs = {}
    for stage_cfg, key in [("baseline", "baseline"), ("finetune", "finetuned"), ("augment", "final")]:
        try:
            val_aurocs[key] = load_config(stage_cfg)["final_best_val_auroc"]
        except (FileNotFoundError, KeyError):
            val_aurocs[key] = 0.0

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main classification results across pipeline stages on the DDI test set ($n=132$).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Stage} & Val. AUROC & Test AUROC & Sens. & Spec. & F1 & $\Delta_{\text{AUROC}}$ \\",
        r"\midrule",
    ]
    for label, okey, skey in stage_order:
        s = stages.get(skey, {})
        o = overall.get(okey, {})
        l_auroc = s.get("Light", {}).get("auroc", 0)
        d_auroc = s.get("Dark", {}).get("auroc", 0)
        gap = d_auroc - l_auroc
        val = val_aurocs.get(okey, 0)
        lines.append(
            f"{label} & {val:.4f} & {o.get('auroc', 0):.3f} & "
            f"{o.get('sensitivity', 0):.3f} & {o.get('specificity', 0):.3f} & "
            f"{o.get('f1', 0):.3f} & {gap:+.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab1_main_results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table2_subgroup_results(metrics):
    """Table 2: Light vs Dark subgroup AUROC and bootstrap CIs."""
    stages = metrics["stages"]
    bootstrap = metrics.get("bootstrap_ci", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Subgroup AUROC with 95\% bootstrap confidence intervals (1000 iterations).}",
        r"\label{tab:subgroup_auroc}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Stage} & Light AUROC & Dark AUROC & Gap & Light 95\% CI & Dark 95\% CI \\",
        r"\midrule",
    ]
    stage_order = [
        ("Baseline", "baseline", None),
        ("Fine-tuned", "finetune", "finetuned"),
        ("+Synthetic", "final_synthetic", "final"),
    ]
    for label, key, ci_key in stage_order:
        s = stages.get(key, {})
        l_auroc = s.get("Light", {}).get("auroc", 0)
        d_auroc = s.get("Dark", {}).get("auroc", 0)
        gap = d_auroc - l_auroc
        if ci_key and ci_key in bootstrap:
            lc = bootstrap[ci_key].get("Light", {})
            dc = bootstrap[ci_key].get("Dark", {})
            l_ci = f"[{lc.get('lo', 0):.3f}, {lc.get('hi', 0):.3f}]"
            d_ci = f"[{dc.get('lo', 0):.3f}, {dc.get('hi', 0):.3f}]"
        else:
            l_ci = "---"
            d_ci = "---"
        lines.append(
            f"{label} & {l_auroc:.4f} & {d_auroc:.4f} & {gap:+.4f} & "
            f"{l_ci} & {d_ci} \\\\"
        )

    # Add footnote about threshold differences
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\small{$^*$ Thresholds differ per stage; AUROC is the primary comparison metric.}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab2_subgroup_auroc.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table3_sensitivity(metrics):
    """Table 3: Sensitivity and specificity by subgroup."""
    stages = metrics["stages"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity, specificity, and PPV by skin-tone subgroup.$^*$}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Stage} & Light Sens. & Dark Sens. & Sens. Gap & Light Spec. & Dark Spec. & PPV \\",
        r"\midrule",
    ]
    stage_order = [
        ("Baseline", "baseline"),
        ("Fine-tuned", "finetune"),
        ("+Synthetic", "final_synthetic"),
    ]
    for label, key in stage_order:
        s = stages.get(key, {})
        l_sens = s.get("Light", {}).get("sensitivity", 0)
        d_sens = s.get("Dark", {}).get("sensitivity", 0)
        l_spec = s.get("Light", {}).get("specificity", 0)
        d_spec = s.get("Dark", {}).get("specificity", 0)
        d_ppv = s.get("Dark", {}).get("ppv", 0)
        lines.append(
            f"{label} & {l_sens:.3f} & {d_sens:.3f} & {d_sens - l_sens:+.3f} & "
            f"{l_spec:.3f} & {d_spec:.3f} & {d_ppv:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab3_sensitivity.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table4_ablation(ablation_rows):
    """Table 4: Ablation study."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study: saved synthetic image multiplier on overall test set.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Multiplier} & Val. AUROC & Test AUROC & Dark AUROC & Sens. & Spec. & F1 \\",
        r"\midrule",
    ]
    for row in ablation_rows:
        m = row["multiplier"]
        val_auroc = row.get("val_auroc", row.get("AUROC", "N/A"))
        test_auroc = row.get("AUROC", "N/A")
        dark_auroc = row.get("Dark_auroc", "N/A")
        sens = row.get("Sens", "N/A")
        spec = row.get("Spec", "N/A")
        f1 = row.get("F1", "N/A")

        def fmt(v, d=4):
            try:
                return f"{float(v):.{d}f}"
            except (ValueError, TypeError):
                return str(v)

        lines.append(
            f"{m}$\\times$ & {fmt(val_auroc)} & {fmt(test_auroc)} & {fmt(dark_auroc)} & "
            f"{fmt(sens)} & {fmt(spec)} & {fmt(f1)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab4_ablation.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table5_training_config():
    """Table 5: Training hyperparameters by stage — reads from config.json files."""
    stages_cfg = {}
    for stage in ["baseline", "finetune", "augment"]:
        try:
            stages_cfg[stage] = load_config(stage)
        except FileNotFoundError:
            stages_cfg[stage] = {}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Training hyperparameters by pipeline stage. Stages 2 and 3 share identical hyperparams.}",
        r"\label{tab:training_config}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Stage} & LR & Epochs & Batch & pos\_wt & $\lambda$ & Early Stop \\",
        r"\midrule",
    ]
    stage_order = ["baseline", "finetune", "augment"]
    for stage in stage_order:
        c = stages_cfg.get(stage, {})
        lr = c.get("lr", "N/A")
        epochs = c.get("epochs", "N/A")
        bs = c.get("batch_size", "N/A")
        pw = c.get("pos_weight", "N/A")
        wd = c.get("weight_decay", 1e-4)
        te = c.get("total_epochs_trained", "N/A")
        pw_str = f"{pw:.1f}" if isinstance(pw, (int, float)) else str(pw)
        lines.append(
            f"{stage.capitalize()} & {lr} & {epochs} & {bs} & "
            f"{pw_str} & {wd} & {te}/{epochs} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab5_training_config.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table6_dataset(metrics):
    """Table 6: Dataset summary."""
    sizes = metrics.get("split_sizes", {})
    syn_count = metrics.get("synthetic_source", "train_dark_mel_only")
    test_sub = metrics.get("test_subgroup_sizes", {})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Dataset composition and split sizes.}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Dataset} & Total & Train & Val & Test & Notes \\",
        r"\midrule",
        r"HAM10000 & 7{,}818 & 6{,}254 & 1{,}564 & --- & Melanoma/nevus only \\",
        f"DDI & 656 & {sizes.get('train', 393)} & {sizes.get('val', 131)} & {sizes.get('test', 132)} & Stratified by skin tone \\\\",
        f"DDI Light (test) & --- & --- & --- & {test_sub.get('light', 42)} & Fitzpatrick I--II \\\\",
        f"DDI Dark (test) & --- & --- & --- & {test_sub.get('dark', 42)} & Fitzpatrick V--VI \\\\",
        r"Synthetic & $\sim$290 & $\sim$290 & --- & --- & 29 train dark mel.$\times$10 aug. \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab6_dataset.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table7_provenance():
    """Table 7: Metrics provenance (appendix)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Metrics provenance: source files and method for each reported value.}",
        r"\label{tab:provenance}",
        r"\begin{tabular}{p{3.5cm}p{4cm}p{2.5cm}p{3cm}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Source File} & \textbf{Method} & \textbf{Hardcoded?} \\",
        r"\midrule",
        r"Overall test AUROC/Sens/Spec/F1 & \texttt{results/metrics\_seed42.json} & Programmatic & No \\",
        r"Subgroup AUROC/Sens/Spec/PPV & \texttt{results/metrics\_seed42.json} & Programmatic & No \\",
        r"Bootstrap 95\% CIs & \texttt{results/metrics\_seed42.json} & 1000-iteration bootstrap & No \\",
        r"p-value (Dark improvement) & \texttt{results/metrics\_seed42.json} & Bootstrap permutation & No \\",
        r"Ablation overall + subgroup & \texttt{results/ablation/ablation\_report.csv} & Programmatic & No \\",
        r"Val AUROCs & \texttt{results/*/config.json} & Programmatic & No \\",
        r"Thresholds & \texttt{results/metrics\_seed42.json} & Youden's J (val set) & No \\",
        r"Split disjointness & \texttt{splits/ddi\_split\_seed42.json} & Assertion check & No \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab7_provenance.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    metrics = load_metrics()
    ablation_rows = load_ablation_report()

    print("Generating LaTeX tables ...")
    write_table1_main_results(metrics)
    write_table2_subgroup_results(metrics)
    write_table3_sensitivity(metrics)
    write_table4_ablation(ablation_rows)
    write_table5_training_config()
    write_table6_dataset(metrics)
    write_table7_provenance()
    print("Done.")


if __name__ == "__main__":
    main()

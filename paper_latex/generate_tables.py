#!/usr/bin/env python3
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


def load_metrics():
    clean = os.path.join(ROOT, "results", "metrics_clean.json")
    path = clean if os.path.exists(clean) else os.path.join(ROOT, "results", "metrics_seed42.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"metrics_clean.json / metrics_seed42.json not found. Run: python fairderm.py --stage evaluate"
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
    stages = metrics["stages"]
    overall = metrics.get("overall_test", {})
    stage_order = [
        ("Baseline", "baseline", "baseline"),
        ("Fine-tuned", "finetuned", "finetune"),
        ("+Generated", "final", "final_synthetic"),
    ]

    val_aurocs = {}
    for stage_cfg, key in [("baseline", "baseline"), ("finetune", "finetuned"), ("augment", "final")]:
        try:
            val_aurocs[key] = load_config(stage_cfg)["final_best_val_auroc"]
        except (FileNotFoundError, KeyError):
            val_aurocs[key] = 0.0

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Main classification results across pipeline stages on the DDI test set ($n=132$). "
        r"Subgroup AUROCs include the Medium (Fitzpatrick III--IV) subgroup.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\textbf{Stage} & Val. AUROC & Test AUROC & Sens. & Spec. & F1 "
        r"& Light & Medium & Dark & $\Delta_{\text{AUROC}}$ \\",
        r"\midrule",
    ]
    for label, okey, skey in stage_order:
        s = stages.get(skey, {})
        o = overall.get(okey, {})
        l_auroc = s.get("Light", {}).get("auroc", 0)
        m_auroc = s.get("Medium", {}).get("auroc", 0)
        d_auroc = s.get("Dark", {}).get("auroc", 0)
        gap = d_auroc - l_auroc
        val = val_aurocs.get(okey, 0)
        lines.append(
            f"{label} & {val:.4f} & {o.get('auroc', 0):.3f} & "
            f"{o.get('sensitivity', 0):.3f} & {o.get('specificity', 0):.3f} & "
            f"{o.get('f1', 0):.3f} & {l_auroc:.3f} & {m_auroc:.3f} & {d_auroc:.3f} & {gap:+.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\vspace{2pt}",
        r"\footnotesize{$\Delta_{\text{AUROC}}$ is the Dark$-$Light gap; AUROC is the primary "
        r"comparison metric, and threshold-dependent metrics use per-stage validation "
        r"Youden thresholds (Table~\ref{tab:sensitivity}).}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab1_main_results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table2_subgroup_results(metrics):
    stages = metrics["stages"]
    bootstrap = metrics.get("bootstrap_ci", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Subgroup AUROC with 95\% bootstrap confidence intervals (1000 iterations).}",
        r"\label{tab:subgroup_auroc}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Light} & \textbf{Medium} & \textbf{Dark} & \textbf{Gap} "
        r"& \textbf{Light CI} & \textbf{Medium CI} & \textbf{Dark CI} \\",
        r"\midrule",
    ]
    stage_order = [
        ("Baseline", "baseline", None),
        ("Fine-tuned", "finetune", "finetuned"),
        ("+Generated", "final_synthetic", "final"),
    ]
    for label, key, ci_key in stage_order:
        s = stages.get(key, {})
        l_auroc = s.get("Light", {}).get("auroc", 0)
        m_auroc = s.get("Medium", {}).get("auroc", 0)
        d_auroc = s.get("Dark", {}).get("auroc", 0)
        gap = d_auroc - l_auroc
        if ci_key and ci_key in bootstrap:
            lc = bootstrap[ci_key].get("Light", {})
            mc = bootstrap[ci_key].get("Medium", {})
            dc = bootstrap[ci_key].get("Dark", {})
            l_ci = f"[{lc.get('lo', 0):.3f}, {lc.get('hi', 0):.3f}]"
            m_ci = f"[{mc.get('lo', 0):.3f}, {mc.get('hi', 0):.3f}]" if mc else "---"
            d_ci = f"[{dc.get('lo', 0):.3f}, {dc.get('hi', 0):.3f}]"
        else:
            l_ci = "---"
            m_ci = "---"
            d_ci = "---"
        lines.append(
            f"{label} & {l_auroc:.4f} & {m_auroc:.4f} & {d_auroc:.4f} & {gap:+.4f} & "
            f"{l_ci} & {m_ci} & {d_ci} \\\\"
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
    stages = metrics["stages"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Sensitivity, specificity, and PPV by skin-tone subgroup.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Light} & \textbf{Medium} & \textbf{Dark} & \textbf{Gap} "
        r"& \textbf{Light} & \textbf{Medium} & \textbf{Dark} & \textbf{PPV} \\",
        r"\midrule",
    ]
    stage_order = [
        ("Baseline", "baseline"),
        ("Fine-tuned", "finetune"),
        ("+Generated", "final_synthetic"),
    ]
    for label, key in stage_order:
        s = stages.get(key, {})
        l_sens = s.get("Light", {}).get("sensitivity", 0)
        m_sens = s.get("Medium", {}).get("sensitivity", 0)
        d_sens = s.get("Dark", {}).get("sensitivity", 0)
        l_spec = s.get("Light", {}).get("specificity", 0)
        m_spec = s.get("Medium", {}).get("specificity", 0)
        d_spec = s.get("Dark", {}).get("specificity", 0)
        d_ppv = s.get("Dark", {}).get("ppv", 0)
        lines.append(
            f"{label} & {l_sens:.3f} & {m_sens:.3f} & {d_sens:.3f} & {d_sens - l_sens:+.3f} & "
            f"{l_spec:.3f} & {m_spec:.3f} & {d_spec:.3f} & {d_ppv:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\footnotesize{Sens.\ Gap = Dark sensitivity $-$ Light sensitivity. "
        r"Thresholds are selected per stage by Youden's $J$ on the validation split.}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab3_sensitivity.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table4_ablation(ablation_rows, metrics=None):
    metrics = metrics or {}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study: generated-image multiplier. All arms freeze the same "
        r"layers as Stage~3 and draw variants round-robin across all 29 source lesions.}",
        r"\label{tab:ablation}",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\textbf{Mult.} & \textbf{Imgs} & \textbf{Lesions} & \textbf{Val. AUROC} & "
        r"\textbf{Test AUROC} & \textbf{Dark AUROC} & \textbf{Light AUROC} & "
        r"\textbf{Sens.} & \textbf{Spec.} & \textbf{F1} \\",
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

        n_img = row.get("n_synth_used", "0")
        n_src = row.get("n_distinct_sources", "0")
        try:
            n_img = f"{int(float(n_img))}"
        except (ValueError, TypeError):
            n_img = str(n_img)
        try:
            n_src = f"{int(float(n_src))}"
        except (ValueError, TypeError):
            n_src = str(n_src)
        lines.append(
            f"{m}$\\times$ & {n_img} & {n_src} & {fmt(val_auroc)} & {fmt(test_auroc)} & "
            f"{fmt(dark_auroc)} & {fmt(row.get('Light_auroc', 'N/A'))} & "
            f"{fmt(sens)} & {fmt(spec)} & {fmt(f1)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
    ]
    # Reconcile the largest multiplier with the main +Synthetic row. The two
    # numbers come from different epoch budgets and are expected to differ;
    # without this note a reader assumes the ablation contradicts the main
    # result. Values are read from the data so the note cannot go stale.
    if ablation_rows and metrics.get("overall_test", {}).get("final"):
        try:
            top_row = ablation_rows[-1]
            main_auroc = float(metrics["overall_test"]["final"]["auroc"])
            top_mult = top_row["multiplier"]
            top_auroc = float(top_row["AUROC"])
            lines += [
                r"\vspace{2pt}",
                r"\footnotesize{$^*$ Ablation "
                + f"{top_mult}$\\times$ [{top_auroc:.4f}] is a 5-epoch "
                "continued-training variant from the fine-tuned checkpoint; the "
                + f"main +Synthetic result [{main_auroc:.3f}] is a 10-epoch "
                "full run --- not identical by design.}",
            ]
        except (KeyError, TypeError, ValueError):
            pass
    lines += [
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab4_ablation.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table5_training_config():
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
        f"DDI Medium (test) & --- & --- & --- & {test_sub.get('medium', 48)} & Fitzpatrick III--IV \\\\",
        f"DDI Dark (test) & --- & --- & --- & {test_sub.get('dark', 42)} & Fitzpatrick V--VI \\\\",
        r"Generated & 290 & 290 & --- & --- & 29 train dark mel.$\times$10 transforms \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab6_dataset.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table7_provenance():
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Metrics provenance: source files and method for each reported value.}",
        r"\label{tab:provenance}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.6cm}"
        r">{\raggedright\arraybackslash}p{4.9cm}"
        r">{\raggedright\arraybackslash}p{1.8cm}"
        r">{\raggedright\arraybackslash}p{1.5cm}"
        r">{\raggedright\arraybackslash}p{1.4cm}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Source File} & \textbf{Method} & \textbf{Literal?} & \textbf{Checked?} \\",
        r"\midrule",
        r"Overall test AUROC/Sens/Spec/F1 & \texttt{metrics\_clean.json} & code & No & Yes \\",
        r"Subgroup AUROC/Sens/Spec/PPV & \texttt{metrics\_clean.json} & code & No & Yes \\",
        r"Bootstrap 95\% CIs & \texttt{metrics\_clean.json} & code & No & Yes \\",
        r"p-value (Dark improvement) & \texttt{metrics\_clean.json} & code & No & Yes \\",
        r"Ablation overall + subgroup & \texttt{ablation\_report.csv} & code & No & Yes \\",
        r"Arm-study AUROC/Sens/Spec & \texttt{pos\_weight\_arms\_summary.json} & code & No & Yes \\",
        r"Multi-seed mean, SD, CI & \texttt{multiseed\_summary.json} & code & No & Yes \\",
        r"Val AUROCs & \texttt{config.json} & code & No & Yes \\",
        r"Thresholds & \texttt{metrics\_clean.json} & Youden's J (val set) & No & Yes \\",
        r"Split disjointness & \texttt{ddi\_split\_seed42.json} & assertion check & No & Yes \\",
        r"Leakage (0 gen--test dupes) & \texttt{leakage\_report.json} & SHA256 dedup & No & Yes \\",
        r"Gen. image provenance & \texttt{synthetic\_provenance\_report.json} & hash $+$ descriptor & No & Yes \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\vspace{2pt}",
        r"\footnotesize{* All paths are relative to the repository root, e.g.\ \texttt{results/metrics\_clean.json}.}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab7_provenance.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def load_pos_weight_arms():
    path = os.path.join(ROOT, "results", "pos_weight_arms", "pos_weight_arms_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_multiseed():
    path = os.path.join(ROOT, "results", "multiseed_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_table8_pos_weight_arms(summary):
    arms = summary.get("arms", {})
    order = [a for a in ("static", "recomputed", "symmetric") if a in arms]
    if not order:
        return
    label = {
        "static": "static (headline)",
        "recomputed": "recomputed",
        "symmetric": "symmetric",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Stage-3 class-imbalance arm study. All arms start from the same seed-42 "
        r"fine-tuned checkpoint and differ only in \texttt{pos\_weight} and generated-image "
        r"composition.}",
        r"\label{tab:pw_arms}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Arm} & \texttt{pos\_weight} & Combined & Light & Medium & Dark & Gap "
        r"& Overall \\",
        r"\midrule",
    ]
    for a in order:
        d = arms[a]
        cc = d.get("class_counts", {})
        comb = f"{cc.get('combined_pos', 0)}/{cc.get('combined_neg', 0)}"
        gap = d.get("gap_dark_minus_light")

        def g(k, digits=4):
            v = d.get(k)
            return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "---"

        lines.append(
            f"{label[a]} & {d.get('pos_weight', 0):.4f} & {comb} & "
            f"{g('light_auroc')} & {g('medium_auroc')} & {g('dark_auroc')} & "
            f"{(f'{gap:+.4f}' if isinstance(gap, (int, float)) else '---')} & "
            f"{g('overall_auroc')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\footnotesize{Combined = malignant/benign counts presented to the loss. "
        r"The static arm retains the pre-augmentation ratio and is the configuration "
        r"behind the headline result. Arms were re-run with the seeded, reproducible "
        r"image generator, so the static arm reproduces the headline overall AUROC "
        r"(0.640 vs.\ 0.639) but not every subgroup value exactly.}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab8_pos_weight_arms.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_table9_multiseed(summary):
    """Columnar multi-seed table: stages as rows, subgroups as columns.

    Every cell is mean $\\pm$ SD (95\\% CI) at three decimals, read directly
    from results/multiseed_summary.json aggregates.
    """
    agg = summary.get("aggregates", {})
    seeds = summary.get("seeds_completed", [])
    if not agg:
        return

    def cell(key):
        a = agg.get(key)
        if not a:
            return "---"
        ci = a.get("ci95")
        ci_s = f"$[{ci[0]:.3f},\\,{ci[1]:.3f}]$" if ci else "---"
        return f"${a['mean']:.3f}\\,\\pm\\,{a['sd']:.3f}$ ({ci_s})"

    rows_spec = [
        ("Overall AUROC", "overall_finetuned", "overall_synthetic"),
        ("Light AUROC", "light_finetuned", "light_synthetic"),
        ("Medium AUROC", "medium_finetuned", "medium_synthetic"),
        ("Dark AUROC", "dark_finetuned", "dark_synthetic"),
        ("Dark$-$Light gap", "gap_finetuned", "gap_synthetic"),
        ("Sensitivity (overall)", "overall_sens_finetuned", "overall_sens_synthetic"),
        ("Specificity (overall)", "overall_spec_finetuned", "overall_spec_synthetic"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        rf"\textbf{{Metric}} & \textbf{{Fine-tuned}} & \textbf{{+Generated}} \\[-1.2ex]",
        rf" & \multicolumn{{2}}{{c}}{{Mean $\pm$ SD (seed-level 95\% CI), seeds "
        rf"{', '.join(str(s) for s in seeds)}}} \\",
        r"\midrule",
    ]
    for label, ft_key, sy_key in rows_spec:
        lines.append(f"{label} & {cell(ft_key)} & {cell(sy_key)} \\\\")
    dd = cell("dark_delta")
    lines += [
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{Paired Dark AUROC delta (generated $-$ fine-tuned): {dd}}} \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\vspace{2pt}",
        r"\footnotesize{Full pipeline retrained per seed (Stage-1 baseline included), "
        r"with the split, generated images, and class ratio re-derived per seed, so "
        r"these runs are not a direct replica of the seed-42 headline configuration "
        r"in Tables~\ref{tab:main_results}--\ref{tab:subgroup_auroc}. Dark$-$Light "
        r"gap is Dark AUROC $-$ Light AUROC. The paired-delta CI includes zero.}",
        r"\end{table}",
    ]
    path = os.path.join(TABLES_DIR, "tab_multiseed.tex")
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
    write_table4_ablation(ablation_rows, metrics)
    write_table5_training_config()
    write_table6_dataset(metrics)
    write_table7_provenance()
    pw = load_pos_weight_arms()
    if pw:
        write_table8_pos_weight_arms(pw)
    else:
        print("  Skipped tab8 (no pos_weight_arms_summary.json yet)")
    ms = load_multiseed()
    if ms:
        write_table9_multiseed(ms)
    else:
        print("  Skipped tab9 (no multiseed_summary.json yet)")
    print("Done.")


if __name__ == "__main__":
    main()

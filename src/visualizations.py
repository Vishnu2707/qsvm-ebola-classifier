"""Publication-ready research figures."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)
sns.set_theme(style="whitegrid")

PALETTE = {
    "BUNDIBUGYO": "#E63946",
    "SUDAN": "#457B9D",
    "ZAIRE": "#2D6A4F",
    "NON_EBOLA_HF": "#F4A261",
    "QSVM": "#9B2226",
    "Linear_SVM": "#6D6875",
    "RBF_SVM": "#005F73",
    "RandomForest": "#0A9396",
    "LogisticRegression": "#94D2BD",
    "XGBoost": "#EE9B00",
}

os.makedirs("results/figures", exist_ok=True)


def _save(fig, name):
    fig.savefig(f"results/figures/{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"results/figures/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {name}")


def fig_bandwidth_sweep(sweep_data):
    """Dual-axis bandwidth sweep figure for KTA and off-diagonal spread."""
    sweep = sweep_data["sweep"]
    best_bw = sweep_data["optimal_bandwidth"]

    bws = [r["bandwidth"] for r in sweep]
    ktas = [r["kta"] for r in sweep]
    stds = [r["offdiag_std"] for r in sweep]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color1 = "#9B2226"
    ax1.plot(
        bws,
        ktas,
        color=color1,
        linewidth=2.4,
        marker="o",
        markersize=6,
        label="Kernel-target alignment",
    )
    ax1.set_xlabel("Bandwidth lambda", fontsize=12)
    ax1.set_ylabel("Kernel-Target Alignment", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "#005F73"
    ax2.plot(
        bws,
        stds,
        color=color2,
        linewidth=2.0,
        marker="s",
        markersize=5,
        linestyle="--",
        label="Off-diagonal sigma",
    )
    ax2.set_ylabel("Kernel Off-Diagonal sigma", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(0.10, color=color2, linestyle=":", linewidth=0.9, alpha=0.6)

    ax1.axvspan(1.0, 2.0, alpha=0.08, color="gray")
    ax1.axvline(best_bw, color="#0A9396", linewidth=1.5, linestyle="-.")
    ax1.annotate(
        f"lambda* = {best_bw}",
        xy=(best_bw, max(ktas)),
        xytext=(best_bw * 1.3, max(ktas) * 0.85),
        fontsize=10,
        color="#0A9396",
        arrowprops=dict(arrowstyle="->", color="#0A9396"),
    )
    ax1.annotate("degenerate\nregion", xy=(1.4, min(ktas)), fontsize=9, color="gray", ha="center")

    fig.tight_layout()
    _save(fig, "bandwidth_sweep")


def fig_kernel_before_after(K_default, K_tuned, y_train, label_names):
    """Side-by-side kernel heatmaps for default and tuned bandwidths."""
    del label_names
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sort_idx = np.argsort(y_train)
    for ax, K, label in [
        (axes[0], K_default, "lambda = 1.0 (default)"),
        (axes[1], K_tuned, "lambda = lambda* (tuned)"),
    ]:
        Ks = K[np.ix_(sort_idx, sort_idx)]
        im = ax.imshow(Ks, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            -0.05,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, "kernel_before_after")


def fig_binary_roc_pr(all_results, qsvm_results, y_test, all_predictions, qsvm_proba, label_encoder):
    """Binary mode: ROC curve + precision-recall curve side by side."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {
        "Linear_SVM": "#888780",
        "RBF_SVM": "#005F73",
        "RandomForest": "#0A9396",
        "LogisticRegression": "#94D2BD",
        "XGBoost": "#EE9B00",
        "QSVM": "#9B2226",
    }
    all_models = dict(all_results)
    all_models["QSVM"] = qsvm_results
    all_probas = dict(all_predictions)
    all_probas["QSVM"] = {
        "y_proba": qsvm_proba.tolist() if hasattr(qsvm_proba, "tolist") else qsvm_proba
    }

    pos_idx = list(label_encoder.classes_).index("BUNDIBUGYO")
    for model_name in all_models:
        proba_data = all_probas.get(model_name, {}).get("y_proba")
        if proba_data is None:
            continue
        y_proba = np.array(proba_data)
        scores = y_proba[:, pos_idx]
        color = colors.get(model_name, "#333333")
        lw = 2.5 if model_name == "QSVM" else 1.5
        ls = "-" if model_name == "QSVM" else "--"

        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=pos_idx)
        roc_auc_val = auc(fpr, tpr)
        axes[0].plot(
            fpr,
            tpr,
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=f"{model_name.replace('_', ' ')} ({roc_auc_val:.2f})",
        )

        prec, rec, _ = precision_recall_curve(y_test, scores, pos_label=pos_idx)
        pr_auc_val = auc(rec, prec)
        axes[1].plot(
            rec,
            prec,
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=f"{model_name.replace('_', ' ')} (AP={pr_auc_val:.2f})",
        )

    axes[0].plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])

    baseline = (y_test == pos_idx).mean()
    axes[1].axhline(
        baseline,
        color="gray",
        linestyle=":",
        linewidth=0.9,
        alpha=0.6,
        label=f"Random ({baseline:.2f})",
    )
    axes[1].set_xlabel("Recall (Bundibugyo)", fontsize=11)
    axes[1].set_ylabel("Precision (Bundibugyo)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    fig.tight_layout(pad=2.5)
    _save(fig, "binary_roc_pr")


def fig_precision_recall_grid(all_results, qsvm_results, y_test, all_predictions, qsvm_proba, label_names, label_encoder):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    all_models = dict(all_results)
    all_models["QSVM"] = qsvm_results
    all_probas = dict(all_predictions)
    all_probas["QSVM"] = {"y_proba": qsvm_proba.tolist() if hasattr(qsvm_proba, "tolist") else qsvm_proba}
    n_classes = len(label_names)
    n_models = len(all_models)
    fig, axes = plt.subplots(n_classes, n_models, figsize=(3.6 * n_models, 3.1 * n_classes), sharey=True)
    y_bin = np.eye(n_classes)[y_test]
    for col_idx, model_name in enumerate(all_models.keys()):
        proba = all_probas.get(model_name, {}).get("y_proba")
        if proba is None:
            continue
        y_proba = np.asarray(proba)
        for row_idx, class_name in enumerate(label_names):
            ax = axes[row_idx, col_idx]
            class_idx = label_encoder.transform([class_name])[0]
            precision, recall, _ = precision_recall_curve(y_bin[:, class_idx], y_proba[:, class_idx])
            ap = average_precision_score(y_bin[:, class_idx], y_proba[:, class_idx])
            color = PALETTE.get(model_name, "#333333")
            ax.plot(recall, precision, color=color, linewidth=1.8)
            ax.fill_between(recall, precision, color=color, alpha=0.15)
            ax.axhline(y_bin[:, class_idx].mean(), color="gray", linestyle="--", linewidth=0.8)
            ax.text(0.97, 0.95, f"AP={ap:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
            if row_idx == 0:
                ax.set_xlabel(model_name.replace("_", " "), fontsize=9, fontweight="bold")
                ax.xaxis.set_label_position("top")
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=9, color=PALETTE.get(class_name, "#333333"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "precision_recall_grid")


def fig_sample_complexity(sample_complexity_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_key, color, label in [
        ("qsvm", PALETTE["QSVM"], "QSVM"),
        ("rbf_svm", PALETTE["RBF_SVM"], "RBF-SVM"),
    ]:
        data = sample_complexity_data[model_key]
        n_vals = sorted(int(k) for k in data.keys())
        means = [data[str(n)][0] for n in n_vals]
        stds = [data[str(n)][1] for n in n_vals]
        axes[0].plot(n_vals, means, marker="o", color=color, linewidth=2.0, label=label)
        axes[0].fill_between(n_vals, np.asarray(means) - stds, np.asarray(means) + stds, color=color, alpha=0.18)
    axes[0].set_xlabel("Training Set Size (n)")
    axes[0].set_ylabel("Macro Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)

    n_vals = sorted(int(k) for k in sample_complexity_data["qsvm"].keys())
    diffs = [sample_complexity_data["qsvm"][str(n)][0] - sample_complexity_data["rbf_svm"][str(n)][0] for n in n_vals]
    axes[1].bar(range(len(n_vals)), diffs, color=[PALETTE["QSVM"] if d > 0 else PALETTE["RBF_SVM"] for d in diffs])
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(range(len(n_vals)))
    axes[1].set_xticklabels(n_vals, rotation=45)
    axes[1].set_xlabel("Training Set Size (n)")
    axes[1].set_ylabel("QSVM - RBF Recall Advantage")
    fig.tight_layout()
    _save(fig, "sample_complexity")


def fig_kernel_matrix_heatmap(K_train, y_train, label_names, label_encoder):
    del label_encoder
    sort_idx = np.argsort(y_train)
    K_sorted = K_train[np.ix_(sort_idx, sort_idx)]
    y_sorted = y_train[sort_idx]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(K_sorted, cmap="RdYlBu_r", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Kernel Value K(xi, xj)")
    counts = [np.sum(y_sorted == i) for i in range(len(label_names))]
    boundaries = np.cumsum(counts)
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1.2)
        ax.axvline(b - 0.5, color="white", linewidth=1.2)
    starts = np.concatenate([[0], boundaries[:-1]])
    ticks = [(s + e - 1) / 2 for s, e in zip(starts, boundaries)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(label_names, rotation=35, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(label_names)
    fig.tight_layout()
    _save(fig, "kernel_matrix_heatmap")


def fig_confusion_matrix_comparison(all_predictions, qsvm_preds, y_test, label_names, label_encoder):
    del label_encoder
    preds = {k: np.asarray(v["y_pred"]) for k, v in all_predictions.items()}
    preds["QSVM"] = np.asarray(qsvm_preds)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    from sklearn.metrics import confusion_matrix

    for idx, (model_name, y_pred) in enumerate(preds.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds" if model_name == "QSVM" else "Blues", xticklabels=label_names, yticklabels=label_names, ax=ax, vmin=0, vmax=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.text(0.5, 1.08, model_name.replace("_", " "), transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
    for idx in range(len(preds), len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    _save(fig, "confusion_matrix_comparison")


def fig_symptom_correlation_network(df, label_names):
    import networkx as nx

    symptom_cols = [
        c
        for c in df.columns
        if c
        not in {
            "label",
            "died",
            "source_citation",
            "reconstruction_method",
            "age_years",
            "days_since_onset",
            "haemorrhage_score",
            "gi_score",
            "neuro_score",
        }
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, label in enumerate(label_names):
        ax = axes.flatten()[idx]
        sub = df[df["label"] == label][symptom_cols]
        corr = sub.corr()
        graph = nx.Graph()
        for feat in symptom_cols:
            graph.add_node(feat, frequency=sub[feat].mean())
        for i, f1 in enumerate(symptom_cols):
            for f2 in symptom_cols[i + 1 :]:
                r = corr.loc[f1, f2]
                if abs(r) > 0.25:
                    graph.add_edge(f1, f2, weight=abs(r), sign=np.sign(r))
        pos = nx.spring_layout(graph, seed=42, k=2.2)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=[graph.nodes[n]["frequency"] * 1600 for n in graph], node_color=PALETTE.get(label), alpha=0.85)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)
        nx.draw_networkx_edges(graph, pos, ax=ax, width=[d["weight"] * 3 for _, _, d in graph.edges(data=True)], edge_color=["#E63946" if d["sign"] > 0 else "#457B9D" for _, _, d in graph.edges(data=True)], alpha=0.6)
        ax.set_axis_off()
        ax.text(0.5, 0.02, label, transform=ax.transAxes, ha="center", fontweight="bold", color=PALETTE.get(label))
    fig.tight_layout()
    _save(fig, "symptom_correlation_network")


def fig_outbreak_context(hdx_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_total = hdx_data.get("total") if hdx_data else None
    if df_total is not None:
        date_col = [c for c in df_total.columns if "date" in c.lower() or "week" in c.lower()]
        conf_col = [c for c in df_total.columns if "confirm" in c.lower() or "conf" in c.lower()]
        susp_col = [c for c in df_total.columns if "suspect" in c.lower() or "susp" in c.lower()]
        if date_col and conf_col:
            df_sorted = df_total.sort_values(date_col[0])
            x = range(len(df_sorted))
            if susp_col:
                y_susp = pd.to_numeric(df_sorted[susp_col[0]], errors="coerce").fillna(0).values
                axes[0].fill_between(x, y_susp, alpha=0.3, color="#E63946", label="Suspected")
            y_conf = pd.to_numeric(df_sorted[conf_col[0]], errors="coerce").fillna(0).values
            axes[0].fill_between(x, y_conf, alpha=0.7, color="#9B2226", label="Confirmed")
            axes[0].legend(frameon=False)
    axes[0].set_xlabel("Weeks Since Outbreak Start")
    axes[0].set_ylabel("Cumulative Cases")

    timeline_dates = [0, 10, 13, 17, 24]
    suspected = [4, 246, 528, 746, 906]
    confirmed = [0, 8, 12, 14, 125]
    deaths = [4, 80, 132, 177, 240]
    axes[1].fill_between(timeline_dates, suspected, alpha=0.25, color="#E63946", label="Suspected")
    axes[1].fill_between(timeline_dates, confirmed, alpha=0.7, color="#9B2226", label="Confirmed")
    axes[1].plot(timeline_dates, deaths, color="#333333", linestyle="--", marker="^", label="Deaths")
    axes[1].axvline(10, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Days Since First Alert (May 5 2026)")
    axes[1].set_ylabel("Cumulative Cases / Deaths")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "outbreak_context")


def fig_model_recall_comparison_bar(all_results, qsvm_results, label_names):
    combined = dict(all_results)
    combined["QSVM"] = qsvm_results
    model_names = list(combined.keys())
    x = np.arange(len(label_names))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model_name in enumerate(model_names):
        recalls = [combined[model_name]["classification_report"].get(label, {}).get("recall", 0.0) for label in label_names]
        bars = ax.bar(x + (i - len(model_names) / 2 + 0.5) * width, recalls, width * 0.9, color=PALETTE.get(model_name, "#888888"), label=model_name.replace("_", " "))
        if "BUNDIBUGYO" in label_names:
            bars[label_names.index("BUNDIBUGYO")].set_edgecolor("#9B2226")
            bars[label_names.index("BUNDIBUGYO")].set_linewidth(2.0)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    _save(fig, "recall_comparison_bar")


def generate_all_figures(all_results, qsvm_results, y_test, all_predictions, qsvm_proba, qsvm_preds, K_train, y_train, label_names, label_encoder, df, hdx_data, sample_complexity_data, binary=False):
    print("\n=== Generating Research Figures ===")
    if binary:
        fig_binary_roc_pr(all_results, qsvm_results, y_test, all_predictions, qsvm_proba, label_encoder)
    else:
        fig_precision_recall_grid(all_results, qsvm_results, y_test, all_predictions, qsvm_proba, label_names, label_encoder)
    fig_sample_complexity(sample_complexity_data)
    fig_kernel_matrix_heatmap(K_train, y_train, label_names, label_encoder)
    fig_confusion_matrix_comparison(all_predictions, qsvm_preds, y_test, label_names, label_encoder)
    fig_symptom_correlation_network(df, label_names)
    fig_outbreak_context(hdx_data)
    fig_model_recall_comparison_bar(all_results, qsvm_results, label_names)
    print("\n=== All figures saved to results/figures/ ===")

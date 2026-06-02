"""Bandwidth sweep and tuned-QSVM training."""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC

from quantum_kernel_tuned import (
    kernel_offdiag_std,
    kernel_target_alignment,
    quantum_kernel_matrix_tuned,
)

RANDOM_SEED = 42
BANDWIDTHS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def sweep_bandwidth(X_train, y_train, label_names):
    """Find the bandwidth that maximizes kernel-target alignment."""
    del label_names
    os.makedirs("results/metrics", exist_ok=True)
    results = []

    print("\n=== Bandwidth Sweep ===")
    for bw in BANDWIDTHS:
        K = quantum_kernel_matrix_tuned(X_train, X_train, bw, verbose=False)
        kta = kernel_target_alignment(K, y_train)
        sigma = kernel_offdiag_std(K)
        results.append({"bandwidth": bw, "kta": kta, "offdiag_std": sigma})
        print(f"  lambda={bw:.3f} | KTA={kta:.4f} | off-diag std={sigma:.4f}")

    best = max(results, key=lambda r: r["kta"])
    best_bw = best["bandwidth"]
    default = next(r for r in results if r["bandwidth"] == 1.0)
    print(
        f"\nOptimal bandwidth: lambda*={best_bw:.3f} "
        f"(KTA={best['kta']:.4f}, std={best['offdiag_std']:.4f})"
    )
    print(
        f"Compare to default lambda=1.0: "
        f"KTA={default['kta']:.4f}, std={default['offdiag_std']:.4f}"
    )

    with open("results/metrics/bandwidth_sweep.json", "w") as f:
        json.dump({"sweep": results, "optimal_bandwidth": best_bw}, f, indent=2)

    return results, best_bw


def train_tuned_qsvm(X_train, y_train, X_test, y_test, best_bw, label_names):
    """Train QSVM at the optimal bandwidth and evaluate."""
    print(f"\n=== Bandwidth-Tuned QSVM (lambda*={best_bw:.3f}) ===")

    K_train = quantum_kernel_matrix_tuned(X_train, X_train, best_bw, verbose=True)
    K_test = quantum_kernel_matrix_tuned(X_test, X_train, best_bw, verbose=True)

    K_train_reg = K_train.copy()
    np.fill_diagonal(K_train_reg, K_train_reg.diagonal() + 1e-4)

    qsvm = SVC(
        kernel="precomputed",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    qsvm.fit(K_train_reg, y_train)

    y_pred = qsvm.predict(K_test)
    y_proba = qsvm.predict_proba(K_test)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        target_names=label_names,
        zero_division=0,
    )
    macro_recall = report["macro avg"]["recall"]

    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
    else:
        roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))

    sigma = kernel_offdiag_std(K_train)
    results = {
        "model": "QSVM_bandwidth_tuned",
        "optimal_bandwidth": best_bw,
        "offdiag_std": sigma,
        "macro_recall": macro_recall,
        "roc_auc": roc_auc,
        "classification_report": report,
    }

    print(f"  Macro recall: {macro_recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Kernel off-diag std: {sigma:.4f} (default lambda=1.0 in sweep)")

    with open("results/metrics/qsvm_tuned_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save("results/metrics/K_train_tuned.npy", K_train)

    return results, qsvm, y_pred, y_proba, K_train

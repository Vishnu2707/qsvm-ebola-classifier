"""QSVM training and sample-complexity analysis."""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from quantum_kernel import geometric_difference, quantum_kernel_matrix


def _one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def train_qsvm(X_train, y_train, X_test, y_test, label_names, C=0.1, verbose=True):
    """Train a precomputed-kernel SVC using the quantum kernel."""
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    t0 = time.time()
    K_train = quantum_kernel_matrix(X_train, X_train, verbose=verbose)
    # Kernel matrix diagnostics
    diag_vals = K_train.diagonal()
    off_diag = K_train[np.triu_indices_from(K_train, k=1)]
    print(f"  Kernel diag mean: {diag_vals.mean():.4f} (should be ~1.0)")
    print(f"  Kernel off-diag mean: {off_diag.mean():.4f}")
    print(f"  Kernel off-diag std:  {off_diag.std():.4f} (low = near-constant = bad)")
    K_test = quantum_kernel_matrix(X_test, X_train, verbose=verbose)
    t_kernel = time.time() - t0

    gd = geometric_difference(K_train, rbf_kernel(X_train, X_train))
    # Regularise kernel diagonal to improve numerical conditioning.
    K_train_reg = K_train.copy()
    np.fill_diagonal(K_train_reg, K_train_reg.diagonal() + 1e-4)

    qsvm = SVC(
        kernel="precomputed",
        C=C,
        probability=True,
        class_weight="balanced",
        random_state=42,
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
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        avg_prec = float(average_precision_score(y_test, y_proba[:, 1]))
    else:
        roc_auc = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        )
        avg_prec = float(
            average_precision_score(_one_hot(y_test, n_classes), y_proba, average="macro")
        )
    results = {
        "model": "QSVM_ZZFeatureMap",
        "n_qubits": 6,
        "kernel_type": "ZZFeatureMap",
        "kernel_compute_time_s": float(t_kernel),
        "geometric_difference": gd,
        "geometric_difference_favours_quantum": bool(gd > 1.0),
        "C": C,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
    }
    print("\n=== QSVM Results ===")
    print(f"Test Recall (Bundibugyo): {report.get('BUNDIBUGYO', {}).get('recall', 'N/A')}")
    print(f"Macro Recall: {report['macro avg']['recall']:.3f}")
    with open("results/metrics/qsvm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save("results/metrics/K_train.npy", np.vstack([K_train, K_test]))
    np.save("results/metrics/K_test.npy", K_test)
    joblib.dump(qsvm, "results/models/qsvm.pkl")
    return results, qsvm, y_pred, y_proba, K_train


def sample_complexity_qsvm(X, y, label_names, sample_fractions=None):
    """Compare QSVM and RBF-SVM macro recall as training size changes."""
    if sample_fractions is None:
        sample_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    rng = np.random.RandomState(42)
    results = {"qsvm": {}, "rbf_svm": {}}
    for frac in sample_fractions:
        n_samples = max(int(len(X) * frac), 32)
        idx = rng.choice(len(X), n_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        qsvm_recalls, rbf_recalls = [], []
        for train_idx, val_idx in skf.split(X_sub, y_sub):
            Xt, Xv = X_sub[train_idx], X_sub[val_idx]
            yt, yv = y_sub[train_idx], y_sub[val_idx]
            Kt = quantum_kernel_matrix(Xt, Xt, verbose=False)
            Kv = quantum_kernel_matrix(Xv, Xt, verbose=False)
            q = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            q.fit(Kt, yt)
            qsvm_recalls.append(recall_score(yv, q.predict(Kv), average="macro", zero_division=0))

            Kr_t = rbf_kernel(Xt, Xt)
            Kr_v = rbf_kernel(Xv, Xt)
            c = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            c.fit(Kr_t, yt)
            rbf_recalls.append(recall_score(yv, c.predict(Kr_v), average="macro", zero_division=0))
        results["qsvm"][str(n_samples)] = [float(np.mean(qsvm_recalls)), float(np.std(qsvm_recalls))]
        results["rbf_svm"][str(n_samples)] = [float(np.mean(rbf_recalls)), float(np.std(rbf_recalls))]
        print(f"  n={n_samples}: QSVM={np.mean(qsvm_recalls):.3f}, RBF={np.mean(rbf_recalls):.3f}")
    with open("results/metrics/sample_complexity.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

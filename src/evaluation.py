"""Statistical evaluation helpers."""

from __future__ import annotations

import json
import os

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

try:
    from bootstrap_ci import run_bootstrap
except ImportError:  # pragma: no cover - supports package imports in smoke tests
    from .bootstrap_ci import run_bootstrap


def mcnemar_test(y_true, pred_model_a, pred_model_b, model_a_name, model_b_name):
    """Run McNemar's paired test for two classifiers on the same test set."""
    n01 = np.sum((pred_model_a != y_true) & (pred_model_b == y_true))
    n10 = np.sum((pred_model_a == y_true) & (pred_model_b != y_true))
    result = mcnemar([[0, n01], [n10, 0]], exact=True)
    print(f"\nMcNemar Test: {model_a_name} vs {model_b_name}")
    print(f"  n01={n01}, n10={n10}")
    print(f"  p-value={result.pvalue:.4f}")
    return {
        "test": "McNemar",
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n01": int(n01),
        "n10": int(n10),
        "pvalue": float(result.pvalue),
        "significant_alpha_05": bool(result.pvalue < 0.05),
    }


def per_class_recall_comparison(all_results, label_names):
    """Extract per-class recall into a plain dict table."""
    table = {}
    for model_name, res in all_results.items():
        report = res.get("classification_report", {})
        table[model_name] = {
            label: report.get(label, {}).get("recall", 0.0) for label in label_names
        }
    return table


def run_all_statistical_tests(y_test, all_predictions, qsvm_preds, label_names):
    """Run McNemar tests between QSVM and each classical model."""
    del label_names
    os.makedirs("results/metrics", exist_ok=True)
    tests = []
    for model_name, preds in all_predictions.items():
        tests.append(
            mcnemar_test(
                np.asarray(y_test),
                np.asarray(preds["y_pred"]),
                np.asarray(qsvm_preds),
                model_name,
                "QSVM",
            )
        )
    with open("results/metrics/statistical_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    bootstrap_preds = {
        model_name: np.asarray(preds["y_pred"])
        for model_name, preds in all_predictions.items()
    }
    bootstrap_preds["QSVM"] = np.asarray(qsvm_preds)
    run_bootstrap(np.asarray(y_test), bootstrap_preds)
    return tests

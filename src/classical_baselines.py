"""Classical benchmark models for the QSVM comparison."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - depends on native OpenMP runtime
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

RANDOM_SEED = 42

CLASSICAL_MODELS = {
    "Linear_SVM": SVC(
        kernel="linear",
        probability=True,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "RBF_SVM": SVC(
        kernel="rbf",
        probability=True,
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
}

if XGBClassifier is not None:
    CLASSICAL_MODELS["XGBoost"] = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )


def _one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def evaluate_model(model, X_train, y_train, X_test, y_test, cv, model_name, label_names):
    """Evaluate one model with CV recall and holdout metrics."""
    cv_recalls = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="recall_macro",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        target_names=label_names,
        zero_division=0,
    )
    results = {
        "model": model_name,
        "cv_recall_macro_mean": float(cv_recalls.mean()),
        "cv_recall_macro_std": float(cv_recalls.std()),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if y_proba is not None:
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            results["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
            results["avg_precision"] = float(
                average_precision_score(y_test, y_proba[:, 1])
            )
        else:
            results["roc_auc_ovr"] = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            )
            results["avg_precision_macro"] = float(
                average_precision_score(_one_hot(y_test, n_classes), y_proba, average="macro")
            )
    print(
        f"  {model_name}: CV Recall={cv_recalls.mean():.3f}+/-{cv_recalls.std():.3f} | "
        f"Test Recall={report['macro avg']['recall']:.3f}"
    )
    return results, model, y_pred, y_proba


def run_all_baselines(X_train, y_train, X_test, y_test, cv, label_names):
    """Run all classical baselines and persist metrics/models."""
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    if XGBClassifier is None:
        print(f"Skipping XGBoost because it could not be imported: {XGBOOST_IMPORT_ERROR}")
    all_results = {}
    all_predictions = {}
    for name, model in tqdm(CLASSICAL_MODELS.items(), desc="Classical baselines"):
        results, fitted_model, y_pred, y_proba = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
            name,
            label_names,
        )
        all_results[name] = results
        all_predictions[name] = {
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist() if y_proba is not None else None,
        }
        joblib.dump(fitted_model, f"results/models/{name}.pkl")
    with open("results/metrics/classical_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results, all_predictions

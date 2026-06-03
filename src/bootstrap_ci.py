"""
bootstrap_ci.py
Bootstrap 95% CI on macro-averaged recall for all models.
Saves results/metrics/bootstrap_ci.json and prints LaTeX rows.
"""
import json
import pathlib
from typing import Dict, Tuple

import numpy as np


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    return float(
        np.mean(
            [
                (y_pred[y_true == c] == c).mean()
                for c in classes
                if (y_true == c).sum() > 0
            ]
        )
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [
        macro_recall(y_true[idx], y_pred[idx])
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ]
    alpha = (1 - ci) / 2
    return (
        float(np.mean(scores)),
        float(np.percentile(scores, 100 * alpha)),
        float(np.percentile(scores, 100 * (1 - alpha))),
    )


def ci_overlaps(ci_a: Tuple, ci_b: Tuple) -> bool:
    return ci_a[1] < ci_b[2] and ci_a[2] > ci_b[1]


def run_bootstrap(
    y_test: np.ndarray,
    preds: Dict[str, np.ndarray],
    out_dir: str = "results/metrics",
    n_boot: int = 2000,
) -> Dict[str, dict]:
    """
    Run bootstrap CI for all models. Called from evaluation.py.

    Parameters
    ----------
    y_test  : ground-truth test labels
    preds   : {model_name: y_pred array}
    out_dir : where to write bootstrap_ci.json
    n_boot  : number of bootstrap resamples

    Returns
    -------
    results dict {model_name: {point, ci_lo, ci_hi, overlaps_xgb}}
    """
    y_test = np.asarray(y_test)
    results = {}
    for name, y_pred in preds.items():
        y_pred = np.asarray(y_pred)
        point = macro_recall(y_test, y_pred)
        _, lo, hi = bootstrap_ci(y_test, y_pred, n_boot=n_boot)
        results[name] = {
            "point": round(point, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }

    # Overlap check against XGBoost (or best classical).
    xgb_key = next((k for k in results if "xgboost" in k.lower()), None)
    if xgb_key:
        xgb_ci = (None, results[xgb_key]["ci_lo"], results[xgb_key]["ci_hi"])
        for name, v in results.items():
            model_ci = (None, v["ci_lo"], v["ci_hi"])
            v["overlaps_xgb"] = ci_overlaps(model_ci, xgb_ci)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(out_dir) / "bootstrap_ci.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Print LaTeX rows.
    print("\n% ---- bootstrap CI table rows ----")
    for name, v in results.items():
        ol = "Yes" if v.get("overlaps_xgb") else "No"
        print(
            f"{name} & {v['point']:.3f} & "
            f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] & {ol} \\\\"
        )

    print(f"\n[bootstrap_ci] Saved to {out}")
    return results

"""
noise_model.py
Depolarising channel noise sensitivity analysis for the tuned
ZZFeatureMap kernel. Provides an analytical QPU proxy without
hardware access.

Model: K_noisy = eta * K_ideal + (1-eta)/2^n
       eta = (1-p)^n_gates

Cite: Nielsen & Chuang 2010 (depolarising channel);
      Temme, Bravyi & Gambetta 2017 PRL (error mitigation).
"""
import json
import pathlib

import numpy as np
from sklearn.metrics import recall_score
from sklearn.svm import SVC

N_QUBITS = 6
N_GATES = 42  # depth-2 ZZFeatureMap, 6 qubits
# 6H + 6RZ + (CNOT+RZ+CNOT)*5 * 2 layers = 42

GATE_ERROR_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def apply_depolarising(
    K: np.ndarray,
    p: float,
    n_qubits: int = N_QUBITS,
    n_gates: int = N_GATES,
) -> np.ndarray:
    """
    Apply depolarising channel to kernel matrix.
    K_noisy = eta*K + (1-eta)/2^n  where eta=(1-p)^n_gates.
    """
    eta = (1.0 - p) ** n_gates
    uniform = 1.0 / (2**n_qubits)
    return eta * K + (1.0 - eta) * uniform


def kernel_stats(K: np.ndarray) -> dict:
    mask = ~np.eye(len(K), dtype=bool)
    offdiag = K[mask]
    return {
        "diag_mean": float(np.diag(K).mean()),
        "offdiag_mean": float(offdiag.mean()),
        "offdiag_std": float(offdiag.std()),
    }


def qsvm_recall(
    K_tr: np.ndarray,
    K_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    C: float = 1.0,
) -> float:
    K_reg = K_tr + 1e-4 * np.eye(len(K_tr))
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(K_reg, y_tr)
    y_pred = clf.predict(K_te)
    return float(recall_score(y_te, y_pred, average="macro", zero_division=0))


def run_noise_analysis(
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    gate_errors: list = None,
    out_dir: str = "results",
) -> list:
    """
    Run full noise sensitivity sweep. Called from main.py --rescue.

    Returns list of result dicts, one per gate error level.
    """
    grid = gate_errors or GATE_ERROR_GRID
    results = []

    print(f"\n{'p_gate':>10} {'eta':>8} {'sigma':>10} {'Recall':>10} {'Usable?':>10}")
    print("-" * 55)

    for p in grid:
        eta = (1.0 - p) ** N_GATES
        K_tr_n = apply_depolarising(K_train, p)
        K_te_n = apply_depolarising(K_test, p)
        stats = kernel_stats(K_tr_n)
        sigma = stats["offdiag_std"]
        recall = qsvm_recall(K_tr_n, K_te_n, y_train, y_test)
        usable = "YES" if sigma >= 0.05 else ("MARGINAL" if sigma >= 0.02 else "NO")

        tag = ""
        if abs(p - 0.001) < 1e-4:
            tag = "  <- IBM Eagle / superconducting"
        if abs(p - 0.010) < 1e-4:
            tag = "  <- NISQ average"

        print(f"{p:>10.4f} {eta:>8.4f} {sigma:>10.4f} {recall:>10.3f} {usable:>10}{tag}")
        results.append(
            {
                "gate_error": p,
                "fidelity": round(eta, 6),
                "offdiag_std": round(sigma, 6),
                "macro_recall": round(recall, 4),
                "usable": usable,
            }
        )

    out = pathlib.Path(out_dir) / "metrics" / "noise_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[noise_model] Saved to {out}")

    _plot(results, pathlib.Path(out_dir) / "figures" / "noise_sensitivity.png")
    return results


def _plot(results: list, out_path: pathlib.Path):
    try:
        import matplotlib.pyplot as plt

        ps = [r["gate_error"] for r in results]
        sigs = [r["offdiag_std"] for r in results]
        recalls = [r["macro_recall"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(ps, sigs, "o-", color="#c0392b", lw=2)
        ax1.axhline(
            0.05,
            color="#e67e22",
            ls="--",
            lw=1.5,
            label=r"$\sigma=0.05$ floor (degenerate below)",
        )
        ax1.axhline(0.087, color="#27ae60", ls=":", lw=1.5, label=r"Ideal $\sigma=0.087$")
        ax1.set_xlabel("Gate error rate $p$", fontsize=11)
        ax1.set_ylabel(r"Off-diagonal $\sigma$", fontsize=11)
        ax1.set_title("Kernel structure vs noise", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(ps, recalls, "s-", color="#2980b9", lw=2)
        ax2.axhline(0.569, color="#27ae60", ls=":", lw=1.5, label="Ideal recall 0.569")
        ax2.axhline(0.500, color="#e74c3c", ls="--", lw=1.5, label="Degenerate floor 0.500")
        ax2.set_xlabel("Gate error rate $p$", fontsize=11)
        ax2.set_ylabel("Macro recall", fontsize=11)
        ax2.set_title("QSVM recall vs noise", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            r"Depolarising noise sensitivity (ZZFeatureMap, $\lambda^*=0.05$, 6 qubits)",
            fontsize=11,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[noise_model] Figure -> {out_path}")
    except ImportError:
        print("matplotlib unavailable -- figure skipped")

"""
vqc_bandwidth_sweep.py
Runs VQC training across the same lambda grid as the QSVM bandwidth sweep,
demonstrating that lambda=0.3 is a defensible choice (not arbitrary).
Produces results/metrics/vqc_bandwidth_sweep.json and
results/figures/vqc_bandwidth_sweep.png.
"""
import json
import pathlib
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import recall_score

N_QUBITS = 6
LAMBDA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]


def _make_circuit(lam: float, n_layers: int):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.AngleEmbedding(lam * x, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return circuit


def _train(circuit, X_tr, y_bin_tr, n_layers, epochs, lr, seed):
    rng = np.random.default_rng(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers, N_QUBITS)
    W = pnp.array(rng.uniform(-np.pi, np.pi, shape), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    pw = (y_bin_tr == 0).sum() / max((y_bin_tr == 1).sum(), 1)

    def cost(W):
        p = pnp.array([circuit(x, W) for x in X_tr])
        lb = pnp.array([1.0 if y == 1 else -1.0 for y in y_bin_tr])
        wt = pnp.array([pw if y == 1 else 1.0 for y in y_bin_tr])
        return pnp.mean(wt * (p - lb) ** 2)

    for ep in range(epochs):
        W, loss = opt.step_and_cost(cost, W)
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep + 1}/{epochs} loss={float(loss):.4f}")
    return W


def _eval(circuit, W, X_te, y_bin_te):
    raw = np.array([float(circuit(x, W)) for x in X_te])
    pred = (raw > 0).astype(int)
    bund = recall_score(y_bin_te, pred, pos_label=1, zero_division=0)
    macro = recall_score(y_bin_te, pred, average="macro", zero_division=0)
    return float(bund), float(macro)


def sweep_vqc_bandwidth(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bundibugyo_class: int = 0,
    n_layers: int = 3,
    epochs: int = 40,
    lr: float = 0.05,
    seed: int = 42,
    lambdas: list = None,
    out_dir: str = "results",
) -> dict:
    """
    Run VQC at each bandwidth lambda. Binary: bundibugyo_class vs rest.
    Returns sweep dict keyed by str(lambda).
    """
    grid = lambdas or LAMBDA_GRID
    y_bin_tr = (y_train == bundibugyo_class).astype(int)
    y_bin_te = (y_test == bundibugyo_class).astype(int)

    sweep = {}
    for lam in grid:
        print(f"\n[vqc_bw_sweep] lambda={lam}")
        t0 = time.time()
        cir = _make_circuit(lam, n_layers)
        W = _train(cir, X_train, y_bin_tr, n_layers, epochs, lr, seed)
        bund, macro = _eval(cir, W, X_test, y_bin_te)
        elapsed = time.time() - t0
        sweep[str(lam)] = {
            "lambda": lam,
            "bundibugyo_recall": bund,
            "macro_recall": macro,
            "elapsed_sec": round(elapsed, 1),
        }
        print(f"  bund={bund:.3f} macro={macro:.3f} t={elapsed:.0f}s")

    # Save.
    out = pathlib.Path(out_dir) / "metrics" / "vqc_bandwidth_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"[vqc_bw_sweep] Saved to {out}")

    # Figure.
    _plot(sweep, pathlib.Path(out_dir) / "figures" / "vqc_bandwidth_sweep.png")
    return sweep


def _plot(sweep: dict, out_path: pathlib.Path):
    try:
        import matplotlib.pyplot as plt

        lams = [float(k) for k in sweep]
        bund = [sweep[k]["bundibugyo_recall"] for k in sweep]
        macro = [sweep[k]["macro_recall"] for k in sweep]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lams, bund, "o-", color="#c0392b", lw=2, label="Bundibugyo recall")
        ax.plot(lams, macro, "s--", color="#2980b9", lw=2, label="Macro recall")
        ax.axvline(0.30, color="#7f8c8d", ls=":", label=r"Paper $\lambda=0.30$")
        best = max(sweep, key=lambda k: sweep[k]["bundibugyo_recall"])
        ax.axvline(float(best), color="#27ae60", ls="--", label=rf"Best $\lambda^*={best}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Bandwidth $\lambda$", fontsize=12)
        ax.set_ylabel("Recall", fontsize=12)
        ax.set_title("VQC Bandwidth Sensitivity (binary Bundibugyo task)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[vqc_bw_sweep] Figure -> {out_path}")
    except ImportError:
        print("matplotlib unavailable -- figure skipped")

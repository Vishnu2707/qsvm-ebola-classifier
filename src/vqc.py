"""Variational quantum classifier for Bundibugyo-vs-rest rescue experiments."""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score

N_QUBITS = 6
N_LAYERS = 3
RANDOM_SEED = 42

try:
    import pennylane as qml
    from pennylane import numpy as pnp

    dev_vqc = qml.device("default.qubit", wires=N_QUBITS)
except Exception as exc:  # pragma: no cover - exercised only without dependency
    qml = None
    pnp = None
    dev_vqc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_pennylane():
    if qml is None:
        raise ImportError("PennyLane is required for the VQC. Install requirements.txt first.") from _IMPORT_ERROR


if qml is not None:

    @qml.qnode(dev_vqc, interface="autograd")
    def vqc_circuit(x, weights):
        """Feature encoding followed by trainable entangling layers."""
        qml.AngleEmbedding(0.3 * x, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))


def vqc_predict_proba(X, weights):
    """Map circuit output from [-1, 1] to probability [0, 1]."""
    require_pennylane()
    outputs = np.array([vqc_circuit(x, weights) for x in X])
    return (outputs + 1) / 2


def train_vqc(X_train, y_train, X_test, y_test, label_names, epochs=40, lr=0.05):
    """Train a binary VQC for BUNDIBUGYO vs rest."""
    require_pennylane()
    print("\n=== Variational Quantum Classifier ===")

    bund_idx = list(label_names).index("BUNDIBUGYO") if "BUNDIBUGYO" in label_names else 0
    y_train_bin = (np.asarray(y_train) == bund_idx).astype(float)
    y_test_bin = (np.asarray(y_test) == bund_idx).astype(float)

    rng = np.random.default_rng(RANDOM_SEED)
    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    weights = pnp.array(rng.uniform(0, np.pi, weights_shape), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(w):
        preds = pnp.array([vqc_circuit(x, w) for x in X_train])
        targets = 2 * y_train_bin - 1
        pos_w = len(y_train_bin) / (2 * y_train_bin.sum() + 1e-9)
        neg_w = len(y_train_bin) / (2 * (len(y_train_bin) - y_train_bin.sum()) + 1e-9)
        weights_vec = pnp.where(y_train_bin == 1, pos_w, neg_w)
        return pnp.mean(weights_vec * (preds - targets) ** 2)

    print(f"  Training {epochs} epochs...")
    for epoch in range(epochs):
        weights, c = opt.step_and_cost(cost, weights)
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}/{epochs} | loss={float(c):.4f}")

    proba_test = vqc_predict_proba(X_test, weights)
    y_pred_bin = (proba_test > 0.5).astype(int)

    recall_bund = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    macro_recall = recall_score(y_test_bin, y_pred_bin, average="macro", zero_division=0)
    try:
        roc_auc = float(roc_auc_score(y_test_bin, proba_test))
    except ValueError:
        roc_auc = float("nan")

    results = {
        "model": "VQC",
        "n_layers": N_LAYERS,
        "bundibugyo_recall": float(recall_bund),
        "macro_recall": float(macro_recall),
        "roc_auc": roc_auc,
    }

    print(f"  Bundibugyo recall: {recall_bund:.4f}")
    print(f"  Macro recall: {macro_recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/vqc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, weights, y_pred_bin, proba_test

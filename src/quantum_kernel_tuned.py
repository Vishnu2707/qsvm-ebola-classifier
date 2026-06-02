"""Bandwidth-tuned PennyLane ZZFeatureMap-style quantum kernel."""

from __future__ import annotations

import time

import numpy as np

N_QUBITS = 6

try:
    import pennylane as qml

    dev_sim = qml.device("default.qubit", wires=N_QUBITS)
except Exception as exc:  # pragma: no cover - exercised only without dependency
    qml = None
    dev_sim = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_pennylane():
    if qml is None:
        raise ImportError(
            "PennyLane is required for the tuned quantum kernel. Install requirements.txt first."
        ) from _IMPORT_ERROR


def _zz_feature_map_tuned(x, wires, bandwidth):
    """Bandwidth-scaled ZZFeatureMap.

    Small bandwidth keeps encoded states closer together; large bandwidth
    recovers the default high-spread regime that can concentrate the kernel.
    """
    x_scaled = bandwidth * x
    n = len(wires)

    for i in wires:
        qml.Hadamard(wires=i)
    for idx, i in enumerate(wires):
        qml.RZ(2.0 * x_scaled[idx], wires=i)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RZ(
            2.0 * (np.pi - x_scaled[i]) * (np.pi - x_scaled[i + 1]),
            wires=wires[i + 1],
        )
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for idx, i in enumerate(wires):
        qml.RZ(2.0 * x_scaled[idx], wires=i)


def make_kernel_circuit(bandwidth):
    require_pennylane()

    @qml.qnode(dev_sim)
    def _kernel_circuit(x1, x2):
        _zz_feature_map_tuned(x1, range(N_QUBITS), bandwidth)
        qml.adjoint(_zz_feature_map_tuned)(x2, range(N_QUBITS), bandwidth)
        return qml.probs(wires=range(N_QUBITS))

    return _kernel_circuit


def quantum_kernel_matrix_tuned(X1, X2, bandwidth, verbose=False):
    """Compute kernel matrix at a given bandwidth."""
    require_pennylane()
    circuit = make_kernel_circuit(bandwidth)

    def kval(x1, x2):
        return float(circuit(x1, x2)[0])

    if verbose:
        print(
            f"Computing kernel matrix at bandwidth={bandwidth:.3f} "
            f"({len(X1)} x {len(X2)})..."
        )
    t0 = time.time()
    K = qml.kernels.kernel_matrix(X1, X2, kval)
    if verbose:
        print(f"  done in {time.time() - t0:.1f}s")
    return np.asarray(K)


def kernel_target_alignment(K, y):
    """Kernel-target alignment against a same-class/different-class target."""
    y = np.asarray(y)
    Y = np.equal.outer(y, y).astype(float) * 2 - 1
    num = np.sum(K * Y)
    den = np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro")
    return float(num / (den + 1e-12))


def kernel_offdiag_std(K):
    """Off-diagonal standard deviation, the kernel concentration diagnostic."""
    off = K[np.triu_indices_from(K, k=1)]
    return float(off.std())

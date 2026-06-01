"""PennyLane ZZFeatureMap-style quantum kernel."""

from __future__ import annotations

import os
import time

import numpy as _np

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
            "PennyLane is required for the quantum kernel. Install requirements.txt first."
        ) from _IMPORT_ERROR


def _zz_feature_map(x, wires):
    n = len(wires)
    for i in wires:
        qml.Hadamard(wires=i)
    for idx, i in enumerate(wires):
        qml.RZ(2.0 * x[idx], wires=i)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RZ(2.0 * (_np.pi - x[i]) * (_np.pi - x[i + 1]), wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for idx, i in enumerate(wires):
        qml.RZ(2.0 * x[idx], wires=i)


if qml is not None:

    @qml.qnode(dev_sim)
    def _kernel_circuit(x1, x2):
        _zz_feature_map(x1, range(N_QUBITS))
        qml.adjoint(_zz_feature_map)(x2, range(N_QUBITS))
        return qml.probs(wires=range(N_QUBITS))


def quantum_kernel_value(x1, x2):
    require_pennylane()
    result = _kernel_circuit(x1, x2)
    return float(result[0])


def quantum_kernel_matrix(X1, X2, verbose=True):
    """Compute K[i,j] = K(X1[i], X2[j])."""
    require_pennylane()
    n1, n2 = len(X1), len(X2)
    if verbose:
        print(f"Computing quantum kernel matrix ({n1} x {n2})...")
    start = time.time()
    K = qml.kernels.kernel_matrix(X1, X2, quantum_kernel_value)
    if verbose:
        print(f"Kernel matrix computed in {time.time() - start:.1f}s")
    return _np.asarray(K)


def geometric_difference(K_quantum, K_classical):
    """Geometric difference test from quantum kernel literature."""
    K_c = _np.asarray(K_classical).copy()
    K_c += 1e-6 * _np.eye(len(K_c))
    try:
        K_c_inv_sqrt = _np.linalg.inv(_np.linalg.cholesky(K_c)).T
        matrix = K_c_inv_sqrt @ K_quantum @ K_c_inv_sqrt
        eigenvalues = _np.linalg.eigvalsh(matrix)
    except _np.linalg.LinAlgError:
        eigenvalues = _np.linalg.eigvalsh(_np.linalg.pinv(K_c) @ K_quantum)
    g = float(_np.sqrt(max(_np.max(eigenvalues), 0.0)))
    print(f"Geometric difference g = {g:.4f}")
    print(f"Quantum kernel {'PREFERRED' if g > 1 else 'NOT preferred'} over classical")
    return g


def draw_circuit(x_sample):
    """Save a circuit diagram for the paper."""
    require_pennylane()
    os.makedirs("results/figures", exist_ok=True)

    @qml.qnode(dev_sim)
    def circuit(x):
        _zz_feature_map(x, range(N_QUBITS))
        return qml.state()

    fig, _ = qml.draw_mpl(circuit)(x_sample)
    fig.savefig("results/figures/quantum_circuit.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("results/figures/quantum_circuit.png", bbox_inches="tight", dpi=300)
    print("Circuit diagram saved")
    return fig

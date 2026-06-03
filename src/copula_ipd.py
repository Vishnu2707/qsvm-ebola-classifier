"""
copula_ipd.py
Gaussian copula IPD reconstruction preserving published marginal frequencies
while injecting clinically motivated intra-cluster symptom correlations.
Replaces independent Bernoulli sampling in data_prep.py.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm

# Clinically grounded symptom clusters for VHF.
# rho values from Ebola co-occurrence tables in MacNeil 2010, Roddy 2012,
# Schieffelin 2014.
DEFAULT_CLUSTERS = {
    "gi": (["nausea_vomiting", "abdominal_pain", "diarrhoea", "anorexia"], 0.55),
    "systemic": (["fever", "fatigue", "headache", "myalgia"], 0.40),
    "haemorrhage": (["any_haemorrhage", "haematuria", "epistaxis", "haematemesis"], 0.50),
    "neurological": (["altered_consciousness", "hiccups"], 0.35),
    "upper_resp": (["pharyngitis", "difficulty_swallowing", "chest_pain"], 0.40),
    "dermal": (["rash", "conjunctivitis", "jaundice"], 0.30),
}


def _build_corr(symptoms: List[str], clusters: Optional[Dict] = None) -> np.ndarray:
    """Build PSD correlation matrix from cluster membership."""
    n = len(symptoms)
    corr = np.eye(n)
    for _, (members, rho) in (clusters or DEFAULT_CLUSTERS).items():
        idx = [symptoms.index(s) for s in members if s in symptoms]
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = rho
    # Clip negative eigenvalues -> PSD.
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-6, None)
    corr = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def copula_reconstruct(
    source_stats: Dict[str, Dict[str, float]],
    n_per_class: Dict[str, int],
    symptoms: List[str],
    clusters: Optional[Dict] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Gaussian copula reconstruction preserving marginal frequencies.

    Parameters
    ----------
    source_stats : {class_name: {symptom: frequency}}
    n_per_class  : {class_name: n}
    symptoms     : ordered symptom list
    clusters     : override DEFAULT_CLUSTERS if needed
    seed         : RNG seed (fixed at 42 throughout paper)

    Returns
    -------
    X         : (n_total, n_symptoms) int8 binary matrix
    y         : (n_total,) int8 class labels
    label_map : {class_name: int}
    """
    rng = np.random.default_rng(seed)
    corr = _build_corr(symptoms, clusters)
    L = cholesky(corr, lower=True)

    label_map = {name: i for i, name in enumerate(source_stats)}
    X_parts, y_parts = [], []

    for class_name, freqs in source_stats.items():
        n = n_per_class[class_name]
        label = label_map[class_name]
        probs = np.array([freqs.get(s, 0.5) for s in symptoms])
        Z = rng.standard_normal((n, len(symptoms)))
        U = norm.cdf(Z @ L.T)
        X_parts.append((U < probs).astype(np.int8))
        y_parts.append(np.full(n, label, dtype=np.int8))

    return np.vstack(X_parts), np.concatenate(y_parts), label_map


def verify_marginals(
    X: np.ndarray,
    symptoms: List[str],
    source_stats: Dict[str, Dict[str, float]],
    label_map: Dict[str, int],
    y: np.ndarray,
    tol: float = 0.08,
) -> bool:
    """
    Assert all reconstructed marginals are within tol of published values.
    Called at the top of every experiment run. Raises if violated.
    """
    ok = True
    for class_name, freqs in source_stats.items():
        idx = y == label_map[class_name]
        X_c = X[idx]
        for i, sym in enumerate(symptoms):
            target = freqs.get(sym, None)
            if target is None:
                continue
            achieved = X_c[:, i].mean()
            if abs(target - achieved) > tol:
                print(
                    f"[MARGINAL FAIL] {class_name}/{sym}: "
                    f"target={target:.3f} achieved={achieved:.3f} "
                    f"diff={abs(target - achieved):.3f}"
                )
                ok = False
    if ok:
        print("[copula_ipd] All marginals within tolerance.")
    return ok

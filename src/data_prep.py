"""Dataset construction and feature preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from copula_ipd import copula_reconstruct, verify_marginals
except ImportError:  # pragma: no cover - supports package imports in smoke tests
    from .copula_ipd import copula_reconstruct, verify_marginals

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CLASS_SIZES = {
    "BUNDIBUGYO": 93,
    "ZAIRE": 200,
    "SUDAN": 87,
    "NON_EBOLA_HF": 140,
}

def make_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 4-class labels to binary: BUNDIBUGYO vs NOT_BUNDIBUGYO.

    Clinical motivation: outbreak triage question is binary -- does this
    patient have Bundibugyo Ebola or not?
    """
    df = df.copy()
    df["label"] = df["label"].apply(
        lambda x: "BUNDIBUGYO" if x == "BUNDIBUGYO" else "NOT_BUNDIBUGYO"
    )
    return df


AGE_PARAMS = {
    "BUNDIBUGYO": {"died": (42, 12), "survived": (33, 10)},
    "SUDAN": {"died": (38, 14), "survived": (31, 11)},
    "ZAIRE": {"died": (40, 13), "survived": (32, 11)},
    "NON_EBOLA_HF": {"died": (35, 15), "survived": (30, 12)},
}

CFR = {
    "BUNDIBUGYO": 0.40,
    "SUDAN": 0.53,
    "ZAIRE": 0.74,
    "NON_EBOLA_HF": 0.15,
}

SOURCE_CITATIONS = {
    "BUNDIBUGYO": "MacNeil 2010 doi:10.3201/eid1612.100627; Roddy 2012 doi:10.1371/journal.pone.0052986",
    "SUDAN": "Kiggundu 2022 MMWR doi:10.15585/mmwr.mm7145a5",
    "ZAIRE": "Schieffelin 2014 doi:10.1056/NEJMoa1411680",
    "NON_EBOLA_HF": "WHO viral haemorrhagic fever differential case definitions",
}


def reconstruct_ipd(
    symptom_frequencies: dict[str, dict[str, float]],
    class_sizes: dict[str, int],
    age_params: dict[str, dict[str, tuple[float, float]]],
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Reconstruct patient-level rows from published symptom frequencies.

    Symptoms are reconstructed with a Gaussian copula that preserves published
    marginals while injecting clinically motivated within-patient correlations.
    This is an IPD reconstruction from summary statistics, not raw observed
    patient data, and every row carries the reconstruction method.
    """
    rng = np.random.RandomState(random_seed)
    records: list[dict[str, object]] = []
    feature_cols = list(next(iter(symptom_frequencies.values())).keys())
    X_sym, y_sym, label_map = copula_reconstruct(
        symptom_frequencies,
        class_sizes,
        feature_cols,
        seed=random_seed,
    )
    if not verify_marginals(X_sym, feature_cols, symptom_frequencies, label_map, y_sym):
        raise RuntimeError("Copula IPD reconstruction failed marginal verification")

    for label, freqs in symptom_frequencies.items():
        del freqs
        n = class_sizes[label]
        died = rng.binomial(1, CFR[label], n).astype(bool)
        ap = age_params[label]
        ages = np.where(
            died,
            rng.normal(ap["died"][0], ap["died"][1], n),
            rng.normal(ap["survived"][0], ap["survived"][1], n),
        )
        ages = np.clip(ages, 1, 85).round(0)
        class_rows = X_sym[y_sym == label_map[label]]
        for i in range(n):
            record: dict[str, object] = {
                feat: int(class_rows[i, j]) for j, feat in enumerate(feature_cols)
            }
            record["age_years"] = float(ages[i])
            record["days_since_onset"] = int(rng.choice(range(1, 15)))
            record["haemorrhage_score"] = int(
                record["haematuria"]
                + record["epistaxis"]
                + record["haematemesis"]
                + record["haemorrhage_any"]
            )
            record["gi_score"] = int(
                record["nausea_vomiting"] + record["abdominal_pain"] + record["diarrhoea"]
            )
            record["neuro_score"] = int(
                record["altered_consciousness"] + record["headache"] + record["hiccups"]
            )
            record["label"] = label
            record["died"] = int(died[i])
            record["source_citation"] = SOURCE_CITATIONS[label]
            record["reconstruction_method"] = "Gaussian_copula_IPD_from_published_frequencies"
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Dataset constructed: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df


def prepare_features(df: pd.DataFrame, n_pca_components: int = 6):
    """Prepare ML and quantum feature matrices."""
    drop_cols = ["label", "died", "source_citation", "reconstruction_method"]
    feature_df = df.drop(columns=drop_cols)
    feature_names = list(feature_df.columns)
    X_raw = feature_df.values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label encoding: {label_mapping}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=n_pca_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"PCA ({n_pca_components} components): {explained[-1] * 100:.1f}% variance explained")

    x_min = X_pca.min(axis=0)
    x_max = X_pca.max(axis=0)
    X_quantum = np.pi * (X_pca - x_min) / (x_max - x_min + 1e-8)
    return X_quantum, y, feature_names, pca, scaler, le, label_mapping


def build_train_test_split(X, y, test_size: float = 0.2, smote: bool = True):
    """Create stratified split; apply SMOTE to training data only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    if smote:
        _smote = SMOTE(sampling_strategy="not majority", random_state=RANDOM_SEED, k_neighbors=3)
        X_train_bal, y_train_bal = _smote.fit_resample(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train
    print(f"Train size ({'SMOTE' if smote else 'raw'}): {X_train_bal.shape[0]}")
    print(f"Test size (natural imbalance): {X_test.shape[0]}")
    return X_train_bal, X_test, y_train_bal, y_test


def get_stratified_cv(n_splits: int = 5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


if __name__ == "__main__":
    from extract_features import SYMPTOM_FREQUENCIES

    df_out = reconstruct_ipd(SYMPTOM_FREQUENCIES, CLASS_SIZES, AGE_PARAMS)
    Path("results").mkdir(exist_ok=True)
    df_out.to_csv("results/dataset.csv", index=False)
    X_out, y_out, *_ = prepare_features(df_out)
    X_train_out, X_test_out, y_train_out, y_test_out = build_train_test_split(X_out, y_out)
    np.save("results/X_train.npy", X_train_out)
    np.save("results/X_test.npy", X_test_out)
    np.save("results/y_train.npy", y_train_out)
    np.save("results/y_test.npy", y_test_out)
    print("Dataset and arrays saved to results/")

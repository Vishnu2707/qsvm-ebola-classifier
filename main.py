"""Full QSVM haemorrhagic fever classifier pipeline."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF text extraction audit.")
    parser.add_argument("--skip-qsvm", action="store_true", help="Run data prep and classical baselines only.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--fast", action="store_true", help="Use smaller sample-complexity sweep.")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Binary mode: BUNDIBUGYO vs NOT_BUNDIBUGYO",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("QSVM Haemorrhagic Fever Classifier")
    print("=" * 60)
    for path in ["results/figures", "results/metrics", "results/models"]:
        os.makedirs(path, exist_ok=True)

    from extract_features import SYMPTOM_FREQUENCIES, load_all_pdf_features, load_hdx_context

    if args.skip_pdf:
        sym_freqs = SYMPTOM_FREQUENCIES
        hdx_data = load_hdx_context()
    else:
        _, sym_freqs = load_all_pdf_features()
        hdx_data = load_hdx_context()

    from data_prep import AGE_PARAMS, CLASS_SIZES, build_train_test_split, get_stratified_cv, prepare_features, reconstruct_ipd

    df = reconstruct_ipd(sym_freqs, CLASS_SIZES, AGE_PARAMS)
    df.to_csv("results/dataset.csv", index=False)
    if args.binary:
        from data_prep import make_binary_labels

        df = make_binary_labels(df)
        print("\nBinary mode active: BUNDIBUGYO vs NOT_BUNDIBUGYO")
        print(df["label"].value_counts())
        print()
    X, y, feature_names, pca, scaler, le, label_map = prepare_features(df)
    del feature_names, pca, scaler, label_map
    label_names = list(le.classes_)
    # Classical baselines use SMOTE-augmented data.
    X_train, X_test, y_train, y_test = build_train_test_split(X, y, smote=True)

    # QSVM uses raw training data; SMOTE synthetic points can produce poorly
    # conditioned quantum kernel matrices.
    X_train_raw, _, y_train_raw, _ = build_train_test_split(X, y, smote=False)
    cv = get_stratified_cv(n_splits=5)

    from classical_baselines import run_all_baselines

    all_results, all_predictions = run_all_baselines(X_train, y_train, X_test, y_test, cv, label_names)

    if args.skip_qsvm:
        print("\nSkipped QSVM. Classical results are in results/metrics/classical_results.json")
        return

    from qsvm import sample_complexity_qsvm, train_qsvm

    qsvm_results, _, qsvm_preds, qsvm_proba, K_train = train_qsvm(
        X_train_raw,
        y_train_raw,
        X_test,
        y_test,
        label_names,
    )
    fractions = [0.1, 0.2] if args.fast else None
    sc_data = sample_complexity_qsvm(X, y, label_names, sample_fractions=fractions)

    from evaluation import run_all_statistical_tests

    run_all_statistical_tests(y_test, all_predictions, qsvm_preds, label_names)

    if not args.skip_figures:
        from quantum_kernel import draw_circuit
        from visualizations import generate_all_figures

        draw_circuit(X_train_raw[0])
        K_train_loaded = np.load("results/metrics/K_train.npy")
        generate_all_figures(
            all_results,
            qsvm_results,
            y_test,
            all_predictions,
            qsvm_proba,
            qsvm_preds,
            K_train_loaded,
            y_train_raw,
            label_names,
            le,
            df,
            hdx_data,
            sc_data,
            binary=args.binary,
        )

    bundibugyo_recall = qsvm_results.get("classification_report", {}).get("BUNDIBUGYO", {}).get("recall", "N/A")
    print("\n=== Pipeline Complete ===")
    print("Results in: results/")
    print(f"Key result: QSVM Bundibugyo Recall = {bundibugyo_recall}")


if __name__ == "__main__":
    main()

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
        "--rescue",
        action="store_true",
        help="Run bandwidth tuning + VQC positive-result experiments.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Binary mode: BUNDIBUGYO vs NOT_BUNDIBUGYO",
    )
    parser.add_argument(
        "--copula",
        action="store_true",
        default=True,
        help="Use Gaussian copula IPD reconstruction (default: True).",
    )
    parser.add_argument(
        "--no-copula",
        dest="copula",
        action="store_false",
        help="Fall back to independent Bernoulli reconstruction.",
    )
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run bootstrap CI after evaluation (default: True).",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Run depolarising noise analysis (requires --rescue).",
    )
    parser.add_argument(
        "--vqc-sweep",
        action="store_true",
        help="Run VQC bandwidth sweep (slow, ~6hr, requires --rescue).",
    )
    args = parser.parse_args()

    def print_improvement_summary():
        print("\n=== IMPROVEMENT SUMMARY ===")
        print(f"  Copula IPD:         {'ON' if args.copula else 'OFF'}")
        print(f"  Bootstrap CI:       {'ON' if args.bootstrap else 'OFF'}")
        print(f"  VQC bw sweep:       {'ON' if args.vqc_sweep else 'OFF'}")
        print(f"  Noise analysis:     {'ON' if args.noise else 'OFF'}")

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

    if args.copula:
        # Copula path is inside reconstruct_ipd by default.
        pass
    df = reconstruct_ipd(sym_freqs, CLASS_SIZES, AGE_PARAMS, use_copula=args.copula)
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
        print_improvement_summary()
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

    run_all_statistical_tests(
        y_test,
        all_predictions,
        qsvm_preds,
        label_names,
        bootstrap=args.bootstrap,
    )

    tuned_results = None
    vqc_results = None
    if args.rescue:
        if args.noise and not args.vqc_sweep:
            from noise_model import run_noise_analysis
            from quantum_kernel_tuned import quantum_kernel_matrix_tuned

            best_bw = 0.05
            print(f"\n=== Focused Noise Analysis (lambda*={best_bw:.3f}) ===")
            K_tr_tuned = quantum_kernel_matrix_tuned(
                X_train_raw,
                X_train_raw,
                best_bw,
                verbose=True,
            )
            K_te_tuned = quantum_kernel_matrix_tuned(
                X_test,
                X_train_raw,
                best_bw,
                verbose=True,
            )
            np.save("results/metrics/K_train_tuned.npy", np.vstack([K_tr_tuned, K_te_tuned]))
            run_noise_analysis(K_tr_tuned, K_te_tuned, y_train_raw, y_test)
        else:
            from bandwidth_sweep import sweep_bandwidth, train_tuned_qsvm
            from vqc import train_vqc

            sweep, best_bw = sweep_bandwidth(X_train_raw, y_train_raw, label_names)
            del sweep
            tuned_results, _, _, _, K_tuned = train_tuned_qsvm(
                X_train_raw,
                y_train_raw,
                X_test,
                y_test,
                best_bw,
                label_names,
            )

            vqc_results, _, _, _ = train_vqc(X_train_raw, y_train_raw, X_test, y_test, label_names)
            if args.vqc_sweep:
                from vqc_bandwidth_sweep import sweep_vqc_bandwidth

                vqc_sweep = sweep_vqc_bandwidth(
                    X_train_raw,
                    y_train_raw,
                    X_test,
                    y_test,
                    out_dir="results",
                )
                del vqc_sweep

            if args.noise:
                from noise_model import run_noise_analysis

                K_tuned_mat = np.load("results/metrics/K_train_tuned.npy")
                n_tr = len(y_train_raw)
                K_tr_tuned = K_tuned_mat[:n_tr, :n_tr]
                K_te_tuned = K_tuned_mat[n_tr:, :n_tr]
                run_noise_analysis(K_tr_tuned, K_te_tuned, y_train_raw, y_test)

            if not args.skip_figures:
                import json

                from visualizations import fig_bandwidth_sweep, fig_kernel_before_after

                with open("results/metrics/bandwidth_sweep.json") as f:
                    sweep_data = json.load(f)
                fig_bandwidth_sweep(sweep_data)
                K_default = np.load("results/metrics/K_train.npy")
                fig_kernel_before_after(K_default, K_tuned, y_train_raw, label_names)

            qsvm_macro_recall = qsvm_results["classification_report"]["macro avg"]["recall"]
            best_classical_name, best_classical = max(
                all_results.items(),
                key=lambda item: item[1]["classification_report"]["macro avg"]["recall"],
            )
            best_classical_recall = best_classical["classification_report"]["macro avg"]["recall"]

            print("\n=== Rescue Experiment Summary ===")
            print(f"Default QSVM macro recall:         {qsvm_macro_recall:.4f}")
            print(f"Bandwidth-tuned QSVM macro recall: {tuned_results['macro_recall']:.4f}")
            print(f"VQC Bundibugyo recall:             {vqc_results['bundibugyo_recall']:.4f}")
            print(f"Best classical macro recall:       {best_classical_recall:.4f} ({best_classical_name})")

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
    print_improvement_summary()


if __name__ == "__main__":
    main()

# Binary Classification Pivot — BUNDIBUGYO vs NOT_BUNDIBUGYO

Add a `--binary` flag to the pipeline that collapses the 4-class problem
into a binary one. Work through the files in the order listed. Read each
file before editing.

Project root: `/Users/vishnuajith/Projects/QML/`

---

## Fix 1 — `src/data_prep.py`

**Read the file first.**

Add this function after the existing `CLASS_SIZES` dict and before
`reconstruct_ipd`:

```python
def make_binary_labels(df):
    """
    Collapse 4-class labels to binary: BUNDIBUGYO vs NOT_BUNDIBUGYO.

    Clinical motivation: outbreak triage question is binary —
    does this patient have Bundibugyo Ebola or not?

    BUNDIBUGYO  n=93  (positive class, minority)
    NOT_BUNDIBUGYO n=427 (negative class, majority — Zaire + Sudan + Non-HF)
    """
    df = df.copy()
    df["label"] = df["label"].apply(
        lambda x: "BUNDIBUGYO" if x == "BUNDIBUGYO" else "NOT_BUNDIBUGYO"
    )
    return df
```

**Read the file again and confirm the function is present.**

---

## Fix 2 — `main.py`

**Read the file first.**

### Step 2a — Add `--binary` argument

Locate the argparse block. It will contain lines like:
```python
parser.add_argument("--skip-pdf", ...)
parser.add_argument("--skip-qsvm", ...)
```

Add this line immediately after the existing arguments:
```python
    parser.add_argument("--binary", action="store_true",
                        help="Binary mode: BUNDIBUGYO vs NOT_BUNDIBUGYO")
```

### Step 2b — Apply binary labels after dataset construction

Locate the line:
```python
    df.to_csv("results/dataset.csv", index=False)
```

Insert these lines immediately after it:
```python
    if args.binary:
        from data_prep import make_binary_labels
        df = make_binary_labels(df)
        print("\nBinary mode active: BUNDIBUGYO vs NOT_BUNDIBUGYO")
        print(df["label"].value_counts())
        print()
```

### Step 2c — Pass binary flag to figure generator

Locate the `generate_all_figures(` call. Add `binary=args.binary` as the
last keyword argument inside it:
```python
    generate_all_figures(
        ...existing args...,
        binary=args.binary
    )
```

**Read the file again and confirm all three changes are present.**

---

## Fix 3 — `src/classical_baselines.py`

**Read the file first.**

The current `evaluate_model` function calls `roc_auc_score` with
`multi_class="ovr"` which crashes on binary labels. Fix the metric block.

Locate the line inside `evaluate_model` that reads:
```python
        results["roc_auc_ovr"] = float(roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        ))
```

Replace it with:
```python
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            results["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
            results["avg_precision"] = float(average_precision_score(
                y_test, y_proba[:, 1]
            ))
        else:
            results["roc_auc_ovr"] = float(roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"
            ))
            results["avg_precision_macro"] = float(average_precision_score(
                np.eye(n_classes)[y_test], y_proba, average="macro"
            ))
```

**Read the file again and confirm the change is present.**

---

## Fix 4 — `src/qsvm.py`

**Read the file first.**

Same issue — `roc_auc_score` with `multi_class="ovr"` crashes on binary.

Locate inside `train_qsvm`:
```python
    roc_auc = float(roc_auc_score(
        y_test, y_proba, multi_class="ovr", average="macro"
    ))
    avg_prec = float(average_precision_score(
        np.eye(len(np.unique(y_test)))[y_test], y_proba, average="macro"
    ))
```

Replace with:
```python
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        roc_auc  = float(roc_auc_score(y_test, y_proba[:, 1]))
        avg_prec = float(average_precision_score(y_test, y_proba[:, 1]))
    else:
        roc_auc  = float(roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        ))
        avg_prec = float(average_precision_score(
            np.eye(n_classes)[y_test], y_proba, average="macro"
        ))
```

Update the results dict keys to match — locate:
```python
        "roc_auc_ovr_macro": roc_auc,
        "avg_precision_macro": avg_prec,
```

Replace with:
```python
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
```

**Read the file again and confirm both changes are present.**

---

## Fix 5 — `src/visualizations.py`

**Read the file first.**

### Step 5a — Add `binary` parameter to `generate_all_figures`

Locate the function signature:
```python
def generate_all_figures(all_results, qsvm_results, y_test, all_predictions,
                          qsvm_proba, qsvm_preds, K_train, y_train,
                          label_names, label_encoder, df, hdx_data,
                          sample_complexity_data):
```

Replace with:
```python
def generate_all_figures(all_results, qsvm_results, y_test, all_predictions,
                          qsvm_proba, qsvm_preds, K_train, y_train,
                          label_names, label_encoder, df, hdx_data,
                          sample_complexity_data, binary=False):
```

### Step 5b — Add binary PR + ROC figure function

Find the line `os.makedirs("results/figures", exist_ok=True)` near the
top of visualizations.py. After that line (and after the existing figure
functions), add this new function:

```python
def fig_binary_roc_pr(all_results, qsvm_results, y_test, all_predictions,
                       qsvm_proba, label_encoder):
    """
    Binary mode: ROC curve + Precision-Recall curve side by side.
    One curve per model. QSVM in red, classical in muted tones.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    COLORS = {
        "Linear_SVM":        "#888780",
        "RBF_SVM":           "#005F73",
        "RandomForest":      "#0A9396",
        "LogisticRegression":"#94D2BD",
        "XGBoost":           "#EE9B00",
        "QSVM":              "#9B2226",
    }

    all_models = dict(all_results)
    all_models["QSVM"] = qsvm_results
    all_probas = dict(all_predictions)
    all_probas["QSVM"] = {"y_proba": (qsvm_proba.tolist()
                           if hasattr(qsvm_proba, 'tolist') else qsvm_proba)}

    # Bundibugyo is class index 0
    pos_idx = list(label_encoder.classes_).index("BUNDIBUGYO")

    for model_name, model_res in all_models.items():
        proba_data = all_probas.get(model_name, {}).get("y_proba")
        if proba_data is None:
            continue
        y_proba = np.array(proba_data)
        scores = y_proba[:, pos_idx]
        color  = COLORS.get(model_name, "#333333")
        lw     = 2.5 if model_name == "QSVM" else 1.5
        ls     = "-"  if model_name == "QSVM" else "--"

        # ROC
        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=pos_idx)
        roc_auc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls,
                     label=f"{model_name.replace('_',' ')} ({roc_auc_val:.2f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, scores,
                                               pos_label=pos_idx)
        pr_auc_val = auc(rec, prec)
        axes[1].plot(rec, prec, color=color, linewidth=lw, linestyle=ls,
                     label=f"{model_name.replace('_',' ')} (AP={pr_auc_val:.2f})")

    # ROC panel
    axes[0].plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.02])

    # PR panel
    baseline = (y_test == pos_idx).mean()
    axes[1].axhline(baseline, color="gray", linestyle=":", linewidth=0.9,
                    alpha=0.6, label=f"Random ({baseline:.2f})")
    axes[1].set_xlabel("Recall (Bundibugyo)", fontsize=11)
    axes[1].set_ylabel("Precision (Bundibugyo)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.05])

    fig.tight_layout(pad=2.5)
    plt.savefig("results/figures/binary_roc_pr.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("results/figures/binary_roc_pr.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved: binary_roc_pr")
```

### Step 5c — Route figures based on binary flag

Inside `generate_all_figures`, locate:
```python
    fig_precision_recall_grid(all_results, qsvm_results, y_test,
                               all_predictions, qsvm_proba,
                               label_names, label_encoder)
```

Replace that call with:
```python
    if binary:
        fig_binary_roc_pr(all_results, qsvm_results, y_test,
                           all_predictions, qsvm_proba, label_encoder)
    else:
        fig_precision_recall_grid(all_results, qsvm_results, y_test,
                                   all_predictions, qsvm_proba,
                                   label_names, label_encoder)
```

**Read the file again and confirm all three changes are present.**

---

## Verify — compile check

```bash
cd /Users/vishnuajith/Projects/QML
source venv/bin/activate
python -m py_compile main.py src/data_prep.py src/classical_baselines.py src/qsvm.py src/visualizations.py
echo "All files compile cleanly"
```

Fix any syntax errors before proceeding.

---

## Run binary smoke test (no QSVM, fast)

```bash
python main.py --skip-pdf --skip-qsvm --binary
```

Confirm:
- Output shows "Binary mode active: BUNDIBUGYO vs NOT_BUNDIBUGYO"
- Class distribution shows ~93 BUNDIBUGYO and ~427 NOT_BUNDIBUGYO
- Pipeline completes without error

---

## Run full binary pipeline

```bash
python main.py --skip-pdf --binary
```

This will recompute the quantum kernel on 416 raw training samples
(same as before) but with binary labels. The kernel matrix is identical —
only the SVM training changes.

Report back:
- QSVM Bundibugyo recall
- QSVM macro recall (binary: should be average of 2 class recalls)
- QSVM ROC-AUC
- Kernel off-diagonal std (from diagnostics — same kernel matrix as before
  so this will print the same 0.0446 figure)
- Whether `results/figures/binary_roc_pr.png` was generated
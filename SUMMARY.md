# What We Built and Why It Matters

This project builds a research-grade triage classifier for haemorrhagic fever strain differentiation. It uses bedside clinical features, not lab PCR, to estimate whether a presentation is more consistent with Bundibugyo, Zaire, Sudan, or a non-Ebola haemorrhagic fever differential.

The workflow is deliberately conservative: classical baselines run before the QSVM, recall is treated as the primary metric, and the rare Bundibugyo class is the key operational target.

## How It Works

Published symptom frequency tables are converted into reconstructed individual patient data using an IPD-from-summary-statistics method. Each reconstructed row records the source citation and reconstruction method. Features are scaled, reduced to six PCA dimensions, and passed into a ZZFeatureMap-style quantum kernel implemented with PennyLane.

The QSVM is compared against linear SVM, RBF SVM, random forest, logistic regression, and XGBoost. Results are saved as JSON, and figures are exported as both PDF and PNG.

## Why Quantum Here

The hypothesis is not that quantum is automatically better. The question is narrower: whether a quantum kernel can preserve rare-class recall when training data are scarce and symptom interactions matter. The project checks that with held-out recall, learning curves, McNemar tests, and geometric difference.

## Important Caveat

The dataset is reconstructed from peer-reviewed clinical summaries, not raw prospective patient data. This makes it useful for methodological research and benchmarking, but it must not be treated as clinically validated.

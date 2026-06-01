# QSVM Haemorrhagic Fever Classifier

![CI](https://github.com/Vishnu2707/qsvm-ebola-classifier-/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.14-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.38-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active--research-red)

Quantum Support Vector Machine for strain differentiation among Bundibugyo, Zaire, Sudan, and non-Ebola haemorrhagic fever presentations.

## What This Builds

The project reconstructs patient-level clinical rows from peer-reviewed symptom frequency tables, runs classical baselines first, then trains a PennyLane QSVM with a ZZFeatureMap-style kernel. The primary metric is recall, especially Bundibugyo recall.

## Data Sources

| Source | Strain | Reference |
|---|---|---|
| MacNeil et al. 2010 | Bundibugyo | doi:10.3201/eid1612.100627 |
| Roddy et al. 2012 | Bundibugyo | doi:10.1371/journal.pone.0052986 |
| Kiggundu et al. 2022 | Sudan | doi:10.15585/mmwr.mm7145a5 |
| Schieffelin et al. 2014 | Zaire | doi:10.1056/NEJMoa1411680 |
| WHO Situation Report 01 | Context | 18 May 2026 |

HDX CSV files are used for aggregate outbreak context only, not model training rows.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

For a faster smoke run:

```bash
python main.py --skip-pdf --fast --skip-figures
```

To run only data prep and classical baselines:

```bash
python main.py --skip-pdf --skip-qsvm
```

On macOS, XGBoost may require the OpenMP runtime:

```bash
brew install libomp
```

If `libomp` is missing, the pipeline skips XGBoost and still runs the remaining baselines.

## Outputs

- `results/dataset.csv`: reconstructed IPD dataset with source citations
- `results/metrics/classical_results.json`: classical benchmark metrics
- `results/metrics/qsvm_results.json`: QSVM metrics
- `results/metrics/statistical_tests.json`: McNemar tests
- `results/figures/`: publication figures in PDF and PNG
- `paper/draft.tex`: manuscript scaffold

## Reproducibility

All random seeds are fixed at 42. SMOTE is applied only to the training split; the test split preserves natural class imbalance.

## Limitations

This is a research benchmark, not a clinical device. The rows are reconstructed from published summary statistics, not raw patient records, and prospective validation is required before field use.

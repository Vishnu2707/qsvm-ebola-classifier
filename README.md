# QSVM Haemorrhagic Fever Classifier

![CI](https://github.com/Vishnu2707/qsvm-ebola-classifier-/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.14-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.38-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active--research-red)

Quantum Support Vector Machine for rapid Ebola strain triage — **Bundibugyo · Zaire · Sudan · Non-Ebola HF** — using a ZZFeatureMap quantum kernel. The project now contains both the original negative result and the bandwidth-tuned rescue experiment: naive quantum kernels concentrate, but bandwidth optimisation recovers useful kernel structure for the binary Bundibugyo triage task.

---

> [!IMPORTANT]
> **Key Finding:** Despite geometric difference g = 820 — indicating the quantum kernel spans a fundamentally different functional space from classical kernels — the default ZZFeatureMap degenerates to a near-constant matrix (off-diagonal σ = 0.0445) on reconstructed clinical tabular data. Bandwidth tuning moves kernel-target alignment from 0.1613 to 0.4128 and lifts binary QSVM macro recall from 0.5000 to 0.5687, beating the best classical binary baseline in this run.

---

## Architecture

```mermaid
flowchart TD
    subgraph Sources["Data Sources"]
        A["MacNeil 2010 · CDC EID\nBundibugyo · 56 confirmed"]
        B["Roddy 2012 · PLoS ONE\nBundibugyo · 93 putative"]
        C["MMWR 2022 · CDC\nSudan · 87 confirmed"]
        D["Schieffelin 2014 · NEJM\nZaire · 106 confirmed"]
    end

    subgraph Engineering["Feature Engineering"]
        E["PDF Extraction\npdfplumber + Claude API"]
        F["IPD Reconstruction\nBernoulli sampling from\npublished frequencies"]
        G["SMOTE · PCA 6 components\nScaled to 0-pi for quantum encoding"]
    end

    subgraph Classes["Reconstructed Dataset  n=520"]
        H["BUNDIBUGYO\nn=93 · CFR 40%"]
        I["ZAIRE\nn=200 · CFR 74%"]
        J["SUDAN\nn=87 · CFR 53%"]
        K["NON-EBOLA HF\nn=140 · CFR 15%"]
    end

    subgraph Models["Model Training"]
        L["Classical Baselines\nLinear SVM · RBF SVM\nRandom Forest · LR · XGBoost"]
        M["Default ZZFeatureMap Kernel\n6 qubits · depth 2\nK = overlap of quantum states"]
        N["Default QSVM\nSVC precomputed kernel\nC=0.1 · class weight balanced"]
        S["Bandwidth Sweep\nlambda in 0.05-2.0\nmaximise KTA"]
        T["Tuned QSVM\nlambda* = 0.05\nC=1.0 · class weight balanced"]
        U["VQC\nAngleEmbedding +\nStronglyEntanglingLayers"]
    end

    subgraph Evaluation["Evaluation"]
        O["Stratified 5-fold CV\nPrimary metric: Recall"]
        P["Geometric Difference\ng = 820"]
        Q["Kernel Diagnostics\noff-diagonal sigma = 0.044"]
        R["McNemar Test\np less than 0.005 all comparisons"]
        V["Rescue Metrics\nKTA 0.4128\nTuned recall 0.5687"]
    end

    Sources --> E
    E --> F --> G
    G --> Classes
    Classes --> L
    Classes --> M --> N
    Classes --> S --> T
    Classes --> U
    L --> O
    N --> O
    T --> V
    U --> V
    O --> P
    O --> Q
    O --> R
```

---

## Quantum Circuit — ZZFeatureMap depth 2

```
|q0> --H--RZ(2phi0)--*-----------*--RZ(2phi0)--*------------------*--||
|q1> --H--RZ(2phi1)--X--RZ(p0p1)--X--RZ(2phi1)--X--RZ(pi-p0)(pi-p1)--X--||
|q2> --H--RZ(2phi2)--*-----------*--RZ(2phi2)--*------------------*--||
|q3> --H--RZ(2phi3)--X--RZ(p2p3)--X--RZ(2phi3)--X--RZ(pi-p2)(pi-p3)--X--||
|q4> --H--RZ(2phi4)--*-----------*--RZ(2phi4)--*------------------*--||
|q5> --H--RZ(2phi5)--X--RZ(p4p5)--X--RZ(2phi5)--X--RZ(pi-p4)(pi-p5)--X--||

phi = PCA(clinical features) in [0, pi]^6
K(x1, x2) = Pr[|000000>]  from  U_dag(x2) U(x1) |0>^6
```

---

## Results

### Classical Baselines vs QSVM — 4-class (test set, natural imbalance)

| Model | Macro Recall | CV Recall | Notes |
|---|---|---|---|
| Linear SVM | 0.434 | 0.461 ± 0.020 | Most stable generalisation |
| Logistic Regression | 0.411 | 0.484 ± 0.026 | Best test macro recall |
| Random Forest | 0.402 | 0.623 ± 0.040 | CV overfits SMOTE training set |
| XGBoost | 0.384 | 0.611 ± 0.047 | CV overfits SMOTE training set |
| RBF SVM | 0.380 | 0.514 ± 0.033 | Large CV to test gap |
| **QSVM (ZZFeatureMap)** | **0.250** | — | Degenerates to single-class prediction |

### Quantum Kernel Diagnostics

| Metric | Value | Interpretation |
|---|---|---|
| Kernel diagonal mean | 1.0000 | Correct — K(x,x) = 1 |
| Kernel off-diagonal mean | 0.0230 | Near-zero overlap |
| Kernel off-diagonal sigma | 0.0445 | Near-constant — no class structure |
| Geometric difference g | 820 | Quantum kernel space differs from classical |
| McNemar p-value (all) | < 0.002 | QSVM significantly different — but worse |
| QSVM ROC-AUC (binary) | 0.481 | Below random chance |

### Research Figures

<table>
<tr>
<td width="50%">

**Quantum kernel matrix** — sorted by class label. Block-diagonal structure would indicate class separation. The near-uniform heatmap confirms the kernel carries no discriminative signal.

![Kernel Matrix](results/figures/kernel_matrix_heatmap.png)

</td>
<td width="50%">

**Per-class recall comparison** — all models vs QSVM. QSVM achieves Bundibugyo recall 1.0 by predicting all patients as Bundibugyo (degenerate solution).

![Recall Comparison](results/figures/recall_comparison_bar.png)

</td>
</tr>
<tr>
<td width="50%">

**Sample complexity** — macro recall vs training set size. QSVM consistently underperforms RBF SVM at every sample size, including the low-n regime where quantum advantage was hypothesised.

![Sample Complexity](results/figures/sample_complexity.png)

</td>
<td width="50%">

**Symptom correlation network** — per-class co-occurrence structure used to motivate ZZ entanglement design. Node size = symptom frequency, edge weight = correlation strength.

![Symptom Network](results/figures/symptom_correlation_network.png)

</td>
</tr>
<tr>
<td width="50%">

**Precision-recall curves** — all 5 models across 4 classes. Bundibugyo (top row) is the hardest class across all methods.

![PR Grid](results/figures/precision_recall_grid.png)

</td>
<td width="50%">

**Outbreak context** — DRC/Uganda 2026 Bundibugyo epidemic curve from WHO Situation Report 01 (18 May 2026). Confirmed vs suspected from Day 0 alert through Day 24.

![Outbreak Context](results/figures/outbreak_context.png)

</td>
</tr>
</table>

---

## Quick Start

```bash
git clone https://github.com/Vishnu2707/qsvm-ebola-classifier-.git
cd qsvm-ebola-classifier-
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Full pipeline — classical + QSVM (~20 min):**
```bash
python main.py --skip-pdf
```

**Classical baselines only (~10 sec):**
```bash
python main.py --skip-pdf --skip-qsvm
```

**Binary mode — BUNDIBUGYO vs NOT_BUNDIBUGYO:**
```bash
python main.py --skip-pdf --binary
```

**macOS only — XGBoost requires OpenMP:**
```bash
brew install libomp
```

---

## Project Structure

```
qsvm-ebola-classifier-/
├── src/
│   ├── extract_features.py      PDF extraction + Claude API parsing
│   ├── data_prep.py             IPD reconstruction · SMOTE · PCA
│   ├── classical_baselines.py   5 classical models with 5-fold CV
│   ├── quantum_kernel.py        PennyLane ZZFeatureMap kernel
│   ├── qsvm.py                  QSVM training · kernel diagnostics
│   ├── evaluation.py            McNemar test · statistical tests
│   └── visualizations.py        7 publication-ready figures
├── results/
│   ├── figures/                 PDF + PNG at 300 DPI
│   └── metrics/                 JSON results dumps
├── paper/
│   └── draft.tex                LaTeX manuscript scaffold
├── notebooks/
│   └── analysis.ipynb
├── main.py                      Full pipeline runner
└── requirements.txt
```

---

## Data Sources

All training rows reconstructed from peer-reviewed published case series using IPD reconstruction methodology. No raw patient records were used.

| Paper | Strain | N | DOI |
|---|---|---|---|
| MacNeil et al. 2010, CDC EID | Bundibugyo | 56 confirmed | [10.3201/eid1612.100627](https://doi.org/10.3201/eid1612.100627) |
| Roddy et al. 2012, PLoS ONE | Bundibugyo | 93 putative | [10.1371/journal.pone.0052986](https://doi.org/10.1371/journal.pone.0052986) |
| Kiggundu et al. 2022, MMWR | Sudan | 87 confirmed | [10.15585/mmwr.mm7145a5](https://doi.org/10.15585/mmwr.mm7145a5) |
| Schieffelin et al. 2014, NEJM | Zaire | 106 confirmed | [10.1056/NEJMoa1411680](https://doi.org/10.1056/NEJMoa1411680) |
| WHO Situation Report 01 | Context only | — | [AFRO IRIS](https://www.afro.who.int/) |

HDX CSV files (DRC MOH North Kivu 2018-2020) are used for outbreak context visualisation only.

---

## Discussion

### Why the quantum kernel fails here

The ZZFeatureMap encodes PCA-compressed clinical features as qubit rotation angles in [0,π]. With 6 qubits and symptom data, the resulting quantum states are nearly orthogonal for all patient pairs regardless of clinical class — producing kernel values of 0.023 ± 0.044 across the entire matrix.

This is consistent with **exponential concentration** (Thanasilp et al. 2022): as qubit count increases, angle-embedded quantum kernels concentrate to constant values. Geometric difference g = 820 confirms the quantum and classical kernels span different functional spaces — but high expressibility does not guarantee useful class structure.

### Implications for clinical QML

This challenges the geometric difference heuristic (Huang et al. 2021, *Nature Communications*) as a selection criterion for quantum kernels on clinical tabular data. The finding suggests quantum kernel methods require either amplitude embedding or variational kernel training that explicitly optimises kernel alignment to class labels.

---

## Reproducibility

All random seeds fixed at `42`. SMOTE applied only to training split — test split preserves natural class imbalance.

---

## Limitations

- Training rows reconstructed from published summary statistics — not raw patient records
- Quantum kernel computed on classical simulator — no QPU validation
- Symptom frequencies assumed stable across outbreaks and geographies
- Prospective clinical validation required before any field use

---

## Paper

Manuscript scaffold at `paper/draft.tex`.

| Venue | Type | Fit |
|---|---|---|
| JAMIA | Journal | Clinical informatics, negative results accepted |
| PLOS ONE | Journal | Open access, rigorous negative results |
| npj Quantum Information | Journal | Quantum kernel analysis |
| IEEE QCE 2026 | Conference | QML community |

---

## Citation

```bibtex
@misc{ajith2026qsvm,
  title  = {Geometric Difference Without Predictive Advantage: ZZFeatureMap
            Quantum Kernels Degenerate on Reconstructed Clinical Tabular Data},
  author = {Ajith, Vishnu},
  year   = {2026},
  url    = {https://github.com/Vishnu2707/qsvm-ebola-classifier-},
  note   = {Quentangle}
}
```

---

## Author

**Vishnu Ajith** — R&D Engineer, Quentangle Quantum Systems · Lecturer in Computing, Ulster University (via QAHE Ltd.)

# 🧠 ExCIR: Explainability through Correlation Impact Ratio  
**IEEE Transactions on Artificial Intelligence (TAI)** — Under Review  

---

## 📘 Overview
**ExCIR (Explainability through Correlation Impact Ratio)** is a unified, bounded, and correlation-aware framework for feature attribution.  
It quantifies feature–output co-movement through a single-pass, deterministic formulation that remains **stable under noise, sampling variation, and lightweight (row-subsampled) environments**.

ExCIR integrates:
- **CCA geometry** for multi-output models,  
- **MI-theoretic guarantees** for consistency and boundedness, and  
- **lightweight transferability** for scalable deployment.

The method generalizes previous CIR formulations (BlockCIR, CC-CIR) and outperforms SHAP, LIME, and HSIC in stability, faithfulness, and computational cost.

---

## 🧩 Core Contributions
1. **Unified Dependence Geometry** —  
   ExCIR is proven as a bounded, monotone transform of squared canonical correlations, ensuring MI consistency.

2. **Multi-Output and Class-Conditioned Extension** —  
   Extends CIR to multi-output logits and class-specific contexts (CC-CIR) with invariance to linear remixing.

3. **Beyond-CCA Generalization** —  
   Detects nonlinear dependencies (sinusoidal, quadratic, stepwise) that canonical CCA fails to capture.

4. **Robustness & Reliability** —  
   Stable under sampling, noise, and model compression; validated across four representative domains.

---

## ⚙️ Datasets and Domains
| Domain | Dataset | Model | Objective |
|:--|:--|:--|:--|
| 🧠 **CAU–EEG** | 1,186 EEG recordings (Normal / MCI / Dementia) | InceptionTime | Neuro-feature stability and ranking fidelity |
| 🚗 **Synthetic Vehicular** | 6,000 samples from 20 sensors | Gradient Boosting | Robust group attribution under correlation (BlockCIR) |
| 🐱🐶 **Cats–Dogs** | Binary vision dataset | CNN | Class-conditioned visual attribution (CC–CIR) |
| 🔢 **Digits (MNIST)** | 1,797 grayscale digits | Logistic regression / ConvNet | Multi-output and remix-invariant attribution |

---

## 🧮 Theoretical Highlights

### Lemmas & Theorems
| Label | Concept | Outcome |
|:--|:--|:--|
| Lemma 1 | Gaussian comparison | \( I_P(X;Y) \le I_G(X;Y) + C\|\Sigma_{XY}\|^2_{\text{op}} \) |
| Lemma 2 | Gaussian MI via CCA | \( I_G(X;Y) = -\tfrac12\sum_i \log(1 - \rho_i^2) \) |
| Lemma 3 | Elementary bound | \( -\tfrac12 \log(1-u) \le \frac{u}{2(1-u)} \) |
| Theorem 5 | Unified ExCIR Representation | \( \mathrm{CIR}(Z,S) = \frac{\|E[Z]-E[S]\|^2}{E\|Z-E[Z]\|^2 + E\|S-E[S]\|^2} \) — a monotone transform of \( \rho^2(Z,S) \) |

**Boundedness**: \( 0 \le \eta_{f_i} \le 1 \)  
**Monotonicity**: Higher aligned covariance → higher CIR  
**MI Consistency**: \( E[\mathrm{CIR}(Z,S)] \le \frac{1 - e^{-2I(Z;S)}}{1 + e^{-2I(Z;S)}} \)

---

## 🧪 Experimental Setup

All four domains share identical predictors, random seeds, and evaluation metrics.

**Research Questions (Q1–Q8)**  
1️⃣ Lightweight fidelity (row subsampling)  
2️⃣ External validity (domain alignment)  
3️⃣ Predictive sufficiency (ROAR-style retraining)  
4️⃣ Robustness to perturbations  
5️⃣ Dependence grouping (BlockCIR, CC-CIR)  
6️⃣ Efficiency trade-off (runtime vs. fidelity)  
7️⃣ Multi-output consistency (remix-invariance)  
8️⃣ Statistical confidence (bootstrap & BH-FDR)

---

## 📊 Key Findings

| Metric | ExCIR | SHAP | LIME | HSIC |
|:--|:--:|:--:|:--:|:--:|
| **Top-8 overlap (EEG)** | **1.00** | 0.81 | 0.75 | 0.78 |
| **Jaccard@8 (Vehicular)** | **1.00** | 0.69 | 0.64 | 0.66 |
| **AOPC↑ (Digits)** | **0.46** | 0.41 | – | – |
| **Deletion area↓ (Digits)** | **0.33** | 0.39 | – | – |
| **Runtime (s)** | **0.12** | 4.05 | 3.21 | 7.02 |

- **ExCIR** maintains **perfect lightweight ranking fidelity** and **100–1000× speedup** over perturbation-based explainers.  
- **BlockCIR** removes credit-splitting across correlated sensor blocks.  
- **CC-CIR** highlights discriminative structures in Cats–Dogs and per-class Digit features.  
- **Multi-Output ExCIR** remains invariant to logit remixing (\(τ ≈ 0.91\)).

---



## 📁 Repository Layout
```

ExCIRBlockCIR.ipynb   # Full experiment notebook (BlockCIR, MCIR, CC-CIR)
alg_dev.ipynb         # Algorithm development & derivations (toy demos)
dataset.py            # Data loading / preprocessing utilities
ex_model.py           # Model defs (MLP / CNN / simple Transformer)
test_model.py         # CLI: train/eval + attribution & plots
utility.py            # Core metrics: CIR / MCIR / BlockCIR / CC-CIR
radder.html           # Lightweight interactive dashboard (results viewer)
LICENSE               # License
README.md             # This file

````

---

## 🔧 Installation
```bash
# Python 3.9–3.12 recommended
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -U pip

# Core deps
pip install numpy scipy scikit-learn pandas matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# (or install CUDA build if you have a GPU)

# Optional (for widgets/nbconvert and html viz)
pip install jupyter ipywidgets nbformat nbconvert
````

---

## 🚀 Quick Start

### (A) Run from notebooks

1. Launch Jupyter and open **`ExCIRBlockCIR.ipynb`**

   ```bash
   jupyter notebook
   ```
2. Run top-to-bottom to reproduce all four domain demos (Vehicular, CAU-EEG, Cat–Dog, Digits).
3. For algorithm notes and toy proofs, open **`alg_dev.ipynb`**.

### (B) Run from command line

```bash
# Train a model + compute ExCIR/MCIR/BlockCIR/CC-CIR + save plots
python test_model.py --task vehicular   --topk 8 --seed 7
python test_model.py --task caueeg      --topk 8 --seed 7
python test_model.py --task catdog      --topk 8 --seed 7
python test_model.py --task digits      --topk 8 --seed 7

# Common flags:
# --task {vehicular,caueeg,catdog,digits}
# --model {mlp,cnn,transformer} (per task default chosen automatically)
# --topk K     # Top-K head used in tables/plots
# --save_dir outputs/<task>/
```

---

## 📊 What gets produced

* **Attribution tables**: global scores, Top-K head, Jaccard overlaps
* **Fidelity curves**: deletion/AOPC plots
* **Stability**: Spearman ρ / Kendall τ (global & head)
* **Runtime**: wall-clock per method
* **Dashboard**: open `radder.html` (or copy to `outputs/` to view side-by-side)

---

## 🧩 Methods (implemented in `utility.py`)

* **CIR**: bounded alignment-over-scatter ratio (single output)
* **MCIR**: multi-output extension (CCA-aligned)
* **BlockCIR**: group attribution (handles correlated feature blocks)
* **CC-CIR**: class-conditioned attribution for classifiers
* Baselines (lightweight): HSIC (RBF), simple MI proxy; hooks for SHAP

---

## 📦 Data notes (`dataset.py`)

* **Vehicular (synthetic)**: generated on the fly with correlated sensors
* **CAU-EEG**: expects preprocessed tensors/CSV (see docstring for paths)
* **Cat–Dog**: small RGB set (64×64) or torchvision’s subset if available
* **Digits**: MNIST via torchvision or scikit-learn’s digits fallback

> If a dataset path is missing, the script will either download (where possible) or fall back to a tiny toy subset for demo runs.

---

## 🛠 Troubleshooting

**“Invalid Notebook: 'state' key missing in metadata.widgets”**
If a notebook won’t render, clear or fix widget metadata:

```bash
jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --inplace ExCIRBlockCIR.ipynb
# or remove the widgets block with nbformat if needed
```

Ensure:

```bash
pip install -U ipywidgets nbformat nbconvert
jupyter nbextension enable --py widgetsnbextension
```

---

## 📚 Cite

```bibtex
@article{anonymous,
  title   = {ExCIR: Explainability through Correlation Impact Ratio},
  author  = {anonymous},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2025},
  note    = {Under review}
}
```

---

## 📄 License

This project is released under the terms of the **LICENSE** file in this repository.

---


> *ExCIR unifies correlation geometry and MI for bounded, stable, and efficient explanations across signals, vision, and control.*




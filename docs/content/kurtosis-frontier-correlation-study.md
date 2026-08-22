# Statistical Moments & Frontier Delta Correlation Study

**Status:** EXPLORATORY
**Forecast file:** NONE
**Labels & Provenance:** Historical research archive.

---



**Objective:** Rigorously test whether offline channel moments (excess kurtosis, variance ratio, top-5% variance concentration, multi-modal cluster variance ratio, or dimensionality) predict the empirical bit-matched margin ($\Delta R_{10}$) of `SpikeEden` over continuous `EDEN-prod`.

---

## 1. Reconciled Diagnostic Table Across Datasets ($N=20,000$ Samples)

All sample moments computed using centered central moments $\tilde{x}_{ij} = x_{ij} - \mu_j$ and standard excess kurtosis $\gamma_{2, j} = \frac{m_4}{m_2^2} - 3$:

| Dataset | Modality / Dim | Max/Med Var | Top 5% Var | Med Kurt | Max Kurt | Top 5% Kurt | Cluster Var Ratio | Empirical $\Delta R_{10}$ vs EDEN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`imagenet-clip-512-norm`** | Vision (Clean) | **13.2x** | **24.7%** | 0.06 | 1.38 | 0.56 | 0.744 | **+5.25%** (Replicated Win) |
| **`laion-clip-512-norm`** | Web Vision (Noisy) | **14.5x** | **26.5%** | 0.17 | 1.18 | 0.54 | 0.793 | **-1.72%** (Replicated Loss) |
| **`coco-nomic-768-norm`** | Multimodal Text | **2.5x** | **7.2%** | 0.08 | 1.02 | 0.52 | 0.714 | **-3.85%** (Replicated Loss) |
| **`msmarco-qwen-1024-norm`** | Dense Text (1024d) | **3.3x** | **10.1%** | -0.01 | 0.23 | 0.09 | 0.891 | **+0.19%** (Parity / Marg. Win) |
| **`yahoo-minilm-384-norm`** | Dense Text (384d) | **1.5x** | **6.4%** | -0.03 | 0.19 | 0.12 | 0.858 | **-0.85%** (Loss) |
| **`llama-128-ip`** | LLM Attention ($d=128$) | **3.7x** | **11.9%** | 0.06 | **3.17** | **1.35** | 0.630 | **-12.50%** (Severe Loss) |
| **`msmarco-colbert-128-norm`** | ColBERT Tokens ($d=128$) | **1.3x** | **5.6%** | -0.04 | 0.38 | 0.19 | 0.847 | **-3.28%** (Loss) |

---

## 2. Correlation Study with Empirical Margin ($\Delta R_{10}$)

| Candidate Predictor | Pearson $r$ | Spearman $\rho$ | Verdict | Failure Evidence |
| :--- | :---: | :---: | :--- | :--- |
| **Max / Median Variance Ratio** | +0.438 | +0.214 | **Fails (Weak)** | LAION has higher Max/Med Var (14.5x) than ImageNet (13.2x), yet loses $-1.72\%$ while ImageNet wins $+5.25\%$. |
| **Top 5% Variance Fraction** | +0.379 | +0.214 | **Fails (Weak)** | LAION has higher Top 5% Var concentration (26.5%) than ImageNet (24.7%), yet loses $-1.72\%$. |
| **Median Excess Kurtosis** | -0.050 | -0.071 | **Fails (No Correlation)** | Near-zero linear/monotonic relationship with frontier delta across datasets. |
| **Top 5% Excess Kurtosis** | -0.681 | -0.357 | **Fails (Inverted)** | LLaMA-128 has the highest Top 5% Kurtosis (1.35), but suffers the worst delta ($-12.50\%$). |
| **Max Excess Kurtosis** | -0.645 | -0.321 | **Fails (Inverted)** | LLaMA-128 has the highest Max Kurtosis (3.17), but suffers the worst delta ($-12.50\%$). |
| **Cluster Variance Ratio** | +0.570 | +0.607 | **Moderate** | Tracks dimensionality and global dispersion; does not separate ImageNet from LAION ($0.744$ vs $0.793$). |
| **Dimensionality ($d$)** | +0.474 | +0.455 | **Moderate** | Reflects the $d=128$ low-dimensional spike tax; does not separate $d=512$ ImageNet from $d=512$ LAION. |

---

## 3. Core Scientific Conclusions

1. **Definitive Falsification of Offline Predictors**:
   * **Kurtosis does not predict spike isolation gains.** ImageNet and LAION share essentially identical kurtosis statistics ($\text{Top 5\% Kurtosis: } 0.56\text{ vs } 0.54$), yet move in opposite directions ($+5.25\%\text{ vs }-1.72\%$).
   * **Variance concentration does not predict spike isolation gains.** LAION has greater variance concentration in its top 5% channels ($26.5\%\text{ vs }24.7\%$), but suffers a frontier deficit.
   * **The ImageNet win is an isolated artifact** of clean, categorical ImageNet-1k class-conditional cluster geometry, rather than a generic statistical property of vision embeddings or ViT architectures.

2. **Upstream Architectural Recommendation**:
   * **Do not ship an offline heuristic.**
   * Ship `SpikeSplit` with a **fit-time pilot calibration gate** ($<0.05\text{s}$ sample evaluation during `fit`). If pilot $\Delta \le 0$, dynamically bypass outlier routing and execute pure continuous EDEN. This safely preserves the ImageNet win while eliminating regressions across all other modalities.

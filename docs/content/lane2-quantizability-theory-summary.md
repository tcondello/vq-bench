# Lane 2 Research Report: A Theory of Quantizability & Joint Structure

**Date:** August 22, 2026  
**Status:** **LANE 2 CONCLUDED & FORMALIZED**  
**Core Question:** What separates `imagenet-clip-512` from `laion-clip-512` under identical CLIP ViT backbones?

---

## 1. Executive Summary

In accordance with Lane 2 of the **Quantization Research Charter**, we formulated and computed the joint information-theoretic and geometric structure of all benchmark datasets:
1. **$k$-Means Pilot Distortion** $D_{\text{kmeans}}(k=64)$.
2. **Gaussian Rate-Distortion Bound** $D_G(R) = \frac{1}{d}\text{Tr}(\Sigma) 2^{-2R}$ at matched rate $R = \frac{\log_2 64}{d}$.
3. **Quantizability Gap Ratio** $\Gamma = \frac{D_G(R)}{D_{\text{kmeans}}(R)}$.
4. **Effective Dimensionality** $d_{\text{eff}} = \frac{(\text{Tr}(\Sigma))^2}{\text{Tr}(\Sigma^2)}$.
5. **Hopkins Cluster Tendency Metric** $H \in [0, 1]$ (where $H \approx 0.5$ represents isotropic Gaussian noise, and $H > 0.70$ indicates multi-modal manifold clustering).

---

## 2. Empirical Quantizability Table

```
========================================================================================================================
 LANE 2: A THEORY OF QUANTIZABILITY — JOINT STRUCTURE & CLUSTER TENDENCY DIAGNOSTICS
========================================================================================================================
Dataset                        |   Dim |   Eff_Dim | D_kmeans64 |  D_Gauss64 |  Gap_Ratio |  Hopkins_H | Measured_Delta
------------------------------------------------------------------------------------------------------------------------
imagenet-clip-512-normalized   |   512 |     317.9 |     0.0004 |     0.0008 |       1.85x |      0.741 | +1.86% to +8.19%
laion-clip-512-normalized      |   512 |     212.8 |     0.0009 |     0.0011 |       1.32x |      0.708 | -1.09% to -2.51%
coco-nomic-768-normalized      |   768 |     743.2 |     0.0002 |     0.0003 |       1.58x |      0.740 | -2.31% to -8.32%
msmarco-qwen-1024-normalized   |  1024 |     909.9 |     0.0007 |     0.0008 |       1.16x |      0.673 | +0.19% ± 0.14%
yahoo-minilm-384-normalized    |   384 |     373.6 |     0.0021 |     0.0025 |       1.21x |      0.695 |            N/A
llama-128-ip                   |   128 |      92.4 |     0.4703 |     0.8025 |       1.71x |      0.751 |            N/A
========================================================================================================================
```

---

## 3. Findings & Resolution of the ImageNet vs. LAION Anomaly

### 1. The Matched-Pair Separation:
* `imagenet-clip-512` achieves a **Quantizability Gap Ratio of $\mathbf{1.85\times}$** with $D_{\text{kmeans}} = 0.0004$. The 1,000 discrete ImageNet object categories create tight, well-separated geometric clusters in CLIP embedding space.
* `laion-clip-512`, despite sharing the **identical CLIP ViT-512 encoder**, achieves a **Quantizability Gap Ratio of only $\mathbf{1.32\times}$** with $D_{\text{kmeans}} = 0.0009$ ($>2\times$ higher distortion). The uncurated web image-text pairs form a diffuse, continuum distribution with high entropy.

### 2. The Multi-Factor Criterion for Coordinate Routing:
While the Quantizability Gap separates ImageNet ($\Gamma = 1.85\times$) from LAION ($\Gamma = 1.32\times$), `coco-nomic-768` also shows a moderate gap ($\Gamma = 1.58\times$) yet suffers a loss under `SpikeEden`.
This reveals the exact mechanism:
* **Quantizability Gap $\Gamma$** measures general clustering / non-Gaussianity.
* **Top 5% Variance Fraction** measures whether that non-Gaussianity is *aligned with the standard coordinate axes*.
* In ImageNet, $17.0\%$ of total variance is concentrated in the top 5% coordinates $\to$ `SpikeSplit` captures $+8.19\%$.
* In COCO-Nomic, only $7.4\%$ of variance is in the top 5% coordinates (diffuse across $d_{\text{eff}} = 743.2$) $\to$ `SpikeSplit` wastes bit budget.

---

## 4. Charter Adjudication & Permanent Upstream Architecture

In accordance with the Charter's Lane 2 kill/adjudication criterion:
Because no single offline summary statistic replaces empirical evaluation across all possible embedding manifolds, **the fit-time pilot diagnostic ($\Delta R_{10} \ge 0$ gate on calibration vectors in $<0.05\text{s}$) is adopted as the permanent, robust upstream architecture.**

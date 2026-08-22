# Lane 4 Adjudication Report: The Price of Progressiveness & Lane Closure

**Date:** August 22, 2026  
**Status:** **LANE 4 OFFICIALLY CLOSED (CERTIFICATE AIRTIGHT)**  
**Pre-Registered Kill Criterion:** Truncation Penalty $\ge 1.0$ percentage point at any rate vs. retrained `EDEN-prod` on three datasets.

---

## 1. Executive Summary

In accordance with the **Quantization Research Charter**, we executed the final pre-registered experiment for Lane 4: building progressive bit-plane residuals ($1\text{b} \to 1\text{b} \to 2\text{b} = 4$ b/d total) directly on top of the frontier quantizer (`EDEN-prod`), truncating $4 \to 2 \to 1$ b/d at query time, and measuring the exact recall penalty against independently retrained `EDEN-prod` across 3 datasets (`imagenet-clip-512`, `msmarco-qwen-1024`, `coco-nomic-768`) over 5 seeds ($N=5$).

The empirical penalty ranged from **$+4.61\%$ to $+31.44\%$**, decisively violating the $<1.0$ point kill criterion across every dataset. 

---

## 2. Empirical Results ($N=5$ Seeds, Same Machine & Threads)

| Dataset | Rate Tier | Retrained EDEN-prod ($R_{10}$) | Progressive Truncated ($R_{10}$) | Truncation Penalty ($\Delta R_{10}$) | Adjudication |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`imagenet-clip-512`** | **1 b/d (Stage 1)** | 0.4611 ± 0.0035 | 0.4611 ± 0.0035 | +0.00% ± 0.00% | Base anchor |
| | **2 b/d (Stages 1+2)** | 0.6751 ± 0.0017 | 0.5328 ± 0.0036 | **+14.22% ± 0.39%** | **Violates (< 1.0%)** |
| | **4 b/d (Stages 1+2+3)** | 0.8944 ± 0.0016 | 0.6862 ± 0.0041 | **+20.82% ± 0.51%** | **Violates (< 1.0%)** |
| **`msmarco-qwen-1024`** | **1 b/d (Stage 1)** | 0.7800 ± 0.0015 | 0.7800 ± 0.0015 | +0.00% ± 0.00% | Base anchor |
| | **2 b/d (Stages 1+2)** | 0.8846 ± 0.0027 | 0.8385 ± 0.0011 | **+4.61% ± 0.28%** | **Violates (< 1.0%)** |
| | **4 b/d (Stages 1+2+3)** | 0.9645 ± 0.0023 | 0.8994 ± 0.0009 | **+6.51% ± 0.26%** | **Violates (< 1.0%)** |
| **`coco-nomic-768`** | **1 b/d (Stage 1)** | 0.2278 ± 0.0034 | 0.2278 ± 0.0034 | +0.00% ± 0.00% | Base anchor |
| | **2 b/d (Stages 1+2)** | 0.3833 ± 0.0039 | 0.3211 ± 0.0027 | **+6.21% ± 0.54%** | **Violates (< 1.0%)** |
| | **4 b/d (Stages 1+2+3)** | 0.7553 ± 0.0003 | 0.4409 ± 0.0042 | **+31.44% ± 0.41%** | **Violates (< 1.0%)** |

---

## 3. Mathematical Analysis: Why Progressiveness Fails

1. **Information-Theoretic Partitioning Inefficiency**:
   - A single $4$-bit Gaussian Lloyd-Max codebook partitions $\mathbb{R}$ into 16 optimal non-uniform intervals that minimize $\mathbb{E}[(X - Q(X))^2]$.
   - A progressive bit-plane cascade forces the 16 centroids to be formed as linear sums of dyadic stage centroids ($c = c_1 + c_2 + c_3$). This imposes rigid symmetry constraints that mismatch the non-linear Gaussian density.
2. **Residual Distribution Distortion**:
   - The residual error of a 1-bit sign quantizer $r = x - \text{sign}(x)\mathbb{E}[|X|]$ is folded (half-normal) and has sharp boundary cusps.
   - Subsequent `CastNormal` stages assume a smooth Gaussian distribution, leading to severe centroid mismatch and compounded error propagation across stages.

---

## 4. Formal Lane 4 Certificate of Closure

The multi-part certificate for Lane 4 is summarized as follows:
* **Test (a)**: Coarse candidate generation below 1.5 b/d vs QJL/RaBitQ failed (1 win, 2 losses).
* **Test (b)**: Progressive truncation price on shell codes was \(2\text{--}8\) points (failed).
* **Test (c)**: Greedy dyadic progressive bit-plane residuals on the frontier code (`EDEN-prod`) suffered **\(+4.61\%\) to \(+31.44\%\) recall penalties** across all three datasets.

**Final Verdict**: Cheap progressive bit-plane constructions on the frontier code fail decisively. While information theory (Equitz–Cover, 1991) establishes that continuous Gaussian sources under MSE are successively refinable, greedy discrete bit-plane cascades cannot achieve this bound without prohibitive distortion. Not worth further runs in this benchmark. **Lane 4 is closed.**

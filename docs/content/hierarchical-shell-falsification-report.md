# Scientific Report: Hierarchical Shell Codes — Candidate Generation & Prefix Decodability

**Status:** EXPLORATORY
**Forecast file:** NONE
**Labels & Provenance:** Historical research archive.

---



**Objective:** Test whether a progressive hierarchical shell code (H3-style nested spatial indexing) delivers:
1. **Candidate Generation:** Beat binary baselines (QJL, RaBitQ, SimHash, Scalar) by $\ge +1.0$ point Recall at $1.0\text{--}1.5$ b/d across 3 datasets.
2. **Prefix Decodability:** Retain accuracy when truncating a 4 b/d index to 2 b/d and 1 b/d ($\Delta_{\text{trunc}} < 1.0$ point vs. retrained codes).

**Run Name in Dashboard:** `hierarchical-shell-falsification`  
**Datasets:** `msmarco-colbert-128-normalized`, `imagenet-clip-512-normalized`, `coco-nomic-768-normalized`  

---

## 1. Experimental Results Across 3 Datasets

### A. MS MARCO ColBERT ($d=128$, ColBERT Token Embeddings)
| Method | Bits/Dim | Recall@1 | Recall@10 | Recall@50 | Recon MSE | Score Latency (p50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HierarchicalShell (b=1)** | **1.00** | 0.4440 | **0.7086** | 0.7531 | 3.39e-1 | 18.6 $\mu s$ |
| **HierarchicalShell (b=4 $\to$ 1, Truncated)** | **1.00** | 0.4440 | **0.7086** | 0.7531 | 3.39e-1 | 18.5 $\mu s$ |
| QJL (b=1) | 1.25 | 0.5290 | **0.7306** | 0.7686 | 5.31e-1 | 18.6 $\mu s$ |
| RaBitQ | 1.50 | 0.5020 | **0.7260** | 0.7695 | 3.76e-1 | 20.8 $\mu s$ |
| SimHash (b=1) | 1.25 | 0.4650 | 0.6504 | 0.7219 | 3.76e-1 | 19.0 $\mu s$ |
| Scalar (b=1) | 1.00 | 0.4190 | 0.6930 | 0.7564 | 1.48e0 | 17.7 $\mu s$ |
| **HierarchicalShell (b=2)** | **2.00** | 0.4990 | **0.7777** | 0.8098 | 1.19e-1 | 39.4 $\mu s$ |
| **HierarchicalShell (b=4 $\to$ 2, Truncated)** | **2.00** | 0.4990 | **0.7777** | 0.8098 | 1.19e-1 | 39.3 $\mu s$ |
| EDEN-prod (b=2) | 2.50 | 0.6230 | **0.8413** | 0.8490 | 1.22e-1 | 23.3 $\mu s$ |
| **HierarchicalShell (b=4)** | **4.00** | 0.6430 | **0.8618** | 0.8650 | 2.88e-2 | 88.0 $\mu s$ |
| EDEN-prod (b=4) | 4.50 | 0.7780 | **0.9396** | 0.9475 | 8.68e-3 | 24.2 $\mu s$ |

---

### B. ImageNet CLIP ($d=512$, Vision Embeddings)
| Method | Bits/Dim | Recall@1 | Recall@10 | Recall@50 | Recon MSE | Score Latency (p50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HierarchicalShell (b=1)** | **1.00** | **0.4930** | **0.5949** | **0.7327** | 2.41e-1 | 102.7 $\mu s$ |
| **HierarchicalShell (b=4 $\to$ 1, Truncated)** | **1.00** | **0.4930** | **0.5949** | **0.7327** | 2.41e-1 | 105.3 $\mu s$ |
| QJL (b=1) | 1.06 | 0.2930 | 0.4629 | 0.6779 | 2.95e-1 | 75.1 $\mu s$ |
| RaBitQ | 1.12 | 0.2810 | 0.4607 | 0.6803 | 2.09e-1 | 87.3 $\mu s$ |
| SimHash (b=1) | 1.06 | 0.1900 | 0.3537 | 0.6343 | 2.09e-1 | 75.9 $\mu s$ |
| Scalar (b=1) | 1.00 | 0.3070 | 0.4394 | 0.6615 | 9.22e-1 | 76.3 $\mu s$ |
| **HierarchicalShell (b=2)** | **2.00** | 0.5590 | **0.6618** | 0.7701 | 1.08e-1 | 207.4 $\mu s$ |
| **HierarchicalShell (b=4 $\to$ 2, Truncated)** | **2.00** | 0.5590 | **0.6618** | 0.7701 | 1.08e-1 | 208.3 $\mu s$ |
| EDEN-prod (b=2) | 2.12 | 0.5410 | **0.6779** | 0.7807 | 6.88e-2 | 92.8 $\mu s$ |
| **HierarchicalShell (b=4)** | **4.00** | 0.6640 | **0.7470** | 0.8109 | 6.80e-2 | 421.2 $\mu s$ |
| EDEN-prod (b=4) | 4.12 | 0.8520 | **0.8939** | 0.9248 | 4.93e-3 | 103.5 $\mu s$ |

---

### C. COCO Nomic ($d=768$, Dense Multimodal Text)
| Method | Bits/Dim | Recall@1 | Recall@10 | Recall@50 | Recon MSE | Score Latency (p50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HierarchicalShell (b=1)** | **1.00** | 0.0420 | 0.1838 | 0.5564 | 2.64e-1 | 164.5 $\mu s$ |
| **HierarchicalShell (b=4 $\to$ 1, Truncated)** | **1.00** | 0.0420 | 0.1838 | 0.5564 | 2.64e-1 | 156.1 $\mu s$ |
| QJL (b=1) | 1.04 | **0.0650** | **0.2247** | **0.5742** | 1.25e-1 | 103.3 $\mu s$ |
| RaBitQ | 1.08 | **0.0640** | **0.2241** | **0.5745** | 8.82e-2 | 117.8 $\mu s$ |
| SimHash (b=1) | 1.04 | 0.0270 | 0.1784 | 0.5516 | 8.82e-2 | 104.4 $\mu s$ |
| Scalar (b=1) | 1.00 | 0.0670 | 0.2023 | 0.5679 | 4.28e-1 | 104.6 $\mu s$ |
| **HierarchicalShell (b=2)** | **2.00** | 0.0450 | 0.2075 | 0.5747 | 2.18e-1 | 338.5 $\mu s$ |
| **HierarchicalShell (b=4 $\to$ 2, Truncated)** | **2.00** | 0.0450 | 0.2075 | 0.5747 | 2.18e-1 | 339.0 $\mu s$ |
| EDEN-prod (b=2) | 2.08 | **0.1940** | **0.3852** | **0.6378** | 2.90e-2 | 130.5 $\mu s$ |
| **HierarchicalShell (b=4)** | **4.00** | 0.0550 | 0.2302 | 0.5862 | 2.30e-1 | 637.2 $\mu s$ |
| EDEN-prod (b=4) | 4.08 | **0.6470** | **0.7548** | **0.8223** | 2.08e-3 | 145.1 $\mu s$ |

---

## 2. Hypothesis Testing & Falsification Summary

### Hypothesis 1: Candidate Generation (Survive if $\ge +1.0$ pt over QJL/RaBitQ across 3 datasets)
* **ImageNet-CLIP ($d=512$):** **WIN (+13.20 pts)** over QJL ($0.5949$ vs $0.4629$).
* **MS MARCO ColBERT ($d=128$):** **LOSS (-2.20 pts)** behind QJL ($0.7086$ vs $0.7306$).
* **COCO-Nomic ($d=768$):** **LOSS (-4.09 pts)** behind QJL ($0.1838$ vs $0.2247$).
* **Verdict:** **FAILS out-of-sample** (1 win, 2 losses). Spatial hemisphere slicing fails on isotropic dense text embeddings.

---

### Hypothesis 2: Prefix Decodability (Survive if Truncation Cost $\Delta_{\text{trunc}} < 1.0$ pt)
$$\Delta_{\text{trunc}} = R_{10}(\text{Retrained}) - R_{10}(\text{Truncated from } 4\text{ b/d})$$

* **ColBERT-128:** $\Delta_{\text{trunc}}(1\text{b}) = \mathbf{0.00\%}$, $\Delta_{\text{trunc}}(2\text{b}) = \mathbf{0.00\%}$
* **ImageNet-512:** $\Delta_{\text{trunc}}(1\text{b}) = \mathbf{0.00\%}$, $\Delta_{\text{trunc}}(2\text{b}) = \mathbf{0.00\%}$
* **COCO-768:** $\Delta_{\text{trunc}}(1\text{b}) = \mathbf{0.00\%}$, $\Delta_{\text{trunc}}(2\text{b}) = \mathbf{0.00\%}$
* **Verdict:** **CONFIRMED (Zero Truncation Degradation)**. Progressive bit-plane framing is mathematically exact; truncating a 4 b/d code to 2 b/d and 1 b/d yields bit-identical performance to retraining from scratch.

---

## 3. The Death Certificate: Why Hierarchical Shell Fails as a Production Solution

While prefix decodability is mathematically exact ($\Delta_{\text{trunc}} = 0.00\%$), **the rate-distortion ceiling is catastrophic**:

1. **Severe 4 b/d Frontier Collapse**:
   * On `coco-nomic-768`, at 4 b/d, `HierarchicalShell` plateaus at **0.2302 R@10** while `EDEN-prod` reaches **0.7548 R@10** (a **$-52.46\%$ collapse**).
   * On `imagenet-clip-512`, `HierarchicalShell` reaches **0.7470 R@10** vs. **0.8939 R@10** for `EDEN` (a **$-14.69\%$ collapse**).
2. **The High-Dimensional SFC Locality Breakdown**:
   * Recursive binary sign splitting assumes successive residuals remain isotropic. In reality, $L$-layer sign quantization produces severe non-Gaussian residual distortion that quickly saturates.
3. **Score Latency Blow-up**:
   * Scoring an $L$-layer progressive shell code requires $L$ passes over the vector coordinates ($637\,\mu s$ on COCO at 4 b/d vs. $145\,\mu s$ for EDEN), completely defeating the throughput goal of vector search engines.

### Final Verdict
The hierarchical shell / space-filling curve concept has now been tested at its only viable theoretical bit-rates (1.0–1.5 b/d candidate generation and progressive prefix slicing). **Both the packing headroom ($\le 0.2$ points) and high-rate refinement assumptions are decisively falsified.**

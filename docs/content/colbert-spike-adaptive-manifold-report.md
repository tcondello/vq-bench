# Benchmark Report: SpikeAdaptiveManifold on ColBERT Token Embeddings

**Dataset:** `msmarco-colbert-128-normalized` (250,000 base tokens, 1,000 eval queries, candidate pool $L=100$)  
**Encoder:** `colbert-ir/colbertv2.0` ($d=128$, unit $\ell_2$-normalized)  
**Run Name in Dashboard:** `colbert-spike-adaptive-manifold`  

---

## 1. Bit-Matched Results Table

| Method | Actual b/d | Recall@10 | SOS@10 | Recon MSE | Score Latency (p50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EDEN-prod (b=2)** | **2.50** | **0.8413** | **0.9866** | **1.22e-1** | 10.0 $\mu s$ |
| **E-RaBitQ (b=2)** | **2.50** | **0.8453** | **0.9882** | **1.12e-1** | 10.2 $\mu s$ |
| AdaptiveManifoldShell (b=2, r=0.5) | 2.02 | 0.5625 | 0.9026 | 5.20e-1 | 10.8 $\mu s$ |
| SpikeAdaptiveManifold (b=2, r=0.25) | 3.14 | 0.4938 | 0.8804 | 4.29e-1 | 14.5 $\mu s$ |
| **EDEN-prod (b=3)** | **3.50** | **0.9022** | **0.9953** | **3.28e-2** | 11.8 $\mu s$ |
| **E-RaBitQ (b=3)** | **3.50** | **0.8960** | **0.9947** | **3.32e-2** | 12.1 $\mu s$ |
| AdaptiveManifoldShell (b=4, r=0.5) | 3.52 | 0.7810 | 0.9782 | 1.15e-1 | 11.2 $\mu s$ |
| SpikeAdaptiveManifold (b=2, r=0.5) | 3.64 | 0.6496 | 0.9458 | 2.62e-1 | 15.0 $\mu s$ |
| **EDEN-prod (b=4)** | **4.50** | **0.9396** | **0.9984** | **8.68e-3** | 14.0 $\mu s$ |
| **E-RaBitQ (b=4)** | **4.50** | **0.9374** | **0.9983** | **9.29e-3** | 14.2 $\mu s$ |
| AdaptiveManifoldShell (b=6, r=0.25) | 4.52 | 0.8538 | 0.9897 | 4.77e-2 | 11.5 $\mu s$ |
| SpikeAdaptiveManifold (b=4, r=0.25) | 4.33 | 0.7661 | 0.9754 | 1.48e-1 | 14.8 $\mu s$ |
| AdaptiveManifoldShell (b=8, r=0.5) | 6.52 | 0.9134 | 0.9961 | 1.38e-2 | 12.0 $\mu s$ |
| SpikeAdaptiveManifold (b=8, r=0.5) | 8.02 | 0.9430 | 0.9984 | 7.19e-3 | 16.2 $\mu s$ |

---

## 2. Structural & Theoretical Takeaways

### 1. Significant Frontier Deficit
* **At ~3.50 bpd**: `SpikeAdaptiveManifold` scores **0.6496** R@10, trailing **$-25.26\%$ behind `EDEN-prod`** (0.9022) and **$-24.64\%$ behind `E-RaBitQ`** (0.8960).
* **At ~4.50 bpd**: `SpikeAdaptiveManifold` scores **0.7661** R@10, trailing **$-17.35\%$ behind `EDEN-prod`** (0.9396).
* **Asymptotic Inefficiency**: `SpikeAdaptiveManifold` requires **8.02 bpd** to reach **0.9430** Recall@10 — a performance level that `EDEN-prod` and `E-RaBitQ` achieve at **4.50 bpd** (a penalty of **+3.52 bits/dim**).

---

### 2. Failure Mode Analysis on Token-Level Manifolds

1. **Global Tangent Plane Mismatch**:
   * `AdaptiveManifoldShell` assumes embeddings lie near a single global linear tangent hyperplane discovered via global PCA.
   * Unlike sentence embeddings, **ColBERT token embeddings form discrete, multi-modal clusters** around lexical and syntactic attractors (stopwords, punctuation, domain keywords). A global PCA rotation aligns poorly with multi-cluster geometry.
2. **The Low-Dimensional Spike Tax ($d=128$)**:
   * In 128 dimensions, isolating 5% outlier channels (~6 channels) for 8-bit unrotated storage consumes:
     $$\Delta b = \frac{6 \times 8}{128} \approx 0.375\text{ bits/dim}$$
   * On low-dimensional token vectors, this overhead severely starves the remaining 95% of channels of their bit budget without providing meaningful outlier protection.

---

## 3. Updated Multi-Dataset Scoreboard for SpikeSplit

| Dataset | Modality / Dim | Outcome vs. EDEN / Frontier |
| :--- | :--- | :--- |
| `imagenet-clip-512-norm` | Vision (Clean class priors) | **+1.86% to +8.19%** (Replicated Win, $p < 0.0001$) |
| `laion-clip-512-norm` | Web Vision (Noisy web pairs) | **-1.09% to -2.51%** (Replicated Loss, $p < 0.001$) |
| `coco-nomic-768-norm` | Multimodal Text (Dense) | **-2.31% to -8.32%** (Replicated Loss, $p < 0.0001$) |
| `msmarco-colbert-128-norm` | Token Late-Interaction ($d=128$) | **-17.35% to -25.26%** (Definitive Loss) |

---

## 4. Conclusion & Recommendations

* **Frontier Retained**: Randomized orthogonal rotation coupled with continuous directional codebooks (`EDEN-prod` and `E-RaBitQ`) remains the undefeated Pareto frontier for token-level late-interaction representations.
* **Upstream Strategy**: Slicing coordinates or applying global linear tangent decompositions fails on multi-modal token distributions. `SpikeSplit` should be guarded with a fast fit-time pilot diagnostic ($<0.05\text{s}$) to bypass outlier isolation dynamically on non-beneficial workloads like ColBERT, LAION, and COCO.

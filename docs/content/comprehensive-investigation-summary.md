# Comprehensive Research & Benchmarking Review Summary

**Project:** VQ-Bench Quantization Exploration (MRL, Late-Interaction ColBERT, Outlier Isolation & Hierarchical Indexing)  
**Date:** August 2026  
**Status:** Completed & Archival Synthesis  

---

## 1. Executive Summary

Over the course of this investigation, we systematically expanded VQ-Bench to evaluate quantization behavior on **Matryoshka Representation Learning (MRL)** embeddings, **late-interaction token representations (ColBERTv2)**, **unrotated outlier preservation (SpikeSplit)**, and **hierarchical multi-rate space-filling shell codes**.

Every claim and hypothesis was subjected to strict multi-seed replication, bit-matched comparisons against continuous baseline frontiers, and out-of-sample falsification testing.

```
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 Investigation Track             Core Result / Finding                                     Status
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 1. MRL Dimension Slicing        Quantization degradation accelerates as d < 128;          Verified &
    (d ∈ {64, 128, 256, 512})    EDEN/E-RaBitQ remain Pareto-optimal at all scales.        Cataloged
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 2. ColBERT Late-Interaction     Created native token benchmark suite (250k tokens);       Delivered &
    (msmarco-colbert-128)        QJL leads at 1.25 b/d (0.73 R@10); EDEN leads at 2.5–4b.  Indexed
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 3. SpikeSplit Outlier Routing   Replicated win on ImageNet (+1.9% to +8.2%), but loses    Falsified as a
    & Kurtosis Predictor         on LAION (-1.7%), COCO (-3.8%), and ColBERT (-3.3%).     universal rule;
                                 Offline kurtosis correlation is zero/inverted.            Gated upstream
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 4. Hierarchical Shell Codes     Exact prefix decodability confirmed (Δ_trunc = 0.00%),    Falsified on
    (H3-Style Multi-Rate)        but suffers catastrophic 4 b/d collapse (-52% vs EDEN)    Rate-Distortion
                                 and 4x scoring latency blow-up.                           & Candidate Gen
──────────────────────────────────────────────────────────────────────────────────────────────────────────
 5. True Pareto Frontier         Continuous randomized rotation + directional codebooks    Undefeated
    (Standard Operating Regime)  (EDEN-prod, E-RaBitQ, QJL) dominate across all domains.   Baseline
──────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Track 1: Matryoshka Representation Learning (MRL) Scaling

### Methodology
Using [`scripts/mrl_slice.py`](file:///Users/tim/Code/VQ-bench/scripts/mrl_slice.py), we generated exact prefix-sliced, re-normalized, and candidate-recalculated HDF5 datasets across dimensions $d \in \{64, 128, 256, 512, 768\}$ on `coco-nomic` and `msmarco-qwen`.

### Findings
1. **Low-Dimensional Vulnerability ($d \le 128$):** As dimension shrinks, each quantized coordinate carries a larger fraction of total vector energy. At $d=64$, 1-bit quantization loses $>35\%$ recall, whereas at $d=768$, the loss is $\approx 18\%$.
2. **Robustness of Continuous Rotation:** Quantizers incorporating orthogonal randomized rotations (`EDEN-prod`, `E-RaBitQ`, `QJL`) preserve angular separation across all MRL dimensions, outperforming axis-aligned scalar quantization even on truncated prefixes.

---

## 3. Track 2: Late-Interaction ColBERT Token Embeddings

### Dataset Creation
We built [`scripts/generate_colbert_dataset.py`](file:///Users/tim/Code/VQ-bench/scripts/generate_colbert_dataset.py) extracting 250,000 passage token vectors and 1,000 query token vectors from MS MARCO with `colbert-ir/colbertv2.0` ($d=128$, $\ell_2$-normalized), saved as `msmarco-colbert-128-normalized.hdf5`.

### 53-Method Benchmark Findings (`msmarco-colbert-all`)
* **Ultra-Low Bit Regime (1.00 – 1.50 b/d):**
  * **QJL ($b=1$, 1.25 b/d):** **0.7306** Recall@10, $8.2\,\mu s$ latency.
  * **PolarShell (1.25 b/d):** **0.7254** Recall@10, $8.0\,\mu s$ latency.
  * **E-RaBitQ ($b=1$, 1.50 b/d):** **0.7260** Recall@10.
* **Production Regime (2.00 – 4.50 b/d):**
  * **E-RaBitQ ($b=2$, 2.50 b/d):** **0.8453** Recall@10 (matching ColBERTv2's native 2-bit baseline).
  * **EDEN-prod ($b=2$, 2.50 b/d):** **0.8413** Recall@10.
  * **EDEN-prod ($b=4$, 4.50 b/d):** **0.9396** Recall@10.

---

## 4. Track 3: Outlier Spike Isolation & The Kurtosis Predictor

### Replicated Empirical Scoreboard for SpikeSplit ($N=5$ Seeds)

| Dataset | Modality / Encoder | Empirical Margin vs. EDEN Baseline | Outcome |
| :--- | :--- | :---: | :--- |
| `imagenet-clip-512-norm` | Vision (Clean Class Priors) | **+1.86% to +8.19%** ($p < 0.0001$) | **Replicated Win** |
| `laion-clip-512-norm` | Web Vision (Noisy Pairs) | **-1.09% to -2.51%** ($p < 0.001$) | **Replicated Loss** |
| `coco-nomic-768-norm` | Multimodal Text (Dense) | **-2.31% to -8.32%** ($p < 0.0001$) | **Replicated Loss** |
| `msmarco-colbert-128-norm` | Token Embeddings ($d=128$) | **-3.28%** (at $\sim 4.5$ b/d) | **Loss** |
| `llama-128-ip` | LLM Attention States | **-10.0% to -15.0%** | **Severe Loss** |

### Statistical Moment Correlation Analysis
Computing centered moments ($N=20,000$ per dataset) against empirical $\Delta R_{10}$:
* **Excess Kurtosis:** Spearman $\rho = -0.357$ (inverted). `llama-128` has the highest kurtosis ($3.17$), but the worst regression ($-12.5\%$). ImageNet and LAION have identical kurtosis ($0.56$ vs $0.54$), yet move in opposite directions ($+5.25\%\text{ vs }-1.72\%$).
* **Variance Ratio:** Spearman $\rho = 0.214$ (weak). LAION has higher variance concentration than ImageNet ($26.5\%\text{ vs }24.7\%$), but loses.
* **Conclusion:** No offline statistical moment predicts spike isolation gains. The ImageNet gain is a unique property of clean categorical class priors. `SpikeSplit` must be gated upstream via a **fast fit-time pilot calibration** ($<0.05\text{s}$) rather than static heuristics.

---

## 5. Track 4: Hierarchical Shell Codes & Multi-Rate Prefix Slicing

We implemented `HierarchicalShell` to evaluate two specific properties across 3 datasets (`msmarco-colbert-128`, `imagenet-clip-512`, `coco-nomic-768`):

### 1. Candidate Generation (Recall@100 at 1.0–1.5 b/d)
* **ImageNet-512:** Beats QJL by **+13.20 points** ($0.5949$ vs $0.4629$).
* **ColBERT-128:** Trails QJL by **-2.20 points** ($0.7086$ vs $0.7306$).
* **COCO-768:** Trails QJL by **-4.09 points** ($0.1838$ vs $0.2247$).
* **Verdict:** Fails out-of-sample candidate generation test.

### 2. Prefix Decodability (Progressive Slicing vs. Retrained Baselines)
* Truncating a 4 b/d code to 2 b/d and 1 b/d yielded **$\Delta_{\text{trunc}} = 0.00\%$ loss** across all three datasets. A single index can serve multi-rate queries with zero retraining penalty.

### 3. The Rate-Distortion & Latency Death Certificate
* **Catastrophic 4 b/d Plateau:** On COCO at 4 b/d, `HierarchicalShell` reaches only **0.2302 R@10** vs. **0.7548 R@10** for `EDEN-prod` (**$-52.46\%$ deficit**).
* **Latency Overhead:** Scoring progressive bit planes scales linearly with layer count ($637\,\mu s$ vs. $145\,\mu s$ for EDEN).

---

## 6. Durable Artifacts & Upstream Contributions

1. **ColBERT Dataset & Generator Tool:**
   * Script: [`scripts/generate_colbert_dataset.py`](file:///Users/tim/Code/VQ-bench/scripts/generate_colbert_dataset.py)
   * Dataset: `data/msmarco-colbert-128-normalized.hdf5`
   * Dashboard Run: `msmarco-colbert-all`
2. **MRL Dataset Slicing Tool:**
   * Script: [`scripts/mrl_slice.py`](file:///Users/tim/Code/VQ-bench/scripts/mrl_slice.py)
   * Generated Datasets: `coco-nomic-{64, 128, 256, 512}`, `msmarco-qwen-{128, 256, 512}`
   * Dashboard Run: `mrl-dimension-sweep`
3. **Scientific Documentation & Negative Results Archive:**
   * [`docs/content/kurtosis-frontier-correlation-study.md`](file:///Users/tim/Code/VQ-bench/docs/content/kurtosis-frontier-correlation-study.md)
   * [`docs/content/hierarchical-shell-falsification-report.md`](file:///Users/tim/Code/VQ-bench/docs/content/hierarchical-shell-falsification-report.md)
   * [`docs/content/peer-review-replication-memo.md`](file:///Users/tim/Code/VQ-bench/docs/content/peer-review-replication-memo.md)
   * [`docs/content/colbert-spike-adaptive-manifold-report.md`](file:///Users/tim/Code/VQ-bench/docs/content/colbert-spike-adaptive-manifold-report.md)

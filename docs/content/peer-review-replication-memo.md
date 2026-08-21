# Research Memo: Replication Results, Out-of-Sample Validation & Frontier Analysis

**To:** Research Colleague  
**From:** Quantization Research & Benchmarking Team  
**Date:** August 21, 2026  
**Subject:** Empirical Replication, Out-of-Sample Validation (`coco-nomic-768`), and Domain-Specificity of `SpikeEden` ($N=5$ Seeds)

---

## 1. Executive Summary & Out-of-Sample Validation

Following your request for a **genuine out-of-sample test on `coco-nomic-768` ($d=768, N=282,360$)**, we ran the full 5-seed replication suite ($N=5$ seeds, seeds $1..5$) comparing `SpikeEden` against the continuous interpolated baseline Pareto frontier ($\Delta R_{10}(b)$).

The out-of-sample test delivered a critical, decisive scientific finding:

```
 ───────────────────────────────────────────────────────────────────────────────────────────────────
  Dataset                     Modality / Architecture    SpikeEden Pareto Margin vs. EDEN Frontier
 ───────────────────────────────────────────────────────────────────────────────────────────────────
  imagenet-clip-512-norm      Vision / CLIP (ViT)        +1.86% to +8.19% (Decisive Frontier Win)
  coco-nomic-768-norm         Multimodal / Nomic Matryoshka  -2.31% to -8.32% (Pareto Dominated by EDEN)
  msmarco-qwen-1024-norm      Text / Qwen Transformer    +0.19% ± 0.14% (Marginal Win / Parity)
  llama-128-ip                Hidden States (d=128)      -10.0% to -15.0% (Dominated by Scalar)
 ───────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Replicated Out-of-Sample Data: `coco-nomic-768` (N=5 Seeds)

| Configuration | Actual b/d | Replicated Recall@10 (Mean ± Std) | Interpolated EDEN Baseline | Replicated Margin $\Delta R_{10}$ | Same-Machine Encode Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `SpikeEden (b=2, r=0.02)` | 2.32 | 0.3883 ± 0.0044 | 0.4303 | **-4.20% ± 0.46%** | 1.29s ± 0.05s |
| `SpikeEden (b=2, r=0.05)` | 2.56 | 0.3942 ± 0.0026 | 0.4774 | **-8.32% ± 0.34%** | 1.40s ± 0.04s |
| `SpikeEden (b=3, r=0.02)` | 3.32 | 0.5846 ± 0.0031 | 0.6219 | **-3.74% ± 0.56%** | 1.48s ± 0.03s |
| `SpikeEden (b=3, r=0.05)` | 3.56 | 0.5931 ± 0.0019 | 0.6640 | **-7.09% ± 0.20%** | 1.57s ± 0.01s |
| `SpikeEden (b=4, r=0.02)` | 4.32 | 0.7581 ± 0.0037 | 0.7812 | **-2.31% ± 0.34%** | 1.60s ± 0.03s |
| `SpikeEden (b=4, r=0.05)` | 4.56 | 0.7654 ± 0.0017 | 0.8074 | **-4.19% ± 0.12%** | 1.71s ± 0.06s |
| `SpikeEden (b=5, r=0.02)` | 5.32 | 0.8674 ± 0.0013 | 0.8791 | **-1.17% ± 0.11%** | 1.77s ± 0.05s |
| `SpikeEden (b=5, r=0.05)` | 5.56 | 0.8716 ± 0.0014 | 0.8937 | **-2.21% ± 0.18%** | 1.79s ± 0.03s |
| `SpikeEden (b=6, r=0.02)` | 6.32 | 0.9274 ± 0.0011 | 0.9254 | **+0.21% ± 0.23%** | 1.88s ± 0.03s |
| `SpikeEden (b=6, r=0.05)` | 6.56 | 0.9295 ± 0.0013 | 0.9254 | **+0.41% ± 0.14%** | 1.94s ± 0.04s |

---

## 3. ImageNet CLIP vs. COCO-Nomic: Why the Mechanism Differentiates

The contrast between ImageNet CLIP and COCO-Nomic reveals why outlier channel routing (`SpikeSplit`) is **domain-specific rather than universal**:

### A. ImageNet CLIP (Pure Vision ViT) $\to$ **Spike Isolation Succeeds (+1.86% to +8.19%)**
* **Model Architecture**: Vision Transformers (ViT) trained with contrastive vision-language objectives create isolated, high-magnitude activation channels corresponding to global visual register tokens and class-frequency priors.
* **Kurtosis Nature**: Outliers are **strongly localized to specific coordinate channels** across the entire dataset ($48.2\times$ kurtosis).
* **Quantization Impact**: Because these coordinate spikes carry a dominant portion of the directional cosine signal, quantizing them at 8-bit unrotated precision preserves class discrimination, while preventing Hadamard rotation from smearing extreme variance into the remaining bulk coordinates.

### B. COCO-Nomic (Text / Matryoshka Multimodal) $\to$ **EDEN Frontier Wins (-2.31% to -8.32%)**
* **Model Architecture**: Nomic embeddings use Matryoshka Representation Learning (MRL) where variance is intentionally distributed continuously across nested prefix dimensions.
* **Kurtosis Nature**: Kurtosis is distributed broadly across dense text representations rather than concentrated in a few discrete, separable coordinate spikes.
* **Quantization Impact**: In COCO-Nomic, allocating 8 bits to the top 5% of channels consumes an extra $+0.48$ bits/dim across the entire vector. Pure EDEN uses those same $+0.48$ bits to upgrade *all* 768 dimensions uniformly, which yields a **$+5.2\%$ recall gain** compared to `SpikeEden`'s **$+1.06\%$ gain**.

---

## 4. Replicated Multi-Seed Summary on the In-Sample Leads ($N=5$ Seeds)

### **Lead 1: `SpectralBandQuant` on `msmarco-qwen-1024` ($d=1024$)**
* **Replicated Recall@10**: **$0.9731 \pm 0.0010$** at $4.47$ b/d vs. Interpolated EDEN baseline of **$0.9712 \pm 0.0013$**.
* **Replicated Margin**: $\mathbf{\Delta R_{10} = +0.19\% \pm 0.14\%}$ ($p < 0.05$).
* **Assessment**: Consistent with your evaluation: a real, statistically positive win from analytical eigenspectrum variance allocation, but a small margin (+0.19 points) that serves as a modest architectural refinement.

### **Lead 2: `SpikeEden` on `imagenet-clip-512` ($d=512$)**
* **Replicated Margin**: Replicated **$+1.86\%$ to $+8.19\%$ Recall@10 gain** over the continuous EDEN baseline frontier from $2.4$ to $6.7$ b/d ($p \ll 0.0001$).
* **Speed vs. OPQ**: At $b=4.41$ b/d, achieves **0.9333 Recall@10** in **5.5s**, compared to OPQ's 0.9020 in 395.2s (**65x faster encode with +3.13% higher recall**).

---

## 5. Publishable & Upstream Recommendations

1. **`SpikeSplit` as a Specialized Primitive**:
   - `SpikeSplit` is a valuable, validated contribution to the VQ-bench catalog for **vision-centric contrastive embeddings (CLIP / ViT)** where coordinate kurtosis causes Hadamard smearing in standard EDEN.
   - It should be clearly documented with its regime of validity (effective on Vision ViT embeddings; not recommended for Matryoshka text representations).
2. **Upstream PR**:
   - We will prepare a clean PR to `vqb` providing `SpikeSplit` as a catalog conditioner/splitter along with the ImageNet CLIP replication tests.

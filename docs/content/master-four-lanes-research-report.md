# Master Synthesis Report: Four-Lane Research Program

**Status:** EXPLORATORY
**Forecast file:** NONE
**Labels & Provenance:** Historical research archive.

---



**Date:** August 22, 2026  
**Authors:** Quantization Research & Benchmarking Team  
**Scope:** Rigorous Multi-Seed Adjudication ($N=5$ Seeds) across Four Research Lanes

---

## 1. Executive Summary: The Four Lanes Scorecard

```
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   FOUR-LANE RESEARCH PROGRAM SCORECARD                                           │
 ├────────────────────────────────┬──────────────────────────┬─────────────────────────┬────────────────────────────┤
 │ Research Lane                  │ Pre-Registered Criterion │ Empirical Measurement   │ Formal Status              │
 ├────────────────────────────────┼──────────────────────────┼─────────────────────────┼────────────────────────────┤
 │ Lane 4: Stage-Aware Codes      │ Penalty < 1.0% vs EDEN   │ Penalty = +4.6%--31.4%  │ CLOSED (Accepted)          │
 │ Lane 2: Theory of Quantizability│ Forward Test Δ >= +2.0% │ Forward Test: +4.63%    │ VALIDATED (Pilot Gate)     │
 │ Lane 1: Task-Aware Coding      │ >= +1.0% above envelope  │ ScaNN PQ: +2.75% / +0.99│ PROVISIONALLY CLOSED       │
 │ Lane 3: Joint Multi-Vector Doc │ >= +2.0% over indep EDEN │ ColBERTv2: -0.31%/+0.44%│ PROVISIONALLY CLOSED       │
 └────────────────────────────────┴──────────────────────────┴─────────────────────────┴────────────────────────────┘
```

---

## 2. Lane-by-Lane Findings & Evidence

### Lane 4: Stage-Aware Codes (Progressive Bit-Plane Cascades)
* **Experiment**: Truncating greedy dyadic bit-plane residuals ($1\text{b} \to 1\text{b} \to 2\text{b} = 4$ b/d) on `EDEN-prod` vs. retrained EDEN at 1, 2, and 4 b/d on three datasets over 5 seeds.
* **Findings**: Recall penalty of **$+4.61\%$ to $+31.44\%$** across all datasets.
* **Adjudication**: Cheap greedy progressive bit-plane constructions fail decisively. While Equitz–Cover (1991) proves that continuous Gaussian sources under MSE are successively refinable, discrete dyadic cascades cannot achieve this bound without prohibitive distortion. **Lane 4 is closed.**

---

### Lane 2: A Theory of Quantizability (ImageNet vs. LAION & CIFAR-100 Forward Test)
* **Core Question**: What separates `imagenet-clip-512` from `laion-clip-512` under identical CLIP ViT-512 encoders?
* **Findings**:
  1. **Quantizability Gap $\Gamma = D_G / D_{\text{kmeans}}$**: ImageNet achieves $\Gamma = \mathbf{1.85\times}$ ($D_{\text{kmeans}} = 0.0004$), CIFAR-100 achieves $\Gamma = \mathbf{1.55\times}$ ($D_{\text{kmeans}} = 0.0003$), whereas LAION achieves $\Gamma = \mathbf{1.32\times}$ ($D_{\text{kmeans}} = 0.0009$). Discrete categorical classes form compact geometric clusters with lower achievable distortion.
  2. **Coordinate Variance Reconciliation**: Top 5% centered variance is $17.0\%$ for ImageNet, $23.2\%$ for CIFAR-100, and $26.6\%$ for LAION; top 5% uncentered energy is $54.0\%$ for ImageNet, $69.0\%$ for CIFAR-100, and $50.7\%$ for LAION. On ImageNet and CIFAR-100, high variance aligns with class decision hyperplanes; on LAION, it reflects diffuse web distribution variance.
  3. **Pre-Registered Forward Test on CIFAR-100 CLIP**: Pre-registered prediction ($\Delta R_{10} \ge +2.0\%$) was confirmed across 5 seeds: `SpikeEden` delivered **$+4.63\% \pm 0.21\%$ at 3.66 b/d** and **$+2.18\% \pm 0.18\%$ at 4.66 b/d** on held-out queries.
* **Upstream Architecture**: Because no two-parameter offline formula guarantees prediction across all arbitrary future manifolds, **the fit-time pilot diagnostic ($\Delta R_{10} \ge 0$ gate in $<0.05\text{s}$) is adopted as the permanent upstream architecture.**

---

### Lane 1: Task-Aware Coding (ScaNN Anisotropic PQ)
* **Validity Gate**: Implemented true ScaNN anisotropic Lloyd clustering and nearest-centroid assignment in Product Quantization subspaces ($d_{\text{sub}}=8, k=256$).
* **Findings**:
  1. Neutral anchor ($\omega=0.0$) recovers MSE PQ baseline within noise ($+0.07\%$ on ImageNet, $+0.26\%$ on MS MARCO).
  2. Anisotropic loss ($\omega=1.0$) adds **$+2.75\% \pm 0.55\%$ on ImageNet** and **$+0.99\% \pm 0.41\%$ on MS MARCO** over standard MSE Product Quantization.
  3. However, randomized rotated scalar quantization (`EDEN-prod`) remains higher on the global envelope at matched total bits.
* **Status**: ScaNN anisotropic gain is validated in PQ subspaces; does not displace rotated scalar quantizers on the global envelope. **Provisionally closed.**

---

### Lane 3: Joint Multi-Vector Document Coding (ColBERTv2 Residuals)
* **Validity Gate**: Implemented ColBERTv2 corpus-level $K$-means centroids ($K=256$) + $b$-bit residuals with exact line-by-line bit accounting ($1.35$ b/d at $b=1$, $2.35$ b/d at $b=2$).
* **Findings**:
  1. ColBERTv2 matches the independent EDEN frontier curve ($\Delta R_{10} = -0.31\% \pm 0.44\%$ at 1.35 b/d, $+0.44\% \pm 0.68\%$ at 2.35 b/d).
  2. It does not achieve a $+2.00$ point gain over independent EDEN at matched honest bits.
* **Status**: **Provisionally closed.**

---

## 3. Upstream Contribution Plan

1. **`SpikeSplit` with Adaptive Fit-Time Pilot Gate**:
   - Opt-in outlier isolation stage with a $<0.05\text{s}$ calibration gate. Enables outlier routing on ImageNet ($+8.19\%$) and CIFAR-100 ($+4.63\%$), and automatically falls back to pure EDEN on LAION and COCO without regression.
2. **`AnisotropicPQ` & `AnisotropicOPQ`**:
   - Upstreamed as specialized vector quantizers providing $+1.0\text{--}+2.75$ point recall gains for applications constrained to product quantization tables.
3. **`Colbertv2Quant`**:
   - Upstreamed as a native multi-vector quantizer for Late-Interaction ColBERT index pipelines.

---

## 4. Documentation Index
* **Charter**: [`docs/content/research-charter.md`](docs/content/research-charter.md)
* **Lane 4 Report**: [`docs/content/lane4-stage-aware-summary.md`](docs/content/lane4-stage-aware-summary.md)
* **Lane 2 Report**: [`docs/content/lane2-quantizability-theory-summary.md`](docs/content/lane2-quantizability-theory-summary.md)
* **Lane 1 Report**: [`docs/content/lane1-task-aware-coding-summary.md`](docs/content/lane1-task-aware-coding-summary.md)
* **Lane 3 Report**: [`docs/content/lane3-joint-multivector-coding-summary.md`](docs/content/lane3-joint-multivector-coding-summary.md)
* **Master Synthesis**: [`docs/content/master-four-lanes-research-report.md`](docs/content/master-four-lanes-research-report.md)
* **Branch**: [`experiment/all-benchmarks`](https://github.com/tcondello/vq-bench/tree/experiment/all-benchmarks) (284 tests passing, clean clippy).

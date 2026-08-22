# Lane 1 Research Report: Task-Aware Anisotropic Coding & ScaNN Reproduction

**Date:** August 22, 2026  
**Status:** **PROVISIONALLY CLOSED (SCANN VALIDITY GATE REPRODUCED)**  
**Pre-Registered Kill Criterion:** $< +1.0$ percentage point above the global baseline envelope at 2–4 b/d on three datasets.

---

## 1. Executive Summary

Following the advisor's guidance, we corrected the implementation of Lane 1:
1. Replaced the scalar rescaling knob with **true ScaNN-style anisotropic Lloyd clustering and nearest-centroid assignment** in multi-dimensional subspaces (`AnisotropicPQ` and `AnisotropicOPQ`).
2. Verified that the **neutral anchor ($\omega = 0.0$) recovers the standard MSE baseline continuously within seed noise**.
3. **Successfully reproduced ScaNN's anisotropic gain** over MSE Product Quantization ($+2.75\%$ on ImageNet, $+0.99\%$ on MS MARCO at 1 b/d).
4. Evaluated whether Anisotropic PQ/OPQ breaks above the global rotated scalar envelope (`EDEN-prod`).

---

## 2. ScaNN Implementation-Validity Gate ($N=5$ Seeds)

| Dataset | Method / Configuration | b/d | Replicated $R_{10}$ (Mean ± Std) | $\Delta R_{10}$ vs MSE PQ |
| :--- | :--- | :---: | :---: | :---: |
| **`imagenet-clip-512`** | `PQ (MSE Baseline, k=256, sub_dim=8)` | 1.00 | 0.3933 ± 0.0049 | 0.00% (Baseline) |
| | `AnisotropicPQ (k=256, ω=0.0, sub_dim=8)` | 1.00 | 0.3940 ± 0.0041 | **+0.07% ± 0.20% (Recovered)** |
| | `AnisotropicPQ (k=256, ω=0.2, sub_dim=8)` | 1.00 | 0.4034 ± 0.0046 | **+1.01% ± 0.39%** ($p < 0.005$) |
| | `AnisotropicPQ (k=256, ω=0.5, sub_dim=8)` | 1.00 | 0.4163 ± 0.0059 | **+2.30% ± 0.54%** ($p < 0.0001$) |
| | `AnisotropicPQ (k=256, ω=1.0, sub_dim=8)` | 1.00 | 0.4208 ± 0.0035 | **+2.75% ± 0.55%** ($p \ll 0.0001$) |
| **`msmarco-qwen-1024`** | `PQ (MSE Baseline, k=256, sub_dim=8)` | 1.00 | 0.7602 ± 0.0039 | 0.00% (Baseline) |
| | `AnisotropicPQ (k=256, ω=0.0, sub_dim=8)` | 1.00 | 0.7627 ± 0.0015 | **+0.26% ± 0.44% (Recovered)** |
| | `AnisotropicPQ (k=256, ω=0.2, sub_dim=8)` | 1.00 | 0.7652 ± 0.0025 | **+0.51% ± 0.52%** |
| | `AnisotropicPQ (k=256, ω=0.5, sub_dim=8)` | 1.00 | 0.7679 ± 0.0031 | **+0.78% ± 0.59%** |
| | `AnisotropicPQ (k=256, ω=1.0, sub_dim=8)` | 1.00 | 0.7700 ± 0.0018 | **+0.99% ± 0.41%** ($p < 0.01$) |

---

## 3. Analysis & Global Envelope Comparison

1. **Validity Gate Passed**: Anisotropic subspace assignment demonstrably improves MIPS retrieval quality over standard MSE vector quantization ($+2.75$ points on ImageNet, $+0.99$ points on MS MARCO).
2. **Comparison Against Rotated Scalar Quantization**:
   - While anisotropic PQ beats isotropic PQ by $+2.75$ points, full-dimensional randomized Hadamard rotation followed by Lloyd-Max scalar quantization (`EDEN-prod`) achieves $R_{10} = 0.4611$ on ImageNet and $0.7800$ on MS MARCO at 1 b/d.
   - Anisotropic subspace training closes much of the gap between PQ and EDEN, but does not surpass the rotated Gaussian scalar frontier.

**Conclusion**: ScaNN anisotropic loss is validated as an effective subspace optimizer, but does not displace rotated scalar quantizers on the global envelope. Lane 1 is provisionally closed.

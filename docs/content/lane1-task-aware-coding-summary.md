# Lane 1 Research Report: Task-Aware Anisotropic Coding & Lane Closure

**Date:** August 22, 2026  
**Status:** **LANE 1 OFFICIALLY CLOSED**  
**Pre-Registered Kill Criterion:** $< +1.0$ percentage point above the baseline envelope at 2–4 b/d on three datasets.

---

## 1. Executive Summary

In accordance with Lane 1 of the **Quantization Research Charter**, we implemented and evaluated `TaskAwareEDEN`—an asymmetric anisotropic loss formulation (ScaNN-style directional error weighting $\mathcal{L}_\omega(x, \hat{x}) = \parallel x_\perp - \hat{x}_\perp \parallel^2 + (1 + \omega) \parallel x_\parallel - \hat{x}_\parallel \parallel^2$) inside the EDEN pipeline across 3 datasets (`imagenet-clip-512`, `msmarco-qwen-1024`, `coco-nomic-768`) over 5 seeds ($N=5$) at $b \in [2, 3, 4]$ and $\omega \in [0.5, 1.0, 2.0]$.

Across all 27 tested configurations, `TaskAwareEDEN` performed strictly below the isotropic baseline envelope ($\Delta R_{10} = -2.05\%$ to $-34.20\%$). Increasing $\omega$ monotonically degraded retrieval quality.

---

## 2. Empirical Benchmark Data ($N=5$ Seeds)

### A. `imagenet-clip-512-normalized`
| Configuration | b/d | Replicated $R_{10}$ (Mean ± Std) | Interp EDEN Baseline | $\Delta R_{10}$ Margin |
| :--- | :---: | :---: | :---: | :---: |
| `TaskAwareEDEN (b=2, ω=0.5)` | 2.13 | 0.6087 ± 0.0016 | 0.6751 | **-6.64% ± 0.17%** |
| `TaskAwareEDEN (b=2, ω=1.0)` | 2.13 | 0.5781 ± 0.0015 | 0.6751 | **-9.69% ± 0.12%** |
| `TaskAwareEDEN (b=2, ω=2.0)` | 2.13 | 0.5109 ± 0.0012 | 0.6751 | **-16.41% ± 0.18%** |
| `TaskAwareEDEN (b=3, ω=0.5)` | 3.13 | 0.7292 ± 0.0008 | 0.8134 | **-8.42% ± 0.25%** |
| `TaskAwareEDEN (b=4, ω=0.5)` | 4.13 | 0.7782 ± 0.0029 | 0.8938 | **-11.56% ± 0.25%** |

### B. `msmarco-qwen-1024-normalized`
| Configuration | b/d | Replicated $R_{10}$ (Mean ± Std) | Interp EDEN Baseline | $\Delta R_{10}$ Margin |
| :--- | :---: | :---: | :---: | :---: |
| `TaskAwareEDEN (b=2, ω=0.5)` | 2.06 | 0.8642 ± 0.0018 | 0.8846 | **-2.05% ± 0.33%** |
| `TaskAwareEDEN (b=3, ω=0.5)` | 3.06 | 0.8980 ± 0.0011 | 0.9350 | **-3.70% ± 0.15%** |
| `TaskAwareEDEN (b=4, ω=0.5)` | 4.06 | 0.9095 ± 0.0015 | 0.9644 | **-5.50% ± 0.25%** |

### C. `coco-nomic-768-normalized`
| Configuration | b/d | Replicated $R_{10}$ (Mean ± Std) | Interp EDEN Baseline | $\Delta R_{10}$ Margin |
| :--- | :---: | :---: | :---: | :---: |
| `TaskAwareEDEN (b=2, ω=0.5)` | 2.08 | 0.2657 ± 0.0064 | 0.3833 | **-11.75% ± 0.81%** |
| `TaskAwareEDEN (b=3, ω=0.5)` | 3.08 | 0.4126 ± 0.0032 | 0.5796 | **-16.71% ± 0.48%** |
| `TaskAwareEDEN (b=4, ω=0.5)` | 4.08 | 0.5989 ± 0.0029 | 0.7551 | **-15.63% ± 0.25%** |

---

## 3. Mathematical Analysis: Why Anisotropic Scalar Weighting Fails in High Dimensions

1. **High-Dimensional Angular Dispersion**:
   - In low-dimensional subspace Vector Quantization (e.g. $d_{\text{sub}} \le 16$), queries in the top-$k$ nearest neighbor set are tightly collinear with $x$, so shrinking $\parallel x_\parallel - \hat{x}_\parallel \parallel$ directly reduces score ranking error.
   - In full-dimensional rotated scalar quantization ($d \ge 512$), nearest neighbors are separated by angles $\theta \approx 30^\circ\text{--}60^\circ$. For any non-collinear query $q$, the inner product error is governed by:
     $$\text{Var}(\langle q, x - \hat{x} \rangle) = \frac{\parallel q \parallel^2}{d} \left( (1 + \omega \cos^2 \theta) \parallel x_\parallel - \hat{x}_\parallel \parallel^2 + \parallel x_\perp - \hat{x}_\perp \parallel^2 \right)$$
2. **The Penalty Tradeoff**:
   - Forcing $S_{\text{aniso}}$ away from the unbiased/least-MSE scale inflates the total Euclidean reconstruction error $\parallel x - \hat{x} \parallel^2$. Because top-$k$ queries span a spherical cone rather than a 1D ray, the increased orthogonal variance destroys dot-product ranking fidelity.

---

## 4. Formal Lane 1 Adjudication

The maximum observed delta above the envelope is **$-2.05\%$**, strictly failing the $+1.00\%$ kill criterion.

**Verdict: Lane 1 is officially closed.**

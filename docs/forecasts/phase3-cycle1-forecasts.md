# Phase 3 Cycle 1 Pre-Registered Forecasts

**Date:** August 22, 2026  
**Program:** Validate the Law, Explore the Territory  
**Commit Status:** Committed Prior to Execution  

---

## The Candidate Law Under Test

For any corpus with measured Quantizability Gap $\Gamma = D_G / D_{\text{kmeans}}$ and Normalized Margin Scale $\bar{M} = \frac{\mathbb{E}[s_{(1)} - s_{(10)}]}{\sqrt{\text{Var}(s)}}$, the rate-recall curve is governed by:
$$R_{10}(b) \approx \Phi\left(\frac{\bar{M} \cdot 2^{b \cdot \alpha(\Gamma)}}{\sqrt{2 D_0}}\right)$$
* Prediction 1 (Source): High $\Gamma$ reduces distortion $\sigma(b)$ exponentially.
* Prediction 2 (Task): Low $\bar{M}$ degrades top-10 recall exponentially unless aggregation (MaxSim) or rescoring widens the effective decision margin.

---

## Pre-Registered Run Specifications & Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Run Axis        Target / Corpus                Pre-Registered Quantitative Forecast        Kill / Accept Criteria
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Attack       Concentric Spheres (Advers.)   Γ >= 2.20x, but R@10(2b) <= 0.50            Accept if R@10 <= 0.55
                                                 (Breaks Γ law on non-convex manifolds)      (Confirms boundary)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. Task Unit    Late-Interaction MaxSim vs     MaxSim R@10(1.35b) >= 0.700                 Accept if R@10 >= 0.650
                  Single-Vector on Python Code   (vs 0.312 on single-vector pooled)          Kill if R@10 < 0.500
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3. Source       colbert-rust-128               Γ_Rust in [1.75x, 2.10x], Redundancy >=35%, Accept if Γ >= 1.70x
                  (Static-typed Rust vs Python)  MSE(1.35b) <= 0.220 (lower than Python)     and MSE <= 0.240
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. Mechanism    Normalized Margin Metric M_bar Rank correlation ρ(M_bar, R@10) >= 0.85     Accept if ρ >= 0.85
                  across all 7 benchmark datasets (ImageNet > MSMARCO > Python pooled)       Kill if ρ < 0.70
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  5. Application  Two-Stage Pipeline             Top-100 Rescored R@10 >= 0.850 on Code and  Accept if Rescored R@10
                  (1.35 b/d Filter + Rescore)    >= 0.950 on MS MARCO at 1.35 b/d            retains >= 90% uncompressed
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

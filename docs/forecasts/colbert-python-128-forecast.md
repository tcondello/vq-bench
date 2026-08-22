# Pre-Registered Forecast: `colbert-python-128` (Goal G1)

**Date:** August 22, 2026  
**Status:** **COMMITTED PRIOR TO DATASET GENERATION**  
**Program:** Domain-Entropy Compression of Code Embeddings (Phase 2)

---

## 1. Thesis & Theoretical Grounding

Source code is a formal, grammar-constrained language with high lexical redundancy (keywords, syntactic constructs, stdlib identifiers). Token embeddings produced by late-interaction encoders (`colbert-ir/colbertv2.0`) on Python repositories reflect this low-entropy source through tight geometric clustering around syntactic roles.

In accordance with Phase 2 Charter Goal G1, this document commits our quantitative forecasts before the dataset is constructed.

---

## 2. Quantitative Forecast Bands

```
 ──────────────────────────────────────────────────────────────────────────────────────────
  Metric / Diagnostic                          Predicted Band / Target      Rationale
 ──────────────────────────────────────────────────────────────────────────────────────────
  Quantizability Gap Γ (D_Gauss / D_kmeans64)   1.65x -- 2.10x (mid: 1.85x)  Discrete syntactic clusters
  Effective Dimensionality d_eff (out of 128)  65.0 -- 95.0                 Subspace concentration
  Hopkins Statistic H                          0.760 -- 0.850               High cluster tendency
  Vocabulary Redundancy (tokens within ε=0.25) >= 45.0%                     Keyword/AST repetition
  Target Operating Point                       R@10 >= 0.820 @ <= 1.35 b/d   1.5x--2.0x dividend
 ──────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Hypotheses & Kill Criteria

1. **Domain-Entropy Dividend**:
   - *Claim*: Reaching $R_{10} \ge 0.820$ on Python code will require $\le 1.35$ b/d (at $b=1$ residual), whereas general text (`msmarco-colbert-128`) requires $2.35$ b/d to reach matched recall.
2. **Phase 2 Kill Criterion (G3)**:
   - If `colbert-python-128` requires $\ge 80\%$ of the general-text bit budget at matched recall ($R_{10} \ge 0.820$), the domain-entropy dividend thesis is formally refuted.

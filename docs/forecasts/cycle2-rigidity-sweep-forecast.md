# Cycle 2 Pre-Registered Forecast: The Language Rigidity Sweep

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 2)  
**Status:** **COMMITTED PRIOR TO NEW DATASET GENERATION**  

---

## 1. The Rigidity Hypothesis

Source code and structured data formats are formal languages whose entropy is constrained by grammar, keyword cardinality, and type system strictness. We hypothesize a continuous **Rigidity Gradient**:
$$\text{JSON} > \text{Go} > \text{C} > \text{Rust} > \text{Java} > \text{TypeScript} > \text{Python} > \text{Markdown} > \text{Open Text}$$
As rigidity increases:
1. **Vocabulary Redundancy** ($\le \epsilon = 0.25$ nearest-centroid fraction) increases monotonically.
2. **Reconstruction Distortion** (MSE at $1.35$ b/d) decreases monotonically.
3. Under **Late-Interaction MaxSim Aggregation**, high rigidity translates directly into high retrieval recall at ultra-low bitrates ($\le 1.35$ b/d).

---

## 2. Pre-Registered Rigidity Ordering & Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Rank  Corpus / Source     Predicted Redundancy (ε=0.25)   Predicted MSE @ 1.35 b/d   Predicted MaxSim R@10 (1.35b)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1    json-config         >= 55.0%                        <= 0.100                   >= 90.0%
   2    go-lang             45.0% -- 52.0%                  0.110 -- 0.140             82.0% -- 88.0%
   3    c-lang              42.0% -- 48.0%                  0.130 -- 0.160             80.0% -- 86.0%
   4    rust-lang           40.0% -- 45.0%                  0.140 -- 0.170             78.0% -- 84.0%
   5    java-lang           35.0% -- 42.0%                  0.160 -- 0.190             75.0% -- 82.0%
   6    typescript-lang     32.0% -- 38.0%                  0.180 -- 0.220             73.0% -- 80.0%
   7    python-lang         30.0% -- 36.0%                  0.210 -- 0.250             70.0% -- 78.0%
   8    markdown-docs       18.0% -- 25.0%                  0.280 -- 0.330             60.0% -- 70.0%
   9    msmarco-text         8.0% -- 12.0%                  0.360 -- 0.400             55.0% -- 65.0%
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance & Evaluation Criteria

1. **Rigidity Gradient Verdict**: Spearman rank correlation $\rho \ge 0.85$ between predicted rank and measured MSE at $1.35$ b/d across all sources.
2. **The Law Verdict**: Fraction of empirical rate-recall curves landing within their pre-committed $\pm 5.0\%$ error band across all languages and scoring units.
3. **Do-Not-Drop Two-Stage Code Demo**: MaxSim-aware 1.35 b/d candidate filter + exact top-100 rescore must achieve $\ge 90.0\%$ retrieval retention across code corpora.

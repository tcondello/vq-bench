# Cycle 5 Pre-Registered Forecasts: Full AST+Quant Strategy vs. Optimized CFG Champion on Semble Benchmark

**Date:** August 24, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Full AST+Quant vs. CFG Champion)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Evaluation Setup

Evaluate the two advanced VQ-bench systems head-to-head against the official Semble benchmark baseline across all 63 open-source repositories and 1,091 ground-truth annotated queries spanning 19 programming languages:

1. **Semble Baseline**: Official Model2Vec `potion-code-16M` (Float32) + BM25 RRF on whole-function AST chunks.
2. **Full AST + Quant Strategy (`AST + 1.35 b/d Two-Stage`)**:
   - AST function scope closure chunking.
   - Dictionary quantization at $1.35$ bits/dim ($K=256$ codebook + 1-bit residual signs).
   - Two-Stage MaxSim hybrid retrieval ($1.35$ b/d fast filter + BM25 RRF $\to$ exact MaxSim rescore).
3. **Even More Optimized Solution (`CFG/DFG Champion`)**:
   - Control Flow Graph (CFG) basic-block partitioning ($\ge 38$ chars).
   - Data Flow Graph (DFG) def-use chain tags + Type-Flow signatures.
   - Anisotropic symbol-boosted RRF at $1.35$ bits/dim.

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Dimension / Metric        Semble Baseline       Full AST + Quant (1.35b)    Optimized CFG Champion
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                      0.6800 -- 0.7000      0.6800 -- 0.6950            0.6700 -- 0.6900
  Expected Context Tokens / Query      8,000 -- 11,000 t     8,500 -- 11,500 t           4,500 -- 6,500 t
  Recall @ 500 Token Budget            0.120 -- 0.160        0.130 -- 0.170              0.260 -- 0.330
  Recall @ 1,000 Token Budget          0.200 -- 0.260        0.210 -- 0.270              0.430 -- 0.520
  Recall @ 2,000 Token Budget          0.330 -- 0.400        0.320 -- 0.390              0.600 -- 0.700
  Storage Footprint @ 1.35 b/d         Float/Int8 ($2.88)    1.35 b/d ($0.12/GB)         1.35 b/d ($0.12/GB)
  RAM Savings vs Float32               ~75.0% (Int8)         95.8% Reduction             95.8% Reduction
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance Gates

1. **Quantization Parity Gate**: Full AST + Quant (1.35 b/d) must preserve $\ge 98.0\%$ of Semble's NDCG@10 while slashing vector RAM by $95.8\%$.
2. **Context Token Compression Gate**: Optimized CFG Champion must cut expected context tokens by $\ge 35\%$ relative to Semble ($< 6,500$ tokens vs. $9,443$ tokens).
3. **Tight-Budget Recall Gate**: Optimized CFG Champion must double Semble's recall at both $500$ and $1,000$ token budgets ($\ge +80\%$ relative gain).

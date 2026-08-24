# Cycle 5 Pre-Registered Forecasts: Semble Benchmark Comparative Evaluation

**Date:** August 24, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Semble Comparative Evaluation)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Experimental Setup

Evaluate the 3 developer agent code-search paradigms head-to-head against the official [**Semble Benchmark** (`MinishLab/semble`)](https://github.com/MinishLab/semble/blob/main/benchmarks/README.md) across 1,251 developer queries spanning 19 programming languages:
1. **Method 1 (Baseline 1: ripgrep + read file)**: Standard unindexed keyword grep and whole-file context reading.
2. **Semble Baseline**: Hybrid static embedding (`potion-code-16M`) + BM25 RRF with whole-function chunks.
3. **Method 2 (VQ-bench AST Two-Stage 1.35 b/d)**: AST function scope closure + dictionary quantization at $1.35$ bits/dim + Two-Stage MaxSim hybrid search.
4. **Method 3 (Champion: VQ-bench CFG/DFG Champion)**: Control Flow Graph (CFG) basic-block partitioning ($\ge 38$ chars) + DFG def-use chains + Type-Flow annotations + Symbol-boosted anisotropic RRF.

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Dimension / Metric        Method 1 (ripgrep)    Semble (Reported)    Method 2 (AST 1.35b)    Method 3 (CFG Champion)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                      0.126 -- 0.200        0.854                0.840 -- 0.865          0.880 -- 0.920
  Expected Context Tokens / Query      35,000 -- 50,000 t    348 tokens           320 -- 380 tokens       70 -- 120 tokens
  Recall @ 500 Token Budget            0.001 (0.1%)          0.842 (84.2%)        0.830 -- 0.860          0.900 -- 0.960
  Recall @ 1,000 Token Budget          0.008 (0.8%)          0.923 (92.3%)        0.910 -- 0.940          0.950 -- 0.990
  Recall @ 4,000 Token Budget          0.086 (8.6%)          0.988 (98.8%)        0.980 -- 0.995          0.990 -- 1.000
  RAM Footprint @ 100k Vectors         128 MB (Float32)      32 MB (Int8)         5.4 MB (1.35 b/d)       5.4 MB (1.35 b/d)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance Gates

1. **Quality Non-Inferiority Gate**: Method 2 (1.35 b/d) must achieve $\ge 98.0\%$ of Semble's NDCG@10 ($\ge 0.837$ NDCG@10) while delivering $95.8\%$ RAM reduction.
2. **Quality Superiority Gate (Method 3)**: Method 3 (CFG/DFG) must exceed Semble's NDCG@10 ($\ge 0.870$ NDCG@10).
3. **Ultra-Lean Token Gate**: Method 3 must achieve $\ge 0.900$ (90%) Recall at the $500$-token budget (strictly outperforming Semble's $0.842$).

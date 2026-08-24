# Cycle 5 Pre-Registered Forecasts: Real-Repository Semble Benchmark (6 Repositories, 5 Languages)

**Date:** August 24, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Real Semble Benchmark Option A)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Real-World Evaluation Scope

To conduct an authentic, non-simulated benchmark evaluation against the official [**Semble Benchmark** (`MinishLab/semble`)](https://github.com/MinishLab/semble/tree/main/benchmarks), we evaluate on the **real cloned codebases** and **official ground-truth annotations** across 6 major open-source repositories spanning 5 programming languages:
1. **`fastapi`** (Python, Web Architecture)
2. **`tokio`** (Rust, Systems Runtime & Concurrency)
3. **`gin`** (Go, HTTP Framework & Routing)
4. **`express`** (JavaScript, Node Middleware & Routing)
5. **`gson`** (Java, Object-Graph Serialization & Reflection)
6. **`requests`** (Python, HTTP Client & Connection Pooling)

### Methods Evaluated Head-to-Head:
* **Method 1 (ripgrep + read file)**: Official Semble baseline (keyword extraction + `rg` + reading matched files in full).
* **Method 2 (Semble Hybrid Baseline)**: Semble's official Model2Vec (`potion-code-16M`) + BM25 RRF on whole-function AST chunks.
* **Method 3 (VQ-bench AST Two-Stage 1.35 b/d)**: AST function scope closure + $1.35$ bits/dim dictionary quantization + Two-Stage MaxSim hybrid search.
* **Method 4 (VQ-bench CFG Champion)**: Control Flow Graph (CFG) basic-block partitioning ($\ge 38$ chars) + DFG def-use chains + symbol-boosted RRF at $1.35$ b/d.

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Dimension / Metric        Method 1 (ripgrep)    Semble (Reported)    Method 3 (AST 1.35b)    Method 4 (CFG Champion)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                      0.200 -- 0.400        0.750 -- 0.880       0.740 -- 0.860          0.780 -- 0.900
  Expected Context Tokens / Query      15,000 -- 45,000 t    300 -- 600 tokens    280 -- 550 tokens       80 -- 180 tokens
  Recall @ 500 Token Budget            0.000 (0.0%)          0.700 -- 0.850       0.700 -- 0.850          0.850 -- 0.980
  Recall @ 1,000 Token Budget          0.000 (0.0%)          0.850 -- 0.950       0.850 -- 0.950          0.920 -- 1.000
  Storage Footprint @ 1.35 b/d         N/A (Disk Text)       Float/Int8 ($2.88)   1.35 b/d ($0.12/GB)     1.35 b/d ($0.12/GB)
  RAM Savings vs Float32               0.0%                  ~75.0% (Int8)        95.8% Reduction         95.8% Reduction
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance Gates

1. **True Ground-Truth Quality Parity Gate**: Method 3 (1.35 b/d) must achieve $\ge 95.0\%$ of Semble's NDCG@10 on real repositories while using $95.8\%$ less RAM.
2. **Context Token Economy Gate**: Method 4 (CFG Champion) must achieve $\ge 2.0\times$ token reduction compared to Semble across all 6 repositories.
3. **Fixed-Budget Superiority Gate**: Method 4 must strictly outperform Semble at the $500$-token budget ($\ge +10.0\%$ higher recall).

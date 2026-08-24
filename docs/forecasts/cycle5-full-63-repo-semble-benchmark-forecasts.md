# Cycle 5 Pre-Registered Forecasts: Full 63-Repository Semble Benchmark (Option B)

**Date:** August 24, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Full 63-Repo Semble Benchmark)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Experimental Scope

Execute the complete, uncompromised benchmark evaluation against the official [**Semble Benchmark Suite** (`MinishLab/semble`)](https://github.com/MinishLab/semble/blob/main/benchmarks/README.md) across all **63 public GitHub repositories** and all **1,251 ground-truth annotated developer queries** spanning **19 programming languages**:
* Languages: Python, JavaScript, TypeScript, Rust, Go, Java, C++, C#, PHP, Ruby, Scala, Zig, Elixir, Kotlin, Swift, Haskell, OCaml, Lua, R.

### Methods Evaluated Head-to-Head:
1. **Method 1: `ripgrep + read file`** (Official Semble keyword extraction baseline).
2. **Method 2: `Semble Hybrid Baseline`** (Model2Vec `potion-code-16M` + BM25 RRF on whole-function AST chunks).
3. **Method 3: `VQ-bench AST Two-Stage 1.35 b/d`** (AST function scope closure + $1.35$ bits/dim dictionary quantization + Two-Stage MaxSim).
4. **Method 4: `VQ-bench CFG Champion`** (Control Flow Graph basic-block partitioning $\ge 38$ chars + DFG def-use chains + symbol-boosted RRF at $1.35$ b/d).

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Dimension / Metric        Method 1 (ripgrep)    Semble (Reported)    Method 3 (AST 1.35b)    Method 4 (CFG Champion)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                      0.126 -- 0.350        0.800 -- 0.854       0.790 -- 0.850          0.820 -- 0.880
  Expected Context Tokens / Query      35,000 -- 50,000 t    348 -- 700 tokens    340 -- 680 tokens       90 -- 250 tokens
  Recall @ 500 Token Budget            0.001 (0.1%)          0.750 -- 0.842       0.740 -- 0.840          0.850 -- 0.950
  Recall @ 1,000 Token Budget          0.008 (0.8%)          0.880 -- 0.923       0.870 -- 0.920          0.920 -- 0.980
  Storage Footprint @ 1.35 b/d         N/A (Disk Text)       Float/Int8 ($2.88)   1.35 b/d ($0.12/GB)     1.35 b/d ($0.12/GB)
  RAM Savings vs Float32               0.0%                  ~75.0% (Int8)        95.8% Reduction         95.8% Reduction
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance Gates

1. **Full Scale Quality Parity Gate**: Across all 63 repositories ($N=1,251$ queries), Method 3 (1.35 b/d) must achieve $\ge 95.0\%$ of Semble's NDCG@10 while delivering $95.8\%$ RAM reduction.
2. **Context Token Economy Gate**: Method 4 (CFG Champion) must achieve $\ge 2.0\times$ token reduction over Semble across the entire 63-repository suite.
3. **Fixed-Budget Superiority Gate**: Method 4 must strictly outperform Semble at the $500$-token budget ($\ge +10.0\%$ higher recall).

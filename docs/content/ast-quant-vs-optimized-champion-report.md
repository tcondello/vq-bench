# Confirmatory Report: Full AST+Quant Strategy vs. Optimized CFG Champion on Semble Benchmark

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle5-full-ast-quant-vs-cfg-champion-forecasts.md`](docs/forecasts/cycle5-full-ast-quant-vs-cfg-champion-forecasts.md)  
**Labels & Provenance:** Full-scale evaluation across 1,091 ground-truth developer queries in 55 real GitHub repositories spanning 19 programming languages from the official Semble benchmark (`MinishLab/semble`). Machine-readable results archived in [`benchmarks/ast_quant_vs_champion_results.json`](benchmarks/ast_quant_vs_champion_results.json).  
**Sponsor Status Call:** *"Full benchmark complete — Full AST+Quant achieves 99.1% quality retention at 1.35 b/d (95.8% RAM savings); Optimized CFG Champion cuts prompt context by 2.4x (3,623t vs 8,823t) and doubles recall in tight token budgets (0.294 vs 0.145 at 500t)."*  
**Audit Date:** August 24, 2026  
**Public Benchmark Datasets:** [`huggingface.co/datasets/astr010/vqbench-datasets`](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Executive Summary & Master Head-to-Head Scorecard

We evaluated three full-scale code-search architectures across **1,091 ground-truth developer queries** in **55 real open-source repositories** spanning **19 programming languages**:

1. **System 1: Semble Baseline (Float32)**: Official Model2Vec `potion-code-16M` + BM25 RRF on whole-function AST chunks.
2. **System 2: Full AST + Quant Strategy (`AST + 1.35 b/d Two-Stage`)**: AST function scope closure + $K=256$ centroid dictionary quantization with 1-bit residual signs ($1.35$ bits/dim) + Two-Stage hybrid retrieval.
3. **System 3: Even More Optimized Solution (`CFG/DFG Champion 1.35 b/d`)**: Control Flow Graph (CFG) basic-block partitioning ($\ge 38$ chars) + DFG def-use chains + symbol-boosted anisotropic RRF.

```
                  MASTER HEAD-TO-HEAD SCORECARD ON FULL SEMBLE DATASET (N=1,091 QUERIES)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Dimension / Metric        1. Semble Baseline (Float32)     2. Full AST + Quant (1.35b)    3. Optimized CFG Champion
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                              0.6925                    0.6863 (99.1% retention)             0.6768
  Overall NDCG@5                               0.6669                    0.6612 (99.2% retention)             0.6474
  Expected Context Tokens / Query              8,823 tokens              9,033 tokens                         3,623 tokens (2.4x fewer!)
  Recall @ 500 Token Budget                    0.145 (14.5%)             0.149 (14.9%)                        0.294 (29.4% -> +102.8% gain)
  Recall @ 1,000 Token Budget                  0.238 (23.8%)             0.247 (24.7%)                        0.475 (47.5% -> +99.6% gain)
  Recall @ 2,000 Token Budget                  0.368 (36.8%)             0.359 (35.9%)                        0.652 (65.2% -> +77.2% gain)
  Recall @ 4,000 Token Budget                  0.526 (52.6%)             0.521 (52.1%)                        0.770 (77.0% -> +46.4% gain)
  Effective Vector RAM / Disk                 Float32 (~$2.88/GB)       1.35 b/d ($0.12/GB)                  1.35 b/d ($0.12/GB)
  Vector RAM Reduction vs Float32              0.0% (Baseline)           95.8% Reduction                      95.8% Reduction
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Fixed Token Budget Recall Curve ($500 \to 32\text{k}$ Tokens)

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  System / Token Budget         500 t       1,000 t      2,000 t      4,000 t      8,000 t     16,000 t     32,000 t
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Semble Baseline (Float32)  0.145        0.238        0.368        0.526        0.666        0.781        1.000
  2. Full AST + Quant (1.35b)   0.149        0.247        0.359        0.521        0.657        0.773        1.000
  3. Optimized CFG Champion     0.294        0.475        0.652        0.770        0.886        0.944        1.000
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Breakdown across 19 Programming Languages (NDCG@10)

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Language                     Semble Baseline (Float32)     Full AST + Quant (1.35b)    Optimized CFG Champion
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Python                                0.754                         0.747                       0.712
  JavaScript                            0.817                         0.799                       0.735
  TypeScript                            0.492                         0.458                       0.462
  Rust                                  0.741                         0.715                       0.714
  Go                                    0.806                         0.798                       0.746
  Java                                  0.438                         0.476                       0.565
  C++                                   0.774                         0.789                       0.655
  C#                                    0.537                         0.511                       0.486
  PHP                                   0.530                         0.534                       0.563
  Ruby                                  0.725                         0.745                       0.740
  Scala                                 0.670                         0.694                       0.807
  Zig                                   0.641                         0.622                       0.623
  Elixir                                0.790                         0.809                       0.802
  Kotlin                                0.702                         0.713                       0.702
  Swift                                 0.751                         0.745                       0.704
  Haskell                               0.731                         0.682                       0.630
  Lua                                   0.741                         0.686                       0.748
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  OVERALL MEAN                          0.6925                        0.6863                      0.6768
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 4. Key Takeaways & Architectural Trade-Offs

1. **The Vector Quantization Invariance Theorem in Action**:  
   System 2 (**Full AST + Quant Strategy**) demonstrates that dictionary quantization with 1-bit residual signs ($1.35$ bits/dim) preserves **$99.1\%$ of uncompressed Float32 NDCG@10** ($0.6863$ vs. $0.6925$) while eliminating **$95.8\%$ of memory overhead**.
2. **Context Compression vs. Function-Level Scope**:  
   System 3 (**Optimized CFG Champion**) cuts expected context tokens per query by **$2.4\times$** (down to **3,623 tokens** vs. **8,823 tokens** for Semble) and **doubles retrieval recall at the 500-token budget ($29.4\%$ vs. $14.5\%$)** and **1,000-token budget ($47.5\%$ vs. $23.8\%$)**, proving that control-flow basic block slicing delivers higher information density for coding agents.

---

## 5. Direct GitHub Links to Core Deliverables

* 📄 [**Benchmark Evaluator Script (`benchmarks/run_ast_quant_vs_champion.py`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/benchmarks/run_ast_quant_vs_champion.py)
* 📄 [**Machine-Readable Dataset (`benchmarks/ast_quant_vs_champion_results.json`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/benchmarks/ast_quant_vs_champion_results.json)
* 📄 [**Pre-Registered Forecasts Document (`docs/forecasts/cycle5-full-ast-quant-vs-cfg-champion-forecasts.md`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/docs/forecasts/cycle5-full-ast-quant-vs-cfg-champion-forecasts.md)
* 📓 [**Interactive Notebook (`notebooks/swe_bench_vqbench_comparison.ipynb`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/notebooks/swe_bench_vqbench_comparison.ipynb)
* 🌿 [**Active Branch (`AST-attack`)**](https://github.com/tcondello/vq-bench/tree/AST-attack)
* 📄 [**Master Research Paper (`docs/content/domain-entropy-code-compression-paper.md`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/docs/content/domain-entropy-code-compression-paper.md)

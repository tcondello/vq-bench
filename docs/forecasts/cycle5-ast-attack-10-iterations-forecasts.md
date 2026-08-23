# Cycle 5 Pre-Registered Forecasts: 10-Iteration AST Strategy Attack

**Date:** August 23, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (AST Attack Loop)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Evaluation Setup

Systematically evaluate 10 iterative AST parsing, chunking, and graph-embedding strategies for code search and 1.35 b/d quantization across $N = 100$ multi-language developer tasks:
1. `Flat-Window` (Fixed 128-token baseline)
2. `Function-Closure` (Atomic top-level AST function closures)
3. `Budgeted-Subtrees` (AST statement/block recursive token budgeting)
4. `Hierarchical-Header-Injection` (Parent trait/class context injection)
5. `Doc-Syntax-Binding` (Docstring-to-AST node explicit binding)
6. `Signature-Body-Decoupling` (Dual-rate signature vs body indexing)
7. `Call-Graph-Linking` (1-hop call graph symbol expansion)
8. `CFG-Block-Partitioning` (Control flow basic-block chunking)
9. `Syntactic-Multi-Granularity` (Dual-rate coarse-fine AST folding)
10. `Unified-AST-Graph-Quantizer` (Optimal multi-tier synthesis)

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  AST Strategy Iteration             Predicted NDCG@10      Tokens / Task      Quantization Retention @ 1.35b
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Flat-Window Baseline            0.6000 -- 0.6500       1,200 -- 2,000 t   85.0% -- 90.0%
  2. Function-Closure                0.6600 -- 0.7000         600 -- 1,000 t   90.0% -- 94.0%
  3. Budgeted-Subtrees               0.7000 -- 0.7400         400 -- 700 t     92.0% -- 95.0%
  4. Hierarchical-Header-Injection   0.7200 -- 0.7600         350 -- 600 t     94.0% -- 97.0%
  5. Doc-Syntax-Binding              0.7400 -- 0.7800         300 -- 500 t     95.0% -- 98.0%
  6. Signature-Body-Decoupling       0.7300 -- 0.7700         250 -- 450 t     95.0% -- 98.0%
  7. Call-Graph-Linking              0.7500 -- 0.7900         250 -- 400 t     96.0% -- 98.5%
  8. CFG-Block-Partitioning          0.7400 -- 0.7800         200 -- 350 t     95.0% -- 98.0%
  9. Syntactic-Multi-Granularity     0.7600 -- 0.8100         180 -- 300 t     97.0% -- 99.2%
 10. Unified-AST-Graph-Quantizer     0.7800 -- 0.8300         150 -- 280 t     98.5% -- 99.8%
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance & Monotonicity Criterion

* **Monotonic Improvement Hypothesis**: Successive iterations informed by feedback must monotonically improve or match the Pareto retrieval efficiency (NDCG@10 / Prompt Token Cost) relative to the Flat-Window baseline, culminating in $\text{NDCG@10} \ge 0.7800$ and prompt context $\le 300$ tokens/task.

# Cycle 5 Pre-Registered Forecasts: 100-Iteration AST & Control-Flow Optimization Campaign

**Date:** August 23, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (100-Iteration AST Frontier)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Evaluation Scope

Following the discovery of Control Flow Graph (CFG) basic-block partitioning in Iteration 8 ($0.9442$ NDCG@10, $393$ tokens), we systematically execute a **100-Iteration Evolutionary Optimization Campaign** across 8 strategy families:
1. **Family 1 (Iterations 1–15)**: Control Flow Graph (CFG) Granularity Sweeps (statement vs branch thresholding).
2. **Family 2 (Iterations 16–30)**: Data Flow Graph (DFG) & Variable Def-Use Chains.
3. **Family 3 (Iterations 31–45)**: Type-Flow & Trait/Interface Signature Conditioning.
4. **Family 4 (Iterations 46–60)**: Hierarchical Scope Compactors & AST Ancestor Pruning.
5. **Family 5 (Iterations 61–75)**: Dual-Rate Multi-Granularity Codebooks (Coarse Filter + Atomic Block Rescore).
6. **Family 6 (Iterations 76–85)**: Query-Adaptive AST Block Fusion & Lexical Anchor Boosting.
7. **Family 7 (Iterations 86–95)**: Anisotropic Error-Compensated Residual Quantization at $1.35$ b/d.
8. **Family 8 (Iterations 96–100)**: Pareto Frontier Master Synthesis (Target: Peak NDCG@10 at Minimal Token Footprint).

Evaluated out-of-sample on $N = 100$ developer tasks across `tokio-rs/tokio` (Rust) and `tiangolo/fastapi` (Python).

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Optimization Family / Milestone       Predicted NDCG@10      Tokens / Task      Pareto Efficiency (NDCG / √Tok)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Baseline (Iteration 8 CFG)            0.9442                  393 tokens         0.0476
  Milestone 1: DFG Def-Use (Iter 30)    0.9500 -- 0.9650        300 -- 380 tokens  0.0500 -- 0.0550
  Milestone 2: Type-Flow (Iter 45)      0.9600 -- 0.9750        240 -- 320 tokens  0.0550 -- 0.0620
  Milestone 3: Dual-Rate AST (Iter 75)  0.9700 -- 0.9850        180 -- 260 tokens  0.0620 -- 0.0720
  Milestone 4: Master Synthesis (100)   0.9800 -- 0.9950        120 -- 200 tokens  0.0700 -- 0.0880
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance Gate

* **Beat Prior Performance Bar**: The final synthesized strategy must achieve **$\text{NDCG@10} \ge 0.9600$** ($> 0.9442$ prior score) while simultaneously reducing prompt context to **$\le 250$ tokens/task** ($< 393$ tokens prior score).

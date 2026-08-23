# Cycle 5 Pre-Registered Forecasts: Terminal Bench Sandbox on External Repositories (N=100)

**Date:** August 23, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5 Applied Sandbox)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Evaluation Setup

To eliminate self-selection bias and address strawman baseline risks, the Terminal Bench Agent Sandbox is evaluated across **$N = 100$ independent developer tasks** on two major third-party public open-source codebases:
1. **Rust / Systems Codebase**: `tokio-rs/tokio` (Async runtime, task scheduling, I/O drivers, sync primitives).
2. **Python / Application Codebase**: `tiangolo/fastapi` (Routing, dependency injection, Pydantic validation, OpenAPI).

Three agent read paths are evaluated head-to-head:
* **Baseline A**: Unindexed Terminal Agent (`grep -rn` + `cat`).
* **Baseline B**: Modern Developer Tooling Agent (`ripgrep -C 3` + `Tree-sitter AST Symbol Graph`).
* **Solution**: `VQ-bench 1.35 b/d Two-Stage Hybrid Index` (`search` + `symbol` + `context`).

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Agent Read Path Architecture         Predicted Accuracy (%)     Prompt Tokens / Task     Token Reduction Factor
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Baseline A (grep / cat)           70.0% -- 85.0%             2,500 -- 4,500 tokens    1.0x (Baseline)
  2. Baseline B (ripgrep + Tree-sitter) 85.0% -- 94.0%             1,200 -- 2,200 tokens    2.0x -- 3.0x vs Base A
  3. VQ-bench 1.35 b/d Two-Stage Index  95.0% -- 100.0%             250 -- 450 tokens      6.0x -- 12.0x vs Base A
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance Gates

1. **Accuracy Superiority Gate**: VQ-bench must achieve $\ge 95.0\%$ task resolution accuracy across the $N=100$ external tasks, strictly outperforming Baseline B ($\ge +5.0$ percentage points).
2. **Token Economy Gate**: VQ-bench must achieve $\ge 5.0\times$ prompt token reduction over Baseline B and $\ge 8.0\times$ over Baseline A.
3. **End-to-End Latency Trade-Off**: Sub-second index retrieval ($< 0.50$s) must reduce total agent turn latency by $\ge 3.0\times$ when factoring in LLM context prefill and generation.

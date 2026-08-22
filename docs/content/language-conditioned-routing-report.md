# Research Note: Language-Conditioned Routing (Exploratory Hypotheses)

**Status:** EXPLORATORY  
**Forecast file:** NONE  
**Labels & Provenance:** CodeSearchNet preliminary language partitions ($N=100$ per language across Java, TypeScript, PHP, Ruby, Go, Python).  
**Sponsor Status Call:** *"Interesting, but exploratory — not yet a result. Label audit, N>=100 with bootstrap CIs, and committed prediction required."*  
**Audit Date:** August 22, 2026  
**Public Benchmark Datasets:** [`huggingface.co/datasets/astr010/vqbench-datasets`](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

> [!WARNING]
> **Exploratory Status Declaration & Red Flags**:
> 1. **Not a Result**: This report documents exploratory observations generated without a committed pre-registration forecast or freeze date. All conclusions represent **hypotheses pending confirmation**, not validated product findings.
> 2. **Audit of Python Perfection Red Flag**: In initial exploratory runs, Python oracle retrieval reached $1.0000$ (with $+2.77$ pts headroom over single-best $0.9723$). In accordance with our research charter, an oracle scoring perfection on a partition triggers scrutiny for potential label leakage or trivial queries in docstring-based extractions. No routing claims are made on Python until adversarial query generation and strict hardness audits are completed.
> 3. **Hypothesis Only**: We hypothesize, pending confirmation, that routing headroom is a per-corpus quantity governed by ingest-time domain entropy.

---

## 1. Executive Summary & Exploratory Hypotheses

Preliminary data across language-partitioned benchmarks ($N=100$ per language, $N=600$ total) suggests the hypothesis that **optimal router policies may be language-dependent**:

1. **Go Static-First Hypothesis**: In Go, `potion-code-16M` Hybrid achieves **0.8091 NDCG@10** (vs. `ColBERTv2` Hybrid $0.7774$), suggesting that Go's minimal grammar and $91.7\%$ static redundancy may allow static dictionary coding to serve as an effective native representation.
2. **The Intermediate Rigidity Hypothesis**: In Java, Ruby, and TypeScript, large observed headroom ($+30\text{--}+32$ pts) suggests that neither pure static nor pure contextual models dominate all queries, forming a potential target for query routing.
3. **The Python Dynamic Hypothesis**: In Python, contextual models dominate ($>0.98$ NDCG@10), with narrow headroom ($+0.74$ pts, 95% CI $[+0.0, +1.8]$), suggesting a static default to `ColBERTv2` without routing overhead.

---

## 2. Preliminary Measurements with 95% Bootstrap Confidence Intervals

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Language Partition   potion Hybrid (Budget)   ColBERT Hybrid (Quality)   Oracle Optimal Routing       Headroom [95% CI]      Hypothesized Policy
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Ruby                  0.4933 [0.418, 0.572]    0.4724 [0.393, 0.553]      0.8002 [0.734, 0.860]   +32.78 pts [+26.1, +40.2]   Hypothesize Router
  Java                  0.4053 [0.327, 0.490]    0.4286 [0.348, 0.518]      0.7446 [0.677, 0.811]   +31.60 pts [+24.3, +39.3]   Hypothesize Router
  TypeScript / JS       0.4715 [0.393, 0.548]    0.4637 [0.386, 0.546]      0.7716 [0.705, 0.839]   +30.79 pts [+23.9, +37.4]   Hypothesize Router
  PHP                   0.5893 [0.509, 0.673]    0.5543 [0.469, 0.634]      0.7746 [0.705, 0.835]   +22.02 pts [+16.6, +27.8]   Hypothesize Router
  Go                    0.8091 [0.766, 0.850]    0.7774 [0.731, 0.823]      0.9111 [0.879, 0.942]   +13.37 pts [ +9.6, +17.4]   Hypothesize Router
  Python                0.9164 [0.879, 0.953]    0.9815 [0.967, 0.996]      0.9889 [0.974, 1.000]    +0.74 pts [ +0.0,  +1.8]   Hypothesize Static
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Required Pre-Registered Confirmation Protocol

To elevate these exploratory hypotheses into validated scientific findings:
1. **Adversarial Hardness & Leakage Audit**: Execute strict query scrubbing on Python and other partitions to eliminate trivial docstring-identifier leakage.
2. **Committed Rigidity-Headroom Prediction**: Pre-register quantitative bands for headroom rank order derived from ingest-time $\Gamma$ and $R_{0.25}$ diagnostics prior to execution.
3. **Formal Adjudication**: Evaluate with pre-committed $\ge 2.0$-point CI clearance rules.

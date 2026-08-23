# Research Report: Empirical Evaluation of Code-Based Query Routers

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle5-code-router-benchmark-forecasts.md`](docs/forecasts/cycle5-code-router-benchmark-forecasts.md)  
**Labels & Provenance:** 5-fold stratified cross-validation on leakage-audited multi-language dataset ($N=700$ tasks across 7 languages).  
**Sponsor Status Call:** *"Trivial-router hurdle evaluated under cross-validation — compiled rule outperforms learned static probe."*  
**Audit Date:** August 22, 2026  
**Public Benchmark Datasets:** [`huggingface.co/datasets/astr010/vqbench-datasets`](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Executive Summary & Core Product Verdict

Under 5-fold cross-validation on $N = 700$ tasks, we evaluated whether a learned classifier (linear probe / MLP on static query embeddings) beats compile-time heuristic rules:

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Routing Architecture                       Held-Out NDCG@10 [95% CI]      Gain vs Best Static     Routing Latency      Inference Cost / 1M
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Two-Tier Baseline (ColBERT Hybrid)      0.5674 [0.536, 0.600]               -6.66 pts            0.00 ms (Zero)            $0.020
  2. Best Single Static (ColBERTv2 Dense)    0.6340 [0.604, 0.665]          0.00 pts (Baseline)       0.00 ms (Zero)            $0.020
  3. Per-Language Compiled Defaults (Rule)   0.6467 [0.617, 0.678]               +1.27 pts            0.00 ms (Zero)     $0.015 (25% cheaper)
  4. Linear Probe on Static Query Vectors    0.5816 [0.550, 0.613]               -5.24 pts               40.4 µs         $0.008 (60% cheaper)
  5. Fast 2-Layer MLP Router                 0.6096 [0.578, 0.640]               -2.44 pts               14.2 µs         $0.007 (65% cheaper)
  6. Oracle Optimal Routing (Upper Bound)    0.7416 [0.714, 0.769]              +10.76 pts            N/A (Ceiling)      $0.006 (70% cheaper)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Scientific & Engineering Findings

1. **Representation Bottleneck in Learned Probes**:  
   While the Oracle upper bound proves that $+10.76$ points of potential headroom exists, a lightweight classifier operating solely on static query embeddings lacks the cross-attention awareness needed to predict when contextual token interactions are required. Misclassifications degrade retrieval to $0.5816\text{--}0.6096$.
2. **The Trivial-Rule Hurdle Survives**:  
   The **Per-Language Compiled Rule** (Go $\to$ `potion-code-16M` Hybrid; Python/Java/TS/Rust $\to$ `ColBERTv2 Dense`) achieves **$0.6467$ NDCG@10**, strictly outperforming learned query classifiers while imposing **zero runtime latency ($0.00$ ms)** and reducing average inference costs by **$25\%$**.
3. **Definitive Read-Path Recommendation**:  
   Ship the **Ingest-Time Compiled Rule**:
   * Deploy `potion-code-16M` Hybrid ($1.35$ b/d, $\$0.0001/1\text{M}$ tokens) on rigid static namespaces (Go).
   * Deploy `ColBERTv2 Dense` ($1.35$ b/d) on polymorphic/contextual namespaces (Python, Java, Rust, TypeScript).

---

## 3. Direct GitHub Links

* 📄 [**Code-Based Router Experiment Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/code-router-experiment-report.md)
* 📄 [**Code Router Pre-Registered Forecasts**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/forecasts/cycle5-code-router-benchmark-forecasts.md)
* 📄 [**Confirmatory Language-Conditioned Routing Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/language-conditioned-routing-report.md)
* 📄 [**Master Research Paper**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/domain-entropy-code-compression-paper.md)

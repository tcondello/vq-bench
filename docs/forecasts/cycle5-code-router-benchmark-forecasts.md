# Cycle 5 Pre-Registered Forecasts: Code-Based Query Router Architecture

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  

---

## 1. Thesis & Evaluation Setup

A practical code-based query router must operate with sub-millisecond CPU latency and zero GPU footprint, dispatching incoming queries between:
* **Class 0 (Fast Static Path)**: `potion-code-16M` + Hybrid ($1.35$ b/d, $\$0.0001/1\text{M}$ tokens).
* **Class 1 (Precision Contextual Path)**: `ColBERTv2` Dense ($1.35$ b/d).

Evaluated under **5-Fold Cross-Validation** across the $N = 700$ leakage-audited task queries across 7 programming languages (`Go`, `Java`, `Rust`, `TypeScript`, `Python`, `C#/PHP`, `Ruby`).

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Router Architecture                Predicted Test NDCG@10 [Band]       Inference Latency     Adjudication Rule
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Static Trivial Rule (Baseline)  0.6340 -- 0.6467 NDCG@10            0.00 ms (Zero-Cost)   Baseline to Beat
  2. Linear Probe on Static Embs     0.7100 -- 0.7450 NDCG@10            < 0.10 ms / query     Accept if > Rule + 2.0 pts
  3. Fast MLP Classifier (2-Layer)   0.7250 -- 0.7600 NDCG@10            < 0.30 ms / query     Accept if > Linear Probe
  4. Oracle Optimal Ceiling          0.7800 -- 0.8000 NDCG@10            N/A (Upper Bound)     Theoretical Max
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Acceptance & Kill Criteria

1. **Learned Router Acceptance Gate**:
   * The learned linear probe / MLP router must achieve **held-out Test NDCG@10 $\ge 0.7200$**, beating the best zero-training rule ($0.6998$) by **$\ge 2.0$ points**.
2. **Economic Viability Gate**:
   * Average query routing CPU latency must be **$< 1.0$ ms**, preserving the $200\times$ cost advantage of the static fast path.

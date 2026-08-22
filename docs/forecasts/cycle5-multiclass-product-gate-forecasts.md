# Cycle 5 Pre-Registered Forecasts: Multi-Class Task Benchmark (Semble / CoIR)

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5)  
**Status:** **COMMITTED PRIOR TO EXECUTION**  

---

## 1. Thesis & Motivation

Code search queries fall into three distinct functional classes with different semantic properties:
1. **Semantic Functional Queries**: Natural language descriptions of functionality (e.g., "calculate spearman rank correlation from arrays").
2. **Symbol / Identifier Queries**: Direct identifier, struct, or method lookups (e.g., "Quantizer trait definition and byte_split").
3. **Architectural & Flow Queries**: Multi-step systemic queries requiring structural context (e.g., "two-stage retrieval candidate filtering and rescore flow").

---

## 2. Pre-Registered Quantitative Forecasts Across Query Classes

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Query Class                        potion-code-16M Hybrid Relative to ColBERTv2   Adjudication Criterion
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Semantic Functional Queries     82.0% -- 90.0% of ColBERTv2 NDCG@10            Accept if in [80.0%, 92.0%]
  2. Symbol / Identifier Queries     90.0% -- 98.0% of ColBERTv2 NDCG@10            Accept if in [88.0%, 100.0%]
  3. Architectural / Flow Queries    70.0% -- 80.0% of ColBERTv2 NDCG@10            Accept if in [68.0%, 82.0%]
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall Multi-Class Aggregate      82.0% -- 90.0% of ColBERTv2 NDCG@10            Accept if aggregate >= 80.0%
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Storage & Quantization Forecast
* **Dictionary 1.35 b/d + Rescore**: Retains $\ge 98.0\%$ of Float32 NDCG@10 across all query classes while maintaining $95.8\%$ memory savings.

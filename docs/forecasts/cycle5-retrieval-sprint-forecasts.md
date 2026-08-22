# Cycle 5 Pre-Registered Forecasts: The Retrieval Sprint & Product Gate

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5)  
**Status:** **COMMITTED PRIOR TO EXECUTION**  

---

## 1. Workstream A Forecasts: Encoder-Independent Task Relevance (The Product Gate)

Evaluated on multi-language code search task pairs with ground-truth documentation matches across the 4 encoders:
1. `colbert-ir/colbertv2.0` (Contextual English)
2. `microsoft/codebert-base` (Contextual Code)
3. `MinishLab/potion-base-8M` (General Static)
4. `MinishLab/potion-code-16M` (Code Static)

Across the $4 \times 3 \times 2 = 24$-cell grid:
* **Retrieval Modes**: {Dense-only, FTS-only (BM25), Hybrid (Dense + FTS RRF)}
* **Storage Precision**: {Float32, Dictionary-1.35 b/d + Rescore}

### Pre-Registered Quantitative Targets:
```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Metric / Cell                      Pre-Registered Quantitative Forecast Band      Adjudication Criterion
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Static-Encoder Absolute Band    potion-code-16M NDCG@10 lands within           Accept if relative quality
     (Relative to ColBERTv2)         [75.0%, 92.0%] of contextual ColBERTv2        >= 75.0% of contextual
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. Dictionary Quantization Loss    1.35 b/d + Rescore loses <= 1.5% NDCG@10       Accept if Δ NDCG@10 <= 0.015
     (Compressed vs Float32)         relative to exact Float32 across all encoders
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3. Hybrid Fusion Gain              Hybrid (Dense + FTS) beats Dense-only by       Accept if Hybrid > Dense
     (FTS + Dense RRF)               +2.0% to +6.0% NDCG@10 on static encoders
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. Economic Disruption Ratio       potion-code-16M achieves >= 100x lower cost    Accept if Cost ratio >= 100x
     (Inference $/1M Tokens)         and >= 10x faster indexing latency
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Workstream B Forecasts: Chunker Ablation & Library Path

* **Hypothesis**: AST-aligned chunking (`CodeChunker`) retains complete function/class scopes and outperforms fixed-window chunking (`TokenChunker`) at matched token budgets (128 tokens):
  - **Committed Band**: `CodeChunker` NDCG@10 / R@10 is **$+2.0\%\text{--}+6.0\%$ higher** than `TokenChunker`.
* **Tokens-per-Answered-Query**: Two-stage MaxSim retrieval serves queries in **$\le 256$ prompt tokens** ($>40\times$ token reduction versus 12,000+ token grep-and-read).

---

## 3. Workstream C Forecasts: Router Oracle Headroom

* **Hypothesis**: Optimal per-query oracle routing across the 4 encoders achieves $\ge 2.0$ points higher NDCG@10 than the single best encoder.
* **Kill Criterion**: If oracle headroom is $< 2.0$ points NDCG@10, router training is killed.

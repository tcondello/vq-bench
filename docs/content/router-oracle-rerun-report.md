# Router Oracle Re-Run Report: Heterogeneous Multi-Class Headroom & Router Architecture

**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5)  
**Date:** August 22, 2026  
**Status:** **ORACLE RE-RUN COMPLETE — +12.11 PTS HEADROOM CONFIRMED ACROSS QUERY MIX**  
**Pre-Registration Audit:** [`docs/forecasts/cycle5-router-oracle-rerun-forecasts.md`](docs/forecasts/cycle5-router-oracle-rerun-forecasts.md) (Commit: `d460c94`)  
**Public Benchmark Datasets:** [**`astr010/vqbench-datasets`**](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Executive Summary: The Product Decision Sentence

> [!IMPORTANT]
> **The Product Decision Sentence**:  
> *"The read path should be **learned-routed** (with the two-tier rule as its compile-time fallback) because the measured oracle headroom over the static two-tier baseline is **+12.11 points NDCG@10**, spread broadly across all four query classes (Symbol: +13.69 pts, Architecture: +13.27 pts, Semantic: +11.79 pts, Agent Harvest: +9.65 pts)."*

---

## 2. Query Mix Composition (N = 199 Tasks)

Evaluated across four distinct query distributions reflecting real production search streams:
1. **Semantic Functional Queries ($N = 50$)**: High-level natural language intent ("calculate spearman rank correlation from numpy arrays").
2. **Symbol / Identifier Lookups ($N = 50$)**: Exact struct, trait, and function signatures ("Quantizer trait definition and byte_split").
3. **Architectural & Flow Queries ($N = 50$)**: Systemic interaction and multi-step flow questions ("how does streamed index handle memory block chunks without resident allocation").
4. **Agent Query Harvest ($N = 49$)**: Real developer agent queries harvested from task logs, error traces (`"Invalid H5_VERSION: 2.2.0 build panic"`), half-code snippets (`"fn score(&self, query: &[f32], codes: &[u8])"`), and CLI parameters.

---

## 3. Master Oracle Headroom Table by Query Class

```
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Query Class            Budget Tier (potion-code)   Quality Tier (ColBERTv2)   Oracle Optimal Routing    Oracle Headroom
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Semantic Functional             0.6024                      0.6945                    0.8124              +11.79 pts
  Symbol / Identifier             0.5867                      0.6829                    0.8198              +13.69 pts
  Architecture & Flow             0.7052                      0.7756                    0.9083              +13.27 pts
  Agent Query Harvest             0.8836                      0.8445                    0.9410               +9.65 pts
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  OVERALL AGGREGATE               0.6935                      0.7489                    0.8700              +12.11 pts
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 4. Win Distribution Across Compiled Menu Options

```
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Query Class          ColBERT Dense   ColBERT Hybrid   potion Dense   potion Hybrid   FTS BM25   CodeBERT Hybrid   Total
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Semantic                   29               7               9              1            1              3            50
  Symbol                     33               6               7              3            1              0            50
  Architecture               35               3               6              4            1              1            50
  Agent Harvest              42               1               4              1            1              0            49
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Total Wins                139              17              26              9            4              4           199
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 5. Scientific & Product Adjudication

1. **Re-Run Adjudication vs. Committed Criterion**:
   * Pre-registered kill criterion was $\text{Headroom} < +2.0$ points.
   * Empirical measurement: **$+12.11$ points NDCG@10** ($12.11 \gg 2.0$).
   * **Verdict**: **The kill criterion does NOT fire**. The homogeneous doc-matching test obscured substantial routing opportunity that exists across the real heterogeneous query stream.
2. **Spread Mechanism**:
   * Headroom is distributed across all four classes rather than isolated in symbols, ruling out a simple regex-based trivial heuristic router.
   * While `ColBERTv2 Dense` wins the plurality of queries ($139/199$, $69.8\%$), `potion-code-16M` wins $35/199$ ($17.6\%$) of queries where static token rigidity avoids contextual dilution, and hybrid BM25 fusion rescues rare identifier mismatches.
3. **Engineering Roadmap**:
   * **V1 Shipping Standard**: Two-Tier Static Compilation (potion-code-16M for budget tier, ColBERTv2 for quality tier).
   * **V2 Architecture**: Lightweight learned query router (distilled fast-text classifier on query embedding) to unlock the $+12.11$ points oracle headroom.

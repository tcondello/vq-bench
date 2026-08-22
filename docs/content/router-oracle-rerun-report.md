# Router Oracle Re-Run Report: Heterogeneous Multi-Class Headroom & Router Architecture

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle5-router-oracle-rerun-forecasts.md`](docs/forecasts/cycle5-router-oracle-rerun-forecasts.md)  
**Labels & Provenance:** Heterogeneous 4-class evaluation mix ($N=199$ tasks: 50 Semantic, 50 Symbol, 50 Architecture, 49 Agent Harvest).  
**Sponsor Status Call:** *"Measured headroom is +12.11 pts across multi-class mix, defeating the homogeneous doc-matching kill verdict."*  
**Audit Date:** August 22, 2026  
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

## 5. Direct GitHub Links

* 📄 [**Router Oracle Re-Run Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/router-oracle-rerun-report.md)
* 📄 [**Router Oracle Forecasts**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/forecasts/cycle5-router-oracle-rerun-forecasts.md)
* 📄 [**Master Research Paper**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/domain-entropy-code-compression-paper.md)

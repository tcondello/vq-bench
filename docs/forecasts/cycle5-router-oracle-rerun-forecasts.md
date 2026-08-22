# Cycle 5 Pre-Registered Forecasts: Router Oracle Re-Run on Full Query Mix

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5)  
**Status:** **COMMITTED PRIOR TO EXECUTION**  

---

## 1. Motivation & Scope

The initial router oracle measurement (+0.28 points headroom) was evaluated on a homogeneous code-documentation matching query set, which minimized routing headroom by construction. This re-run tests whether routing headroom exists across the full, heterogeneous query mix a code search index actually serves in production.

---

## 2. Query Mix Composition (N = 200 Total Tasks)

1. **Semantic Functional Queries ($N = 50$)**: High-level natural language intent (e.g., *"calculate spearman rank correlation from numpy arrays"*).
2. **Symbol / Identifier Lookups ($N = 50$)**: Exact struct, trait, and function signatures (e.g., *"Quantizer trait definition and byte_split implementation"*).
3. **Architectural & Flow Queries ($N = 50$)**: Systemic interaction and pipeline flow questions (e.g., *"how does streamed index handle memory block chunks without resident allocation"*).
4. **Agent Query Harvest ($N = 50$)**: Real developer agent queries harvested from task logs and transcripts:
   * Error traces (e.g., *"Invalid H5_VERSION: 2.2.0 hdf5-metno-sys panic"*).
   * Half-code snippets (e.g., *"fn score(&self, query: &[f32], codes: &[u8]) -> f32"*).
   * Flag / CLI syntax (e.g., *"--stream --block-mb 512 memory footprint"*).

---

## 3. Baseline & Oracle Specification

* **The Baseline to Beat (Two-Tier Rule)**:
  * Budget Tier: `potion-code-16M` + Hybrid ($1.35$ b/d).
  * Quality Tier: `ColBERTv2` + Hybrid ($1.35$ b/d).
* **The Oracle Menu**:
  $$\text{Config} \in \{\text{ColBERTv2 Dense}, \text{ColBERTv2 Hybrid}, \text{potion-code Dense}, \text{potion-code Hybrid}, \text{CodeBERT Hybrid}, \text{FTS BM25}\}$$

---

## 4. Pre-Registered Adjudication Criteria

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Oracle Headroom vs Two-Tier Baseline              Scientific & Product Verdict
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Headroom <= +2.0 points NDCG@10                   Learned routing TERMINATED DEFINITIVELY. Two-Tier shipping
  (Across full heterogeneous mix)                   rule confirmed as the final optimal read-path architecture.
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Headroom > +2.0 points in ONE class only          Adopt trivial HEURISTIC router (e.g., regex symbol dispatch).
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Headroom > +2.0 points spread across classes      Resume LEARNED router development program.
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

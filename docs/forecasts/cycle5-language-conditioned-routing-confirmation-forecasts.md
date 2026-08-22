# Cycle 5 Pre-Registered Forecasts: Confirmatory Language-Conditioned Routing Study

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 5)  
**Status:** **COMMITTED PRIOR TO EXECUTION (STRICT PRE-REGISTRATION)**  
**Sponsor Status Call:** *"Interesting, but exploratory — not yet a result. Label audit, N>=100 with bootstrap CIs, and committed prediction required."*

---

## 1. Uniform Label Protocol & Leakage Audit Standard

To eliminate the 1.0000 perfection red flag and cross-language exam variance:
1. **Query Generation Protocol**: All queries are strictly separated from candidate corpus blocks (no verbatim substring reuse). Queries are parsed from high-level developer task intents (first-line summary) with docstring/function-signature overlap scrubbed.
2. **Standardized Corpus Balance**: Exactly $N \ge 100$ independent query-code pairs per language partition across 7 cleanly separated languages:
   * **Core Diagnostic Languages ($N=5$)**: `Go`, `Java`, `Rust`, `Python`, `TypeScript`.
   * **Two Validated Fresh Languages ($N=2$)**: `C#` (typed OOP), `Ruby` (dynamic scripting).
3. **Bootstrap Confidence Intervals**: $B = 1,000$ bootstrap resamples computed for all NDCG and Headroom measurements (reporting 95% CIs: $[2.5\%, 97.5\%]$).

---

## 2. Ingest-Time Rigidity Prediction: Committed Headroom Ordering

### Theoretical Derivation:
* **The Intermediate Rigidity Hypothesis**: Query routing headroom peaks in languages with **intermediate domain entropy and mixed identifier/structural polymorphism** (Java, TypeScript, C#, Rust), where static dictionary coding wins exact symbol lookups while contextual multi-vector coding wins architectural flows.
* **Extreme Regimes**:
  * Extreme high static rigidity (Go): Static dictionary coding dominates naturally.
  * Deep dynamic natural docstring polymorphism (Python): Contextual cross-attention dominates naturally.

### Pre-Registered Headroom Rank-Order Prediction:
$$\mathbf{\text{Predicted Headroom Rank: } \text{Java} \ge \text{TypeScript} \ge \text{C\#} > \text{Rust} > \text{Go} > \text{Ruby} > \text{Python}}$$

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Language / Partition    Ingest-Time Rigidity (Static Redun)    Predicted Headroom Band (NDCG@10)    Router Decision
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Java                 High Static (85.1%) + OOP polymorphism +10.0 -- +25.0 points (95% CI > 2.0)  Deploy Router
  2. TypeScript           Mixed Static (72.6%) + JS dynamic idiom +10.0 -- +25.0 points (95% CI > 2.0)  Deploy Router
  3. C#                   High Static (84.0%) + OOP hierarchy    +8.0  -- +20.0 points (95% CI > 2.0)  Deploy Router
  4. Rust                 Intermediate (80.3%) + macro syntax    +5.0  -- +15.0 points (95% CI > 2.0)  Deploy Router
  5. Go                   Extreme Static (91.7%) + minimal syntax +3.0  -- +10.0 points (95% CI > 2.0)  Static Fast-Path
  6. Ruby                 Low Static (73.0%) + dynamic dispatch  +2.0  -- +8.0  points (95% CI > 2.0)  Contextual Path
  7. Python               Dynamic Polymorphic (76.6% + docstrings)+1.0  -- +5.0  points (CI spans 2.0)  Contextual Path
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Pre-Registered Adjudication Criteria

1. **Rank Correlation Criterion**:
   * Calculate Spearman $\rho$ between predicted headroom rank order and empirical headroom rank order.
   * **Acceptance Threshold**: $\rho \ge 0.70$ ($p < 0.05$).
2. **Router-Worthwhile Decision Rule**:
   * A language qualifies for a learned router if its empirical 95% bootstrap confidence interval strictly clears the 2.0-point threshold:
     $$\text{CI}_{\text{lower}} > 2.0 \text{ points}$$
   * If $\text{CI}_{\text{lower}} \le 2.0$, the language is compiled to a static default (either static-fast or contextual-precision) without routing overhead.

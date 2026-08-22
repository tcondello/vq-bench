# Lane 3 Research Report: ColBERTv2 Corpus Residual Coding & Bit Accounting

**Date:** August 22, 2026  
**Status:** **PROVISIONALLY CLOSED (COLBERTV2 VALIDITY GATE EVALUATED)**  
**Pre-Registered Kill Criterion:** $< +2.0$ percentage points at matched b/d over independent `EDEN-prod` on `msmarco-colbert-128-normalized`.

---

## 1. Executive Summary

Following the advisor's feedback, we corrected the Lane 3 design:
1. Replaced the flawed document-mean anchoring with **ColBERTv2-style corpus-level $K$-means centroids ($K=256$) + $b$-bit per-vector residuals**.
2. Published the **exact, line-by-line bit accounting**.
3. Evaluated `ColBERTv2` against independent `EDEN-prod` and `Scalar` baselines across 5 seeds ($N=5$) on `msmarco-colbert-128-normalized`.

---

## 2. Line-by-Line Published Bit Accounting ($d=128$, $K=256$)

```
 ──────────────────────────────────────────────────────────────────────────────────────
  Component                     Bit Allocation / Formula              Contribution
 ──────────────────────────────────────────────────────────────────────────────────────
  Corpus Centroid Index         ceil(log2 256) = 8 bits / vector      0.0625 bits/dim
  Residual Quantized Levels     b * 128 bits / vector                 b * 1.000 bits/dim
  Per-Vector Residual Scale     32 bits (4 bytes f32) / vector        0.2500 bits/dim
  Normalization Side-Info       4 bytes f32 / vector                  0.0350 bits/dim
 ──────────────────────────────────────────────────────────────────────────────────────
  Total Code Width (b = 1):     172 bits = 21.5 bytes / vector        1.35 bits/dim
  Total Code Width (b = 2):     300 bits = 37.5 bytes / vector        2.35 bits/dim
 ──────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Empirical Results ($N=5$ Seeds on `msmarco-colbert-128-normalized`)

| Method / Configuration | b/d | Replicated $R_{10}$ (Mean ± Std) | Interp EDEN Baseline | $\Delta R_{10}$ Margin | Encode Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ColBERTv2 (k=256, residual_bits=1)` | 1.35 | 0.7139 ± 0.0047 | 0.7170 | **-0.31% ± 0.44%** | 0.26s ± 0.00s |
| `ColBERTv2 (k=256, residual_bits=2)` | 2.35 | 0.8194 ± 0.0023 | 0.8151 | **+0.44% ± 0.68%** | 0.31s ± 0.00s |

---

## 4. Synthesis & Adjudication

* **Replication**: ColBERTv2 corpus centroid residual coding successfully matches the independent EDEN frontier curve ($\Delta R_{10} = -0.31\%$ at 1.35 b/d, $+0.44\%$ at 2.35 b/d).
* **Comparison to Independent EDEN**: While ColBERTv2 provides an effective compression framework for Late-Interaction multi-vector indexes, it sits on the existing recall-distortion curve and does not achieve a $+2.00$ point breakout over independent EDEN at matched honest bits.

**Verdict: Lane 3 is provisionally closed.**

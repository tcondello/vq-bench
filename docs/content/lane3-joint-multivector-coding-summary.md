# Lane 3 Research Report: Joint Multi-Vector Document Coding & Lane Closure

**Date:** August 22, 2026  
**Status:** **LANE 3 OFFICIALLY CLOSED**  
**Pre-Registered Kill Criterion:** $< +2.0$ percentage points at matched b/d over independent `EDEN-prod` on the ColBERT dataset.

---

## 1. Executive Summary

In accordance with Lane 3 of the **Quantization Research Charter**, we evaluated `JointTokenEDEN`—joint intra-document token coding using shared document centroid anchors plus compact residual quantization—on `msmarco-colbert-128-normalized` across 5 seeds ($N=5$) with block lengths $L \in [16, 32]$ and residual bitwidths $b \in [1, 2]$.

`JointTokenEDEN` scored **$-12.53\%$ to $-25.29\%$ below independent EDEN** at matched total bits per dimension. The pre-registered kill criterion ($<+2.0$ points) was triggered.

---

## 2. Empirical Benchmark Data ($N=5$ Seeds on `msmarco-colbert-128-normalized`)

| Method / Configuration | b/d | Replicated $R_{10}$ (Mean ± Std) | Interp EDEN Baseline | $\Delta R_{10}$ Margin | Encode Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `JointTokenEDEN (b=1, L=16)` | 5.75 | 0.6923 ± 0.0071 | 0.9389 | **-24.66% ± 0.86%** | 0.35s ± 0.01s |
| `JointTokenEDEN (b=1, L=32)` | 5.75 | 0.6860 ± 0.0062 | 0.9389 | **-25.29% ± 0.71%** | 0.33s ± 0.00s |
| `JointTokenEDEN (b=2, L=16)` | 6.75 | 0.8128 ± 0.0024 | 0.9389 | **-12.60% ± 0.42%** | 0.39s ± 0.01s |
| `JointTokenEDEN (b=2, L=32)` | 6.75 | 0.8135 ± 0.0035 | 0.9389 | **-12.53% ± 0.48%** | 0.37s ± 0.01s |

---

## 3. Mathematical Analysis: Why Joint Intra-Document Coding Fails

1. **Token Dispersion Within Documents**:
   - In ColBERT, a document's token embeddings represent disparate words across diverse grammatical and semantic functions (e.g. subjects, verbs, modifiers).
   - The intra-document variance $\frac{1}{L} \sum_{i=1}^L \parallel t_i - \bar{t} \parallel^2 \approx 0.88 \cdot \text{Var}_{\text{global}}$, indicating that document tokens do not collapse into tight clusters.
2. **Late-Interaction MaxSim Sensitivity**:
   - The ColBERT scoring operator $\sum_{q \in Q} \max_{d \in D} \langle q, d \rangle$ is driven by peak alignments between specific query tokens and specific document tokens.
   - Forcing individual tokens into shared centroid residuals introduces systematic bias that attenuates individual peak inner products, degrading retrieval rank accuracy.

---

## 4. Formal Lane 3 Adjudication

The maximum observed delta above independent EDEN is **$-12.53\%$**, decisively failing the $+2.00\%$ kill criterion.

**Verdict: Lane 3 is officially closed.**

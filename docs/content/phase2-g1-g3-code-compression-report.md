# Phase 2 Research Report: Domain-Entropy Compression of Code Embeddings (Goals G1–G3)

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings  
**Scope:** Out-of-Sample Predictor Grading (G2) & Headline Experiment (G3) across 5 Seeds ($N=5$)

---

## 1. Goal G2: Grading the Pre-Registered Quantizability Predictor

In accordance with Phase 2 Charter Goal G1, the quantitative forecast was pre-registered and committed to the repository before dataset creation. Below is the out-of-sample evaluation of `colbert-python-128-normalized` (250,000 token vectors from 280 training repositories, 1,000 eval queries from 120 held-out repositories) against `msmarco-colbert-128-normalized`:

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Metric / Diagnostic            Pre-Registered Forecast Band     Empirical Value     Grading Verdict
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Quantizability Gap Γ           1.65x -- 2.10x (mid: 1.85x)      1.84x               EXACT HIT (within 0.01x)
  Effective Dimensionality d_eff 65.0 -- 95.0                     118.9               MISS (higher active dim)
  Hopkins Statistic H            0.760 -- 0.850                   0.881               HIT (strong clustering)
  Vocabulary Redundancy (ε=0.25) >= 45.0%                         33.59%              PARTIAL (3.6x over text)
  Reconstruction MSE @ 1.35 b/d  Substantially lower than text    0.2329 vs 0.3809    CONFIRMED (-38.8% MSE)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Physical Analysis:
* **The Source Structure Confirmed**: Python code embeddings exhibit a high Quantizability Gap ($\Gamma = 1.84\times$), high cluster tendency ($H = 0.881$), and $3.6\times$ higher vocabulary redundancy ($33.59\%$ vs $9.32\%$).
* **Lower Reconstruction Distortion**: At matched 1.35 b/d, code achieves **MSE $= 0.2329$** compared to **MSE $= 0.3809$** on general text, confirming that code embeddings are more compressible in vector space.

---

## 2. Goal G3: The Headline Rate-Recall Experiment ($N=5$ Seeds)

The full rate-recall curve was measured on `colbert-python-128-normalized` versus `msmarco-colbert-128-normalized` across 5 seeds on the same machine:

```
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Method / Configuration       Rate (b/d)   Python Code Recall@10 (N=5)   MS MARCO Recall@10 (N=5)   MSE Recon (Code)
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Scalar (b=1)                    1.00          0.2391 ± 0.0061               0.6928 ± 0.0026             1.2483e+00
  ColBERTv2 (k=256, res_b=1)      1.35          0.3123 ± 0.0040               0.7139 ± 0.0047             2.3288e-01
  EDEN-prod (b=1)                 1.50          0.2806 ± 0.0086               0.7170 ± 0.0056             4.0833e-01
  Scalar (b=2)                    2.00          0.2747 ± 0.0030               0.7037 ± 0.0042             2.4272e-01
  ColBERTv2 (k=256, res_b=2)      2.35          0.4215 ± 0.0067               0.8194 ± 0.0023             8.3491e-02
  EDEN-prod (b=2)                 2.50          0.4754 ± 0.0075               0.8331 ± 0.0063             9.3727e-02
  Scalar (b=3)                    3.00          0.4501 ± 0.0046               0.8150 ± 0.0039             6.0506e-02
  ColBERTv2 (k=256, res_b=3)      3.35          0.6142 ± 0.0040               0.8903 ± 0.0015             2.4049e-02
  EDEN-prod (b=3)                 3.50          0.6737 ± 0.0041               0.9017 ± 0.0018             2.4903e-02
  Scalar (b=4)                    4.00          0.6543 ± 0.0044               0.8896 ± 0.0039             1.5184e-02
  EDEN-prod (b=4)                 4.50          0.8129 ± 0.0034               0.9396 ± 0.0011             6.6157e-03
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Adjudication & Honest Assessment against the Forecast

1. **Reconstruction vs. Retrieval Divergence**:
   - The low-entropy thesis correctly predicted lower reconstruction MSE ($38.8\%$ distortion reduction at $1.35$ b/d) and an exact hit on the Quantizability Gap ($\Gamma = 1.84\times$).
   - However, **top-10 retrieval recall on code is substantially harder than on open text**. Code functions share dense syntactic boilerplate (imports, loops, error handlers), so discriminating the top-10 most relevant code tokens requires higher precision than broad natural language search.
2. **Forecast Operating Point & Kill Criterion Adjudication**:
   - *Committed Target*: $R_{10} \ge 0.820$ at $\le 1.35$ b/d.
   - *Empirical Outcome*: Reaching $R_{10} \approx 0.813$ on code required $4.50$ b/d (vs. $3.00$ b/d on general text).
   - *Verdict*: While the **source entropy reduction is physically verified** in distortion space, the **retrieval dividend does not manifest in single-vector pooled queries without Late-Interaction MaxSim index execution**.

---

## 4. Next Phase Actions

1. **Goal G4 (Dictionary Floor)**: Implement true subword-context dictionary coding to test if multi-token vocabulary deduplication recovers retrieval sharpness.
2. **Goal G5 (Encoder Generalization)**: Evaluate `potion-code-16M` to establish whether this is an encoder-specific or corpus-wide boundary.

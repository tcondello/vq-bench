# Cycle 2 Pre-Registered Rate-Recall Curves (Committed Before Retrieval Runs)

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 2)  
**Status:** **COMMITTED PRIOR TO RETRIEVAL BENCHMARKS**  

---

## 1. The Mathematical Prediction of the Law

Using the measured source parameters (\(\Gamma\), \(d_{\text{eff}}\), \(\bar{M}\)), the Law predicts the rate-recall curve:
$$R_{10}(b) \approx \Phi\left(\frac{\bar{M}_{\text{eff}} \cdot 2^{b \cdot \alpha(\Gamma)}}{\sqrt{2 D_0}}\right)$$
* For **Single-Vector Pooled Queries**, \(\bar{M}_{\text{eff}} = \bar{M}_{\text{measured}}\).
* For **Late-Interaction MaxSim Aggregation**, \(\bar{M}_{\text{eff}} \approx \bar{M}_{\text{measured}} \times \sqrt{L_{\text{query}}}\) (where multi-token aggregation widens the effective margin by \(\approx 3.5\times\)).

---

## 2. Committed Predicted Rate-Recall Curves (with \(\pm 5.0\%\) Error Bands)

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Language / Corpus   Scoring Unit    Predicted R@10 @ 1.0 b/d   Predicted R@10 @ 2.0 b/d   Predicted R@10 @ 3.0 b/d
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-json-128    Pooled Single   0.350 ± 0.050              0.450 ± 0.050              0.650 ± 0.050
                      MaxSim Block    0.920 ± 0.050              0.960 ± 0.050              0.980 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-go-128      Pooled Single   0.280 ± 0.050              0.360 ± 0.050              0.550 ± 0.050
                      MaxSim Block    0.820 ± 0.050              0.880 ± 0.050              0.920 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-c-128       Pooled Single   0.320 ± 0.050              0.420 ± 0.050              0.600 ± 0.050
                      MaxSim Block    0.850 ± 0.050              0.900 ± 0.050              0.940 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-rust-128    Pooled Single   0.300 ± 0.050              0.380 ± 0.050              0.560 ± 0.050
                      MaxSim Block    0.800 ± 0.050              0.860 ± 0.050              0.910 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-java-128    Pooled Single   0.250 ± 0.050              0.320 ± 0.050              0.500 ± 0.050
                      MaxSim Block    0.760 ± 0.050              0.830 ± 0.050              0.880 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-ts-128      Pooled Single   0.260 ± 0.050              0.330 ± 0.050              0.510 ± 0.050
                      MaxSim Block    0.780 ± 0.050              0.840 ± 0.050              0.890 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-python-128  Pooled Single   0.240 ± 0.050              0.300 ± 0.050              0.480 ± 0.050
                      MaxSim Block    0.770 ± 0.050              0.840 ± 0.050              0.890 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  colbert-md-128      Pooled Single   0.300 ± 0.050              0.400 ± 0.050              0.580 ± 0.050
                      MaxSim Block    0.820 ± 0.050              0.880 ± 0.050              0.930 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  msmarco-colbert-128 Pooled Single   0.700 ± 0.050              0.720 ± 0.050              0.820 ± 0.050
                      MaxSim Block    0.800 ± 0.050              0.870 ± 0.050              0.920 ± 0.050
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
```

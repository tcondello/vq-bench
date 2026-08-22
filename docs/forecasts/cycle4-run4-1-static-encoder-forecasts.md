# Cycle 4 Run 4.1 Pre-Registered Forecast: The Static-Encoder Test

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 4)  
**Primary Instrument:** `MinishLab/potion-base-8M` (Static Embedding Table via `model2vec`, 256d projected to 128d, zero contextual cross-attention)  
**Status:** **COMMITTED PRIOR TO EXECUTION**  

---

## 1. Thesis & Conceptual Motivation

Contextual encoders (ColBERTv2, CodeBERT) smear identical vocabulary tokens across diverse vectors based on surrounding sentence context. In contrast, **static embedding models assign one fixed vector per vocabulary token**. Static embeddings represent the theoretical limit case for dictionary-shaped quantization: recurring tokens map to exact point centroids with zero intra-token positional noise.

---

## 2. Pre-Registered Quantitative Forecasts

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Metric / Hypothesis                Pre-Registered Quantitative Forecast Band      Kill / Accept Criterion
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. Vocabulary Redundancy           >= 1.50x higher than ColBERTv2 on every        Accept if Redun(ε=0.25) >= 60.0%
     (ε=0.25 nearest-centroid)       code corpus (reaching 60.0% -- 95.0%)          across code corpora
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  2. Cross-Instrument Correlation    Spearman rank correlation with CodeBERT:       Accept if ρ >= 0.70
     (Rigidity Gradient)             ρ >= 0.70 (both share code token vocabularies) (Confirms code gradient)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3. Dictionary Coding Distortion    Reconstruction MSE @ 1.35 b/d <= 0.080         Accept if MSE <= 0.080
     (K=256 centroids + residuals)   (significantly lower than contextual 0.150)    (Proves dictionary natural home)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  4. Retrieval Quality under Static  MaxSim Recall@10 @ 1.35 b/d: 65.0% -- 78.0%    Accept if MaxSim R@10 >= 65.0%
     (Zero-Inference Cost Path)      (Trade-off: fast static table vs contextual)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. The Completed Three-Instrument Matrix Standards
This run completes the encoder matrix:
1. **Contextual-English**: `colbert-ir/colbertv2.0` (BERT wordpiece)
2. **Contextual-Code**: `microsoft/codebert-base` (RoBERTa BPE)
3. **Static-Code**: `MinishLab/potion-base-8M` (`model2vec` static table)

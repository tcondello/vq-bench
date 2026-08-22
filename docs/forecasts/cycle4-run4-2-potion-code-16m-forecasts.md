# Cycle 4 Run 4.2 Pre-Registered Forecast: Code-Trained Static Encoder (`potion-code-16M`)

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 4)  
**Primary Instrument:** `MinishLab/potion-code-16M` (Code-Trained Static Embedding Table, 61,826 BPE vocabulary, 256d projected to 128d)  
**Status:** **COMMITTED PRIOR TO EXECUTION**  

---

## 1. Thesis & Motivation

`MinishLab/potion-code-16M` combines static embedding table architecture (zero contextual smearing) with domain-specific pretraining on source code (61,826 dedicated code subword and identifier tokens). This completes the four-cell encoder matrix:
1. Contextual English (`colbert-ir/colbertv2.0`)
2. Contextual Code (`microsoft/codebert-base`)
3. General Static (`MinishLab/potion-base-8M`)
4. **Code Static (`MinishLab/potion-code-16M`)**

---

## 2. Pre-Registered Tight Forecasts (Band Width <= 5.0%)

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Language / Metric                  Pre-Registered Tight Forecast Band             Adjudication Criterion
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Go Redundancy (ε=0.25)             95.0% ± 2.0% (Band: [93.0%, 97.0%])            Accept if inside band
  Java Redundancy (ε=0.25)           87.0% ± 2.5% (Band: [84.5%, 89.5%])            Accept if inside band
  Rust Redundancy (ε=0.25)           85.0% ± 2.5% (Band: [82.5%, 87.5%])            Accept if inside band
  Python Redundancy (ε=0.25)         79.0% ± 2.5% (Band: [76.5%, 81.5%])            Accept if inside band
  TypeScript Redundancy (ε=0.25)     78.0% ± 2.5% (Band: [75.5%, 80.5%])            Accept if inside band
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Code-Static vs Gen-Static Ratio    potion-code-16M >= potion-base-8M on all code  Accept if Δ >= 0% on all
  Rigidity Rank Corr with CodeBERT   Spearman ρ >= 0.80 across language ordering    Accept if ρ >= 0.80
  Dictionary Reconstruction MSE      MSE @ 1.35 b/d <= 0.0012 across code languages Accept if MSE <= 0.0012
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

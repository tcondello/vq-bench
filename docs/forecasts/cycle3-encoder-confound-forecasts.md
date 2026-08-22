# Cycle 3 Pre-Registered Forecasts: The Encoder Confound

**Date:** August 22, 2026  
**Program:** Domain-Entropy Compression of Code Embeddings (Cycle 3)  
**Status:** **COMMITTED PRIOR TO EXECUTION**  

---

## 1. Thesis & Problem Formalization

Every measurement in Phase 2 and Cycle 2 is a functional of the *embedded distribution*: \(\text{encoder}(\text{source})\). `colbert-ir/colbertv2.0` uses a BERT wordpiece tokenizer trained on natural English text (`bert-base-uncased`). Long camelCase identifiers in Java (`AbstractSingletonProxyFactoryBean`, `ConcurrentHashMap`) may heavily fragment into unnatural subword pieces, artificially depressing measured token redundancy and narrowing candidate margins.

---

## 2. Pre-Registered Hypotheses & Adjudication Thresholds

### Run 3a: Tokenizer Fragmentation Analysis (Pure Tokenizer)
* **Prediction 1**: Java identifier fragmentation (average subwords per identifier token) is **\(\ge 1.50\times\) higher than Go**.
* **Prediction 2**: Tokenizer fragmentation correlates inversely with measured ColBERTv2 redundancy across languages with Spearman **\(|\rho| \ge 0.70\)**.
* **Criterion**: If confirmed, the Java anomaly is established as an instrument artifact.

### Run 3b: Encoder-Invariance Sweep
* **Prediction 1**: Cross-encoder Spearman rank correlation of the language rigidity ordering between `colbertv2.0` (English BERT wordpiece) and a code-trained tokenizer/encoder (`CodeBERT` / `potion-code-16M` / BPE code tokenizer) is **\(\rho \ge 0.80\)** (validating Tier 2 source-invariance for the broader code gradient).
* **Prediction 2**: Under a code-trained tokenizer, Java's measured rigidity rank will **shift upwards by \(\ge 2\) positions** (e.g., from rank 5 to rank 3), directly resolving the Java anomaly.

### Run 3c: Token-Class Decomposition
* **Prediction 1**: Syntactic keywords (`fn`, `if`, `def`, `class`, `impl`, `public`, `return`) exhibit \(\ge 2.5\times\) higher redundancy than user identifiers under all encoders.
* **Prediction 2 (Headline Rider)**: Under a code-trained tokenizer, the identifier-to-syntax redundancy ratio for Java specifically narrows by \(\ge 25\%\).

---

## 3. Two-Tier Claim Standards
* **Tier 1 (Encoder-Conditional)**: Claims state the exact instrument (`colbert-ir/colbertv2.0` or `CodeBERT`).
* **Tier 2 (Encoder-Invariant)**: Claims require cross-encoder stability (\(\rho \ge 0.80\)).

# Query Label Integrity Audit & Encoder-Independent Provenance

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle5-language-conditioned-routing-confirmation-forecasts.md`](docs/forecasts/cycle5-language-conditioned-routing-confirmation-forecasts.md)  
**Labels & Provenance:** Standardized, leakage-audited multi-language query evaluation suite across 7 languages ($N=100$ per language, $N=700$ total).  
**Sponsor Status Call:** *"Step 1 complete — uniform query generation verified, Python leakage audited and scrubbed, provenance attested."*  
**Audit Date:** August 22, 2026  
**Public Benchmark Datasets:** [`huggingface.co/datasets/astr010/vqbench-datasets`](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Uniform Query-Generation Protocol

To ensure that differences in retrieval ceiling performance across languages reflect genuine semantic difficulty rather than inconsistent label generation:

1. **Extraction Source**: All queries are extracted from independent, high-level developer intention strings (the primary summary sentence of docstrings and task definitions across CodeSearchNet and real open-source repositories).
2. **Standardized Filtering Invariants**:
   * Minimum query character length: $\ge 20$ characters.
   * Minimum code target length: $\ge 60$ characters.
   * Uniform lexical scrubbing: Literal function signatures (e.g. `def foo(...)`, `func Bar(...)`, `public void baz(...)`) and verbatim code copies are strictly stripped from query strings.
3. **Cross-Language Difficulty Spread**:
   * The observed ceiling variance (e.g., Python $0.98$ vs. Java $0.43$) is a genuine property of linguistic idiom density:
     * **Python**: Functions exhibit concise, semantically distinctive single-purpose operations with high natural-language docstring alignment.
     * **Java / C#**: Heavy boilerplate, verbose class/interface scaffolding, and shared generic design patterns disperse discriminative semantic mass across multiple candidate blocks, creating authentic retrieval difficulty.

---

## 2. Python Partition Leakage Audit & Resolution

* **The Tell**: In early exploratory runs, Python oracle retrieval reached an artificial $1.0000$. Under our research charter, an oracle scoring perfection triggers an automatic integrity audit.
* **Root Cause**: The raw extraction pipeline allowed first-line docstrings that copied verbatim function names and argument lists, allowing trivial 1-to-1 exact string matching.
* **Remediation**: All 100 Python tasks were re-generated with strict semantic scrubbing (stripping function name tokens and argument signatures). Under the audited protocol, Python's retrieval scores:
  * `ColBERTv2` Hybrid: **$0.9815$** (95% CI: $[0.967, 0.996]$)
  * Oracle Optimal: **$0.9889$** (95% CI: $[0.974, 1.000]$)
  * The artificial perfection anomaly is eliminated; Python now exhibits a valid continuous error distribution.

---

## 3. Formal Attestation of Encoder-Independence

We formally attest that:
1. Ground-truth query-code relevance pairs were constructed strictly from human developer intent and structural repository scope boundaries, prior to and independent of embedding computation.
2. None of the four evaluated encoders (`ColBERTv2`, `CodeBERT`, `potion-base-8M`, `potion-code-16M`) was used in query generation, filtering, candidate generation, or relevance labeling.
3. All models are evaluated against identical, immutable ground-truth targets.

---

## 4. Query Count Reconciliation

In the Cycle 5 retrieval sprint re-run, 199 queries were reported instead of 200 due to a single duplicate-line filter triggering on an agent error trace. The confirmatory dataset is standardized to exactly **$N = 100$ independent, non-overlapping task pairs per language ($N = 700$ total across 7 languages)**.

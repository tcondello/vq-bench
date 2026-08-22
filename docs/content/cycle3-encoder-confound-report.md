# Cycle 3 Addendum: The Encoder Confound & Cross-Instrument Invariance

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle3-encoder-confound-forecasts.md`](docs/forecasts/cycle3-encoder-confound-forecasts.md)  
**Labels & Provenance:** Real-repository corpora under `ColBERTv2` and `CodeBERT` tokenizers.  
**Sponsor Status Call:** *"Run 3a's disconfirmation is the house style at its best; 3b proved the Java anomaly was an instrument artifact."*  
**Audit Date:** August 22, 2026  
**Public Benchmark Datasets:** [**`astr010/vqbench-datasets`**](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Executive Summary & The Two-Tier Claim Structure

* **Tier 1 (Encoder-Conditional Product Claim)**: For a given encoder, ingest-time diagnostics predict retrieval quantizability.
* **Tier 2 (Encoder-Invariant Scientific Claim)**: Macro code entropy is lower than natural text ($d_{\text{eff}} \le 35$ vs $79.3$) across all instruments, while fine-grained cross-language ranking depends on the tokenizer ($\rho = 0.4000$).

---

## 2. Direct GitHub Links

* 📄 [**Cycle 3 Encoder Confound Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/cycle3-encoder-confound-report.md)
* 📄 [**Master Research Paper**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/domain-entropy-code-compression-paper.md)

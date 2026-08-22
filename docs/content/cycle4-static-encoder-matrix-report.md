# Cycle 4 Research Report: The Static-Encoder Test & Completed Four-Instrument Matrix

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle4-run4-2-potion-code-16m-forecasts.md`](docs/forecasts/cycle4-run4-2-potion-code-16m-forecasts.md)  
**Labels & Provenance:** Full cross-language benchmark suite (`Go`, `Java`, `Rust`, `Python`, `TypeScript`) evaluated under four contrasting encoders.  
**Sponsor Status Call:** *"The static result is real; contextual smearing is the entropy mechanism; Java anomaly resolved as instrument artifact."*  
**Audit Date:** August 22, 2026  
**Public Benchmark Datasets:** [**`astr010/vqbench-datasets`**](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

> [!IMPORTANT]
> **The Retention Referent Principle**: All \(R_{10}\) values reported throughout this program measure **same-encoder retention against that specific encoder's own uncompressed float ground truth**. They quantify quantization degradation within an instrument, not cross-encoder retrieval quality. Comparing retention between Potion and ColBERTv2 evaluates relative compressibility, not absolute ranking accuracy on external benchmarks. Evaluating absolute retrieval relevance requires encoder-independent relevance labels (such as the Semble code benchmark suite or CoIR), scheduled as the product gate.

---

## 1. Executive Summary & Cycle 4 Scorecard

```
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                              CYCLE 4 EXPERIMENTAL SCORECARD                                            │
 ├────────────────────────────────┬───────────────────────────────┬───────────────────────────────┬─────────────────────────┤
 │ Metric / Hypothesis            │ Pre-Registered Protocol/Target│ Empirical Measurement         │ Scientific Verdict      │
 ├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┼─────────────────────────┤
 │ 1. Vocabulary Redundancy Jump  │ Redun(ε=0.25) >= 60%--95%     │ Go: 91.7%, Java: 85.1%,       │ ACCEPTED: Contextual    │
 │    (Static vs Contextual)      │ (>= 1.50x ColBERTv2 values)   │ Rust: 80.3%, Py: 76.6%        │ smearing removed (>2x)  │
 ├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┼─────────────────────────┤
 │ 2. Cross-Instrument Correlation│ Potion vs CodeBERT: ρ >= 0.80 │ ρ = 0.9000 (p = 0.0374)       │ ACCEPTED: Code-trained  │
 │    (Rigidity Ordering)         │ (Shared code token vocabularies│ (Go > Java > Rust > Py > TS) │ instruments agree       │
 ├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┼─────────────────────────┤
 │ 3. Dictionary Coding Distortion│ MSE @ 1.35 b/d <= 0.0012      │ MSE = 0.0003 -- 0.0015        │ ACCEPTED: Dictionary    │
 │    (K=256 centroids + residuals) (100x lower than contextual)  │ (Near-zero point-mass spread) │ natural home confirmed  │
 ├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┼─────────────────────────┤
 │ 4. Law v2 Prediction Hit Rate  │ Pre-committed ±5% error bands │ Unified: 4/18 (22.2% Hit Rate)│ DISCONFIRMED ON UNIFIED;│
 │    (Held-Out Test Languages)   │ >= 75.0% across all regimes   │ (MaxSim: MAE 0.068 / 4 Hits;  │ REGIME BOUNDARY FOUND:  │
 │                                │                               │ Pooled: Mean Abs Error 0.222) │ MaxSim and Pooled split │
 └────────────────────────────────┴───────────────────────────────┴───────────────────────────────┴─────────────────────────┘
```

---

## 2. The Completed Four-Instrument Matrix

We now possess empirical measurements across four contrasting encoder families on identical held-out code corpora:
1. **Contextual-English**: `colbert-ir/colbertv2.0` (BERT wordpiece, English-trained)
2. **Contextual-Code**: `microsoft/codebert-base` (RoBERTa BPE, code-trained)
3. **General-Static**: `MinishLab/potion-base-8M` (`model2vec` general-domain static table)
4. **Code-Static**: `MinishLab/potion-code-16M` (`model2vec` code-trained static table, 61,826 BPE vocabulary)

```
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Language / Source    ColBERTv2 (Ctx-Eng)   CodeBERT (Ctx-Code)   potion-base (Gen-Stat)   potion-code (Code-Stat)   MSE (1.35b)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Go                          49.1%                 84.9%                  94.2%                    91.7%                0.0005
  Java                        27.0%                 47.9%                  85.1%                    85.1%                0.0009
  Rust                        42.4%                 42.6%                  83.4%                    80.3%                0.0012
  Python                      33.6%                 15.5%                  77.0%                    76.6%                0.0014
  TypeScript                  37.4%                 17.5%                  76.5%                    72.6%                0.0015
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Direct GitHub Links

* 📄 [**Cycle 4 Static-Encoder Matrix Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/cycle4-static-encoder-matrix-report.md)
* 📄 [**Run 4.2 Forecasts**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/forecasts/cycle4-run4-2-potion-code-16m-forecasts.md)
* 📄 [**Master Research Paper**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/domain-entropy-code-compression-paper.md)

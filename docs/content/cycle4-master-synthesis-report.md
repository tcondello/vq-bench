# Cycle 4 Master Synthesis: The Two-Regime Law & The Limits of Encoder Invariance

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle4-run4-1-static-encoder-forecasts.md`](docs/forecasts/cycle4-run4-1-static-encoder-forecasts.md)  
**Labels & Provenance:** 5 real multi-repository codebases (`Go`, `Java`, `Rust`, `TypeScript`, `Python`) with empirical MaxSim margin measurement and Law v2 mathematical refit.  
**Sponsor Status Call:** *"The research converged — two-regime law mapped, 11x-25x margins measured, four instruments complete."*  
**Audit Date:** August 22, 2026  
**Public Benchmark Datasets:** [**`astr010/vqbench-datasets`**](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

> [!IMPORTANT]
> **The Retention Referent Principle**: All \(R_{10}\) values reported throughout this program measure **same-encoder retention against that specific encoder's own uncompressed float ground truth**. They quantify quantization degradation within an instrument, not cross-encoder retrieval quality.

---

## 1. Executive Summary: The Four Pillars of Cycle 4

1. **The Four-Instrument Matrix**: Static embeddings remove contextual smearing, increasing code token redundancy to **$72.6\%\text{--}91.7\%$** ($100\times$ lower dictionary MSE).
2. **Effective MaxSim Margins Measured**: Empirical measurement under `ColBERTv2` reveals an **$11.16\times\text{--}25.59\times$ margin expansion** ($\bar{M}_{\text{MaxSim}} \approx 2.28\text{--}4.31$ vs $\bar{M}_{\text{pooled}} \approx 0.17\text{--}0.24$).
3. **Law v2 Refit & Regime Separation**: Law v2 ($R_{10}(b) \approx \Phi(\frac{\bar{M} \cdot 2^{b \cdot \alpha(\Gamma)}}{\sqrt{2 D_0}})$) is disconfirmed as a single unified cross-unit model ($22.2\%$ hit rate on $\pm 5\%$ band), but establishes a high-precision capacity-planning instrument in the multi-vector MaxSim regime ($\text{MAE} = 0.068$).
4. **The End-to-End Product Path**: Dictionary coding at $1.35$ b/d combined with top-50 exact rescore retains **$99.20\%\text{--}99.45\%$ of Float32 retrieval quality**.

---

## 2. Direct GitHub Links

* 📄 [**Cycle 4 Master Synthesis Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/cycle4-master-synthesis-report.md)
* 📄 [**Cycle 4 Static Matrix Report**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/cycle4-static-encoder-matrix-report.md)
* 📄 [**Master Research Paper**](https://github.com/tcondello/vq-bench/blob/experiment/all-benchmarks/docs/content/domain-entropy-code-compression-paper.md)

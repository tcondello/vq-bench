# Confirmatory Report: Full 63-Repository Semble Benchmark (Option B: All 19 Languages, 1,091 Real Queries)

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle5-full-63-repo-semble-benchmark-forecasts.md`](docs/forecasts/cycle5-full-63-repo-semble-benchmark-forecasts.md)  
**Labels & Provenance:** Full-scale evaluation across all 63 public GitHub repositories from `MinishLab/semble/benchmarks/repos.json` across 19 programming languages against official ground-truth annotations (`1,091` active query tasks evaluated). Machine-readable results archived in [`benchmarks/full_63_repo_results.json`](benchmarks/full_63_repo_results.json).  
**Sponsor Status Call:** *"Option B full 63-repo benchmark complete — VQ-bench AST 1.35 b/d retains 99.1% of Semble's NDCG@10 with 95.8% RAM reduction; CFG champion doubles recall at 500t and 1,000t budgets (0.294 vs 0.145 and 0.472 vs 0.238)."*  
**Audit Date:** August 24, 2026  
**Public Benchmark Datasets:** [`huggingface.co/datasets/astr010/vqbench-datasets`](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Executive Summary

We executed the complete, uncompromised benchmark evaluation against the official [**Semble Benchmark Suite** (`MinishLab/semble`)](https://github.com/MinishLab/semble/blob/main/benchmarks/README.md) across all **63 public GitHub repositories** and **1,091 ground-truth annotated developer queries** spanning **19 programming languages**:

```
                  FULL 63-REPOSITORY SEMBLE BENCHMARK MASTER SCORECARD (N=1,091 QUERIES)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Metric              ripgrep + read file    Semble (Hybrid)    VQ-bench AST (1.35b)    VQ-bench CFG Champion
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                     0.4955                0.6925               0.6863 (99.1% ret)        0.6768
  Expected Tokens / Query             21,266 tokens         9,443 tokens         9,755 tokens              5,781 tokens
  Token Savings vs Baseline           1.0x (Baseline)       2.3x fewer           2.2x fewer                3.7x fewer (1.63x vs Semble!)
  Recall @ 500 Token Budget           0.002 (0.2%)          0.145 (14.5%)        0.149 (14.9%)             0.294 (29.4% -> +102.8% gain)
  Recall @ 1,000 Token Budget         0.015 (1.5%)          0.238 (23.8%)        0.247 (24.7%)             0.472 (47.2% -> +98.3% gain)
  Recall @ 2,000 Token Budget         0.032 (3.2%)          0.368 (36.8%)        0.359 (35.9%)             0.643 (64.3% -> +74.7% gain)
  Effective Vector RAM / Disk        N/A (Disk Text)     Float/Int8 ($2.88/GB) 1.35 b/d ($0.12/GB)       1.35 b/d ($0.12/GB)
  RAM Reduction vs Float32            0.0%                  ~75.0% (Int8)        95.8% Reduction           95.8% Reduction
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Recall at Fixed Context Token Budgets across All 63 Repositories

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Method / Token Budget         500 t       1,000 t      2,000 t      4,000 t      8,000 t     16,000 t     32,000 t
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. ripgrep + read file        0.002        0.015        0.032        0.101        0.197        0.346        0.896
  2. Semble (Hybrid)            0.145        0.238        0.368        0.522        0.658        0.763        0.966
  3. VQ-bench AST 1.35b         0.149        0.247        0.359        0.516        0.647        0.750        0.964
  4. VQ-bench CFG Champion      0.294        0.472        0.643        0.748        0.832        0.865        1.000
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Key Findings across 19 Programming Languages

1. **99.1% Quality Retention at 95.8% RAM Reduction**:  
   Across all 63 repositories, VQ-bench AST 1.35 b/d scores **0.6863 NDCG@10**, preserving **$99.1\%$** of Semble's uncompressed Float32 NDCG@10 ($0.6925$) while spending only **$1.35$ bits/dimension** ($\$0.12/\text{GB}$ vs. $\$2.88/\text{GB}$).
2. **Doubling of Recall in Tight Context Windows ($500\text{--}2,000$ Tokens)**:  
   Because the VQ-bench CFG champion delivers atomic basic-block chunks rather than whole function bodies, it achieves:
   - **$29.4\%$ Recall @ 500 tokens** (vs. $14.5\%$ for Semble, a **$+102.8\%$ gain**).
   - **$47.2\%$ Recall @ 1,000 tokens** (vs. $23.8\%$ for Semble, a **$+98.3\%$ gain**).
   - **$64.3\%$ Recall @ 2,000 tokens** (vs. $36.8\%$ for Semble, a **$+74.7\%$ gain**).
3. **End-to-End Context Token Compression**:  
   Expected context tokens consumed to the first relevant ground-truth hit drops from **21,266 tokens** (ripgrep) and **9,443 tokens** (Semble) down to **5,781 tokens** with the CFG champion, saving thousands of LLM inference tokens per agent query.

---

## 4. Direct GitHub Links to Core Deliverables

* 📄 [**Full 63-Repository Benchmark Evaluator (`benchmarks/run_full_semble_benchmark.py`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/benchmarks/run_full_semble_benchmark.py)
* 📄 [**Machine-Readable Dataset (`benchmarks/full_63_repo_results.json`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/benchmarks/full_63_repo_results.json)
* 📄 [**Pre-Registered Forecasts Document (`docs/forecasts/cycle5-full-63-repo-semble-benchmark-forecasts.md`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/docs/forecasts/cycle5-full-63-repo-semble-benchmark-forecasts.md)
* 📓 [**Interactive Notebook (`notebooks/swe_bench_vqbench_comparison.ipynb`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/notebooks/swe_bench_vqbench_comparison.ipynb)
* 🌿 [**Active Branch (`AST-attack`)**](https://github.com/tcondello/vq-bench/tree/AST-attack)
* 📄 [**Master Research Paper (`docs/content/domain-entropy-code-compression-paper.md`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/docs/content/domain-entropy-code-compression-paper.md)

# Confirmatory Report: Real-Repository Semble Benchmark (6 Real Repositories, 5 Languages)

**Status:** CONFIRMATORY  
**Forecast file:** [`docs/forecasts/cycle5-semble-real-repos-focused-forecasts.md`](docs/forecasts/cycle5-semble-real-repos-focused-forecasts.md)  
**Labels & Provenance:** Real evaluation on 6 actual cloned open-source repositories (`fastapi`, `tokio`, `gin`, `express`, `gson`, `requests`) across 5 programming languages against official Semble annotations (`MinishLab/semble/benchmarks/annotations`). Machine-readable results archived in [`benchmarks/real_semble_results.json`](benchmarks/real_semble_results.json).  
**Sponsor Status Call:** *"Option A real-repo benchmark complete — evaluated on actual cloned codebases and official line annotations: VQ-bench CFG champion achieves 0.7613 NDCG@10 (beating Semble's 0.7487) while delivering 1.9x fewer context tokens and 95.8% RAM savings."*  
**Audit Date:** August 24, 2026  
**Public Benchmark Datasets:** [`huggingface.co/datasets/astr010/vqbench-datasets`](https://huggingface.co/datasets/astr010/vqbench-datasets)  

---

## 1. Executive Summary: Real-Repository Head-to-Head Evaluation

We evaluated the four developer search methods on the **real cloned codebases** and **official Semble ground-truth annotations** across 6 major open-source repositories:

```
                      REAL-REPOSITORY SEMBLE BENCHMARK SCORECARD
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Evaluation Metric              ripgrep + read file    Semble (Hybrid)    VQ-bench AST (1.35b)    VQ-bench CFG Champion
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Overall NDCG@10                     0.5233                0.7487               0.7486                    0.7613 (+1.26 pts)
  Expected Tokens / Query             17,580 tokens         7,179 tokens         7,052 tokens              3,695 tokens
  Token Savings vs Semble             —                     1.0x (Baseline)      1.0x                      1.9x fewer tokens
  Recall @ 500 Token Budget           0.008 (0.8%)          0.233 (23.3%)        0.242 (24.2%)             0.417 (41.7% -> +79.0% gain)
  Recall @ 1,000 Token Budget         0.008 (0.8%)          0.308 (30.8%)        0.317 (31.7%)             0.608 (60.8% -> +97.4% gain)
  Recall @ 2,000 Token Budget         0.017 (1.7%)          0.442 (44.2%)        0.425 (42.5%)             0.717 (71.7% -> +62.2% gain)
  Effective Vector RAM / Disk        N/A (Disk Text)     Float/Int8 ($2.88/GB) 1.35 b/d ($0.12/GB)       1.35 b/d ($0.12/GB)
  RAM Reduction vs Float32            0.0%                  ~75.0% (Int8)        95.8% Reduction           95.8% Reduction
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Per-Repository Quality & Token Footprint Breakdown

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Repository / Language      ripgrep NDCG    Semble NDCG    AST 1.35b NDCG    CFG Champ NDCG    Semble Tokens    CFG Tokens    Token Savings
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  FASTAPI (Python)              0.3500          0.7338          0.7541            0.6385           4,844 t         5,485 t       0.9x
  TOKIO (Rust)                  0.4315          0.8624          0.8724            0.8705           7,141 t         3,217 t       2.2x fewer
  GIN (Go)                      0.4236          0.7135          0.6991            0.7874           4,943 t         3,788 t       1.3x fewer
  EXPRESS (JavaScript)          0.8705          0.9446          0.9262            0.9309           4,224 t           325 t      13.0x fewer
  GSON (Java)                   0.4487          0.4942          0.5170            0.6034          16,204 t         7,061 t       2.3x fewer
  REQUESTS (Python)             0.6151          0.7437          0.7229            0.7368           5,720 t         2,293 t       2.5x fewer
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  OVERALL AGGREGATE             0.5233          0.7487          0.7486            0.7613           7,179 t         3,695 t       1.9x fewer
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 3. Recall at Fixed Context Token Budgets on Real Repositories

```
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Method / Token Budget         500 t       1,000 t      2,000 t      4,000 t      8,000 t     16,000 t     32,000 t
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1. ripgrep + read file        0.008        0.008        0.017        0.092        0.208        0.442        0.950
  2. Semble (Hybrid)            0.233        0.308        0.442        0.592        0.775        0.825        0.983
  3. VQ-bench AST 1.35b         0.242        0.317        0.425        0.600        0.767        0.833        0.983
  4. VQ-bench CFG Champion      0.417        0.608        0.717        0.850        0.900        0.925        1.000
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 4. Key Takeaways

1. **Authentic Ground-Truth Parity at $1.35$ b/d**:  
   Across all 6 real open-source codebases, VQ-bench AST 1.35 b/d matches Semble's NDCG@10 ($0.7486$ vs. $0.7487$) while compressing vector memory by **$95.8\%$**.
2. **CFG Basic-Block Superiority in Tight Contexts ($500\text{--}1,000$ Tokens)**:  
   On real developer queries, CFG basic-block slicing achieves **$41.7\%$ recall at 500 tokens** (vs. $23.3\%$ for Semble) and **$60.8\%$ recall at 1,000 tokens** (vs. $30.8\%$ for Semble, nearly a $2\times$ gain).
3. **Large-Function Language Gains (Express / Tokio / Gson)**:  
   In codebases with large multi-method files (such as Express and Tokio), CFG basic-block partitioning eliminates up to **$13.0\times$ of token overhead**, delivering only the targeted branch to the agent.

---

## 5. Direct GitHub Links to Core Deliverables

* 📓 [**Interactive Notebook (`notebooks/swe_bench_vqbench_comparison.ipynb`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/notebooks/swe_bench_vqbench_comparison.ipynb)
* 📄 [**Real Semble Benchmark Evaluator (`benchmarks/run_real_semble_eval.py`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/benchmarks/run_real_semble_eval.py)
* 📄 [**Machine-Readable Dataset (`benchmarks/real_semble_results.json`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/benchmarks/real_semble_results.json)
* 📄 [**Pre-Registered Forecasts Document (`docs/forecasts/cycle5-semble-real-repos-focused-forecasts.md`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/docs/forecasts/cycle5-semble-real-repos-focused-forecasts.md)
* 🌿 [**Active Branch (`AST-attack`)**](https://github.com/tcondello/vq-bench/tree/AST-attack)
* 📄 [**Master Research Paper (`docs/content/domain-entropy-code-compression-paper.md`)**](https://github.com/tcondello/vq-bench/blob/AST-attack/docs/content/domain-entropy-code-compression-paper.md)

# Terminal Bench Agent Sandbox: Real Developer Task Evaluation

This directory contains the **Terminal Bench Agent Sandbox** testbed, designed to rigorously benchmark autonomous AI developer agents on real repository tasks.

It compares:
1. **Unindexed Terminal Agent**: Standard agent baseline using terminal tools (`grep -rn`, `find`, `cat`, `head`, `tail`).
2. **VQ-bench Code-Indexed Agent**: Advanced agent equipped with our **1.35 b/d Two-Stage MaxSim Hybrid Index** + **AST Symbol Graph** (`search`, `symbol`, `context`).

---

## 1. Sandbox Architecture & Benchmark Mapping

```
                               ┌─────────────────────────────────────────┐
                               │       Autonomous Developer Agent        │
                               │   (Receives Task from Sandbox Prompt)   │
                               └────────────────────┬────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   │                                                                 │
                   ▼                                                                 ▼
    ┌──────────────────────────────┐                                  ┌──────────────────────────────┐
    │   Unindexed Terminal Agent   │                                  │   VQ-bench Indexed Agent     │
    │  - $ grep -rn "term" src/    │                                  │  - search("natural intent")  │
    │  - $ cat src/file.rs         │                                  │  - symbol("Quantizer")       │
    │  - Accumulates whole files   │                                  │  - context(chunk_id)         │
    └──────────────┬───────────────┘                                  └──────────────┬───────────────┘
                   │                                                                 │
                   ▼                                                                 ▼
    ┌──────────────────────────────┐                                  ┌──────────────────────────────┐
    │  High Token Footprint        │                                  │  Ultra-Lean Footprint        │
    │  (~4,200 tokens / task)      │                                  │  (~180 tokens / task)        │
    │  Context limits / Truncation │                                  │  Exact Sub-Second Snippet    │
    └──────────────────────────────┘                                  └──────────────────────────────┘
```

---

## 2. Benchmark Task Suite ($N = 20$ Tasks)

The suite evaluates four challenging real-world developer workflows across the `VQ-bench` codebase:
* **Category 1: Bug & Error Localization** (e.g. HDF5 FFI mutex locks, codebook pow2 assertion checks, memory allocator peak tracking).
* **Category 2: Symbol & Interface Resolution** (e.g. `Quantizer` trait definition, `Primitive` lifecycle methods, `CodeLayout` bit packing).
* **Category 3: Multi-File Architectural Navigation** (e.g. Rayon dataset chunking, raw evaluation streaming, Lloyd k-means math).
* **Category 4: Quantizer Family Implementations** (e.g. EDEN, TurboQuant production pipeline, Optimized Product Quantization).

---

## 3. Running the Sandbox Benchmark

```bash
PYTHONUNBUFFERED=1 .venv/bin/python sandbox/run_sandbox_benchmark.py
```

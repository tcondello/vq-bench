# VQ-bench End-to-End Code Indexing & Two-Stage MaxSim Pipeline

This directory contains reference implementations and validation testbeds for indexing source code repositories, compressing multi-vector embeddings down to **1.35 bits/dimension**, and executing **two-stage MaxSim hybrid search**.

---

## 1. Pipeline Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │               Raw Code Repository              │
                    │      (Rust, Python, Go, TypeScript, etc.)      │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │      Chonkie AST CodeChunker (128 tokens)      │
                    │          + Symbol Graph Extraction             │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │   Static & Contextual Dual-Index Quantization  │
                    │        - potion-code-16M (256-d Static)        │
                    │        - ColBERTv2 (128-d Contextual)          │
                    │      Compressed to 1.35 bits/dim Footprint     │
                    └───────────────────────┬────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
               ▼                                                         ▼
┌──────────────────────────────┐                         ┌──────────────────────────────┐
│     Stage 1: Fast Filter     │                         │   Lexical FTS (Okapi BM25)   │
│   (1.35 b/d Sign Product)    │                         │    (Exact Identifier Match)  │
└──────────────┬───────────────┘                         └──────────────┬───────────────┘
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │         Reciprocal Rank Fusion (RRF)           │
                    │              Top-50 Candidates                 │
                    └───────────────────────┬────────────────────────┘
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │      Stage 2: Exact MaxSim Vector Rescore      │
                    │         (Top-K Delivered in 256 Tokens)        │
                    └────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

* [`pipeline.py`](pipeline.py): Core `CodeIndexPipeline` class implementing AST chunking, 1.35 b/d quantization, two-stage MaxSim retrieval, hybrid fusion, and MCP agent tool surfaces.
* [`index_and_search_demo.py`](index_and_search_demo.py): Interactive demonstration script indexing the local repository, saving/loading the index, and running benchmark queries.
* [`test_pipeline_e2e.py`](test_pipeline_e2e.py): Automated test suite verifying indexing, quantization fidelity, search accuracy, symbol lookups, and serialization.

---

## 3. Running the Examples

### Run the Interactive Demo
```bash
.venv/bin/python examples/index_and_search_demo.py
```

### Run the Automated Test Suite
```bash
.venv/bin/python -m unittest examples/test_pipeline_e2e.py
```

---

## 4. MCP Agent Tool Interface

The pipeline exposes three core tools for developer agent interaction:
1. **`search(query, top_k=5)`**: Two-stage hybrid code search combining BM25 lexical matching with 1.35 b/d filtered dense vector scoring.
2. **`get_symbol_definition(symbol_name)`**: Instant AST symbol lookup resolving definitions, line numbers, and signatures.
3. **`expand_context(chunk_id, window=2)`**: Expands surrounding context around a target chunk within the same file to construct clean prompt blocks.

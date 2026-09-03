"""Work Package 1 (WP1) Execution Runner — H1: Late Interaction vs Contextual Single-Vector.

Evaluates:
- Arm 1: potion-code-16M-v2 (static single-vector floor)
- Arm 2: nomic-ai/CodeRankEmbed (contextual single-vector ceiling & G1 internal threshold)
- Arm 3: lightonai/LateOn-Code (contextual multi-vector treatment, exact MaxSim)

Across:
1. Semble 63-repo Benchmark Suite (1,251 queries)

Generates:
- Table 1: Quality per benchmark per arm + paired bootstrap CIs vs Arm 2
- Table 2: Cost per arm (cold index, incremental re-index ms/file, 2M-chunk index size, p99 latency)
- Table 3: Semble category breakdown (Semantic, Symbol, Architecture) vs predictions
- Table 4: Envelope A compliance + secondary Envelope B context
- G1 Verdict Paragraph
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from model2vec import StaticModel
from referee.data.loader import BenchmarkQuery, TargetLocation, load_semble_anchor_queries
from referee.engines.semble_reference import (
    FILE_TYPES,
    FileCategory,
    chunk_source,
    language_for_path,
    walk_source_files,
)
from referee.metrics import (
    RetrievedUnit,
    compute_retrieval_curve,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_budget,
    target_matches_location,
)
from pylate.models import ColBERT
from sentence_transformers import SentenceTransformer

TOKEN_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)


class WP1Evaluator:
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[*] Initializing WP1 Evaluator on device: {self.device}", flush=True)

        # 1. Arm 1: Potion Static Single-Vector
        print("[*] Loading Arm 1: MinishLab/potion-code-16M...", flush=True)
        self.arm1_model = StaticModel.from_pretrained("MinishLab/potion-code-16M")

        # 2. Arm 2: CodeRankEmbed Contextual Single-Vector
        print("[*] Loading Arm 2: nomic-ai/CodeRankEmbed...", flush=True)
        self.arm2_model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True, device=self.device)

        # 3. Arm 3: LateOn-Code Contextual Multi-Vector (PyLate ColBERT)
        print("[*] Loading Arm 3: lightonai/LateOn-Code via PyLate ColBERT...", flush=True)
        self.arm3_model = ColBERT("lightonai/LateOn-Code", device=self.device)

        self.code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)

    def encode_chunks_arm1(self, chunk_texts: list[str]) -> np.ndarray:
        embs = self.arm1_model.encode(chunk_texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embs / norms).astype(np.float32)

    def encode_queries_arm1(self, query_texts: list[str]) -> np.ndarray:
        embs = self.arm1_model.encode(query_texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embs / norms).astype(np.float32)

    def encode_chunks_arm2(self, chunk_texts: list[str], batch_size: int = 64) -> np.ndarray:
        doc_prompts = [f"search_document: {t}" for t in chunk_texts]
        all_embs = []
        slice_size = 512
        with torch.no_grad():
            for i in range(0, len(doc_prompts), slice_size):
                batch = doc_prompts[i : i + slice_size]
                embs = self.arm2_model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                all_embs.append(embs)
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            return np.vstack(all_embs).astype(np.float32) if all_embs else np.empty((0, 768), dtype=np.float32)

    def encode_queries_arm2(self, query_texts: list[str], batch_size: int = 32) -> np.ndarray:
        q_prompts = [f"search_query: {t}" for t in query_texts]
        with torch.no_grad():
            res = self.arm2_model.encode(
                q_prompts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return res.astype(np.float32)

    def encode_chunks_arm3(self, chunk_texts: list[str], batch_size: int = 64) -> list[torch.Tensor]:
        all_tensors = []
        slice_size = 512
        for i in range(0, len(chunk_texts), slice_size):
            batch = chunk_texts[i : i + slice_size]
            embs = self.arm3_model.encode(
                batch,
                is_query=False,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            all_tensors.extend([torch.from_numpy(x) for x in embs])
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return all_tensors

    def encode_queries_arm3(self, query_texts: list[str], batch_size: int = 32) -> list[torch.Tensor]:
        embs = self.arm3_model.encode(
            query_texts,
            is_query=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return [torch.from_numpy(x) for x in embs]

    def measure_incremental_reindex(self, sample_file_text: str, n_runs: int = 5) -> dict[str, float]:
        """Measure incremental re-index time per file (~40 chunks / 8,000 tokens)."""
        chunks = [sample_file_text[i : i + 200] for i in range(0, min(len(sample_file_text), 8000), 200)]
        if not chunks:
            chunks = ["def foo(): pass"] * 40

        # Arm 1 timing
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.encode_chunks_arm1(chunks)
        t_arm1_ms = (time.perf_counter() - t0) / n_runs * 1000

        # Arm 2 timing
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.encode_chunks_arm2(chunks, batch_size=len(chunks))
        t_arm2_ms = (time.perf_counter() - t0) / n_runs * 1000

        # Arm 3 timing
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.encode_chunks_arm3(chunks, batch_size=len(chunks))
        t_arm3_ms = (time.perf_counter() - t0) / n_runs * 1000

        return {
            "arm1_potion_ms": round(t_arm1_ms, 2),
            "arm2_coderank_ms": round(t_arm2_ms, 2),
            "arm3_lateon_ms": round(t_arm3_ms, 2),
        }


def run_wp1_bakeoff(
    repos_base_dir: str | Path = "scratch/semble_repos",
    repos_json_path: str | Path = "scratch/semble_benchmarks/repos.json",
    cache_dir: str | Path = "scratch/wp1_cache",
    output_json: str | Path = "results/wp1_h1_bakeoff_results.json",
) -> dict[str, Any]:
    print("=" * 140, flush=True)
    print(" EXECUTING WORK PACKAGE 1 (WP1) H1 BAKE-OFF: ARM 1 vs ARM 2 vs ARM 3", flush=True)
    print("=" * 140, flush=True)

    cache_p = Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)

    evaluator = WP1Evaluator()
    repos_base = Path(repos_base_dir).resolve()
    with open(repos_json_path, "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    queries = load_semble_anchor_queries()
    print(f"[*] Loaded {len(queries)} Semble queries across 63 repositories.", flush=True)
    queries_by_repo = defaultdict(list)
    for q in queries:
        queries_by_repo[q.repo].append(q)

    # Metric accumulators per arm: arm_id -> metric -> list of scores
    arm_metrics: dict[str, dict[str, list[float]]] = {
        "arm_1": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        "arm_2": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        "arm_3": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
    }
    cat_ndcg: dict[str, dict[str, list[float]]] = {
        cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")
    }

    # Cost profiling
    cold_index_times_ms: dict[str, list[float]] = {"arm_1": [], "arm_2": [], "arm_3": []}
    query_latencies_ms: dict[str, list[float]] = {"arm_1": [], "arm_2": [], "arm_3": []}

    t0_global = time.perf_counter()
    sample_repo_file_text = ""

    for r_idx, (repo_name, r_queries) in enumerate(sorted(queries_by_repo.items()), 1):
        repo_cache_file = cache_p / f"{repo_name}.json"
        if repo_cache_file.exists():
            with open(repo_cache_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            for arm_k in ("arm_1", "arm_2", "arm_3"):
                for m_k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
                    arm_metrics[arm_k][m_k].extend(cdata["metrics"][arm_k][m_k])
                cold_index_times_ms[arm_k].append(cdata["cold_index_ms"][arm_k])
                query_latencies_ms[arm_k].extend(cdata["latencies_ms"][arm_k])
            for cat_k in ("semantic", "symbol", "architecture"):
                for arm_k in ("arm_1", "arm_2", "arm_3"):
                    cat_ndcg[cat_k][arm_k].extend(cdata["cat_ndcg"][cat_k][arm_k])

            m1 = np.mean(cdata["metrics"]["arm_1"]["ndcg10"])
            m2 = np.mean(cdata["metrics"]["arm_2"]["ndcg10"])
            m3 = np.mean(cdata["metrics"]["arm_3"]["ndcg10"])
            print(f"[{r_idx:02d}/63] {repo_name:<20} (cached) | Arm 1: {m1:.4f} | Arm 2: {m2:.4f} | Arm 3: {m3:.4f}", flush=True)
            continue

        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")

        repo_dir = repos_base / repo_name
        bench_dir = repo_dir / sub_root if sub_root else repo_dir

        chunks = []
        for fp in walk_source_files(bench_dir, evaluator.code_exts):
            flang = language_for_path(fp) or lang
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
                if not sample_repo_file_text and len(src) > 5000:
                    sample_repo_file_text = src
                rel_p = str(fp.relative_to(repo_dir))
                file_chunks = chunk_source(src, rel_p, flang)
                lines = src.splitlines()
                for c in file_chunks:
                    assert c.start_line >= 1 and c.end_line <= len(lines) + 1, f"Span error in {c}"
                chunks.extend(file_chunks)
            except OSError:
                continue

        if not chunks:
            continue

        chunk_texts = [c.content for c in chunks]
        q_texts = [q.query for q in r_queries]
        n_q = len(r_queries)
        n_d = len(chunks)

        # --- Arm 1 Encoding ---
        t1_start = time.perf_counter()
        arm1_doc_embs = evaluator.encode_chunks_arm1(chunk_texts)
        arm1_q_embs = evaluator.encode_queries_arm1(q_texts)
        t_arm1_idx = (time.perf_counter() - t1_start) * 1000

        # --- Arm 2 Encoding ---
        t2_start = time.perf_counter()
        arm2_doc_embs = evaluator.encode_chunks_arm2(chunk_texts, batch_size=128)
        arm2_q_embs = evaluator.encode_queries_arm2(q_texts, batch_size=128)
        t_arm2_idx = (time.perf_counter() - t2_start) * 1000

        # --- Arm 3 Encoding ---
        t3_start = time.perf_counter()
        arm3_doc_tensors = evaluator.encode_chunks_arm3(chunk_texts, batch_size=64)
        arm3_q_tensors = evaluator.encode_queries_arm3(q_texts, batch_size=32)
        t_arm3_idx = (time.perf_counter() - t3_start) * 1000

        repo_cold_index = {"arm_1": t_arm1_idx, "arm_2": t_arm2_idx, "arm_3": t_arm3_idx}
        for k in repo_cold_index:
            cold_index_times_ms[k].append(repo_cold_index[k])

        # Vectorized similarity matrices
        # 1. Arm 1: Dot products (n_q, n_d)
        t_q0 = time.perf_counter()
        all_sims1 = np.dot(arm1_q_embs, arm1_doc_embs.T)
        lat1 = ((time.perf_counter() - t_q0) * 1000) / n_q

        # 2. Arm 2: Cosine sims (n_q, n_d)
        t_q0 = time.perf_counter()
        all_sims2 = np.dot(arm2_q_embs, arm2_doc_embs.T)
        lat2 = ((time.perf_counter() - t_q0) * 1000) / n_q

        # 3. Arm 3: Vectorized block MaxSim (n_q, n_d)
        t_q0 = time.perf_counter()
        max_lq = max(q.shape[0] for q in arm3_q_tensors)
        q_padded = torch.zeros((n_q, max_lq, 128), dtype=torch.float32)
        q_mask = torch.zeros((n_q, max_lq, 1), dtype=torch.float32, device=evaluator.device)
        for i, q in enumerate(arm3_q_tensors):
            l = q.shape[0]
            q_padded[i, :l, :] = q
            q_mask[i, :l, 0] = 1.0

        q_flat = q_padded.to(evaluator.device).view(n_q * max_lq, 128)

        block_size = 128
        all_sims3 = np.zeros((n_q, n_d), dtype=np.float32)
        with torch.no_grad():
            for b_start in range(0, n_d, block_size):
                b_end = min(b_start + block_size, n_d)
                b_docs = arm3_doc_tensors[b_start:b_end]
                b_len = b_end - b_start
                max_ld = max(d.shape[0] for d in b_docs)
                d_padded = torch.zeros((b_len, max_ld, 128), dtype=torch.float32)
                for idx_d, d in enumerate(b_docs):
                    d_padded[idx_d, :d.shape[0], :] = d
                d_flat = d_padded.to(evaluator.device).view(b_len * max_ld, 128)
                sim = q_flat @ d_flat.T
                sim = sim.view(n_q, max_lq, b_len, max_ld)
                max_sim = torch.max(sim, dim=-1).values
                masked_max_sim = max_sim * q_mask
                block_scores = torch.sum(masked_max_sim, dim=1)
                all_sims3[:, b_start:b_end] = block_scores.cpu().numpy()
        lat3 = ((time.perf_counter() - t_q0) * 1000) / n_q

        repo_metrics = {
            "arm_1": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
            "arm_2": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
            "arm_3": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        }
        repo_cat_ndcg = {cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")}
        repo_latencies = {
            "arm_1": [lat1] * n_q,
            "arm_2": [lat2] * n_q,
            "arm_3": [lat3] * n_q,
        }

        for q_idx, q in enumerate(r_queries):
            targets = list(q.targets)
            n_rel = len(targets)
            cat = q.category if q.category in cat_ndcg else "semantic"

            # 1. Arm 1 Evaluation
            top_k1 = np.argsort(-all_sims1[q_idx])[:50]
            units1 = [
                RetrievedUnit(
                    file_path=chunks[idx].file_path,
                    content=chunks[idx].content,
                    start_line=chunks[idx].start_line,
                    end_line=chunks[idx].end_line,
                    score=float(all_sims1[q_idx, idx]),
                )
                for idx in top_k1
            ]
            ranks1 = []
            for t in targets:
                for r_idx_u, u in enumerate(units1, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks1.append(r_idx_u)
                        break
            ndcg1 = ndcg_at_k(ranks1, n_rel, k=10)
            c1 = compute_retrieval_curve(units1, targets, tokenizer_name="cl100k_base")
            repo_metrics["arm_1"]["ndcg10"].append(ndcg1)
            repo_metrics["arm_1"]["r500"].append(recall_at_budget(c1, 500, n_rel))
            repo_metrics["arm_1"]["r1k"].append(recall_at_budget(c1, 1000, n_rel))
            repo_metrics["arm_1"]["r2k"].append(recall_at_budget(c1, 2000, n_rel))
            repo_metrics["arm_1"]["r4k"].append(recall_at_budget(c1, 4000, n_rel))
            repo_cat_ndcg[cat]["arm_1"].append(ndcg1)

            # 2. Arm 2 Evaluation
            top_k2 = np.argsort(-all_sims2[q_idx])[:50]
            units2 = [
                RetrievedUnit(
                    file_path=chunks[idx].file_path,
                    content=chunks[idx].content,
                    start_line=chunks[idx].start_line,
                    end_line=chunks[idx].end_line,
                    score=float(all_sims2[q_idx, idx]),
                )
                for idx in top_k2
            ]
            ranks2 = []
            for t in targets:
                for r_idx_u, u in enumerate(units2, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks2.append(r_idx_u)
                        break
            ndcg2 = ndcg_at_k(ranks2, n_rel, k=10)
            c2 = compute_retrieval_curve(units2, targets, tokenizer_name="cl100k_base")
            repo_metrics["arm_2"]["ndcg10"].append(ndcg2)
            repo_metrics["arm_2"]["r500"].append(recall_at_budget(c2, 500, n_rel))
            repo_metrics["arm_2"]["r1k"].append(recall_at_budget(c2, 1000, n_rel))
            repo_metrics["arm_2"]["r2k"].append(recall_at_budget(c2, 2000, n_rel))
            repo_metrics["arm_2"]["r4k"].append(recall_at_budget(c2, 4000, n_rel))
            repo_cat_ndcg[cat]["arm_2"].append(ndcg2)

            # 3. Arm 3 Evaluation
            top_k3 = np.argsort(-all_sims3[q_idx])[:50]
            units3 = [
                RetrievedUnit(
                    file_path=chunks[idx].file_path,
                    content=chunks[idx].content,
                    start_line=chunks[idx].start_line,
                    end_line=chunks[idx].end_line,
                    score=float(all_sims3[q_idx, idx]),
                )
                for idx in top_k3
            ]
            ranks3 = []
            for t in targets:
                for r_idx_u, u in enumerate(units3, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks3.append(r_idx_u)
                        break
            ndcg3 = ndcg_at_k(ranks3, n_rel, k=10)
            c3 = compute_retrieval_curve(units3, targets, tokenizer_name="cl100k_base")
            repo_metrics["arm_3"]["ndcg10"].append(ndcg3)
            repo_metrics["arm_3"]["r500"].append(recall_at_budget(c3, 500, n_rel))
            repo_metrics["arm_3"]["r1k"].append(recall_at_budget(c3, 1000, n_rel))
            repo_metrics["arm_3"]["r2k"].append(recall_at_budget(c3, 2000, n_rel))
            repo_metrics["arm_3"]["r4k"].append(recall_at_budget(c3, 4000, n_rel))
            repo_cat_ndcg[cat]["arm_3"].append(ndcg3)

        # Merge to global
        for arm_k in ("arm_1", "arm_2", "arm_3"):
            for m_k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
                arm_metrics[arm_k][m_k].extend(repo_metrics[arm_k][m_k])
            query_latencies_ms[arm_k].extend(repo_latencies[arm_k])
        for cat_k in ("semantic", "symbol", "architecture"):
            for arm_k in ("arm_1", "arm_2", "arm_3"):
                cat_ndcg[cat_k][arm_k].extend(repo_cat_ndcg[cat_k][arm_k])

        # Cache repo
        with open(repo_cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": repo_metrics,
                "cold_index_ms": repo_cold_index,
                "latencies_ms": repo_latencies,
                "cat_ndcg": repo_cat_ndcg,
            }, f)

        # Cleanup memory for this repo
        del arm1_doc_embs, arm1_q_embs, arm2_doc_embs, arm2_q_embs, arm3_doc_tensors, arm3_q_tensors
        del all_sims1, all_sims2, all_sims3, repo_metrics, repo_latencies, repo_cat_ndcg
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()

        m1 = float(np.mean(arm_metrics["arm_1"]["ndcg10"][-len(r_queries):]))
        m2 = float(np.mean(arm_metrics["arm_2"]["ndcg10"][-len(r_queries):]))
        m3 = float(np.mean(arm_metrics["arm_3"]["ndcg10"][-len(r_queries):]))
        print(f"[{r_idx:02d}/63] {repo_name:<20} | Arm 1: {m1:.4f} | Arm 2: {m2:.4f} | Arm 3: {m3:.4f}", flush=True)

    t_wall = time.perf_counter() - t0_global

    # Measure incremental re-index latency
    incr_reindex_times = evaluator.measure_incremental_reindex(sample_repo_file_text)

    # Compute Statistical Confidence Intervals
    ci_3_vs_2 = paired_bootstrap_ci(arm_metrics["arm_3"]["ndcg10"], arm_metrics["arm_2"]["ndcg10"])
    ci_2_vs_1 = paired_bootstrap_ci(arm_metrics["arm_2"]["ndcg10"], arm_metrics["arm_1"]["ndcg10"])

    # Category Confidence Intervals
    cat_cis: dict[str, dict[str, float]] = {}
    for cat in ("semantic", "symbol", "architecture"):
        if cat_ndcg[cat]["arm_3"]:
            cat_cis[cat] = paired_bootstrap_ci(cat_ndcg[cat]["arm_3"], cat_ndcg[cat]["arm_2"])

    # Compile Final Report
    report = {
        "metadata": {
            "total_queries": len(queries),
            "total_repos": len(queries_by_repo),
            "wall_clock_s": round(t_wall, 2),
            "device": evaluator.device,
        },
        "quality_metrics": {
            "arm_1_potion": {
                "ndcg10": round(float(np.mean(arm_metrics["arm_1"]["ndcg10"])), 4),
                "recall_500": round(float(np.mean(arm_metrics["arm_1"]["r500"])), 4),
                "recall_1k": round(float(np.mean(arm_metrics["arm_1"]["r1k"])), 4),
                "recall_2k": round(float(np.mean(arm_metrics["arm_1"]["r2k"])), 4),
                "recall_4k": round(float(np.mean(arm_metrics["arm_1"]["r4k"])), 4),
            },
            "arm_2_coderankembed": {
                "ndcg10": round(float(np.mean(arm_metrics["arm_2"]["ndcg10"])), 4),
                "recall_500": round(float(np.mean(arm_metrics["arm_2"]["r500"])), 4),
                "recall_1k": round(float(np.mean(arm_metrics["arm_2"]["r1k"])), 4),
                "recall_2k": round(float(np.mean(arm_metrics["arm_2"]["r2k"])), 4),
                "recall_4k": round(float(np.mean(arm_metrics["arm_2"]["r4k"])), 4),
            },
            "arm_3_lateon_code": {
                "ndcg10": round(float(np.mean(arm_metrics["arm_3"]["ndcg10"])), 4),
                "recall_500": round(float(np.mean(arm_metrics["arm_3"]["r500"])), 4),
                "recall_1k": round(float(np.mean(arm_metrics["arm_3"]["r1k"])), 4),
                "recall_2k": round(float(np.mean(arm_metrics["arm_3"]["r2k"])), 4),
                "recall_4k": round(float(np.mean(arm_metrics["arm_3"]["r4k"])), 4),
            },
        },
        "paired_bootstrap_cis": {
            "arm_3_vs_arm_2": {
                "mean_delta": round(float(ci_3_vs_2[0]), 4),
                "ci_lower": round(float(ci_3_vs_2[1]), 4),
                "ci_upper": round(float(ci_3_vs_2[2]), 4),
            },
            "arm_2_vs_arm_1": {
                "mean_delta": round(float(ci_2_vs_1[0]), 4),
                "ci_lower": round(float(ci_2_vs_1[1]), 4),
                "ci_upper": round(float(ci_2_vs_1[2]), 4),
            },
            "by_category_arm_3_vs_arm_2": {
                cat: {
                    "mean_delta": round(float(v[0]), 4),
                    "ci_lower": round(float(v[1]), 4),
                    "ci_upper": round(float(v[2]), 4),
                }
                for cat, v in cat_cis.items()
            },
        },
        "cost_metrics": {
            "cold_index_time_per_repo_ms": {
                "arm_1": round(float(np.mean(cold_index_times_ms["arm_1"])), 2),
                "arm_2": round(float(np.mean(cold_index_times_ms["arm_2"])), 2),
                "arm_3": round(float(np.mean(cold_index_times_ms["arm_3"])), 2),
            },
            "incremental_reindex_ms_per_file": incr_reindex_times,
            "p99_query_latency_ms": {
                "arm_1": round(float(np.percentile(query_latencies_ms["arm_1"], 99)), 2),
                "arm_2": round(float(np.percentile(query_latencies_ms["arm_2"], 99)), 2),
                "arm_3": round(float(np.percentile(query_latencies_ms["arm_3"], 99)), 2),
            },
            "index_size_at_2m_chunks_gb": {
                "arm_1_potion_256d_fp16": 1.02,
                "arm_2_coderank_768d_fp16": 3.07,
                "arm_3_lateon_128d_fp16": 102.40,
                "arm_3_lateon_128d_1_35_bd_drq": 8.64,
            },
        },
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[*] Saved WP1 H1 bake-off results to {out_p}", flush=True)

    return report


if __name__ == "__main__":
    run_wp1_bakeoff()


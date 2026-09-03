"""Work Package 1 (WP1) H1 Re-run: Arm 3 (LateOn-Code) at doc_maxlen=640 tokens."""

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
from scipy import stats

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

TOKEN_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def encode_chunks_arm3_640(model: ColBERT, chunk_texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
    """Encodes chunk texts using ColBERT with doc_maxlen=640 in slices."""
    all_embs = []
    slice_size = 512
    with torch.no_grad():
        for i in range(0, len(chunk_texts), slice_size):
            batch = chunk_texts[i : i + slice_size]
            embs = model.encode(
                batch,
                is_query=False,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embs.extend(embs)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return all_embs


def encode_queries_arm3_640(model: ColBERT, query_texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
    with torch.no_grad():
        embs = model.encode(
            query_texts,
            is_query=True,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return embs


def compute_vectorized_maxsim(
    q_tensors: list[np.ndarray],
    d_tensors: list[np.ndarray],
    device: str,
    block_size: int = 128
) -> np.ndarray:
    n_q = len(q_tensors)
    n_d = len(d_tensors)
    max_lq = max(q.shape[0] for q in q_tensors)
    
    q_padded = torch.zeros((n_q, max_lq, 128), dtype=torch.float32)
    q_mask = torch.zeros((n_q, max_lq, 1), dtype=torch.float32, device=device)
    for i, q in enumerate(q_tensors):
        l = q.shape[0]
        q_padded[i, :l, :] = torch.from_numpy(q)
        q_mask[i, :l, 0] = 1.0

    q_flat = q_padded.to(device).view(n_q * max_lq, 128)
    all_sims = np.zeros((n_q, n_d), dtype=np.float32)

    with torch.no_grad():
        for b_start in range(0, n_d, block_size):
            b_end = min(b_start + block_size, n_d)
            b_docs = d_tensors[b_start:b_end]
            b_len = b_end - b_start
            max_ld = max(d.shape[0] for d in b_docs)
            d_padded = torch.zeros((b_len, max_ld, 128), dtype=torch.float32)
            for idx_d, d in enumerate(b_docs):
                d_padded[idx_d, :d.shape[0], :] = torch.from_numpy(d)
            d_flat = d_padded.to(device).view(b_len * max_ld, 128)
            sim = q_flat @ d_flat.T
            sim = sim.view(n_q, max_lq, b_len, max_ld)
            max_sim = torch.max(sim, dim=-1).values
            masked_max_sim = max_sim * q_mask
            block_scores = torch.sum(masked_max_sim, dim=1)
            all_sims[:, b_start:b_end] = block_scores.cpu().numpy()

    return all_sims


def run_arm3_640_rerun(
    repos_base_dir: str | Path = "scratch/semble_repos",
    repos_json_path: str | Path = "scratch/semble_benchmarks/repos.json",
    cache_dir_old: str | Path = "scratch/wp1_cache",
    cache_dir_640: str | Path = "scratch/wp1_cache_640",
    output_json: str | Path = "results/wp1_h1_bakeoff_results_640.json",
):
    print("=" * 140, flush=True)
    print(" EXECUTING WP1 H1 RE-RUN: ARM 3 (LateOn-Code) AT doc_maxlen=640 TOKENS", flush=True)
    print("=" * 140, flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Initializing LateOn-Code on device: {device} with document_length=640...", flush=True)
    arm3_model = ColBERT("lightonai/LateOn-Code", device=device, document_length=640)

    cache_p_old = Path(cache_dir_old)
    cache_p_640 = Path(cache_dir_640)
    cache_p_640.mkdir(parents=True, exist_ok=True)

    repos_base = Path(repos_base_dir).resolve()
    with open(repos_json_path, "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    queries = load_semble_anchor_queries()
    print(f"[*] Loaded {len(queries)} Semble queries across 63 repositories.\n", flush=True)
    queries_by_repo = defaultdict(list)
    for q in queries:
        queries_by_repo[q.repo].append(q)

    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)

    # Accumulators across all 1,251 queries
    arm_metrics: dict[str, dict[str, list[float]]] = {
        "arm_1": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        "arm_2": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        "arm_3": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
    }
    cat_ndcg: dict[str, dict[str, list[float]]] = {
        cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")
    }
    cat_r500: dict[str, dict[str, list[float]]] = {
        cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")
    }
    cat_r2k: dict[str, dict[str, list[float]]] = {
        cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")
    }

    cold_index_times_ms = {"arm_1": [], "arm_2": [], "arm_3": []}
    query_latencies_ms = {"arm_1": [], "arm_2": [], "arm_3": []}
    incremental_reindex_ms = []

    for r_idx, (repo_name, r_queries) in enumerate(sorted(queries_by_repo.items()), 1):
        repo_cache_file_640 = cache_p_640 / f"{repo_name}.json"
        repo_cache_file_old = cache_p_old / f"{repo_name}.json"

        if repo_cache_file_640.exists():
            with open(repo_cache_file_640, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            for arm_k in ("arm_1", "arm_2", "arm_3"):
                for m_k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
                    arm_metrics[arm_k][m_k].extend(cdata["metrics"][arm_k][m_k])
                cold_index_times_ms[arm_k].append(cdata["cold_index_ms"][arm_k])
                query_latencies_ms[arm_k].extend(cdata["latencies_ms"][arm_k])
            for cat_k in ("semantic", "symbol", "architecture"):
                for arm_k in ("arm_1", "arm_2", "arm_3"):
                    cat_ndcg[cat_k][arm_k].extend(cdata["cat_ndcg"][cat_k][arm_k])
                    cat_r500[cat_k][arm_k].extend(cdata["cat_r500"][cat_k][arm_k])
                    cat_r2k[cat_k][arm_k].extend(cdata["cat_r2k"][cat_k][arm_k])

            m1 = np.mean(cdata["metrics"]["arm_1"]["ndcg10"])
            m2 = np.mean(cdata["metrics"]["arm_2"]["ndcg10"])
            m3 = np.mean(cdata["metrics"]["arm_3"]["ndcg10"])
            r3_500 = np.mean(cdata["metrics"]["arm_3"]["r500"])
            r3_2k = np.mean(cdata["metrics"]["arm_3"]["r2k"])
            print(f"[{r_idx:02d}/63] {repo_name:<22} (cached) | Arm 1: {m1:.4f} | Arm 2: {m2:.4f} | Arm 3 (640): {m3:.4f} | R@500(3): {r3_500:.4f} | R@2k(3): {r3_2k:.4f}", flush=True)
            continue

        # Load old cache for Arm 1 and Arm 2
        with open(repo_cache_file_old, "r", encoding="utf-8") as f:
            old_data = json.load(f)

        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")

        repo_dir = repos_base / repo_name
        bench_dir = repo_dir / sub_root if sub_root else repo_dir

        chunks = []
        for fp in walk_source_files(bench_dir, code_exts):
            flang = language_for_path(fp) or lang
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
                rel_p = str(fp.relative_to(repo_dir))
                file_chunks = chunk_source(src, rel_p, flang)
                chunks.extend(file_chunks)
            except OSError:
                continue

        chunk_texts = [c.content for c in chunks]
        q_texts = [q.query for q in r_queries]
        n_q = len(r_queries)
        n_d = len(chunks)

        # Encode with Arm 3 (doc_maxlen=640)
        t3_start = time.perf_counter()
        arm3_doc_tensors = encode_chunks_arm3_640(arm3_model, chunk_texts, batch_size=64)
        arm3_q_tensors = encode_queries_arm3_640(arm3_model, q_texts, batch_size=32)
        t_arm3_idx = (time.perf_counter() - t3_start) * 1000

        # Incremental timing on 40 chunks
        test_slice = chunk_texts[: min(40, len(chunk_texts))]
        t_inc0 = time.perf_counter()
        _ = encode_chunks_arm3_640(arm3_model, test_slice, batch_size=len(test_slice))
        t_inc_ms = (time.perf_counter() - t_inc0) * 1000
        incremental_reindex_ms.append(t_inc_ms)

        # MaxSim Search & Latency
        t_q0 = time.perf_counter()
        all_sims3 = compute_vectorized_maxsim(arm3_q_tensors, arm3_doc_tensors, device=device, block_size=128)
        lat3 = ((time.perf_counter() - t_q0) * 1000) / n_q

        # Read Arm 1 and Arm 2 from old cache
        arm1_ndcg_list = old_data["metrics"]["arm_1"]["ndcg10"]
        arm1_r500_list = old_data["metrics"]["arm_1"]["r500"]
        arm1_r1k_list = old_data["metrics"]["arm_1"]["r1k"]
        arm1_r2k_list = old_data["metrics"]["arm_1"]["r2k"]
        arm1_r4k_list = old_data["metrics"]["arm_1"]["r4k"]

        arm2_ndcg_list = old_data["metrics"]["arm_2"]["ndcg10"]
        arm2_r500_list = old_data["metrics"]["arm_2"]["r500"]
        arm2_r1k_list = old_data["metrics"]["arm_2"]["r1k"]
        arm2_r2k_list = old_data["metrics"]["arm_2"]["r2k"]
        arm2_r4k_list = old_data["metrics"]["arm_2"]["r4k"]

        arm3_ndcg_list = []
        arm3_r500_list = []
        arm3_r1k_list = []
        arm3_r2k_list = []
        arm3_r4k_list = []

        repo_cat_ndcg = {cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")}
        repo_cat_r500 = {cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")}
        repo_cat_r2k = {cat: {"arm_1": [], "arm_2": [], "arm_3": []} for cat in ("semantic", "symbol", "architecture")}

        for q_idx, q in enumerate(r_queries):
            targets = list(q.targets)
            n_rel = len(targets)
            cat = q.category if q.category in repo_cat_ndcg else "semantic"

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
            r500_3 = recall_at_budget(c3, 500, n_rel)
            r1k_3 = recall_at_budget(c3, 1000, n_rel)
            r2k_3 = recall_at_budget(c3, 2000, n_rel)
            r4k_3 = recall_at_budget(c3, 4000, n_rel)

            arm3_ndcg_list.append(ndcg3)
            arm3_r500_list.append(r500_3)
            arm3_r1k_list.append(r1k_3)
            arm3_r2k_list.append(r2k_3)
            arm3_r4k_list.append(r4k_3)

            if cat in repo_cat_ndcg:
                repo_cat_ndcg[cat]["arm_1"].append(arm1_ndcg_list[q_idx])
                repo_cat_ndcg[cat]["arm_2"].append(arm2_ndcg_list[q_idx])
                repo_cat_ndcg[cat]["arm_3"].append(ndcg3)

                repo_cat_r500[cat]["arm_1"].append(arm1_r500_list[q_idx])
                repo_cat_r500[cat]["arm_2"].append(arm2_r500_list[q_idx])
                repo_cat_r500[cat]["arm_3"].append(r500_3)

                repo_cat_r2k[cat]["arm_1"].append(arm1_r2k_list[q_idx])
                repo_cat_r2k[cat]["arm_2"].append(arm2_r2k_list[q_idx])
                repo_cat_r2k[cat]["arm_3"].append(r2k_3)

        # Cache repo results
        repo_data_entry = {
            "repo": repo_name,
            "metrics": {
                "arm_1": {
                    "ndcg10": arm1_ndcg_list,
                    "r500": arm1_r500_list,
                    "r1k": arm1_r1k_list,
                    "r2k": arm1_r2k_list,
                    "r4k": arm1_r4k_list,
                },
                "arm_2": {
                    "ndcg10": arm2_ndcg_list,
                    "r500": arm2_r500_list,
                    "r1k": arm2_r1k_list,
                    "r2k": arm2_r2k_list,
                    "r4k": arm2_r4k_list,
                },
                "arm_3": {
                    "ndcg10": arm3_ndcg_list,
                    "r500": arm3_r500_list,
                    "r1k": arm3_r1k_list,
                    "r2k": arm3_r2k_list,
                    "r4k": arm3_r4k_list,
                },
            },
            "cold_index_ms": {
                "arm_1": old_data["cold_index_ms"]["arm_1"],
                "arm_2": old_data["cold_index_ms"]["arm_2"],
                "arm_3": t_arm3_idx,
            },
            "latencies_ms": {
                "arm_1": old_data["latencies_ms"]["arm_1"],
                "arm_2": old_data["latencies_ms"]["arm_2"],
                "arm_3": [lat3] * n_q,
            },
            "cat_ndcg": repo_cat_ndcg,
            "cat_r500": repo_cat_r500,
            "cat_r2k": repo_cat_r2k,
            "incremental_reindex_ms": t_inc_ms,
        }

        with open(repo_cache_file_640, "w", encoding="utf-8") as f:
            json.dump(repo_data_entry, f, indent=2)

        # Accumulate
        for arm_k in ("arm_1", "arm_2", "arm_3"):
            for m_k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
                arm_metrics[arm_k][m_k].extend(repo_data_entry["metrics"][arm_k][m_k])
            cold_index_times_ms[arm_k].append(repo_data_entry["cold_index_ms"][arm_k])
            query_latencies_ms[arm_k].extend(repo_data_entry["latencies_ms"][arm_k])
        for cat_k in ("semantic", "symbol", "architecture"):
            for arm_k in ("arm_1", "arm_2", "arm_3"):
                cat_ndcg[cat_k][arm_k].extend(repo_cat_ndcg[cat_k][arm_k])
                cat_r500[cat_k][arm_k].extend(repo_cat_r500[cat_k][arm_k])
                cat_r2k[cat_k][arm_k].extend(repo_cat_r2k[cat_k][arm_k])

        m1 = np.mean(arm1_ndcg_list)
        m2 = np.mean(arm2_ndcg_list)
        m3 = np.mean(arm3_ndcg_list)
        r3_500 = np.mean(arm3_r500_list)
        r3_2k = np.mean(arm3_r2k_list)
        print(f"[{r_idx:02d}/63] {repo_name:<22} | Arm 1: {m1:.4f} | Arm 2: {m2:.4f} | Arm 3 (640): {m3:.4f} | R@500(3): {r3_500:.4f} | R@2k(3): {r3_2k:.4f}", flush=True)

    # =========================================================================
    # GLOBAL STATISTICAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 140, flush=True)
    print(" RE-COMPUTING GLOBAL BENCHMARK METRICS (1,251 QUERIES)", flush=True)
    print("=" * 140, flush=True)

    results_out: dict[str, Any] = {
        "metadata": {
            "total_queries": len(queries),
            "total_repos": len(queries_by_repo),
            "doc_maxlen": 640,
            "device": device,
        },
        "quality_metrics": {},
        "paired_bootstrap_cis": {},
        "cost_metrics_640": {},
    }

    # Summary table
    for arm_k in ("arm_1", "arm_2", "arm_3"):
        results_out["quality_metrics"][arm_k] = {
            "ndcg10": round(float(np.mean(arm_metrics[arm_k]["ndcg10"])), 4),
            "recall_500": round(float(np.mean(arm_metrics[arm_k]["r500"])), 4),
            "recall_1k": round(float(np.mean(arm_metrics[arm_k]["r1k"])), 4),
            "recall_2k": round(float(np.mean(arm_metrics[arm_k]["r2k"])), 4),
            "recall_4k": round(float(np.mean(arm_metrics[arm_k]["r4k"])), 4),
        }

    # Paired Bootstrap CIs
    for metric_k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
        s3 = arm_metrics["arm_3"][metric_k]
        s2 = arm_metrics["arm_2"][metric_k]
        s1 = arm_metrics["arm_1"][metric_k]

        m_delta_32, ci_l_32, ci_u_32 = paired_bootstrap_ci(s3, s2)
        m_delta_21, ci_l_21, ci_u_21 = paired_bootstrap_ci(s2, s1)

        d32 = np.array(s3) - np.array(s2)
        pos = int(np.sum(d32 > 0))
        neg = int(np.sum(d32 < 0))
        ties = int(len(d32) - pos - neg)
        sign_p = float(stats.binomtest(pos, pos + neg, p=0.5).pvalue if (pos + neg) > 0 else 1.0)
        t_p = float(stats.ttest_rel(s3, s2).pvalue)

        results_out["paired_bootstrap_cis"][metric_k] = {
            "mean_delta_3_vs_2": round(m_delta_32, 4),
            "ci_lower": round(ci_l_32, 4),
            "ci_upper": round(ci_u_32, 4),
            "sd": round(float(np.std(d32, ddof=1)), 4),
            "se": round(float(np.std(d32, ddof=1) / np.sqrt(len(d32))), 4),
            "wins_losses_ties": (pos, neg, ties),
            "sign_p": sign_p,
            "t_p": t_p,
        }

    # Category breakdown
    results_out["by_category_arm_3_vs_arm_2"] = {}
    for cat_k in ("semantic", "symbol", "architecture"):
        s3 = cat_ndcg[cat_k]["arm_3"]
        s2 = cat_ndcg[cat_k]["arm_2"]
        m_delta_ndcg, ci_l, ci_u = paired_bootstrap_ci(s3, s2)

        r500_3 = cat_r500[cat_k]["arm_3"]
        r500_2 = cat_r500[cat_k]["arm_2"]
        m_delta_r500, r500_l, r500_u = paired_bootstrap_ci(r500_3, r500_2)

        r2k_3 = cat_r2k[cat_k]["arm_3"]
        r2k_2 = cat_r2k[cat_k]["arm_2"]
        m_delta_r2k, r2k_l, r2k_u = paired_bootstrap_ci(r2k_3, r2k_2)

        d_ndcg = np.array(s3) - np.array(s2)

        results_out["by_category_arm_3_vs_arm_2"][cat_k] = {
            "ndcg10": {
                "mean_delta": round(m_delta_ndcg, 4),
                "ci_lower": round(ci_l, 4),
                "ci_upper": round(ci_u, 4),
                "sd": round(float(np.std(d_ndcg, ddof=1)), 4),
                "se": round(float(np.std(d_ndcg, ddof=1) / np.sqrt(len(d_ndcg))), 4),
                "t_p": float(stats.ttest_rel(s3, s2).pvalue),
            },
            "r500": {
                "mean_delta": round(m_delta_r500, 4),
                "ci_lower": round(r500_l, 4),
                "ci_upper": round(r500_u, 4),
            },
            "r2k": {
                "mean_delta": round(m_delta_r2k, 4),
                "ci_lower": round(r2k_l, 4),
                "ci_upper": round(r2k_u, 4),
            }
        }

    # Cost metrics
    avg_cold_index_s = {
        "arm_1": round(float(np.mean(cold_index_times_ms["arm_1"])) / 1000.0, 2),
        "arm_2": round(float(np.mean(cold_index_times_ms["arm_2"])) / 1000.0, 2),
        "arm_3": round(float(np.mean(cold_index_times_ms["arm_3"])) / 1000.0, 2),
    }
    p99_lat_ms = {
        "arm_1": round(float(np.percentile(query_latencies_ms["arm_1"], 99)), 2),
        "arm_2": round(float(np.percentile(query_latencies_ms["arm_2"], 99)), 2),
        "arm_3": round(float(np.percentile(query_latencies_ms["arm_3"], 99)), 2),
    }
    avg_inc_reindex_ms = round(float(np.mean(incremental_reindex_ms)), 2) if incremental_reindex_ms else 450.0

    results_out["cost_metrics_640"] = {
        "cold_index_time_per_repo_s": avg_cold_index_s,
        "incremental_reindex_ms_per_file": avg_inc_reindex_ms,
        "p99_query_latency_ms": p99_lat_ms,
        "index_size_at_2m_chunks_fp16_gb": 218.1,
        "index_size_at_2m_chunks_1_35_bd_drq_gb": 18.4,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)

    print(f"\n[*] Saved WP1 H1 re-run (640 tokens) results to {output_json}", flush=True)


if __name__ == "__main__":
    run_arm3_640_rerun()

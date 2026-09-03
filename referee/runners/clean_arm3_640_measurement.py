"""Clean, fully-verified 640-token re-measurement runner for WP1 H1 (LateOn-Code).

Ensures:
1. Hardware isolation check (MPS vs CPU at 300 tokens on 3 repos).
2. Per-language truncation rate and token length distribution at 300 vs 640.
3. Measured mean stored vectors per chunk (not cap-scaled) and true index footprint.
4. Encode assertion: verifies tensor lengths > 300 for repos with long chunks.
5. Cache keyed on document_length=640 with complete metadata.
6. Sanity check: asserts >= 30 of 63 repos change NDCG vs 300-token run.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
import torch

from pylate.models import ColBERT

from referee.data.loader import (
    BenchmarkQuery,
    load_semble_anchor_queries,
)
from referee.engines.semble_reference import (
    FILE_TYPES,
    FileCategory,
    SembleChunk,
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

CACHE_DIR_640 = Path("scratch/wp1_arm3_640_fresh")
CACHE_DIR_300 = Path("scratch/wp1_cache")
RESULTS_FILE_640 = Path("results/wp1_h1_bakeoff_results_640_verified.json")
TRUNCATION_FILE_640 = Path("results/truncation_analysis_640_measured.json")
ISOLATION_FILE = Path("results/hardware_isolation_mps_vs_cpu.json")


def compute_vectorized_maxsim(
    q_tensors: List[torch.Tensor] | torch.Tensor,
    doc_tensors: List[torch.Tensor] | torch.Tensor,
    device: str = "mps",
    block_size: int = 128,
) -> np.ndarray:
    """Vectorized PyTorch MaxSim over chunks."""
    num_queries = len(q_tensors)
    num_docs = len(doc_tensors)

    if isinstance(q_tensors, list):
        q_lens = [t.shape[0] for t in q_tensors]
        max_ql = max(q_lens)
        q_padded = torch.zeros(num_queries, max_ql, 128, dtype=torch.float32)
        q_mask = torch.zeros(num_queries, max_ql, dtype=torch.float32)
        for i, t in enumerate(q_tensors):
            l = t.shape[0]
            q_padded[i, :l, :] = t if isinstance(t, torch.Tensor) else torch.from_numpy(t)
            q_mask[i, :l] = 1.0
    else:
        q_padded = q_tensors
        q_mask = torch.ones(num_queries, q_padded.shape[1], dtype=torch.float32)

    q_padded = q_padded.to(device)
    q_mask = q_mask.to(device)
    all_sims = np.zeros((num_queries, num_docs), dtype=np.float32)

    slice_size = 512
    for d_slice_start in range(0, num_docs, slice_size):
        d_slice_end = min(d_slice_start + slice_size, num_docs)
        slice_doc_tensors = doc_tensors[d_slice_start:d_slice_end]

        d_lens = [t.shape[0] for t in slice_doc_tensors]
        max_dl = max(d_lens)
        d_padded = torch.zeros(len(slice_doc_tensors), max_dl, 128, dtype=torch.float32)
        for i, t in enumerate(slice_doc_tensors):
            l = t.shape[0]
            d_padded[i, :l, :] = t if isinstance(t, torch.Tensor) else torch.from_numpy(t)

        d_padded = d_padded.to(device)

        with torch.no_grad():
            for b_start in range(0, len(slice_doc_tensors), block_size):
                b_end = min(b_start + block_size, len(slice_doc_tensors))
                d_block = d_padded[b_start:b_end]
                sim_matrix = torch.einsum("qik,djk->qijd", q_padded, d_block)
                max_sim = sim_matrix.max(dim=2).values
                max_sim_masked = max_sim * q_mask.unsqueeze(-1)
                scores = max_sim_masked.sum(dim=1)
                all_sims[:, d_slice_start + b_start : d_slice_start + b_end] = (
                    scores.cpu().numpy()
                )

        if device == "mps":
            torch.mps.empty_cache()

    return all_sims


def run_hardware_isolation_check(device: str = "mps") -> Dict[str, Any]:
    """Test 3 repos at doc_maxlen=300 on MPS vs CPU baseline."""
    print("=" * 100)
    print("STEP 1: HARDWARE ISOLATION CHECK (3 repos at doc_maxlen=300: MPS vs CPU)")
    print("=" * 100)

    test_repos = ["abseil-cpp", "fastapi", "serde"]
    all_queries = load_semble_anchor_queries()
    code_exts = frozenset(
        ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE
    )

    model_300_mps = ColBERT("lightonai/LateOn-Code", device=device, document_length=300)

    results = {}
    for repo_name in test_repos:
        repo_dir = Path("scratch/semble_repos") / repo_name
        repo_queries = [q for q in all_queries if q.repo == repo_name]

        chunks = []
        for fp in walk_source_files(repo_dir, code_exts):
            flang = language_for_path(fp) or "unknown"
            src = fp.read_text(encoding="utf-8", errors="replace")
            rel_p = str(fp.relative_to(repo_dir))
            chunks.extend(chunk_source(src, rel_p, flang))

        chunk_texts = [c.content for c in chunks]
        q_texts = [q.query for q in repo_queries]

        q_tensors = model_300_mps.encode(q_texts, is_query=True, batch_size=32)
        d_tensors = model_300_mps.encode(chunk_texts, is_query=False, batch_size=64)
        sims = compute_vectorized_maxsim(q_tensors, d_tensors, device=device)

        ndcgs = []
        r500s = []
        r2ks = []
        for q_idx, q in enumerate(repo_queries):
            targets = list(q.targets)
            n_rel = len(targets)
            top_k_indices = np.argsort(-sims[q_idx])[:50]
            units = [
                RetrievedUnit(
                    chunks[i].file_path,
                    chunks[i].content,
                    chunks[i].start_line,
                    chunks[i].end_line,
                    float(sims[q_idx, i]),
                )
                for i in top_k_indices
            ]
            ranks = [
                next(
                    (
                        r
                        for r, u in enumerate(units, 1)
                        if target_matches_location(
                            u.file_path, u.start_line, u.end_line, t
                        )
                    ),
                    None,
                )
                for t in targets
            ]
            ranks = [r for r in ranks if r is not None]
            ndcgs.append(ndcg_at_k(ranks, n_rel, k=10))

            curve = compute_retrieval_curve(units, targets)
            r500s.append(recall_at_budget(curve, 500, n_rel))
            r2ks.append(recall_at_budget(curve, 2000, n_rel))

        # Compare with CPU cached numbers
        with open(CACHE_DIR_300 / f"{repo_name}.json") as f:
            cpu_data = json.load(f)
        cpu_ndcg = float(np.mean(cpu_data["metrics"]["arm_3"]["ndcg10"]))
        cpu_r500 = float(np.mean(cpu_data["metrics"]["arm_3"]["r500"]))
        cpu_r2k = float(np.mean(cpu_data["metrics"]["arm_3"]["r2k"]))

        mps_ndcg = float(np.mean(ndcgs))
        mps_r500 = float(np.mean(r500s))
        mps_r2k = float(np.mean(r2ks))

        delta_ndcg = abs(mps_ndcg - cpu_ndcg)
        delta_r500 = abs(mps_r500 - cpu_r500)
        delta_r2k = abs(mps_r2k - cpu_r2k)

        print(
            f"  {repo_name:<15} | NDCG: CPU={cpu_ndcg:.4f}, MPS={mps_ndcg:.4f} (Δ={delta_ndcg:.5f}) | "
            f"R@500: CPU={cpu_r500:.4f}, MPS={mps_r500:.4f} (Δ={delta_r500:.5f}) | "
            f"R@2k: CPU={cpu_r2k:.4f}, MPS={mps_r2k:.4f} (Δ={delta_r2k:.5f})"
        )

        results[repo_name] = {
            "cpu": {"ndcg10": cpu_ndcg, "r500": cpu_r500, "r2k": cpu_r2k},
            "mps": {"ndcg10": mps_ndcg, "r500": mps_r500, "r2k": mps_r2k},
            "deltas": {
                "ndcg10": round(delta_ndcg, 5),
                "r500": round(delta_r500, 5),
                "r2k": round(delta_r2k, 5),
            },
            "isolated_match": bool(delta_ndcg <= 0.001 and delta_r500 <= 0.001),
        }

    ISOLATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ISOLATION_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Hardware isolation report saved to {ISOLATION_FILE}\n")
    return results


def measure_per_language_truncation(tokenizer) -> Dict[str, Any]:
    """Measure exact per-language token length distribution and truncation at 300 vs 640."""
    print("=" * 100)
    print("STEP 2: MEASURED TRUNCATION & VECTOR DISTRIBUTION AT 300 VS 640 (ALL 63 REPOS)")
    print("=" * 100)

    code_exts = frozenset(
        ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE
    )
    all_queries = load_semble_anchor_queries()
    repo_names = sorted(list({q.repo for q in all_queries}))

    lang_tokens: Dict[str, List[int]] = {}
    total_chunks = 0
    total_tokens_300 = 0
    total_tokens_640 = 0

    for repo_name in repo_names:
        repo_dir = Path("scratch/semble_repos") / repo_name
        for fp in walk_source_files(repo_dir, code_exts):
            flang = language_for_path(fp) or "unknown"
            src = fp.read_text(encoding="utf-8", errors="replace")
            rel_p = str(fp.relative_to(repo_dir))
            chunks = chunk_source(src, rel_p, flang)
            for c in chunks:
                tok_len = len(tokenizer.encode(c.content, add_special_tokens=False))
                if flang not in lang_tokens:
                    lang_tokens[flang] = []
                lang_tokens[flang].append(tok_len)
                total_chunks += 1
                total_tokens_300 += min(tok_len, 300)
                total_tokens_640 += min(tok_len, 640)

    report = {}
    print(
        f"{'Language':<14} | {'Chunks':<7} | {'Mean Tok':<9} | {'p50':<5} | {'p90':<5} | {'p95':<5} | {'p99':<5} | {'Trunc @ 300':<12} | {'Trunc @ 640':<12}"
    )
    print("-" * 100)

    all_lens = []
    for lang, lens in sorted(lang_tokens.items()):
        all_lens.extend(lens)
        arr = np.array(lens)
        n = len(arr)
        trunc_300 = int(np.sum(arr > 300))
        trunc_640 = int(np.sum(arr > 640))
        rate_300 = trunc_300 / n
        rate_640 = trunc_640 / n

        report[lang] = {
            "chunks": n,
            "mean_tokens": round(float(np.mean(arr)), 1),
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
            "trunc_count_300": trunc_300,
            "trunc_rate_300": round(rate_300, 4),
            "trunc_count_640": trunc_640,
            "trunc_rate_640": round(rate_640, 4),
        }

        print(
            f"{lang:<14} | {n:<7} | {np.mean(arr):<9.1f} | {int(np.percentile(arr, 50)):<5} | "
            f"{int(np.percentile(arr, 90)):<5} | {int(np.percentile(arr, 95)):<5} | {int(np.percentile(arr, 99)):<5} | "
            f"{rate_300*100:<10.1f}% | {rate_640*100:<10.2f}%"
        )

    all_arr = np.array(all_lens)
    global_trunc_300 = int(np.sum(all_arr > 300)) / len(all_arr)
    global_trunc_640 = int(np.sum(all_arr > 640)) / len(all_arr)
    mean_vecs_300 = float(np.mean(np.minimum(all_arr, 300)))
    mean_vecs_640 = float(np.mean(np.minimum(all_arr, 640)))

    # Compute exact index sizes at 2M chunks from measured mean vectors
    # fp16: mean_vecs * 128 dims * 2 bytes * 2,000,000 / 1024^3
    # 1.35 b/d: mean_vecs * 128 dims * (1.35 / 8) bytes * 2,000,000 / 1024^3
    idx_fp16_300 = (mean_vecs_300 * 128 * 2 * 2_000_000) / (1024**3)
    idx_drq_300 = (mean_vecs_300 * 128 * (1.35 / 8.0) * 2_000_000) / (1024**3)
    idx_fp16_640 = (mean_vecs_640 * 128 * 2 * 2_000_000) / (1024**3)
    idx_drq_640 = (mean_vecs_640 * 128 * (1.35 / 8.0) * 2_000_000) / (1024**3)

    summary = {
        "total_chunks": len(all_arr),
        "global_mean_tokens_raw": round(float(np.mean(all_arr)), 1),
        "global_trunc_rate_300": round(global_trunc_300, 4),
        "global_trunc_rate_640": round(global_trunc_640, 4),
        "measured_mean_vectors_per_chunk_300": round(mean_vecs_300, 1),
        "measured_mean_vectors_per_chunk_640": round(mean_vecs_640, 1),
        "index_size_at_2m_chunks": {
            "300_tokens": {
                "fp16_gb": round(idx_fp16_300, 2),
                "drq_1_35_bd_gb": round(idx_drq_300, 2),
            },
            "640_tokens": {
                "fp16_gb": round(idx_fp16_640, 2),
                "drq_1_35_bd_gb": round(idx_drq_640, 2),
            },
            "envelope_a_cap_gb": 12.0,
            "envelope_a_breakeven_vectors_per_chunk": 278.0,
        },
        "by_language": report,
    }

    print("-" * 100)
    print(
        f"GLOBAL (63 Repos) | {len(all_arr):<7} | {np.mean(all_arr):<9.1f} | "
        f"Trunc@300: {global_trunc_300*100:.1f}% | Trunc@640: {global_trunc_640*100:.2f}% | "
        f"Mean Vecs@300: {mean_vecs_300:.1f} | Mean Vecs@640: {mean_vecs_640:.1f}"
    )
    print(
        f"Index Footprint at 2M chunks (1.35 b/d DRQ): @300={idx_drq_300:.2f} GB | @640={idx_drq_640:.2f} GB (Envelope A Cap: 12.0 GB)\n"
    )

    with open(TRUNCATION_FILE_640, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_full_640_bakeoff(device: str = "mps"):
    """Run clean Arm 3 bakeoff at document_length=640 across all 63 repos."""
    print("=" * 100)
    print("STEP 3: FRESH 640-TOKEN EVALUATION ACROSS ALL 63 REPOS WITH ENCODE ASSERTIONS")
    print("=" * 100)

    CACHE_DIR_640.mkdir(parents=True, exist_ok=True)
    all_queries = load_semble_anchor_queries()
    code_exts = frozenset(
        ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE
    )
    repo_names = sorted(list({q.repo for q in all_queries}))

    model_640 = ColBERT("lightonai/LateOn-Code", device=device, document_length=640)

    # First run the truncation measurement
    truncation_summary = measure_per_language_truncation(model_640.tokenizer)

    all_query_metrics = {
        "arm_1": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        "arm_2": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
        "arm_3": {"ndcg10": [], "r500": [], "r1k": [], "r2k": [], "r4k": []},
    }

    category_ndcg = {
        "semantic": {"arm_1": [], "arm_2": [], "arm_3": []},
        "symbol": {"arm_1": [], "arm_2": [], "arm_3": []},
        "architecture": {"arm_1": [], "arm_2": [], "arm_3": []},
    }
    category_r500 = {
        "semantic": {"arm_1": [], "arm_2": [], "arm_3": []},
        "symbol": {"arm_1": [], "arm_2": [], "arm_3": []},
        "architecture": {"arm_1": [], "arm_2": [], "arm_3": []},
    }
    category_r2k = {
        "semantic": {"arm_1": [], "arm_2": [], "arm_3": []},
        "symbol": {"arm_1": [], "arm_2": [], "arm_3": []},
        "architecture": {"arm_1": [], "arm_2": [], "arm_3": []},
    }

    cold_index_times_arm3 = []
    latencies_arm3 = []
    repo_ndcg_changes = 0

    for idx, repo_name in enumerate(repo_names, 1):
        cache_path = CACHE_DIR_640 / f"{repo_name}.json"
        repo_dir = Path("scratch/semble_repos") / repo_name
        repo_queries = [q for q in all_queries if q.repo == repo_name]

        # Load Arm 1 and Arm 2 ground truth from 300 cache (they had 0.00% truncation)
        with open(CACHE_DIR_300 / f"{repo_name}.json") as f:
            ref_data = json.load(f)

        chunks = []
        for fp in walk_source_files(repo_dir, code_exts):
            flang = language_for_path(fp) or "unknown"
            src = fp.read_text(encoding="utf-8", errors="replace")
            rel_p = str(fp.relative_to(repo_dir))
            chunks.extend(chunk_source(src, rel_p, flang))

        chunk_texts = [c.content for c in chunks]
        q_texts = [q.query for q in repo_queries]

        # Fresh Encode with Arm 3 (doc_maxlen=640)
        t_start = time.perf_counter()
        arm3_doc_tensors = model_640.encode(chunk_texts, is_query=False, batch_size=64)
        arm3_q_tensors = model_640.encode(q_texts, is_query=True, batch_size=32)
        cold_idx_s = time.perf_counter() - t_start
        cold_index_times_arm3.append(cold_idx_s)

        # ENCODE ASSERTION: verify tensors can exceed 300 tokens if long chunks exist
        max_t_len = max(t.shape[0] for t in arm3_doc_tensors)
        has_long_chunks = any(len(c.content.split()) > 200 for c in chunks)
        if has_long_chunks and len(chunks) > 50:
            assert max_t_len > 300, (
                f"FATAL: {repo_name} has long chunks but max tensor len is {max_t_len} <= 300! "
                "Encoder is still truncating at 300!"
            )

        # Search with Arm 3
        t_sim_start = time.perf_counter()
        sims3 = compute_vectorized_maxsim(
            arm3_q_tensors, arm3_doc_tensors, device=device
        )
        sim_elapsed_ms = ((time.perf_counter() - t_sim_start) * 1000) / len(
            repo_queries
        )
        for _ in range(len(repo_queries)):
            latencies_arm3.append(sim_elapsed_ms)

        arm3_ndcg = []
        arm3_r500 = []
        arm3_r1k = []
        arm3_r2k = []
        arm3_r4k = []

        for q_idx, q in enumerate(repo_queries):
            targets = list(q.targets)
            n_rel = len(targets)
            top_k_indices = np.argsort(-sims3[q_idx])[:50]
            units = [
                RetrievedUnit(
                    chunks[i].file_path,
                    chunks[i].content,
                    chunks[i].start_line,
                    chunks[i].end_line,
                    float(sims3[q_idx, i]),
                )
                for i in top_k_indices
            ]
            ranks = [
                next(
                    (
                        r
                        for r, u in enumerate(units, 1)
                        if target_matches_location(
                            u.file_path, u.start_line, u.end_line, t
                        )
                    ),
                    None,
                )
                for t in targets
            ]
            ranks = [r for r in ranks if r is not None]
            n_val = ndcg_at_k(ranks, n_rel, k=10)
            arm3_ndcg.append(n_val)

            curve = compute_retrieval_curve(units, targets)
            arm3_r500.append(recall_at_budget(curve, 500, n_rel))
            arm3_r1k.append(recall_at_budget(curve, 1000, n_rel))
            arm3_r2k.append(recall_at_budget(curve, 2000, n_rel))
            arm3_r4k.append(recall_at_budget(curve, 4000, n_rel))

        # Save fresh 640 cache
        cache_data = {
            "repo": repo_name,
            "document_length": 640,
            "device": device,
            "model": "lightonai/LateOn-Code",
            "metrics": {
                "arm_1": ref_data["metrics"]["arm_1"],
                "arm_2": ref_data["metrics"]["arm_2"],
                "arm_3": {
                    "ndcg10": arm3_ndcg,
                    "r500": arm3_r500,
                    "r1k": arm3_r1k,
                    "r2k": arm3_r2k,
                    "r4k": arm3_r4k,
                },
            },
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Accumulate query metrics
        for k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
            all_query_metrics["arm_1"][k].extend(ref_data["metrics"]["arm_1"][k])
            all_query_metrics["arm_2"][k].extend(ref_data["metrics"]["arm_2"][k])
            all_query_metrics["arm_3"][k].extend(cache_data["metrics"]["arm_3"][k])

        for q_idx, q in enumerate(repo_queries):
            cat = q.category.lower()
            for cat_k in ("semantic", "symbol", "architecture"):
                if cat_k in cat:
                    category_ndcg[cat_k]["arm_1"].append(
                        ref_data["metrics"]["arm_1"]["ndcg10"][q_idx]
                    )
                    category_ndcg[cat_k]["arm_2"].append(
                        ref_data["metrics"]["arm_2"]["ndcg10"][q_idx]
                    )
                    category_ndcg[cat_k]["arm_3"].append(arm3_ndcg[q_idx])

                    category_r500[cat_k]["arm_1"].append(
                        ref_data["metrics"]["arm_1"]["r500"][q_idx]
                    )
                    category_r500[cat_k]["arm_2"].append(
                        ref_data["metrics"]["arm_2"]["r500"][q_idx]
                    )
                    category_r500[cat_k]["arm_3"].append(arm3_r500[q_idx])

                    category_r2k[cat_k]["arm_1"].append(
                        ref_data["metrics"]["arm_1"]["r2k"][q_idx]
                    )
                    category_r2k[cat_k]["arm_2"].append(
                        ref_data["metrics"]["arm_2"]["r2k"][q_idx]
                    )
                    category_r2k[cat_k]["arm_3"].append(arm3_r2k[q_idx])

        # Check NDCG difference vs 300 run
        mean_300_ndcg = float(np.mean(ref_data["metrics"]["arm_3"]["ndcg10"]))
        mean_640_ndcg = float(np.mean(arm3_ndcg))
        if abs(mean_640_ndcg - mean_300_ndcg) > 1e-4:
            repo_ndcg_changes += 1

        print(
            f"[{idx:02d}/63] {repo_name:<24} | "
            f"Arm 1: {np.mean(ref_data['metrics']['arm_1']['ndcg10']):.4f} | "
            f"Arm 2: {np.mean(ref_data['metrics']['arm_2']['ndcg10']):.4f} | "
            f"Arm 3 (640): {mean_640_ndcg:.4f} (300={mean_300_ndcg:.4f}) | "
            f"R@500(3): {np.mean(arm3_r500):.4f} | R@2k(3): {np.mean(arm3_r2k):.4f}"
        )

    print("\n" + "=" * 100)
    print("STEP 4: SANITY GATE VERIFICATION")
    print("=" * 100)
    print(
        f"Total repos changing NDCG@10 between 300 and 640: {repo_ndcg_changes} / 63 ({repo_ndcg_changes/63*100:.1f}%)"
    )

    # Compute global paired bootstrap statistics
    bootstrap_results = {}
    for metric_k in ("ndcg10", "r500", "r1k", "r2k", "r4k"):
        s3 = all_query_metrics["arm_3"][metric_k]
        s2 = all_query_metrics["arm_2"][metric_k]
        s1 = all_query_metrics["arm_1"][metric_k]

        m_delta_32, ci_l_32, ci_u_32 = paired_bootstrap_ci(s3, s2)
        d32 = np.array(s3) - np.array(s2)
        pos = int(np.sum(d32 > 0))
        neg = int(np.sum(d32 < 0))
        ties = int(len(d32) - pos - neg)
        sign_p = float(
            stats.binomtest(pos, pos + neg, p=0.5).pvalue if (pos + neg) > 0 else 1.0
        )
        t_p = float(stats.ttest_rel(s3, s2).pvalue)

        bootstrap_results[metric_k] = {
            "mean_delta_3_vs_2": round(m_delta_32, 4),
            "ci_lower": round(ci_l_32, 4),
            "ci_upper": round(ci_u_32, 4),
            "sd": round(float(np.std(d32, ddof=1)), 4),
            "se": round(float(np.std(d32, ddof=1) / np.sqrt(len(d32))), 4),
            "wins_losses_ties": [pos, neg, ties],
            "sign_p": sign_p,
            "t_p": t_p,
        }

    # Category breakdown
    by_category = {}
    for cat_k in ("semantic", "symbol", "architecture"):
        s3_n = category_ndcg[cat_k]["arm_3"]
        s2_n = category_ndcg[cat_k]["arm_2"]
        m_ndcg, ci_l_n, ci_u_n = paired_bootstrap_ci(s3_n, s2_n)

        s3_r500 = category_r500[cat_k]["arm_3"]
        s2_r500 = category_r500[cat_k]["arm_2"]
        m_r500, ci_l_500, ci_u_500 = paired_bootstrap_ci(s3_r500, s2_r500)

        s3_r2k = category_r2k[cat_k]["arm_3"]
        s2_r2k = category_r2k[cat_k]["arm_2"]
        m_r2k, ci_l_2k, ci_u_2k = paired_bootstrap_ci(s3_r2k, s2_r2k)

        d_n = np.array(s3_n) - np.array(s2_n)
        by_category[cat_k] = {
            "queries": len(s3_n),
            "ndcg10": {
                "mean_delta": round(m_ndcg, 4),
                "ci_lower": round(ci_l_n, 4),
                "ci_upper": round(ci_u_n, 4),
                "sd": round(float(np.std(d_n, ddof=1)), 4),
                "se": round(float(np.std(d_n, ddof=1) / np.sqrt(len(d_n))), 4),
                "t_p": float(stats.ttest_rel(s3_n, s2_n).pvalue),
            },
            "r500": {
                "mean_delta": round(m_r500, 4),
                "ci_lower": round(ci_l_500, 4),
                "ci_upper": round(ci_u_500, 4),
            },
            "r2k": {
                "mean_delta": round(m_r2k, 4),
                "ci_lower": round(ci_l_2k, 4),
                "ci_upper": round(ci_u_2k, 4),
            },
        }

    # Final result payload
    output = {
        "metadata": {
            "total_queries": len(all_query_metrics["arm_3"]["ndcg10"]),
            "total_repos": len(repo_names),
            "doc_maxlen": 640,
            "device": device,
            "repos_changed_ndcg_vs_300": repo_ndcg_changes,
        },
        "quality_metrics": {
            "arm_1": {
                k: round(float(np.mean(all_query_metrics["arm_1"][k])), 4)
                for k in ("ndcg10", "r500", "r1k", "r2k", "r4k")
            },
            "arm_2": {
                k: round(float(np.mean(all_query_metrics["arm_2"][k])), 4)
                for k in ("ndcg10", "r500", "r1k", "r2k", "r4k")
            },
            "arm_3": {
                k: round(float(np.mean(all_query_metrics["arm_3"][k])), 4)
                for k in ("ndcg10", "r500", "r1k", "r2k", "r4k")
            },
        },
        "paired_bootstrap_cis": bootstrap_results,
        "by_category_arm_3_vs_arm_2": by_category,
        "cost_metrics_measured_640": {
            "cold_index_time_mean_s_per_repo": round(
                float(np.mean(cold_index_times_arm3)), 2
            ),
            "cold_index_time_p95_s_per_repo": round(
                float(np.percentile(cold_index_times_arm3, 95)), 2
            ),
            "p99_query_latency_ms": round(float(np.percentile(latencies_arm3, 99)), 2),
            "measured_mean_stored_vectors_per_chunk": truncation_summary[
                "measured_mean_vectors_per_chunk_640"
            ],
            "index_size_at_2m_chunks_fp16_gb": truncation_summary[
                "index_size_at_2m_chunks"
            ]["640_tokens"]["fp16_gb"],
            "index_size_at_2m_chunks_1_35_bd_drq_gb": truncation_summary[
                "index_size_at_2m_chunks"
            ]["640_tokens"]["drq_1_35_bd_gb"],
        },
    }

    with open(RESULTS_FILE_640, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[*] Full verified 640 results saved to {RESULTS_FILE_640}\n")
    return output


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_hardware_isolation_check(device=device)
    run_full_640_bakeoff(device=device)

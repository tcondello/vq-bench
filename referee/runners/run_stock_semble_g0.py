"""G0 Benchmark Runner comparing Stock Installed Semble (Arm A) vs Semble Reimpl (Arm B).

Evaluates all 1,251 queries across 63 repositories under both:
1. n_relevant = len(task.relevant)
2. n_relevant = len(task.relevant) + len(task.secondary)

Produces the exact numbers for G0-1 and G0-2.
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
from referee.data.loader import BenchmarkQuery, load_semble_anchor_queries
from referee.engines.semble_reference import SembleReferenceRetriever
from referee.metrics import (
    RetrievedUnit,
    TargetLocation,
    compute_retrieval_curve,
    ndcg_at_k,
    recall_at_budget,
    target_matches_location,
)
from semble import SembleIndex

TOKEN_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def run_g0_comparison(
    repos_base_dir: str | Path = "scratch/semble_repos",
    repos_json_path: str | Path = "scratch/semble_benchmarks/repos.json",
    annotations_dir: str | Path = "scratch/semble_benchmarks/annotations",
    output_json: str | Path = "results/g0_stock_vs_reimpl.json",
) -> dict[str, Any]:
    print("=" * 140)
    print(" RUNNING G0 COMPARISON: STOCK SEMBLE (ARM A) vs SEMBLE REIMPL (ARM B) ACROSS 63 REPOS (1,251 QUERIES)")
    print("=" * 140)

    repos_base = Path(repos_base_dir).resolve()
    with open(repos_json_path, "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    # Load raw annotations to have exact separation of relevant vs secondary
    annot_dir = Path(annotations_dir).resolve()
    repo_tasks: dict[str, list[dict[str, Any]]] = {}
    for af in sorted(annot_dir.glob("*.json")):
        repo_tasks[af.stem] = json.load(af.open(encoding="utf-8"))

    # Initialise Arm B retriever
    retriever_b = SembleReferenceRetriever()

    # Metrics storage: arm -> denominator_mode -> list of values
    # denominator_mode: "primary_only" (|relevant|) vs "primary_and_secondary" (|relevant| + |secondary|)
    ndcg_results: dict[str, dict[str, list[float]]] = {
        "arm_a_stock": {"primary_only": [], "primary_and_secondary": []},
        "arm_b_reimpl": {"primary_only": [], "primary_and_secondary": []},
    }
    rec2k_results: dict[str, dict[str, list[float]]] = {
        "arm_a_stock": {"primary_only": [], "primary_and_secondary": []},
        "arm_b_reimpl": {"primary_only": [], "primary_and_secondary": []},
    }
    budget_recalls: dict[str, dict[int, list[float]]] = {
        "arm_a_stock": {b: [] for b in TOKEN_BUDGETS},
        "arm_b_reimpl": {b: [] for b in TOKEN_BUDGETS},
    }

    t0_global = time.perf_counter()

    for r_idx, (repo_name, tasks) in enumerate(sorted(repo_tasks.items()), 1):
        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")

        repo_dir = repos_base / repo_name
        bench_dir = repo_dir / sub_root if sub_root else repo_dir

        # 1. Index Arm A (Stock Semble)
        t_a_start = time.perf_counter()
        try:
            index_a = SembleIndex.from_path(bench_dir)
            t_a_index = (time.perf_counter() - t_a_start) * 1000
        except Exception as e:
            print(f"[-] Arm A failed to index {repo_name}: {e}")
            index_a = None
            t_a_index = 0.0

        # 2. Index Arm B (Semble Reimpl)
        t_b_start = time.perf_counter()
        stats_b = retriever_b.index_directory(
            directory_path=bench_dir,
            language=lang,
            repo_name=repo_name,
            display_root=repo_dir,
        )
        t_b_index = (time.perf_counter() - t_b_start) * 1000

        for task in tasks:
            query = task["query"]
            rel_targets = [
                TargetLocation(
                    path=str(t["path"]) if isinstance(t, dict) else str(t),
                    start_line=int(t["start_line"]) if isinstance(t, dict) and t.get("start_line") is not None else None,
                    end_line=int(t["end_line"]) if isinstance(t, dict) and t.get("end_line") is not None else None,
                )
                for t in task.get("relevant", [])
            ]
            sec_targets = [
                TargetLocation(
                    path=str(t["path"]) if isinstance(t, dict) else str(t),
                    start_line=int(t["start_line"]) if isinstance(t, dict) and t.get("start_line") is not None else None,
                    end_line=int(t["end_line"]) if isinstance(t, dict) and t.get("end_line") is not None else None,
                )
                for t in task.get("secondary", [])
            ]
            all_targets = rel_targets + sec_targets
            n_prim = len(rel_targets)
            n_all = len(all_targets)

            # --- Evaluate Arm A (Stock) ---
            if index_a is not None:
                search_res_a = index_a.search(query, top_k=50)
                units_a = [
                    RetrievedUnit(
                        file_path=r.chunk.file_path,
                        content=r.chunk.content,
                        start_line=r.chunk.start_line,
                        end_line=r.chunk.end_line,
                        score=r.score,
                    )
                    for r in search_res_a
                ]
            else:
                units_a = []

            # Determine ranks for Arm A
            ranks_prim_a = []
            for t in rel_targets:
                for rank_idx, u in enumerate(units_a, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks_prim_a.append(rank_idx)
                        break

            ranks_all_a = []
            for t in all_targets:
                for rank_idx, u in enumerate(units_a, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks_all_a.append(rank_idx)
                        break

            # NDCG for Arm A
            ndcg_prim_a = ndcg_at_k(ranks_prim_a, n_prim, k=10) if n_prim > 0 else 0.0
            ndcg_all_a = ndcg_at_k(ranks_all_a, n_all, k=10) if n_all > 0 else 0.0
            ndcg_results["arm_a_stock"]["primary_only"].append(ndcg_prim_a)
            ndcg_results["arm_a_stock"]["primary_and_secondary"].append(ndcg_all_a)

            # Token Budget Recall for Arm A
            curve_prim_a = compute_retrieval_curve(units_a, rel_targets, tokenizer_name="cl100k_base")
            curve_all_a = compute_retrieval_curve(units_a, all_targets, tokenizer_name="cl100k_base")
            rec2k_results["arm_a_stock"]["primary_only"].append(recall_at_budget(curve_prim_a, 2000, n_prim) if n_prim > 0 else 0.0)
            rec2k_results["arm_a_stock"]["primary_and_secondary"].append(recall_at_budget(curve_all_a, 2000, n_all) if n_all > 0 else 0.0)
            for b in TOKEN_BUDGETS:
                budget_recalls["arm_a_stock"][b].append(recall_at_budget(curve_all_a, b, n_all) if n_all > 0 else 0.0)

            # --- Evaluate Arm B (Reimpl) ---
            units_b = retriever_b.search(query, top_k=50)

            ranks_prim_b = []
            for t in rel_targets:
                for rank_idx, u in enumerate(units_b, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks_prim_b.append(rank_idx)
                        break

            ranks_all_b = []
            for t in all_targets:
                for rank_idx, u in enumerate(units_b, 1):
                    if target_matches_location(u.file_path, u.start_line, u.end_line, t):
                        ranks_all_b.append(rank_idx)
                        break

            ndcg_prim_b = ndcg_at_k(ranks_prim_b, n_prim, k=10) if n_prim > 0 else 0.0
            ndcg_all_b = ndcg_at_k(ranks_all_b, n_all, k=10) if n_all > 0 else 0.0
            ndcg_results["arm_b_reimpl"]["primary_only"].append(ndcg_prim_b)
            ndcg_results["arm_b_reimpl"]["primary_and_secondary"].append(ndcg_all_b)

            curve_prim_b = compute_retrieval_curve(units_b, rel_targets, tokenizer_name="cl100k_base")
            curve_all_b = compute_retrieval_curve(units_b, all_targets, tokenizer_name="cl100k_base")
            rec2k_results["arm_b_reimpl"]["primary_only"].append(recall_at_budget(curve_prim_b, 2000, n_prim) if n_prim > 0 else 0.0)
            rec2k_results["arm_b_reimpl"]["primary_and_secondary"].append(recall_at_budget(curve_all_b, 2000, n_all) if n_all > 0 else 0.0)
            for b in TOKEN_BUDGETS:
                budget_recalls["arm_b_reimpl"][b].append(recall_at_budget(curve_all_b, b, n_all) if n_all > 0 else 0.0)

        mean_ndcg_a = np.mean(ndcg_results["arm_a_stock"]["primary_and_secondary"][-len(tasks):])
        mean_ndcg_b = np.mean(ndcg_results["arm_b_reimpl"]["primary_and_secondary"][-len(tasks):])
        print(f"[{r_idx:02d}/63] {repo_name:<20} | Arm A (Stock) NDCG: {mean_ndcg_a:.4f} | Arm B (Reimpl) NDCG: {mean_ndcg_b:.4f}")

    t_wall = time.perf_counter() - t0_global

    # Compute Final Summary
    a_ndcg_all = float(np.mean(ndcg_results["arm_a_stock"]["primary_and_secondary"]))
    a_ndcg_prim = float(np.mean(ndcg_results["arm_a_stock"]["primary_only"]))
    a_rec2k_all = float(np.mean(rec2k_results["arm_a_stock"]["primary_and_secondary"]))
    a_rec2k_prim = float(np.mean(rec2k_results["arm_a_stock"]["primary_only"]))

    b_ndcg_all = float(np.mean(ndcg_results["arm_b_reimpl"]["primary_and_secondary"]))
    b_ndcg_prim = float(np.mean(ndcg_results["arm_b_reimpl"]["primary_only"]))
    b_rec2k_all = float(np.mean(rec2k_results["arm_b_reimpl"]["primary_and_secondary"]))
    b_rec2k_prim = float(np.mean(rec2k_results["arm_b_reimpl"]["primary_only"]))

    summary = {
        "metadata": {
            "total_queries": len(ndcg_results["arm_a_stock"]["primary_and_secondary"]),
            "total_repos": len(repo_tasks),
            "wall_clock_s": round(t_wall, 2),
        },
        "published_figures": {
            "ndcg10": 0.854,
            "recall_2k": 0.938,
        },
        "arm_a_stock_semble": {
            "ndcg10_with_all_targets": round(a_ndcg_all, 4),
            "ndcg10_with_primary_only": round(a_ndcg_prim, 4),
            "recall2k_with_all_targets": round(a_rec2k_all, 4),
            "recall2k_with_primary_only": round(a_rec2k_prim, 4),
            "recall_at_token_budgets": {str(b): round(float(np.mean(budget_recalls["arm_a_stock"][b])), 4) for b in TOKEN_BUDGETS},
        },
        "arm_b_semble_reimpl": {
            "ndcg10_with_all_targets": round(b_ndcg_all, 4),
            "ndcg10_with_primary_only": round(b_ndcg_prim, 4),
            "recall2k_with_all_targets": round(b_rec2k_all, 4),
            "recall2k_with_primary_only": round(b_rec2k_prim, 4),
            "recall_at_token_budgets": {str(b): round(float(np.mean(budget_recalls["arm_b_reimpl"][b])), 4) for b in TOKEN_BUDGETS},
        },
        "deltas": {
            "arm_a_minus_published_ndcg": round(a_ndcg_all - 0.854, 4),
            "arm_a_minus_published_recall2k": round(a_rec2k_all - 0.938, 4),
            "arm_b_minus_arm_a_ndcg": round(b_ndcg_all - a_ndcg_all, 4),
            "arm_b_minus_arm_a_recall2k": round(b_rec2k_all - a_rec2k_all, 4),
        },
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[*] Saved full G0 comparison results to {out_file}")

    print("\n" + "=" * 140)
    print(" G0 EXIT COMPARISON SCORECARD")
    print("=" * 140)
    print(f"Arm A (Stock Semble)   | NDCG@10 (|rel|+|sec|): {a_ndcg_all:.4f} | NDCG@10 (|rel| only): {a_ndcg_prim:.4f} | Recall@2k: {a_rec2k_all:.4f}")
    print(f"Arm B (Semble Reimpl)  | NDCG@10 (|rel|+|sec|): {b_ndcg_all:.4f} | NDCG@10 (|rel| only): {b_ndcg_prim:.4f} | Recall@2k: {b_rec2k_all:.4f}")
    print(f"Published (Semble)     | NDCG@10:               0.8540 |                         --   | Recall@2k: 0.9380")
    print("-" * 140)
    print(f"Delta (A - P) NDCG: {a_ndcg_all - 0.854:+.4f} | Recall@2k: {a_rec2k_all - 0.938:+.4f}")
    print(f"Delta (B - A) NDCG: {b_ndcg_all - a_ndcg_all:+.4f} | Recall@2k: {b_rec2k_all - a_rec2k_all:+.4f}")
    print("=" * 140)

    return summary


if __name__ == "__main__":
    run_g0_comparison()

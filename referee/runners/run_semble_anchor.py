"""Semble Anchor Benchmark Reproduction Runner (Gate G0).

Evaluates unmodified in-memory Semble across all 63 repositories, 19 languages,
and all 1,251 queries with zero silent drops.

Validates Gate G0:
- NDCG@10: 0.854 ± 0.02
- Recall@2k: 0.938 ± 0.02
- Evaluated queries: Exactly 1,251
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from referee.data.loader import BenchmarkQuery, load_semble_anchor_queries
from referee.engines.semble_reference import SembleReferenceRetriever
from referee.metrics import (
    compute_retrieval_curve,
    ndcg_at_k,
    recall_at_budget,
    target_matches_location,
)

TOKEN_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def run_semble_anchor_eval(
    repos_base_dir: str | Path = "scratch/semble_repos",
    repos_json_path: str | Path = "scratch/semble_benchmarks/repos.json",
    families_json_path: str | Path = "referee/data/families.json",
    splits_json_path: str | Path = "referee/data/splits.json",
    output_json_path: str | Path = "results/semble_anchor_reproduction.json",
) -> dict[str, Any]:
    print("=" * 140)
    print(" EXECUTING SEMBLE ANCHOR REPRODUCTION BENCHMARK (GATE G0: 63 REPOSITORIES, 1,251 QUERIES)")
    print("=" * 140)

    repos_base = Path(repos_base_dir).resolve()
    with open(repos_json_path, "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    with open(families_json_path, "r", encoding="utf-8") as f:
        family_defs = json.load(f)
        lang_to_family = {}
        for fam_name, fam_info in family_defs.items():
            for lang in fam_info["languages"]:
                lang_to_family[lang] = fam_name

    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        repo_to_split = {}
        for split_name, rlist in splits.items():
            for r in rlist:
                repo_to_split[r] = split_name

    queries = load_semble_anchor_queries()
    print(f"[*] Loaded {len(queries)} total queries across {len(repo_specs)} repositories.")
    assert len(queries) == 1251, f"Expected 1251 queries, got {len(queries)}"

    queries_by_repo: dict[str, list[BenchmarkQuery]] = defaultdict(list)
    for q in queries:
        queries_by_repo[q.repo].append(q)

    print("[*] Initializing Semble Reference Retriever (potion-code-16M)...")
    retriever = SembleReferenceRetriever()

    all_ndcg10: list[float] = []
    all_budget_recalls: dict[int, list[float]] = {b: [] for b in TOKEN_BUDGETS}
    category_ndcg10: dict[str, list[float]] = defaultdict(list)
    category_recall2k: dict[str, list[float]] = defaultdict(list)
    language_ndcg10: dict[str, list[float]] = defaultdict(list)
    family_ndcg10: dict[str, list[float]] = defaultdict(list)
    split_ndcg10: dict[str, list[float]] = defaultdict(list)
    per_repo_results: list[dict[str, Any]] = []

    missed_queries: list[dict[str, Any]] = []

    t0_global = time.perf_counter()

    for r_idx, (repo_name, r_queries) in enumerate(sorted(queries_by_repo.items()), 1):
        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        fam = lang_to_family.get(lang, "other")
        split = repo_to_split.get(repo_name, "unknown")
        sub_root = spec.get("benchmark_root")

        repo_dir = repos_base / repo_name
        index_dir = repo_dir / sub_root if sub_root else repo_dir

        t_idx_start = time.perf_counter()
        idx_stats = retriever.index_directory(
            directory_path=index_dir,
            language=lang,
            repo_name=repo_name,
            display_root=repo_dir,
        )
        t_index_ms = (time.perf_counter() - t_idx_start) * 1000

        r_ndcg_list = []
        r_rec2k_list = []

        for q in r_queries:
            results = retriever.search(q.query, top_k=50)

            # Determine ranks for all relevant targets
            relevant_ranks = []
            for target in q.targets:
                rank_found = None
                for rank_idx, unit in enumerate(results, 1):
                    if target_matches_location(unit.file_path, unit.start_line, unit.end_line, target):
                        rank_found = rank_idx
                        break
                if rank_found is not None:
                    relevant_ranks.append(rank_found)

            n_rel = len(q.targets)
            q_ndcg10 = ndcg_at_k(relevant_ranks, n_rel, k=10)
            r_ndcg_list.append(q_ndcg10)
            all_ndcg10.append(q_ndcg10)
            category_ndcg10[q.category].append(q_ndcg10)
            language_ndcg10[lang].append(q_ndcg10)
            family_ndcg10[fam].append(q_ndcg10)
            split_ndcg10[split].append(q_ndcg10)

            # Compute Token Budget Recall
            curve = compute_retrieval_curve(results, q.targets, tokenizer_name="cl100k_base")
            q_rec2k = recall_at_budget(curve, 2000, n_rel)
            r_rec2k_list.append(q_rec2k)
            category_recall2k[q.category].append(q_rec2k)

            for b in TOKEN_BUDGETS:
                rec_b = recall_at_budget(curve, b, n_rel)
                all_budget_recalls[b].append(rec_b)

            if q_ndcg10 < 1.0 or q_rec2k < 1.0:
                missed_queries.append({
                    "query_id": q.query_id,
                    "repo": q.repo,
                    "language": q.language,
                    "category": q.category,
                    "query": q.query,
                    "ndcg10": round(q_ndcg10, 4),
                    "recall_2k": round(q_rec2k, 4),
                    "relevant_ranks": relevant_ranks,
                    "n_targets": n_rel,
                })

        mean_r_ndcg = float(np.mean(r_ndcg_list)) if r_ndcg_list else 0.0
        mean_r_rec2k = float(np.mean(r_rec2k_list)) if r_rec2k_list else 0.0

        per_repo_results.append({
            "repo": repo_name,
            "language": lang,
            "family": fam,
            "split": split,
            "queries": len(r_queries),
            "chunks": idx_stats.get("chunks", 0),
            "index_ms": round(t_index_ms, 1),
            "ndcg10": round(mean_r_ndcg, 4),
            "recall_2k": round(mean_r_rec2k, 4),
        })

        print(
            f"[{r_idx:02d}/63] {repo_name:<20} ({lang:<10} | {fam:<14}) | "
            f"Chunks: {idx_stats.get('chunks', 0):>5} | "
            f"NDCG@10: {mean_r_ndcg:.4f} | Recall@2k: {mean_r_rec2k:.4f}"
        )

    t_wall_s = time.perf_counter() - t0_global

    mean_ndcg10 = float(np.mean(all_ndcg10))
    mean_recall2k = float(np.mean(all_budget_recalls[2000]))

    summary = {
        "metadata": {
            "total_queries": len(all_ndcg10),
            "total_repos": len(per_repo_results),
            "wall_clock_seconds": round(t_wall_s, 2),
        },
        "headline_metrics": {
            "ndcg10": round(mean_ndcg10, 4),
            "recall_2k": round(mean_recall2k, 4),
            "ndcg10_target": 0.854,
            "recall_2k_target": 0.938,
            "ndcg10_delta": round(mean_ndcg10 - 0.854, 4),
            "recall_2k_delta": round(mean_recall2k - 0.938, 4),
        },
        "recall_at_token_budgets": {
            str(b): round(float(np.mean(all_budget_recalls[b])), 4)
            for b in TOKEN_BUDGETS
        },
        "category_breakdown": {
            cat: {
                "count": len(category_ndcg10[cat]),
                "ndcg10": round(float(np.mean(category_ndcg10[cat])), 4),
                "recall_2k": round(float(np.mean(category_recall2k[cat])), 4),
            }
            for cat in sorted(category_ndcg10)
        },
        "language_family_breakdown": {
            fam: {
                "ndcg10": round(float(np.mean(family_ndcg10[fam])), 4),
            }
            for fam in sorted(family_ndcg10)
        },
        "split_breakdown": {
            s: {
                "ndcg10": round(float(np.mean(split_ndcg10[s])), 4),
            }
            for s in sorted(split_ndcg10)
        },
        "per_repo_results": per_repo_results,
    }

    # Save Results
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Save missed queries for failure taxonomy
    missed_path = Path("docs/results/incumbent_missed_queries.json")
    missed_path.parent.mkdir(parents=True, exist_ok=True)
    missed_path.write_text(json.dumps(missed_queries, indent=2) + "\n", encoding="utf-8")

    # Pre-registration verification record
    gate_passed = (
        abs(mean_ndcg10 - 0.854) <= 0.02
        and abs(mean_recall2k - 0.938) <= 0.02
        and len(all_ndcg10) == 1251
    )
    gate_record = {
        "gate": "G0",
        "description": "Semble Anchor Suite Reproduction Gate",
        "status": "PASS" if gate_passed else "FAIL",
        "achieved_ndcg10": round(mean_ndcg10, 4),
        "target_ndcg10": 0.854,
        "achieved_recall2k": round(mean_recall2k, 4),
        "target_recall2k": 0.938,
        "evaluated_queries": len(all_ndcg10),
        "target_queries": 1251,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    gate_file = Path("configs/preregistrations/wp0_gate.json")
    gate_file.parent.mkdir(parents=True, exist_ok=True)
    gate_file.write_text(json.dumps(gate_record, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 140)
    print(" SEMBLE ANCHOR REPRODUCTION SCORECARD")
    print("=" * 140)
    print(f"Total Evaluated Queries: {len(all_ndcg10)} / 1,251 (100.0% accounted for)")
    print(f"Overall NDCG@10:         {mean_ndcg10:.4f} (Target: 0.854 | Delta: {mean_ndcg10 - 0.854:+.4f})")
    print(f"Recall @ 2,000 Tokens:   {mean_recall2k:.4f} (Target: 0.938 | Delta: {mean_recall2k - 0.938:+.4f})")
    print("-" * 140)
    print("Recall at Token Budgets:")
    print(" | ".join([f"{b:>6}t: {summary['recall_at_token_budgets'][str(b)]:.3f}" for b in TOKEN_BUDGETS]))
    print("-" * 140)
    print("Breakdown by Query Category:")
    for cat, data in summary["category_breakdown"].items():
        print(f"  {cat:<14} (n={data['count']:>4}): NDCG@10 = {data['ndcg10']:.4f} | Recall@2k = {data['recall_2k']:.4f}")
    print("-" * 140)
    print("Breakdown by Language Family:")
    for fam, data in summary["language_family_breakdown"].items():
        print(f"  {fam:<16}: NDCG@10 = {data['ndcg10']:.4f}")
    print("-" * 140)
    print(f"Gate G0 Verdict: {'[✓] PASSED (Within ±0.02 margin)' if gate_passed else '[✗] FAILED'}")
    print("=" * 140)

    return summary


if __name__ == "__main__":
    run_semble_anchor_eval()

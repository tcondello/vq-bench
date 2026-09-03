"""Run Semble Semantic Ablation (alpha=1.0, dense-only) across 63 repos / 1,251 queries."""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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

def run_semantic_ablation():
    print("=" * 110)
    print(" EXECUTING SEMBLE DENSE-ONLY SEMANTIC ABLATION (alpha=1.0, 63 REPOS, 1,251 QUERIES)")
    print("=" * 110)

    repos_base = Path("scratch/semble_repos").resolve()
    with open("scratch/semble_benchmarks/repos.json", "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    queries = load_semble_anchor_queries()
    queries_by_repo = defaultdict(list)
    for q in queries:
        queries_by_repo[q.repo].append(q)

    retriever = SembleReferenceRetriever()

    all_ndcg10 = []
    all_rec2k = []
    cat_ndcg = defaultdict(list)

    for r_idx, (repo_name, r_queries) in enumerate(sorted(queries_by_repo.items()), 1):
        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")
        repo_dir = repos_base / repo_name
        index_dir = repo_dir / sub_root if sub_root else repo_dir

        retriever.index_directory(
            directory_path=index_dir,
            language=lang,
            repo_name=repo_name,
            display_root=repo_dir,
        )

        r_ndcg = []
        for q in r_queries:
            results = retriever.search(q.query, top_k=50, alpha=1.0)
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
            curve = compute_retrieval_curve(results, q.targets, tokenizer_name="cl100k_base")
            q_rec2k = recall_at_budget(curve, 2000, n_rel)

            r_ndcg.append(q_ndcg10)
            all_ndcg10.append(q_ndcg10)
            all_rec2k.append(q_rec2k)
            cat_ndcg[q.category].append(q_ndcg10)

        print(f"[{r_idx:02d}/63] {repo_name:<20} | Semble Dense NDCG@10: {np.mean(r_ndcg):.4f}")

    mean_ndcg = float(np.mean(all_ndcg10))
    mean_rec2k = float(np.mean(all_rec2k))

    print("\n" + "=" * 110)
    print(" SEMBLE DENSE-ONLY (alpha=1.0) RESULTS")
    print("=" * 110)
    print(f"Overall NDCG@10: {mean_ndcg:.4f}")
    print(f"Overall Recall@2k: {mean_rec2k:.4f}")
    for cat in sorted(cat_ndcg.keys()):
        print(f"  {cat:<14}: NDCG@10 = {np.mean(cat_ndcg[cat]):.4f}")
    print("=" * 110)

    out_file = Path("results/semble_dense_semantic_ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "ndcg10": round(mean_ndcg, 4),
        "recall2k": round(mean_rec2k, 4),
        "categories": {cat: round(float(np.mean(cat_ndcg[cat])), 4) for cat in cat_ndcg},
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    run_semantic_ablation()

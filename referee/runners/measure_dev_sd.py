"""Measure paired per-query SD between encoders on the dev split.

Quantifies the empirical resolution floor and MDE across Semantic, Symbol,
and Architecture queries to calibrate WP1 prediction intervals.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from referee.data.loader import TargetLocation, load_semble_anchor_queries
from referee.engines.semble_reference import (
    FILE_TYPES,
    FileCategory,
    chunk_source,
    language_for_path,
    walk_source_files,
)
from referee.metrics import RetrievedUnit, ndcg_at_k, target_matches_location
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer

TOKEN_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def measure_dev_sd(
    repos_base_dir: str | Path = "scratch/semble_repos",
    repos_json_path: str | Path = "scratch/semble_benchmarks/repos.json",
    splits_json_path: str | Path = "referee/data/splits.json",
    output_json: str | Path = "results/wp1_dev_sd_calibration.json",
) -> dict[str, Any]:
    print("=" * 140)
    print(" MEASURING PAIRED PER-QUERY SD ON DEV SPLIT (ARM 1 POTION vs ARM 2 CODERANKEMBED)")
    print("=" * 140)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Compute Device: {device}")

    repos_base = Path(repos_base_dir).resolve()
    with open(repos_json_path, "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        dev_repos = set(splits.get("train_pool_51", []))

    all_queries = load_semble_anchor_queries()
    dev_queries = [q for q in all_queries if q.repo in dev_repos]
    print(f"[*] Loaded {len(dev_queries)} queries across {len(dev_repos)} dev repositories.")

    queries_by_repo = defaultdict(list)
    for q in dev_queries:
        queries_by_repo[q.repo].append(q)

    # Load Encoders
    print("[*] Loading Arm 1: StaticModel (MinishLab/potion-code-16M)...")
    potion_model = StaticModel.from_pretrained("MinishLab/potion-code-16M")

    print("[*] Loading Arm 2: SentenceTransformer (nomic-ai/CodeRankEmbed)...")
    coderank_model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True, device=device)

    deltas_by_cat: dict[str, list[float]] = defaultdict(list)
    arm1_ndcg_by_cat: dict[str, list[float]] = defaultdict(list)
    arm2_ndcg_by_cat: dict[str, list[float]] = defaultdict(list)

    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)

    t0_start = time.perf_counter()

    for r_idx, (repo_name, r_queries) in enumerate(sorted(queries_by_repo.items()), 1):
        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")

        repo_dir = repos_base / repo_name
        bench_dir = repo_dir / sub_root if sub_root else repo_dir

        # Chunk files
        chunks = []
        for fp in walk_source_files(bench_dir, code_exts):
            flang = language_for_path(fp) or lang
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
                rel_p = str(fp.relative_to(repo_dir))
                chunks.extend(chunk_source(src, rel_p, flang))
            except OSError:
                continue

        if not chunks:
            continue

        chunk_texts = [c.content for c in chunks]

        # 1. Arm 1: Potion Embeddings
        arm1_chunk_embs = potion_model.encode(chunk_texts)
        arm1_norms = np.linalg.norm(arm1_chunk_embs, axis=1, keepdims=True)
        arm1_norms[arm1_norms == 0] = 1.0
        arm1_chunk_embs = (arm1_chunk_embs / arm1_norms).astype(np.float32)

        # 2. Arm 2: CodeRankEmbed Embeddings
        doc_prompts = [f"search_document: {t}" for t in chunk_texts]
        with torch.no_grad():
            arm2_chunk_embs = coderank_model.encode(
                doc_prompts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        # Encode Queries
        q_texts = [q.query for q in r_queries]
        arm1_q_embs = potion_model.encode(q_texts)
        arm1_q_norms = np.linalg.norm(arm1_q_embs, axis=1, keepdims=True)
        arm1_q_norms[arm1_q_norms == 0] = 1.0
        arm1_q_embs = (arm1_q_embs / arm1_q_norms).astype(np.float32)

        q_prompts = [f"search_query: {t}" for t in q_texts]
        with torch.no_grad():
            arm2_q_embs = coderank_model.encode(
                q_prompts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        # Score per query
        for q_idx, q in enumerate(r_queries):
            # Arm 1 search
            sims1 = np.dot(arm1_chunk_embs, arm1_q_embs[q_idx])
            top_k1 = np.argsort(-sims1)[:50]
            ranks1 = []
            for t in q.targets:
                for rank_idx, c_idx in enumerate(top_k1, 1):
                    c = chunks[c_idx]
                    if target_matches_location(c.file_path, c.start_line, c.end_line, t):
                        ranks1.append(rank_idx)
                        break
            n_rel = len(q.targets)
            ndcg1 = ndcg_at_k(ranks1, n_rel, k=10)

            # Arm 2 search
            sims2 = (arm2_chunk_embs @ arm2_q_embs[q_idx]).cpu().numpy()
            top_k2 = np.argsort(-sims2)[:50]
            ranks2 = []
            for t in q.targets:
                for rank_idx, c_idx in enumerate(top_k2, 1):
                    c = chunks[c_idx]
                    if target_matches_location(c.file_path, c.start_line, c.end_line, t):
                        ranks2.append(rank_idx)
                        break
            ndcg2 = ndcg_at_k(ranks2, n_rel, k=10)

            delta = ndcg2 - ndcg1
            cat = q.category
            deltas_by_cat[cat].append(delta)
            arm1_ndcg_by_cat[cat].append(ndcg1)
            arm2_ndcg_by_cat[cat].append(ndcg2)

        print(f"[{r_idx:02d}/51] {repo_name:<20} | Chunks: {len(chunks):>5} | Processed {len(r_queries)} queries.")

    t_wall = time.perf_counter() - t0_start

    # Compute Statistics per Category
    results = {}
    print("\n" + "=" * 140)
    print(" EMPIRICAL PAIRED SD & MDE RESOLUTION TABLE (ARM 2 CodeRankEmbed vs ARM 1 Potion)")
    print("=" * 140)
    print(f"{'Category':<16} | {'n':>5} | {'Arm 1 (Dense)':>13} | {'Arm 2 (Dense)':>13} | {'Mean Delta':>11} | {'Paired SD':>10} | {'MDE (80% Power)':>16}")
    print("-" * 140)

    total_deltas = []
    for cat in sorted(deltas_by_cat):
        d_arr = np.array(deltas_by_cat[cat])
        total_deltas.extend(deltas_by_cat[cat])
        n = len(d_arr)
        mean_d = float(np.mean(d_arr))
        sd_d = float(np.std(d_arr, ddof=1))
        # MDE at 80% power, alpha=0.05 two-tailed: (z_0.025 + z_0.20) * sd / sqrt(n) = (1.96 + 0.84) * sd / sqrt(n) = 2.80 * sd / sqrt(n)
        mde = 2.80 * sd_d / np.sqrt(n)
        mean1 = float(np.mean(arm1_ndcg_by_cat[cat]))
        mean2 = float(np.mean(arm2_ndcg_by_cat[cat]))

        results[cat] = {
            "n": n,
            "arm1_dense_ndcg10": round(mean1, 4),
            "arm2_dense_ndcg10": round(mean2, 4),
            "mean_delta": round(mean_d, 4),
            "paired_sd": round(sd_d, 4),
            "mde_80_power": round(mde, 4),
        }
        print(f"{cat:<16} | {n:>5} | {mean1:13.4f} | {mean2:13.4f} | {mean_d:+11.4f} | {sd_d:10.4f} | {mde:16.4f}")

    all_d = np.array(total_deltas)
    overall_sd = float(np.std(all_d, ddof=1))
    overall_mde = 2.80 * overall_sd / np.sqrt(len(all_d))
    print("-" * 140)
    print(f"{'OVERALL DEV':<16} | {len(all_d):>5} | {np.mean([results[c]['arm1_dense_ndcg10'] for c in results]):13.4f} | {np.mean([results[c]['arm2_dense_ndcg10'] for c in results]):13.4f} | {np.mean(all_d):+11.4f} | {overall_sd:10.4f} | {overall_mde:16.4f}")
    print("=" * 140)

    results["overall"] = {
        "n": len(all_d),
        "mean_delta": round(float(np.mean(all_d)), 4),
        "paired_sd": round(overall_sd, 4),
        "mde_80_power": round(overall_mde, 4),
        "wall_clock_s": round(t_wall, 2),
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[*] Saved SD calibration record to {out_p}")

    return results


if __name__ == "__main__":
    measure_dev_sd()

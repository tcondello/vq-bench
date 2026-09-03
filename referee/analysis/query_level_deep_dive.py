"""Query-level statistical deep dive and Guzzle investigation for WP1 H1 bake-off."""

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy import stats
from referee.data.loader import load_semble_anchor_queries

def run_deep_dive():
    cache_dir = Path("scratch/wp1_cache")
    queries = load_semble_anchor_queries()
    queries_by_repo = defaultdict(list)
    for q in queries:
        queries_by_repo[q.repo].append(q)

    # Accumulate all query scores
    # query_data: list of dicts: { query_id, repo, category, query, arm1_ndcg, arm2_ndcg, arm3_ndcg, ... }
    all_query_data = []

    for repo_file in sorted(cache_dir.glob("*.json")):
        repo_name = repo_file.stem
        r_queries = queries_by_repo.get(repo_name, [])
        with open(repo_file, "r", encoding="utf-8") as f:
            cdata = json.load(f)

        m1 = cdata["metrics"]["arm_1"]["ndcg10"]
        m2 = cdata["metrics"]["arm_2"]["ndcg10"]
        m3 = cdata["metrics"]["arm_3"]["ndcg10"]

        assert len(m1) == len(r_queries), f"Mismatch in {repo_name}: {len(m1)} vs {len(r_queries)}"

        for q, s1, s2, s3 in zip(r_queries, m1, m2, m3):
            all_query_data.append({
                "query_id": q.query_id,
                "repo": repo_name,
                "category": q.category,
                "query": q.query,
                "arm1": s1,
                "arm2": s2,
                "arm3": s3,
                "delta_3_2": s3 - s2,
                "delta_2_1": s2 - s1,
                "delta_3_1": s3 - s1,
            })

    assert len(all_query_data) == 1251, f"Expected 1251 queries, got {len(all_query_data)}"

    # 1. Statistical calculations
    def compute_stats(q_list, name=""):
        d32 = np.array([q["delta_3_2"] for q in q_list])
        d21 = np.array([q["delta_2_1"] for q in q_list])
        s1 = np.array([q["arm1"] for q in q_list])
        s2 = np.array([q["arm2"] for q in q_list])
        s3 = np.array([q["arm3"] for q in q_list])
        n = len(q_list)

        # Bootstrap 95% CI for delta_3_2
        rng = np.random.default_rng(42)
        boot_means_32 = [np.mean(rng.choice(d32, size=n, replace=True)) for _ in range(10000)]
        ci_32 = (np.percentile(boot_means_32, 2.5), np.percentile(boot_means_32, 97.5))

        # Bootstrap 95% CI for delta_2_1
        boot_means_21 = [np.mean(rng.choice(d21, size=n, replace=True)) for _ in range(10000)]
        ci_21 = (np.percentile(boot_means_21, 2.5), np.percentile(boot_means_21, 97.5))

        sd_32 = np.std(d32, ddof=1)
        se_32 = sd_32 / np.sqrt(n)

        sd_21 = np.std(d21, ddof=1)
        se_21 = sd_21 / np.sqrt(n)

        # Sign test for 3 vs 2
        pos_32 = np.sum(d32 > 0)
        neg_32 = np.sum(d32 < 0)
        sign_p_32 = stats.binomtest(pos_32, pos_32 + neg_32, p=0.5).pvalue if (pos_32 + neg_32) > 0 else 1.0

        # Paired t-test
        t_stat_32, t_p_32 = stats.ttest_rel(s3, s2)
        t_stat_21, t_p_21 = stats.ttest_rel(s2, s1)

        return {
            "name": name,
            "n": n,
            "mean_arm1": float(np.mean(s1)),
            "mean_arm2": float(np.mean(s2)),
            "mean_arm3": float(np.mean(s3)),
            "delta_3_2": float(np.mean(d32)),
            "sd_32": float(sd_32),
            "se_32": float(se_32),
            "ci_32": (float(ci_32[0]), float(ci_32[1])),
            "wins_3_over_2": int(pos_32),
            "losses_3_over_2": int(neg_32),
            "ties_3_2": int(n - pos_32 - neg_32),
            "sign_p_32": float(sign_p_32),
            "t_p_32": float(t_p_32),
            "delta_2_1": float(np.mean(d21)),
            "sd_21": float(sd_21),
            "se_21": float(se_21),
            "ci_21": (float(ci_21[0]), float(ci_21[1])),
            "t_p_21": float(t_p_21),
        }

    overall_stats = compute_stats(all_query_data, "OVERALL (1,251 queries)")
    semantic_stats = compute_stats([q for q in all_query_data if q["category"] == "semantic"], "Semantic (703 queries)")
    symbol_stats = compute_stats([q for q in all_query_data if q["category"] == "symbol"], "Symbol (204 queries)")
    arch_stats = compute_stats([q for q in all_query_data if q["category"] == "architecture"], "Architecture (344 queries)")

    print("\n" + "=" * 110)
    print(" QUERY-LEVEL STATISTICAL DEEP DIVE (1,251 QUERIES)")
    print("=" * 110)
    for st in (overall_stats, semantic_stats, symbol_stats, arch_stats):
        print(f"--- {st['name']} ---")
        print(f"  Means: Arm 1={st['mean_arm1']:.4f} | Arm 2={st['mean_arm2']:.4f} | Arm 3={st['mean_arm3']:.4f}")
        print(f"  Arm 3 vs Arm 2: Delta = {st['delta_3_2']:+.4f} | SD(d) = {st['sd_32']:.4f} | SE = {st['se_32']:.4f}")
        print(f"  95% Bootstrap CI: [{st['ci_32'][0]:+.4f}, {st['ci_32'][1]:+.4f}]")
        print(f"  Wins/Losses/Ties (Arm 3 vs 2): {st['wins_3_over_2']} / {st['losses_3_over_2']} / {st['ties_3_2']} (Sign Test p = {st['sign_p_32']:.4e}, Paired t p = {st['t_p_32']:.4e})")
        print(f"  Arm 2 vs Arm 1: Delta = {st['delta_2_1']:+.4f} | SD(d) = {st['sd_21']:.4f} | SE = {st['se_21']:.4f} | 95% CI: [{st['ci_21'][0]:+.4f}, {st['ci_21'][1]:+.4f}] (t p = {st['t_p_21']:.4e})")
        print()

    # 2. Guzzle Investigation
    print("=" * 110)
    print(" GUZZLE INVESTIGATION (20 QUERIES)")
    print("=" * 110)
    guzzle_queries = [q for q in all_query_data if q["repo"] == "guzzle"]
    print(f"{'Query ID':<10} | {'Cat':<12} | {'Arm 1':>7} | {'Arm 2':>7} | {'Arm 3':>7} | {'Δ(2-1)':>7} | {'Δ(3-2)':>7} | Query")
    print("-" * 110)
    for q in guzzle_queries:
        print(f"{q['query_id']:<10} | {q['category']:<12} | {q['arm1']:>7.4f} | {q['arm2']:>7.4f} | {q['arm3']:>7.4f} | {q['delta_2_1']:>+7.4f} | {q['delta_3_2']:>+7.4f} | {q['query']}")
    print("-" * 110)
    m1_g = np.mean([q["arm1"] for q in guzzle_queries])
    m2_g = np.mean([q["arm2"] for q in guzzle_queries])
    m3_g = np.mean([q["arm3"] for q in guzzle_queries])
    print(f"Guzzle Means: Arm 1 = {m1_g:.4f} | Arm 2 = {m2_g:.4f} | Arm 3 = {m3_g:.4f} | Δ(2-1) = {m2_g - m1_g:+.4f} | Δ(3-2) = {m3_g - m2_g:+.4f}")
    print("=" * 110)

    # Save deep dive JSON
    out_file = Path("results/query_level_statistical_deep_dive.json")
    out_file.write_text(json.dumps({
        "overall": overall_stats,
        "semantic": semantic_stats,
        "symbol": symbol_stats,
        "architecture": arch_stats,
        "guzzle": {
            "mean_arm1": float(m1_g),
            "mean_arm2": float(m2_g),
            "mean_arm3": float(m3_g),
            "queries": guzzle_queries,
        }
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    run_deep_dive()

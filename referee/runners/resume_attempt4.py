import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from pathlib import Path
import numpy as np
import torch
from pylate.models import ColBERT

from referee.data.loader import load_semble_anchor_queries
from referee.engines.semble_reference import (
    FILE_TYPES, FileCategory, chunk_source, language_for_path, walk_source_files
)
from referee.metrics import (
    RetrievedUnit, compute_retrieval_curve, ndcg_at_k, recall_at_budget, target_matches_location
)

def compute_vectorized_maxsim_norm_q(q_embs: list[np.ndarray], d_embs: list[np.ndarray], device: str) -> np.ndarray:
    """
    Computes Query-Length Normalised MaxSim:
      Score(Q, D) = (1 / |Q|) * sum_{q in Q} max_{d in D} <q, d>
    """
    if not d_embs or not q_embs:
        return np.zeros((len(q_embs), len(d_embs)), dtype=np.float32)

    d_cat_np = np.concatenate(d_embs, axis=0)
    d_splits = [t.shape[0] for t in d_embs]
    split_indices = np.cumsum(d_splits)[:-1]

    d_cat = torch.from_numpy(d_cat_np).to(device=device, dtype=torch.float32)
    num_q = len(q_embs)
    num_d = len(d_splits)
    sim_matrix = np.zeros((num_q, num_d), dtype=np.float32)

    for qi, q_np in enumerate(q_embs):
        l_q = q_np.shape[0]
        q_t = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)
        dots_t = torch.matmul(q_t, d_cat.T)
        dots_np = dots_t.cpu().numpy()
        splits = np.split(dots_np, split_indices, axis=1)
        scores_q = np.empty(num_d, dtype=np.float32)
        for di, s in enumerate(splits):
            scores_q[di] = np.sum(np.max(s, axis=1)) / max(1, l_q)
        sim_matrix[qi] = scores_q

    del d_cat
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return sim_matrix

def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = a - b
    mean_delta = float(np.mean(deltas))
    n = len(deltas)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    p_val = float(2.0 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    return mean_delta, ci_low, ci_high, p_val

def resume_and_complete():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 100)
    print(f"RESUMING ATTEMPT 4: LENGTH-NORMALISED LATE INTERACTION ON {device.upper()}")
    print("=" * 100)
    
    cache_dir = Path("scratch/wp1_arm3_640_norm_len")
    cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache_dir = Path("scratch/wp1_cache")
    
    with open("scratch/semble_benchmarks/repos.json", "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}
    
    all_queries = load_semble_anchor_queries()
    all_repo_names = sorted(list({q.repo for q in all_queries}))
    
    # Sort remaining repos: process already cached first, then order remaining by chunk size (zig-clap, zls, zod, zig)
    cached_repos = [r for r in all_repo_names if (cache_dir / f"{r}.json").exists()]
    uncached_repos = [r for r in all_repo_names if not (cache_dir / f"{r}.json").exists()]
    uncached_order = ["zig-clap", "zls", "zod", "zig"]
    uncached_repos = sorted(uncached_repos, key=lambda r: uncached_order.index(r) if r in uncached_order else 99)
    
    ordered_repos = cached_repos + uncached_repos
    print(f"Total Repositories: {len(ordered_repos)} (Cached: {len(cached_repos)}, Remaining: {len(uncached_repos)})")
    
    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)
    
    model = None
    all_latencies = []
    
    for idx, repo_name in enumerate(ordered_repos, 1):
        cached_file = cache_dir / f"{repo_name}.json"
        baseline_file = baseline_cache_dir / f"{repo_name}.json"
        
        b_data = json.load(baseline_file.open()) if baseline_file.exists() else {}
        b_metrics = b_data.get("metrics", {})
        arm1_ndcg = float(np.mean(b_metrics.get("arm_1", {}).get("ndcg10", [0.0])))
        arm2_ndcg = float(np.mean(b_metrics.get("arm_2", {}).get("ndcg10", [0.0])))
        arm3_300 = float(np.mean(b_metrics.get("arm_3", {}).get("ndcg10", [0.0])))
        
        arm1_r500 = float(np.mean(b_metrics.get("arm_1", {}).get("r500", [0.0])))
        arm2_r500 = float(np.mean(b_metrics.get("arm_2", {}).get("r500", [0.0])))
        arm3_300_r500 = float(np.mean(b_metrics.get("arm_3", {}).get("r500", [0.0])))
        
        arm1_r2k = float(np.mean(b_metrics.get("arm_1", {}).get("r2k", [0.0])))
        arm2_r2k = float(np.mean(b_metrics.get("arm_2", {}).get("r2k", [0.0])))
        arm3_300_r2k = float(np.mean(b_metrics.get("arm_3", {}).get("r2k", [0.0])))
        
        if cached_file.exists():
            c_data = json.load(cached_file.open())
            c_data["arm1_ndcg"] = arm1_ndcg
            c_data["arm2_ndcg"] = arm2_ndcg
            c_data["arm3_ndcg_300"] = arm3_300
            c_data["arm1_r500"] = arm1_r500
            c_data["arm2_r500"] = arm2_r500
            c_data["arm3_300_r500"] = arm3_300_r500
            c_data["arm1_r2k"] = arm1_r2k
            c_data["arm2_r2k"] = arm2_r2k
            c_data["arm3_300_r2k"] = arm3_300_r2k
            cached_file.write_text(json.dumps(c_data, indent=2))
            print(f"[{idx:02d}/63] [CACHED] {repo_name:<20} | Arm 1: {arm1_ndcg:.4f} | Arm 2: {arm2_ndcg:.4f} | Arm 3(640 norm): {c_data['arm3_ndcg_640_norm']:.4f} (300={arm3_300:.4f}) | R@500: {c_data['arm3_r500']:.4f} | R@2k: {c_data['arm3_r2k']:.4f}", flush=True)
            continue
            
        if model is None:
            print("\n[*] Loading ColBERT model (lightonai/LateOn-Code, document_length=640)...", flush=True)
            model = ColBERT("lightonai/LateOn-Code", device=device, document_length=640)
            
        spec = repo_specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")
        
        repo_dir = Path("scratch/semble_repos") / repo_name
        bench_dir = repo_dir / sub_root if sub_root else repo_dir
        repo_queries = [q for q in all_queries if q.repo == repo_name]
        
        chunks = []
        for fp in walk_source_files(bench_dir, code_exts):
            flang = language_for_path(fp) or lang
            src = fp.read_text(encoding="utf-8", errors="replace")
            rel_p = str(fp.relative_to(repo_dir))
            chunks.extend(chunk_source(src, rel_p, flang))
            
        chunk_texts = [c.content for c in chunks]
        q_texts = [q.query for q in repo_queries]
        
        print(f"[*] Encoding {repo_name}: {len(q_texts)} queries, {len(chunk_texts)} chunks (batch_size=64)...", flush=True)
        t0_enc = time.perf_counter()
        q_np = model.encode(q_texts, is_query=True, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        d_np = model.encode(chunk_texts, is_query=False, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
        t_enc = time.perf_counter() - t0_enc
        print(f"[*] Encoded {len(chunk_texts)} chunks in {t_enc:.1f}s ({len(chunk_texts)/max(0.1, t_enc):.1f} chunks/s)", flush=True)
        
        max_doc_len = max(t.shape[0] for t in d_np) if d_np else 0
        if any(len(c.content.split()) > 120 for c in chunks):
            assert max_doc_len > 300, f"FATAL ERROR on {repo_name}: max token length is {max_doc_len} <= 300."
            
        t0_score = time.perf_counter()
        sim_matrix = compute_vectorized_maxsim_norm_q(q_np, d_np, device=device)
        score_time = (time.perf_counter() - t0_score) / max(1, len(repo_queries))
        
        repo_ndcgs, repo_r500s, repo_r1ks, repo_r2ks, repo_r4ks = [], [], [], [], []
        
        for q_idx, q in enumerate(repo_queries):
            t0_q = time.perf_counter()
            targets = list(q.targets)
            n_rel = len(targets)
            top_k = np.argsort(-sim_matrix[q_idx])[:50]
            units = [
                RetrievedUnit(chunks[i].file_path, chunks[i].content, chunks[i].start_line, chunks[i].end_line, float(sim_matrix[q_idx, i]))
                for i in top_k
            ]
            ranks = [next((r for r, u in enumerate(units, 1) if target_matches_location(u.file_path, u.start_line, u.end_line, t)), None) for t in targets]
            ranks = [rk for rk in ranks if rk is not None]
            
            ndcg_10 = ndcg_at_k(ranks, n_rel, k=10)
            curve = compute_retrieval_curve(units, targets)
            r500 = recall_at_budget(curve, 500, n_rel)
            r1k = recall_at_budget(curve, 1000, n_rel)
            r2k = recall_at_budget(curve, 2000, n_rel)
            r4k = recall_at_budget(curve, 4000, n_rel)
            lat_ms = (time.perf_counter() - t0_q + score_time) * 1000.0
            
            repo_ndcgs.append(ndcg_10)
            repo_r500s.append(r500)
            repo_r1ks.append(r1k)
            repo_r2ks.append(r2k)
            repo_r4ks.append(r4k)
            all_latencies.append(lat_ms)
            
        m_ndcg = float(np.mean(repo_ndcgs))
        m_r500 = float(np.mean(repo_r500s))
        m_r1k = float(np.mean(repo_r1ks))
        m_r2k = float(np.mean(repo_r2ks))
        m_r4k = float(np.mean(repo_r4ks))
        
        repo_res = {
            "repo": repo_name,
            "n_chunks": len(chunks),
            "max_doc_len": max_doc_len,
            "mean_vecs_per_chunk": float(np.mean([t.shape[0] for t in d_np])) if d_np else 0.0,
            "arm1_ndcg": arm1_ndcg,
            "arm2_ndcg": arm2_ndcg,
            "arm3_ndcg_300": arm3_300,
            "arm3_ndcg_640_norm": m_ndcg,
            "arm1_r500": arm1_r500,
            "arm2_r500": arm2_r500,
            "arm3_300_r500": arm3_300_r500,
            "arm3_r500": m_r500,
            "arm3_r1k": m_r1k,
            "arm1_r2k": arm1_r2k,
            "arm2_r2k": arm2_r2k,
            "arm3_300_r2k": arm3_300_r2k,
            "arm3_r2k": m_r2k,
            "arm3_r4k": m_r4k,
        }
        cached_file.write_text(json.dumps(repo_res, indent=2))
        print(f"[{idx:02d}/63] [FRESH ] {repo_name:<20} | Arm 1: {arm1_ndcg:.4f} | Arm 2: {arm2_ndcg:.4f} | Arm 3(640 norm): {m_ndcg:.4f} (300={arm3_300:.4f}) | R@500: {m_r500:.4f} | R@2k: {m_r2k:.4f}", flush=True)

    # Load all 63 finished repo files
    all_63_results = []
    total_stored_vectors = 0
    total_measured_chunks = 0
    for repo_name in sorted(all_repo_names):
        f = cache_dir / f"{repo_name}.json"
        data = json.load(f.open())
        all_63_results.append(data)
        total_stored_vectors += int(data["n_chunks"] * data["mean_vecs_per_chunk"])
        total_measured_chunks += data["n_chunks"]

    # Complete Aggregates
    print("\n" + "=" * 100)
    print("FINAL ATTEMPT 4 FULL 63-REPO AGGREGATE SUMMARY")
    print("=" * 100)
    
    changed_repos = sum(1 for r in all_63_results if abs(r["arm3_ndcg_640_norm"] - r["arm3_ndcg_300"]) > 1e-4)
    print(f"Sanity Gate: Changed Repositories vs 300-tok baseline: {changed_repos} / {len(all_63_results)}")
    
    r_a2_ndcg = np.array([r["arm2_ndcg"] for r in all_63_results])
    r_a3_ndcg = np.array([r["arm3_ndcg_640_norm"] for r in all_63_results])
    r_a1_ndcg = np.array([r["arm1_ndcg"] for r in all_63_results])
    
    r_a2_r500 = np.array([r["arm2_r500"] for r in all_63_results])
    r_a3_r500 = np.array([r["arm3_r500"] for r in all_63_results])
    r_a1_r500 = np.array([r["arm1_r500"] for r in all_63_results])
    
    r_a2_r2k = np.array([r["arm2_r2k"] for r in all_63_results])
    r_a3_r2k = np.array([r["arm3_r2k"] for r in all_63_results])
    r_a1_r2k = np.array([r["arm1_r2k"] for r in all_63_results])
    
    d_ndcg, ndcg_l, ndcg_h, ndcg_p = paired_bootstrap_ci(r_a3_ndcg, r_a2_ndcg)
    d_r500, r500_l, r500_h, r500_p = paired_bootstrap_ci(r_a3_r500, r_a2_r500)
    d_r2k, r2k_l, r2k_h, r2k_p = paired_bootstrap_ci(r_a3_r2k, r_a2_r2k)
    
    mean_vecs_global = total_stored_vectors / max(1, total_measured_chunks)
    index_size_2m_fp16 = (2_000_000 * mean_vecs_global * 128 * 2) / (1024**3)
    index_size_2m_drq = (2_000_000 * mean_vecs_global * 128 * (1.35 / 8.0)) / (1024**3)
    
    print("\n--- PRIMARY & SECONDARY METRICS (ARM 3 VS ARM 2, N=63 REPOS) ---")
    print(f"NDCG@10:    Arm 2 = {np.mean(r_a2_ndcg):.4f} | Arm 3 = {np.mean(r_a3_ndcg):.4f} | Δ = {d_ndcg:+.4f} [{ndcg_l:+.4f}, {ndcg_h:+.4f}] (p = {ndcg_p:.4f})")
    print(f"Recall@500: Arm 2 = {np.mean(r_a2_r500):.4f} | Arm 3 = {np.mean(r_a3_r500):.4f} | Δ = {d_r500:+.4f} [{r500_l:+.4f}, {r500_h:+.4f}] (p = {r500_p:.4f})")
    print(f"Recall@2k:  Arm 2 = {np.mean(r_a2_r2k):.4f} | Arm 3 = {np.mean(r_a3_r2k):.4f} | Δ = {d_r2k:+.4f} [{r2k_l:+.4f}, {r2k_h:+.4f}] (p = {r2k_p:.4f})")
    
    print("\n--- MEASURED RESOURCE FOOTPRINT & ENVELOPE A ---")
    print(f"Measured Mean Vectors / Chunk: {mean_vecs_global:.2f}")
    print(f"Index Size @ 2M chunks (fp16): {index_size_2m_fp16:.2f} GB")
    print(f"Index Size @ 2M chunks (1.35 b/d DRQ): {index_size_2m_drq:.2f} GB (Cap: 12.0 GB -> {'PASS' if index_size_2m_drq <= 12.0 else 'FAIL'})")
    
    final_payload = {
        "summary": {
            "n_repos": len(all_63_results),
            "changed_repos": changed_repos,
            "mean_vecs_per_chunk": mean_vecs_global,
            "index_size_2m_fp16_gb": index_size_2m_fp16,
            "index_size_2m_drq_gb": index_size_2m_drq,
            "arm1": {"ndcg_at_10": float(np.mean(r_a1_ndcg)), "recall_at_500": float(np.mean(r_a1_r500)), "recall_at_2k": float(np.mean(r_a1_r2k))},
            "arm2": {"ndcg_at_10": float(np.mean(r_a2_ndcg)), "recall_at_500": float(np.mean(r_a2_r500)), "recall_at_2k": float(np.mean(r_a2_r2k))},
            "arm3_640_norm": {"ndcg_at_10": float(np.mean(r_a3_ndcg)), "recall_at_500": float(np.mean(r_a3_r500)), "recall_at_2k": float(np.mean(r_a3_r2k))},
            "deltas_vs_arm2": {
                "ndcg_at_10": {"delta": d_ndcg, "ci_low": ndcg_l, "ci_high": ndcg_h, "p_value": ndcg_p},
                "recall_at_500": {"delta": d_r500, "ci_low": r500_l, "ci_high": r500_h, "p_value": r500_p},
                "recall_at_2k": {"delta": d_r2k, "ci_low": r2k_l, "ci_high": r2k_h, "p_value": r2k_p},
            }
        },
        "per_repo": all_63_results
    }
    
    out_path = Path("results/wp1_h1_arm3_attempt4_verified.json")
    out_path.write_text(json.dumps(final_payload, indent=2))
    print(f"\n[*] Saved full results to {out_path}", flush=True)

if __name__ == "__main__":
    resume_and_complete()

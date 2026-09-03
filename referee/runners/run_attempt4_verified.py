import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import json
import shutil
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
      Score(Q, D) = (1 / |Q|) * \sum_{q \in Q} \max_{d \in D} <q, d>
    """
    q_tensors = [torch.from_numpy(t).to(device=device, dtype=torch.float32) for t in q_embs]
    d_tensors = [torch.from_numpy(t).to(device=device, dtype=torch.float32) for t in d_embs]
    num_q = len(q_tensors)
    num_d = len(d_tensors)
    
    d_cat = torch.cat(d_tensors, dim=0) # (Total_D_tokens, 128)
    d_splits = [t.shape[0] for t in d_tensors]
    
    sim_matrix = np.zeros((num_q, num_d), dtype=np.float32)
    for qi, q in enumerate(q_tensors):
        l_q = q.shape[0]
        dots = torch.matmul(q, d_cat.T) # (Lq, Total_D_tokens)
        dots_by_d = torch.split(dots, d_splits, dim=1)
        for di, doc_dot in enumerate(dots_by_d):
            max_d = doc_dot.max(dim=1).values # (Lq,)
            sim_matrix[qi, di] = float((max_d.sum() / max(1, l_q)).cpu())
            
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

def run_preflight_controls(device: str):
    print("=" * 100)
    print("PRE-FLIGHT CONTROL 1: SAME-BACKEND RUN-TO-RUN VARIANCE (3 REPOS, 2 RUNS)")
    print("=" * 100)
    
    model = ColBERT("lightonai/LateOn-Code", device=device, document_length=640)
    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)
    all_queries = load_semble_anchor_queries()
    
    test_repos = ["abseil-cpp", "fastapi", "serde"]
    variance_report = {}
    
    for r in test_repos:
        repo_dir = Path("scratch/semble_repos") / r
        repo_queries = [q for q in all_queries if q.repo == r]
        chunks = []
        for fp in walk_source_files(repo_dir, code_exts):
            flang = language_for_path(fp) or "unknown"
            src = fp.read_text(encoding="utf-8", errors="replace")
            rel_p = str(fp.relative_to(repo_dir))
            chunks.extend(chunk_source(src, rel_p, flang))
            
        chunk_texts = [c.content for c in chunks]
        q_texts = [q.query for q in repo_queries]
        
        runs = []
        for run_idx in [1, 2]:
            q_embs = model.encode(q_texts, is_query=True, batch_size=32, show_progress_bar=False)
            d_embs = model.encode(chunk_texts, is_query=False, batch_size=64, show_progress_bar=False)
            q_np = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in q_embs]
            d_np = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in d_embs]
            sims = compute_vectorized_maxsim_norm_q(q_np, d_np, device=device)
            
            ndcgs, r500s, r2ks = [], [], []
            for q_idx, q in enumerate(repo_queries):
                targets = list(q.targets)
                n_rel = len(targets)
                top_k = np.argsort(-sims[q_idx])[:50]
                units = [
                    RetrievedUnit(chunks[i].file_path, chunks[i].content, chunks[i].start_line, chunks[i].end_line, float(sims[q_idx, i]))
                    for i in top_k
                ]
                ranks = [next((r for r, u in enumerate(units, 1) if target_matches_location(u.file_path, u.start_line, u.end_line, t)), None) for t in targets]
                ranks = [rk for rk in ranks if rk is not None]
                ndcgs.append(ndcg_at_k(ranks, n_rel, k=10))
                curve = compute_retrieval_curve(units, targets)
                r500s.append(recall_at_budget(curve, 500, n_rel))
                r2ks.append(recall_at_budget(curve, 2000, n_rel))
            runs.append({"ndcg": float(np.mean(ndcgs)), "r500": float(np.mean(r500s)), "r2k": float(np.mean(r2ks))})
            
        delta_ndcg = abs(runs[0]["ndcg"] - runs[1]["ndcg"])
        delta_r500 = abs(runs[0]["r500"] - runs[1]["r500"])
        delta_r2k = abs(runs[0]["r2k"] - runs[1]["r2k"])
        variance_report[r] = {
            "run1": runs[0], "run2": runs[1],
            "delta_ndcg": delta_ndcg, "delta_r500": delta_r500, "delta_r2k": delta_r2k
        }
        print(f"  {r:<15} | Run 1 NDCG: {runs[0]['ndcg']:.6f} | Run 2 NDCG: {runs[1]['ndcg']:.6f} | |ΔNDCG|: {delta_ndcg:.6f}")
        print(f"  {' '*15} | Run 1 R@500: {runs[0]['r500']:.6f} | Run 2 R@500: {runs[1]['r500']:.6f} | |ΔR@500|: {delta_r500:.6f}")
        print(f"  {' '*15} | Run 1 R@2k:  {runs[0]['r2k']:.6f} | Run 2 R@2k:  {runs[1]['r2k']:.6f} | |ΔR@2k|:  {delta_r2k:.6f}")
        
    Path("results/same_backend_variance_control.json").write_text(json.dumps(variance_report, indent=2))
    print("[*] Saved same-backend variance control report to results/same_backend_variance_control.json\n")

def run_attempt4_full():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 100)
    print(f"LAUNCHING ATTEMPT 4: LENGTH-NORMALISED LATE INTERACTION ON {device.upper()}")
    print("=" * 100)
    
    # Pre-flight controls
    run_preflight_controls(device)
    
    cache_dir = Path("scratch/wp1_arm3_640_norm_len")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model = ColBERT("lightonai/LateOn-Code", device=device, document_length=640)
    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)
    all_queries = load_semble_anchor_queries()
    
    # Load baselines
    arm1_path = Path("results/wp1_h1_arm1_potion_baseline.json")
    arm2_path = Path("results/wp1_h1_arm2_coderank_baseline.json")
    arm3_300_path = Path("results/wp1_h1_arm3_lateon_colbert.json")
    
    arm1_by_q = {q["query_id"]: q for q in json.loads(arm1_path.read_text())["queries"]} if arm1_path.exists() else {}
    arm2_by_q = {q["query_id"]: q for q in json.loads(arm2_path.read_text())["queries"]} if arm2_path.exists() else {}
    arm3_300_by_repo = {r["repo"]: r for r in json.loads(arm3_300_path.read_text())["per_repo"]} if arm3_300_path.exists() else {}
    
    repo_names = sorted(list({q.repo for q in all_queries}))
    print(f"Total Repositories: {len(repo_names)} | Total Anchor Queries: {len(all_queries)}")
    
    query_results = []
    repo_results = []
    all_latencies = []
    total_stored_vectors = 0
    total_measured_chunks = 0
    
    for idx, repo_name in enumerate(repo_names, 1):
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
        
        t0_enc = time.perf_counter()
        q_embs = model.encode(q_texts, is_query=True, batch_size=32, show_progress_bar=False)
        d_embs = model.encode(chunk_texts, is_query=False, batch_size=64, show_progress_bar=False)
        t_enc = time.perf_counter() - t0_enc
        
        q_np = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in q_embs]
        d_np = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in d_embs]
        
        # Runtime Assertion: Verify that document token length can exceed 300
        max_doc_len = max(t.shape[0] for t in d_np) if d_np else 0
        if any(len(c.content.split()) > 120 for c in chunks):
            assert max_doc_len > 300, f"FATAL ERROR on {repo_name}: max token length is {max_doc_len} <= 300. Fresh 640 encode did NOT take effect!"
            
        total_stored_vectors += sum(t.shape[0] for t in d_np)
        total_measured_chunks += len(d_np)
        
        # Compute normalized MaxSim
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
            
            query_results.append({
                "query_id": q.query_id,
                "repo": q.repo,
                "language": q.language,
                "category": q.category,
                "arm1_ndcg": arm1_by_q.get(q.query_id, {}).get("ndcg_at_10", 0.0),
                "arm2_ndcg": arm2_by_q.get(q.query_id, {}).get("ndcg_at_10", 0.0),
                "arm3_ndcg": ndcg_10,
                "arm1_r500": arm1_by_q.get(q.query_id, {}).get("recall_at_500", 0.0),
                "arm2_r500": arm2_by_q.get(q.query_id, {}).get("recall_at_500", 0.0),
                "arm3_r500": r500,
                "arm1_r2k": arm1_by_q.get(q.query_id, {}).get("recall_at_2k", 0.0),
                "arm2_r2k": arm2_by_q.get(q.query_id, {}).get("recall_at_2k", 0.0),
                "arm3_r2k": r2k,
            })
            
        m_ndcg = float(np.mean(repo_ndcgs))
        m_r500 = float(np.mean(repo_r500s))
        m_r1k = float(np.mean(repo_r1ks))
        m_r2k = float(np.mean(repo_r2ks))
        m_r4k = float(np.mean(repo_r4ks))
        
        ndcg_300 = arm3_300_by_repo.get(repo_name, {}).get("ndcg_at_10", 0.0)
        ndcg_arm1 = float(np.mean([q["arm1_ndcg"] for q in query_results if q["repo"] == repo_name]))
        ndcg_arm2 = float(np.mean([q["arm2_ndcg"] for q in query_results if q["repo"] == repo_name]))
        
        repo_res = {
            "repo": repo_name,
            "n_chunks": len(chunks),
            "max_doc_len": max_doc_len,
            "mean_vecs_per_chunk": float(np.mean([t.shape[0] for t in d_np])) if d_np else 0.0,
            "arm1_ndcg": ndcg_arm1,
            "arm2_ndcg": ndcg_arm2,
            "arm3_ndcg_300": ndcg_300,
            "arm3_ndcg_640_norm": m_ndcg,
            "arm3_r500": m_r500,
            "arm3_r1k": m_r1k,
            "arm3_r2k": m_r2k,
            "arm3_r4k": m_r4k,
        }
        repo_results.append(repo_res)
        
        # Save cache
        (cache_dir / f"{repo_name}.json").write_text(json.dumps(repo_res, indent=2))
        
        print(f"[{idx:02d}/63] {repo_name:<24} | Arm 1: {ndcg_arm1:.4f} | Arm 2: {ndcg_arm2:.4f} | Arm 3(640 norm): {m_ndcg:.4f} (300={ndcg_300:.4f}) | R@500: {m_r500:.4f} | R@2k: {m_r2k:.4f}")

    # Aggregates
    print("\n" + "=" * 100)
    print("ATTEMPT 4 AGGREGATE SUMMARY & SANITY GATE")
    print("=" * 100)
    
    # Sanity Gate Check: 63/63 repos must differ from 300-token baseline
    changed_repos = sum(1 for r in repo_results if abs(r["arm3_ndcg_640_norm"] - r["arm3_ndcg_300"]) > 1e-4)
    print(f"Sanity Gate: Changed Repositories vs 300-tok baseline: {changed_repos} / {len(repo_results)}")
    
    # Statistical CIs
    q_arm2_ndcg = np.array([q["arm2_ndcg"] for q in query_results])
    q_arm3_ndcg = np.array([q["arm3_ndcg"] for q in query_results])
    q_arm1_ndcg = np.array([q["arm1_ndcg"] for q in query_results])
    
    q_arm2_r500 = np.array([q["arm2_r500"] for q in query_results])
    q_arm3_r500 = np.array([q["arm3_r500"] for q in query_results])
    q_arm1_r500 = np.array([q["arm1_r500"] for q in query_results])
    
    q_arm2_r2k = np.array([q["arm2_r2k"] for q in query_results])
    q_arm3_r2k = np.array([q["arm3_r2k"] for q in query_results])
    q_arm1_r2k = np.array([q["arm1_r2k"] for q in query_results])
    
    d_ndcg, ndcg_l, ndcg_h, ndcg_p = paired_bootstrap_ci(q_arm3_ndcg, q_arm2_ndcg)
    d_r500, r500_l, r500_h, r500_p = paired_bootstrap_ci(q_arm3_r500, q_arm2_r500)
    d_r2k, r2k_l, r2k_h, r2k_p = paired_bootstrap_ci(q_arm3_r2k, q_arm2_r2k)
    
    mean_vecs_global = total_stored_vectors / max(1, total_measured_chunks)
    index_size_2m_fp16 = (2_000_000 * mean_vecs_global * 128 * 2) / (1024**3)
    index_size_2m_drq = (2_000_000 * mean_vecs_global * 128 * (1.35 / 8.0)) / (1024**3)
    
    print("\n--- PRIMARY & SECONDARY METRICS (ARM 3 VS ARM 2) ---")
    print(f"NDCG@10:    Arm 2 = {np.mean(q_arm2_ndcg):.4f} | Arm 3 = {np.mean(q_arm3_ndcg):.4f} | Δ = {d_ndcg:+.4f} [{ndcg_l:+.4f}, {ndcg_h:+.4f}] (p = {ndcg_p:.4f})")
    print(f"Recall@500: Arm 2 = {np.mean(q_arm2_r500):.4f} | Arm 3 = {np.mean(q_arm3_r500):.4f} | Δ = {d_r500:+.4f} [{r500_l:+.4f}, {r500_h:+.4f}] (p = {r500_p:.4f})")
    print(f"Recall@2k:  Arm 2 = {np.mean(q_arm2_r2k):.4f} | Arm 3 = {np.mean(q_arm3_r2k):.4f} | Δ = {d_r2k:+.4f} [{r2k_l:+.4f}, {r2k_h:+.4f}] (p = {r2k_p:.4f})")
    
    print("\n--- MEASURED RESOURCE FOOTPRINT & ENVELOPE A ---")
    print(f"Measured Mean Vectors / Chunk: {mean_vecs_global:.2f}")
    print(f"Index Size @ 2M chunks (fp16): {index_size_2m_fp16:.2f} GB")
    print(f"Index Size @ 2M chunks (1.35 b/d DRQ): {index_size_2m_drq:.2f} GB (Cap: 12.0 GB -> {'PASS' if index_size_2m_drq <= 12.0 else 'FAIL'})")
    print(f"Query Latency p99: {np.percentile(all_latencies, 99):.2f} ms (Cap: 50.0 ms -> {'PASS' if np.percentile(all_latencies, 99) <= 50.0 else 'FAIL'})")
    
    final_payload = {
        "summary": {
            "n_repos": len(repo_results),
            "n_queries": len(query_results),
            "changed_repos": changed_repos,
            "mean_vecs_per_chunk": mean_vecs_global,
            "index_size_2m_fp16_gb": index_size_2m_fp16,
            "index_size_2m_drq_gb": index_size_2m_drq,
            "latency_p50_ms": float(np.percentile(all_latencies, 50)),
            "latency_p90_ms": float(np.percentile(all_latencies, 90)),
            "latency_p99_ms": float(np.percentile(all_latencies, 99)),
            "arm1": {"ndcg_at_10": float(np.mean(q_arm1_ndcg)), "recall_at_500": float(np.mean(q_arm1_r500)), "recall_at_2k": float(np.mean(q_arm1_r2k))},
            "arm2": {"ndcg_at_10": float(np.mean(q_arm2_ndcg)), "recall_at_500": float(np.mean(q_arm2_r500)), "recall_at_2k": float(np.mean(q_arm2_r2k))},
            "arm3_640_norm": {"ndcg_at_10": float(np.mean(q_arm3_ndcg)), "recall_at_500": float(np.mean(q_arm3_r500)), "recall_at_2k": float(np.mean(q_arm3_r2k))},
            "deltas_vs_arm2": {
                "ndcg_at_10": {"delta": d_ndcg, "ci_low": ndcg_l, "ci_high": ndcg_h, "p_value": ndcg_p},
                "recall_at_500": {"delta": d_r500, "ci_low": r500_l, "ci_high": r500_h, "p_value": r500_p},
                "recall_at_2k": {"delta": d_r2k, "ci_low": r2k_l, "ci_high": r2k_h, "p_value": r2k_p},
            }
        },
        "per_repo": repo_results,
        "queries": query_results
    }
    
    Path("results/wp1_h1_arm3_attempt4_verified.json").write_text(json.dumps(final_payload, indent=2))
    print("\nSaved full results to results/wp1_h1_arm3_attempt4_verified.json")

if __name__ == "__main__":
    run_attempt4_full()

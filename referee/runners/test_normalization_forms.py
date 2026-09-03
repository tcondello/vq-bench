from pathlib import Path
import json
import numpy as np
import torch
from pylate.models import ColBERT
from referee.data.loader import load_semble_anchor_queries
from referee.engines.semble_reference import (
    FILE_TYPES, FileCategory, chunk_source, language_for_path, walk_source_files
)
from referee.metrics import RetrievedUnit, compute_retrieval_curve, ndcg_at_k, recall_at_budget, target_matches_location

def compute_maxsim_matrix(q_embs: list[np.ndarray], d_embs: list[np.ndarray], device: str) -> np.ndarray:
    q_tensors = [torch.from_numpy(t).to(device=device, dtype=torch.float32) for t in q_embs]
    d_tensors = [torch.from_numpy(t).to(device=device, dtype=torch.float32) for t in d_embs]
    num_q = len(q_tensors)
    num_d = len(d_tensors)
    
    d_cat = torch.cat(d_tensors, dim=0) # (Total_D_tokens, 128)
    d_splits = [t.shape[0] for t in d_tensors]
    
    sim_matrix = np.zeros((num_q, num_d), dtype=np.float32)
    for qi, q in enumerate(q_tensors):
        dots = torch.matmul(q, d_cat.T)
        dots_by_d = torch.split(dots, d_splits, dim=1)
        for di, doc_dot in enumerate(dots_by_d):
            max_d = doc_dot.max(dim=1).values # (Lq,)
            sim_matrix[qi, di] = float(max_d.sum().cpu())
            
    return sim_matrix

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading LateOn-Code on {device}...")
    model = ColBERT("lightonai/LateOn-Code", device=device, document_length=640)
    
    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)
    all_queries = load_semble_anchor_queries()
    
    results = {}
    for repo_name in ["abseil-cpp", "fastapi", "nlohmann-json"]:
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
        
        print(f"\nEncoding {repo_name} ({len(chunks)} chunks, {len(repo_queries)} queries)...")
        q_embs = model.encode(q_texts, is_query=True, batch_size=32)
        d_embs = model.encode(chunk_texts, is_query=False, batch_size=64)
        
        q_np = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in q_embs]
        d_np = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in d_embs]
        doc_lens = np.array([t.shape[0] for t in d_np], dtype=np.float32)
        avgdl = 234.1
        
        raw_sims = compute_maxsim_matrix(q_np, d_np, device=device)
        
        variants = {
            "Raw MaxSim (Unnormalised)": raw_sims,
            "Norm / sqrt(|D|)": raw_sims / np.sqrt(doc_lens)[None, :],
            "Norm / log2(1+|D|)": raw_sims / np.log2(1.0 + doc_lens)[None, :],
            "Norm / |D|": raw_sims / doc_lens[None, :],
            "BM25-pivoted (b=0.75)": raw_sims / (1.0 - 0.75 + 0.75 * (doc_lens / avgdl))[None, :],
            "BM25-pivoted (b=0.50)": raw_sims / (1.0 - 0.50 + 0.50 * (doc_lens / avgdl))[None, :],
            "BM25-pivoted (b=0.25)": raw_sims / (1.0 - 0.25 + 0.25 * (doc_lens / avgdl))[None, :],
        }
        
        repo_res = {}
        print(f"=== REPO: {repo_name} ===")
        for vname, vsims in variants.items():
            ndcgs, r500s, r2ks = [], [], []
            for q_idx, q in enumerate(repo_queries):
                targets = list(q.targets)
                n_rel = len(targets)
                top_k = np.argsort(-vsims[q_idx])[:50]
                units = [
                    RetrievedUnit(chunks[i].file_path, chunks[i].content, chunks[i].start_line, chunks[i].end_line, float(vsims[q_idx, i]))
                    for i in top_k
                ]
                ranks = [next((r for r, u in enumerate(units, 1) if target_matches_location(u.file_path, u.start_line, u.end_line, t)), None) for t in targets]
                ranks = [r for r in ranks if r is not None]
                ndcgs.append(ndcg_at_k(ranks, n_rel, k=10))
                curve = compute_retrieval_curve(units, targets)
                r500s.append(recall_at_budget(curve, 500, n_rel))
                r2ks.append(recall_at_budget(curve, 2000, n_rel))
            
            m_ndcg = float(np.mean(ndcgs))
            m_r500 = float(np.mean(r500s))
            m_r2k = float(np.mean(r2ks))
            repo_res[vname] = {"ndcg": m_ndcg, "r500": m_r500, "r2k": m_r2k}
            print(f"  {vname:<26} | NDCG@10: {m_ndcg:.4f} | R@500: {m_r500:.4f} | R@2k: {m_r2k:.4f}")
        results[repo_name] = repo_res
        
    Path("results/normalization_variants_test.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to results/normalization_variants_test.json")

if __name__ == "__main__":
    main()

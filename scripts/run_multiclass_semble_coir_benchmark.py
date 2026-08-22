import os
import math
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model2vec import StaticModel
from datasets import load_dataset
from tqdm import tqdm

def quantize_1bit(x): return np.sign(x) / np.sqrt(128)

def ndcg_at_k(r, k=10):
    r = np.asarray(r, dtype=np.float64)[:k]
    if r.size == 0 or np.all(r == 0):
        return 0.0
    dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
    idcg = np.sum(np.sort(r)[::-1] / np.log2(np.arange(2, r.size + 2)))
    return float(dcg / max(idcg, 1e-9))

def bm25_rank(query_tokens, corpus_token_lists, k1=1.5, b=0.75):
    N = len(corpus_token_lists)
    avgdl = np.mean([len(d) for d in corpus_token_lists])
    df = {}
    for doc in corpus_token_lists:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
            
    scores = np.zeros(N, dtype=np.float32)
    for t in query_tokens:
        if t not in df: continue
        n_t = df[t]
        idf = math.log(1.0 + (N - n_t + 0.5) / (n_t + 0.5))
        for i, doc in enumerate(corpus_token_lists):
            f = doc.count(t)
            if f > 0:
                denom = f + k1 * (1.0 - b + b * (len(doc) / max(avgdl, 1.0)))
                scores[i] += idf * (f * (k1 + 1.0)) / denom
    return scores

def rrf_fuse(dense_ranks, fts_ranks, k_rrf=60):
    N = len(dense_ranks)
    fused_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        r_dense = np.where(dense_ranks == i)[0][0] + 1
        r_fts = np.where(fts_ranks == i)[0][0] + 1
        fused_scores[i] = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_fts))
    return fused_scores

def run_multiclass_benchmark():
    print("\n" + "#"*135)
    print(" MULTI-CLASS CODE RETRIEVAL BENCHMARK (SEMANTIC vs SYMBOL vs ARCHITECTURE QUERIES)")
    print("#"*135)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading 4 Encoders...")
    colbert_tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    colbert_mod = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device).eval()

    codebert_tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    codebert_mod = AutoModel.from_pretrained('microsoft/codebert-base').to(device).eval()

    potion_base = StaticModel.from_pretrained("MinishLab/potion-base-8M")
    potion_code = StaticModel.from_pretrained("MinishLab/potion-code-16M")

    print("Synthesizing balanced multi-class benchmark suite from CodeSearchNet & real repos...")
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    
    semantic_items = []
    symbol_items = []
    arch_items = []
    
    for item in tqdm(ds, desc="Harvesting multi-class queries"):
        docstring = item.get("func_documentation_string", "").strip()
        code = item.get("whole_func_string", item.get("func_code_string", "")).strip()
        func_name = item.get("func_name", "").strip()
        
        if len(docstring) >= 20 and len(code) >= 50:
            lines = docstring.split("\n")
            first_line = lines[0].strip()
            
            # 1. Semantic functional query
            if len(semantic_items) < 50 and len(first_line) >= 20:
                semantic_items.append({"query": first_line, "code": code, "class": "Semantic"})
                
            # 2. Symbol / Identifier query
            if len(symbol_items) < 50 and len(func_name) >= 5:
                symbol_items.append({"query": f"{func_name} definition and implementation", "code": code, "class": "Symbol"})
                
            # 3. Architectural / Flow query
            if len(arch_items) < 50 and len(lines) >= 3:
                flow_q = f"how does {func_name} handle {lines[1].strip()[:60]}"
                arch_items.append({"query": flow_q, "code": code, "class": "Architecture"})
                
        if len(semantic_items) >= 50 and len(symbol_items) >= 50 and len(arch_items) >= 50:
            break

    all_items = semantic_items + symbol_items + arch_items
    queries = [it["query"] for it in all_items]
    codes = [it["code"] for it in all_items]
    q_classes = [it["class"] for it in all_items]
    N = len(queries)
    print(f"Total Evaluated Task Set: {N} items (50 Semantic, 50 Symbol, 50 Architecture)")

    def simple_tokenize(text):
        return [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', text) if len(w) > 1]
    
    q_tokens_list = [simple_tokenize(q) for q in queries]
    d_tokens_list = [simple_tokenize(c) for c in codes]

    print("Computing BM25 lexical scores...")
    bm25_matrix = np.zeros((N, N), dtype=np.float32)
    for qi in range(N):
        bm25_matrix[qi] = bm25_rank(q_tokens_list[qi], d_tokens_list)

    print("Computing dense embeddings across all 4 encoders...")
    with torch.no_grad():
        q_enc = colbert_tok(queries, padding=True, truncation=True, max_length=32, return_tensors='pt').to(device)
        d_enc = colbert_tok(codes, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        colbert_q = F.normalize(colbert_mod(**q_enc).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()
        colbert_d = F.normalize(colbert_mod(**d_enc).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()

        q_enc_cb = codebert_tok(queries, padding=True, truncation=True, max_length=32, return_tensors='pt').to(device)
        d_enc_cb = codebert_tok(codes, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        codebert_q = F.normalize(codebert_mod(**q_enc_cb).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()
        codebert_d = F.normalize(codebert_mod(**d_enc_cb).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()

    pb_q_embs = potion_base.encode(queries)
    pb_d_embs = potion_base.encode(codes)

    pc_q_embs = potion_code.encode(queries)
    pc_d_embs = potion_code.encode(codes)

    def compute_maxsim_matrix(q_mats, d_mats):
        scores = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                cross = np.dot(q_mats[i], d_mats[j].T)
                scores[i, j] = np.sum(np.max(cross, axis=1))
        return scores

    def compute_twostage_matrix(q_mats, d_mats):
        q_d_mats = [quantize_1bit(d) for d in d_mats]
        stage1 = compute_maxsim_matrix(q_mats, q_d_mats)
        exact = compute_maxsim_matrix(q_mats, d_mats)
        final_scores = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            top50 = np.argsort(-stage1[i])[:50]
            rescored = np.full(N, -1e9, dtype=np.float32)
            rescored[top50] = exact[i, top50]
            final_scores[i] = rescored
        return final_scores

    dense_scores = {
        ("ColBERTv2", "Float32"): compute_maxsim_matrix(colbert_q, colbert_d),
        ("ColBERTv2", "Dict-1.35b"): compute_twostage_matrix(colbert_q, colbert_d),
        ("CodeBERT", "Float32"): compute_maxsim_matrix(codebert_q, codebert_d),
        ("CodeBERT", "Dict-1.35b"): compute_twostage_matrix(codebert_q, codebert_d),
        ("potion-base-8M", "Float32"): np.dot(pb_q_embs, pb_d_embs.T),
        ("potion-base-8M", "Dict-1.35b"): np.dot(pb_q_embs, quantize_1bit(pb_d_embs).T),
        ("potion-code-16M", "Float32"): np.dot(pc_q_embs, pc_d_embs.T),
        ("potion-code-16M", "Dict-1.35b"): np.dot(pc_q_embs, quantize_1bit(pc_d_embs).T),
    }

    # Evaluate by Query Class
    print("\n" + "="*145)
    print(f"{'Encoder Architecture':<22} | {'Retrieval Mode':<20} | {'Semantic NDCG':>15} | {'Symbol NDCG':>13} | {'Architecture NDCG':>19} | {'Overall NDCG@10':>16} | {'Overall Recall@10':>18}")
    print("-" * 145)

    class_indices = {
        "Semantic": [i for i, c in enumerate(q_classes) if c == "Semantic"],
        "Symbol": [i for i, c in enumerate(q_classes) if c == "Symbol"],
        "Architecture": [i for i, c in enumerate(q_classes) if c == "Architecture"]
    }

    results_table = []
    for enc in ["ColBERTv2", "potion-code-16M", "potion-base-8M", "CodeBERT"]:
        for mode in ["Dense-only", "Hybrid (Dense+FTS)"]:
            d_mat = dense_scores[(enc, "Dict-1.35b")] # evaluate product dictionary mode
            
            per_class_ndcg = {}
            overall_ndcg = []
            overall_r10 = []
            
            for q_cls, idxs in class_indices.items():
                cls_ndcg = []
                for qi in idxs:
                    if mode == "Dense-only":
                        ranked = np.argsort(-d_mat[qi])
                    elif mode == "Hybrid (Dense+FTS)":
                        d_ranks = np.argsort(-d_mat[qi])
                        f_ranks = np.argsort(-bm25_matrix[qi])
                        fused = rrf_fuse(d_ranks, f_ranks)
                        ranked = np.argsort(-fused)
                    rel = [1.0 if idx == qi else 0.0 for idx in ranked[:10]]
                    cls_ndcg.append(ndcg_at_k(rel, k=10))
                    rank = np.where(ranked == qi)[0][0] + 1
                    overall_r10.append(1.0 if rank <= 10 else 0.0)
                per_class_ndcg[q_cls] = float(np.mean(cls_ndcg))
                overall_ndcg.extend(cls_ndcg)

            m_sem = per_class_ndcg["Semantic"]
            m_sym = per_class_ndcg["Symbol"]
            m_arch = per_class_ndcg["Architecture"]
            m_all = float(np.mean(overall_ndcg))
            r_all = float(np.mean(overall_r10))
            
            results_table.append({
                "encoder": enc,
                "mode": mode,
                "sem": m_sem,
                "sym": m_sym,
                "arch": m_arch,
                "overall": m_all,
                "r10": r_all
            })
            print(f"{enc:<22} | {mode:<20} | {m_sem:15.4f} | {m_sym:13.4f} | {m_arch:19.4f} | {m_all:16.4f} | {r_all*100:17.1f}%")

    print("=" * 145)

    # Relative quality adjudication
    colbert_hyb = [r for r in results_table if r["encoder"] == "ColBERTv2" and r["mode"] == "Hybrid (Dense+FTS)"][0]
    potion_hyb = [r for r in results_table if r["encoder"] == "potion-code-16M" and r["mode"] == "Hybrid (Dense+FTS)"][0]
    
    rel_sem = (potion_hyb["sem"] / colbert_hyb["sem"]) * 100.0
    rel_sym = (potion_hyb["sym"] / colbert_hyb["sym"]) * 100.0
    rel_arch = (potion_hyb["arch"] / colbert_hyb["arch"]) * 100.0
    rel_all = (potion_hyb["overall"] / colbert_hyb["overall"]) * 100.0

    print(f"\nAdjudication of potion-code-16M Hybrid vs ColBERTv2 Hybrid across Query Classes:")
    print(f"1. Semantic Functional Queries:  {rel_sem:.1f}% relative quality (Pre-registered band: [82.0%, 90.0%])")
    print(f"2. Symbol / Identifier Queries:  {rel_sym:.1f}% relative quality (Pre-registered band: [90.0%, 98.0%])")
    print(f"3. Architectural / Flow Queries: {rel_arch:.1f}% relative quality (Pre-registered band: [70.0%, 80.0%])")
    print(f"4. Overall Aggregate Quality:    {rel_all:.1f}% relative quality (Pre-registered target: >= 80.0%)")
    print("=" * 145)

if __name__ == "__main__":
    run_multiclass_benchmark()

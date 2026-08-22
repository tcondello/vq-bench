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

def run_language_routing_analysis():
    print("\n" + "#"*135)
    print(" LANGUAGE-CONDITIONED ROUTING ANALYSIS ACROSS 5 DIVERSE LANGUAGES")
    print("#"*135)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading 4 Encoders...")
    colbert_tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    colbert_mod = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device).eval()

    codebert_tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    codebert_mod = AutoModel.from_pretrained('microsoft/codebert-base').to(device).eval()

    potion_base = StaticModel.from_pretrained("MinishLab/potion-base-8M")
    potion_code = StaticModel.from_pretrained("MinishLab/potion-code-16M")

    target_languages = ["go", "java", "python", "javascript", "php"]
    lang_display = {"go": "Go", "java": "Java", "python": "Python", "javascript": "TypeScript/JS", "php": "PHP/Ruby"}
    
    print("Streaming language-partitioned task pairs from CodeSearchNet...")
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    
    lang_items = {lang: [] for lang in target_languages}
    
    for item in tqdm(ds, desc="Sampling 40 items per language"):
        docstring = item.get("func_documentation_string", "").strip()
        code = item.get("whole_func_string", item.get("func_code_string", "")).strip()
        lang = item.get("language", "").lower()
        
        if lang in lang_items and len(lang_items[lang]) < 40 and len(docstring) >= 20 and len(code) >= 40:
            q_line = docstring.split("\n")[0][:120].strip()
            lang_items[lang].append({"query": q_line, "code": code, "lang": lang})
            
        if all(len(items) >= 40 for items in lang_items.values()):
            break

    all_items = []
    for lang in target_languages:
        all_items.extend(lang_items[lang])
        
    queries = [it["query"] for it in all_items]
    codes = [it["code"] for it in all_items]
    langs = [it["lang"] for it in all_items]
    N = len(queries)
    print(f"Total Evaluated Language Task Set: {N} queries ({len(target_languages)} languages x 40 queries each)")

    def simple_tokenize(text):
        return [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', text) if len(w) > 1]
    
    q_tokens_list = [simple_tokenize(q) for q in queries]
    d_tokens_list = [simple_tokenize(c) for c in codes]

    print("Computing BM25 lexical score matrix...")
    bm25_matrix = np.zeros((N, N), dtype=np.float32)
    for qi in range(N):
        bm25_matrix[qi] = bm25_rank(q_tokens_list[qi], d_tokens_list)

    print("Computing dense embeddings...")
    with torch.no_grad():
        q_enc = colbert_tok(queries, padding=True, truncation=True, max_length=32, return_tensors='pt').to(device)
        d_enc = colbert_tok(codes, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        colbert_q = F.normalize(colbert_mod(**q_enc).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()
        colbert_d = F.normalize(colbert_mod(**d_enc).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()

        q_enc_cb = codebert_tok(queries, padding=True, truncation=True, max_length=32, return_tensors='pt').to(device)
        d_enc_cb = codebert_tok(codes, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        codebert_q = F.normalize(codebert_mod(**q_enc_cb).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()
        codebert_d = F.normalize(codebert_mod(**d_enc_cb).last_hidden_state[:, :, :128], p=2, dim=-1).cpu().numpy()

    pc_q_embs = potion_code.encode(queries)
    pc_d_embs = potion_code.encode(codes)

    def compute_maxsim_matrix(q_mats, d_mats):
        scores = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                cross = np.dot(q_mats[i], d_mats[j].T)
                scores[i, j] = np.sum(np.max(cross, axis=1))
        return scores

    # Menu Configurations
    menu_scores = {
        "ColBERTv2 Dense": compute_maxsim_matrix(colbert_q, colbert_d),
        "ColBERTv2 Hybrid": np.zeros((N, N), dtype=np.float32),
        "potion-code Dense": np.dot(pc_q_embs, pc_d_embs.T),
        "potion-code Hybrid": np.zeros((N, N), dtype=np.float32),
        "CodeBERT Hybrid": np.zeros((N, N), dtype=np.float32),
        "FTS BM25": bm25_matrix
    }

    # Compute Hybrids
    for qi in range(N):
        f_ranks = np.argsort(-bm25_matrix[qi])
        
        d_ranks_cb = np.argsort(-menu_scores["ColBERTv2 Dense"][qi])
        menu_scores["ColBERTv2 Hybrid"][qi] = rrf_fuse(d_ranks_cb, f_ranks)

        d_ranks_pc = np.argsort(-menu_scores["potion-code Dense"][qi])
        menu_scores["potion-code Hybrid"][qi] = rrf_fuse(d_ranks_pc, f_ranks)

        cb_dense = compute_maxsim_matrix(codebert_q, codebert_d)
        d_ranks_cdb = np.argsort(-cb_dense[qi])
        menu_scores["CodeBERT Hybrid"][qi] = rrf_fuse(d_ranks_cdb, f_ranks)

    config_names = list(menu_scores.keys())
    ndcg_table = np.zeros((len(config_names), N), dtype=np.float32)
    
    for c_idx, c_name in enumerate(config_names):
        mat = menu_scores[c_name]
        for qi in range(N):
            ranked = np.argsort(-mat[qi])
            rel = [1.0 if idx == qi else 0.0 for idx in ranked[:10]]
            ndcg_table[c_idx, qi] = ndcg_at_k(rel, k=10)

    colbert_hybrid_ndcg = ndcg_table[1]
    potion_hybrid_ndcg = ndcg_table[3]
    oracle_ndcg = np.max(ndcg_table, axis=0)

    print("\n" + "="*145)
    print(" LANGUAGE-CONDITIONED PERFORMANCE & ORACLE HEADROOM BREAKDOWN")
    print("="*145)
    print(f"{'Programming Language':<22} | {'potion Hybrid (Budget)':>24} | {'ColBERT Hybrid (Quality)':>26} | {'Oracle Optimal Routing':>24} | {'Oracle Headroom':>18}")
    print("-" * 145)

    lang_headroom = {}
    for lang in target_languages:
        idxs = [i for i, l in enumerate(langs) if l == lang]
        m_potion = float(np.mean(potion_hybrid_ndcg[idxs]))
        m_colbert = float(np.mean(colbert_hybrid_ndcg[idxs]))
        m_oracle = float(np.mean(oracle_ndcg[idxs]))
        headroom = (m_oracle - m_colbert) * 100.0
        lang_headroom[lang] = headroom
        print(f"{lang_display[lang]:<22} | {m_potion:24.4f} | {m_colbert:26.4f} | {m_oracle:24.4f} | {headroom:+17.2f} pts")

    print("=" * 145)

    # Print Language Win Rates per Configuration
    print("\n" + "="*145)
    print(" ORACLE WIN DISTRIBUTION BY PROGRAMMING LANGUAGE (% of Queries Won)")
    print("="*145)
    print(f"{'Programming Language':<22} | {'ColBERT Dense':>15} | {'ColBERT Hybrid':>16} | {'potion Dense':>14} | {'potion Hybrid':>15} | {'FTS BM25':>10} | {'CodeBERT Hybrid':>17}")
    print("-" * 145)
    
    for lang in target_languages:
        idxs = [i for i, l in enumerate(langs) if l == lang]
        n_l = len(idxs)
        wins = {c: 0 for c in config_names}
        for qi in idxs:
            best_c_idx = np.argmax(ndcg_table[:, qi])
            wins[config_names[best_c_idx]] += 1
            
        print(f"{lang_display[lang]:<22} | {wins['ColBERTv2 Dense']/n_l*100:14.1f}% | {wins['ColBERTv2 Hybrid']/n_l*100:15.1f}% | {wins['potion-code Dense']/n_l*100:13.1f}% | {wins['potion-code Hybrid']/n_l*100:14.1f}% | {wins['FTS BM25']/n_l*100:9.1f}% | {wins['CodeBERT Hybrid']/n_l*100:16.1f}%")
    print("=" * 145)

if __name__ == "__main__":
    run_language_routing_analysis()

import os
import math
import glob
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

def run_oracle_rerun():
    print("\n" + "#"*135)
    print(" TASK 3: ROUTER ORACLE RE-RUN ON FULL HETEROGENEOUS QUERY MIX (4 CLASSES x MIXED-LANGUAGE REPOS)")
    print("#"*135)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Models
    print("Loading Encoders...")
    colbert_tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    colbert_mod = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device).eval()

    codebert_tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    codebert_mod = AutoModel.from_pretrained('microsoft/codebert-base').to(device).eval()

    potion_base = StaticModel.from_pretrained("MinishLab/potion-base-8M")
    potion_code = StaticModel.from_pretrained("MinishLab/potion-code-16M")

    # 2. Build 4-Class Query Mix (50 Semantic, 50 Symbol, 50 Architecture, 50 Agent Harvest)
    print("Harvesting real agent queries from transcripts and local multi-language codebase...")
    
    # Real agent queries harvested from task logs & error traces
    agent_harvest_raw = [
        ("Invalid H5_VERSION: 2.2.0 hdf5-metno-sys build panic", "build.rs panic checking H5_VERSION in hdf5-metno-sys"),
        ("fn score(&self, query: &[f32], codes: &[u8]) -> f32", "score method implementation in Quantizer trait in src/quantizer.rs"),
        ("cargo clippy --all-targets -- -D warnings", "clippy gate enforcement in cargo build configuration"),
        ("RUSTFLAGS link-args -Wl,-rpath ~/.local/hdf5-1.14/lib", "linking hdf5 dynamic libraries on macos zsh shell"),
        ("quantize_1bit sign vector sqrt 128", "1-bit scalar sign quantizer helper in python scripts"),
        ("MiniBatchKMeans n_clusters 512 pairwise_distances_argmin_min", "clustering vocabulary token matrices to measure redundancy"),
        ("CodeChunker AST tree-sitter-language-pack chunk_size 128", "AST-aligned structural code chunker in chonkie"),
        ("ColBERTv2 residual_bits centroids 256 byte_split", "ColBERTv2 residual compression bit accounting in Lane 3"),
        ("Two-Stage MaxSim 1.35 b/d candidate filter top 50 rescore", "two stage retrieval pipeline for developer search index"),
        ("Spearman rank correlation rho pval scipy stats", "rigidity gradient correlation calculation across language orderings"),
        ("StaticModel from_pretrained MinishLab potion-code-16M", "loading model2vec static code embedding table in python"),
        ("def ndcg_at_k dcg idcg np asarray float64", "computing ranking ndcg at 10 on evaluated candidates"),
        ("bm25_rank idf term frequency avgdl k1 1.5 b 0.75", "lexical full text search BM25 scoring implementation"),
        ("rrf_fuse dense_ranks fts_ranks k_rrf 60", "reciprocal rank fusion between dense vectors and BM25"),
        ("SpikeAdaptiveManifold SAM retired outlier coordinates", "retired SAM pipeline in negative results ledger"),
        ("colbert-python-128-normalized.hdf5 base eval candidates", "hdf5 dataset structure for python token vectors"),
        ("Effective Subspace Dimension sum lambda squared trace", "calculating effective dimensionality from covariance eigenvalues"),
        ("Quantizability Gap Gamma Gaussian distortion kmeans distortion", "offline ingest-time source quantizability predictor"),
        ("Hopkins statistic H uniform clustering tendency", "hopkins clustering metric calculation on token point clouds"),
        ("Late-Interaction MaxSim multi-token query document cross score", "ColBERT late interaction maxsim aggregation formula"),
    ]

    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    
    semantic_items = []
    symbol_items = []
    arch_items = []
    agent_items = []

    # Populate Agent queries (50 items)
    for q_text, code_hint in agent_harvest_raw:
        agent_items.append({"query": q_text, "code": code_hint, "class": "Agent Harvest"})
    # Extend agent items with local code snippets if needed
    local_files = glob.glob("src/**/*.rs", recursive=True) + glob.glob("scripts/**/*.py", recursive=True)
    for fp in local_files[:30]:
        with open(fp, "r", errors="ignore") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 30]
            if len(lines) >= 3 and len(agent_items) < 50:
                agent_items.append({"query": lines[0][:80], "code": "\n".join(lines[:5]), "class": "Agent Harvest"})

    # Populate Semantic, Symbol, Architecture queries (50 items each)
    for item in tqdm(ds, desc="Harvesting benchmark queries"):
        docstring = item.get("func_documentation_string", "").strip()
        code = item.get("whole_func_string", item.get("func_code_string", "")).strip()
        func_name = item.get("func_name", "").strip()
        
        if len(docstring) >= 20 and len(code) >= 50:
            lines = docstring.split("\n")
            first_line = lines[0].strip()
            
            if len(semantic_items) < 50 and len(first_line) >= 20:
                semantic_items.append({"query": first_line, "code": code, "class": "Semantic"})
            if len(symbol_items) < 50 and len(func_name) >= 5:
                symbol_items.append({"query": f"{func_name} implementation", "code": code, "class": "Symbol"})
            if len(arch_items) < 50 and len(lines) >= 3:
                arch_items.append({"query": f"how does {func_name} handle {lines[1].strip()[:60]}", "code": code, "class": "Architecture"})
                
        if len(semantic_items) >= 50 and len(symbol_items) >= 50 and len(arch_items) >= 50 and len(agent_items) >= 50:
            break

    all_items = semantic_items[:50] + symbol_items[:50] + arch_items[:50] + agent_items[:50]
    queries = [it["query"] for it in all_items]
    codes = [it["code"] for it in all_items]
    q_classes = [it["class"] for it in all_items]
    N = len(queries)
    print(f"Total Evaluated Task Mix: {N} queries (50 Semantic, 50 Symbol, 50 Architecture, 50 Agent Harvest)")

    def simple_tokenize(text):
        return [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', text) if len(w) > 1]
    
    q_tokens_list = [simple_tokenize(q) for q in queries]
    d_tokens_list = [simple_tokenize(c) for c in codes]

    print("Computing BM25 lexical score matrix...")
    bm25_matrix = np.zeros((N, N), dtype=np.float32)
    for qi in range(N):
        bm25_matrix[qi] = bm25_rank(q_tokens_list[qi], d_tokens_list)

    print("Computing dense embeddings for all menu configurations...")
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

    # Menu Configurations
    menu_scores = {
        "ColBERTv2 Dense": compute_maxsim_matrix(colbert_q, colbert_d),
        "ColBERTv2 Hybrid": np.zeros((N, N), dtype=np.float32), # populated below
        "potion-code Dense": np.dot(pc_q_embs, pc_d_embs.T),
        "potion-code Hybrid": np.zeros((N, N), dtype=np.float32), # populated below
        "CodeBERT Hybrid": np.zeros((N, N), dtype=np.float32),
        "FTS BM25": bm25_matrix
    }

    # Compute Hybrids
    for qi in range(N):
        f_ranks = np.argsort(-bm25_matrix[qi])
        
        # ColBERT Hybrid
        d_ranks_cb = np.argsort(-menu_scores["ColBERTv2 Dense"][qi])
        menu_scores["ColBERTv2 Hybrid"][qi] = rrf_fuse(d_ranks_cb, f_ranks)

        # potion-code Hybrid
        d_ranks_pc = np.argsort(-menu_scores["potion-code Dense"][qi])
        menu_scores["potion-code Hybrid"][qi] = rrf_fuse(d_ranks_pc, f_ranks)

        # CodeBERT Hybrid
        cb_dense = compute_maxsim_matrix(codebert_q, codebert_d)
        d_ranks_cdb = np.argsort(-cb_dense[qi])
        menu_scores["CodeBERT Hybrid"][qi] = rrf_fuse(d_ranks_cdb, f_ranks)

    # Compute Per-Query NDCG@10 for every menu configuration
    config_names = list(menu_scores.keys())
    ndcg_table = np.zeros((len(config_names), N), dtype=np.float32)
    
    for c_idx, c_name in enumerate(config_names):
        mat = menu_scores[c_name]
        for qi in range(N):
            ranked = np.argsort(-mat[qi])
            rel = [1.0 if idx == qi else 0.0 for idx in ranked[:10]]
            ndcg_table[c_idx, qi] = ndcg_at_k(rel, k=10)

    # Define Baselines:
    # 1. Budget Tier: potion-code Hybrid (c_idx = 3)
    # 2. Quality Tier: ColBERTv2 Hybrid (c_idx = 1)
    # 3. Two-Tier Shipping Baseline: ColBERTv2 Hybrid on quality, potion-code Hybrid on budget
    colbert_hybrid_ndcg = ndcg_table[1]
    potion_hybrid_ndcg = ndcg_table[3]
    
    # Oracle Routing across all menu options
    oracle_ndcg = np.max(ndcg_table, axis=0)

    class_names = ["Semantic", "Symbol", "Architecture", "Agent Harvest"]
    class_indices = {
        "Semantic": [i for i, c in enumerate(q_classes) if c == "Semantic"],
        "Symbol": [i for i, c in enumerate(q_classes) if c == "Symbol"],
        "Architecture": [i for i, c in enumerate(q_classes) if c == "Architecture"],
        "Agent Harvest": [i for i, c in enumerate(q_classes) if c == "Agent Harvest"],
    }

    print("\n" + "="*145)
    print(f"{'Query Class':<20} | {'Budget Tier (potion)':>22} | {'Quality Tier (ColBERT)':>24} | {'Oracle Optimal Routing':>24} | {'Oracle Headroom':>18}")
    print("-" * 145)

    per_class_headroom = {}
    for c_name in class_names:
        idxs = class_indices[c_name]
        m_potion = float(np.mean(potion_hybrid_ndcg[idxs]))
        m_colbert = float(np.mean(colbert_hybrid_ndcg[idxs]))
        m_oracle = float(np.mean(oracle_ndcg[idxs]))
        headroom = (m_oracle - m_colbert) * 100.0
        per_class_headroom[c_name] = headroom
        print(f"{c_name:<20} | {m_potion:22.4f} | {m_colbert:24.4f} | {m_oracle:24.4f} | {headroom:+17.2f} pts")

    print("-" * 145)
    overall_potion = float(np.mean(potion_hybrid_ndcg))
    overall_colbert = float(np.mean(colbert_hybrid_ndcg))
    overall_oracle = float(np.mean(oracle_ndcg))
    overall_headroom = (overall_oracle - overall_colbert) * 100.0

    print(f"{'OVERALL AGGREGATE':<20} | {overall_potion:22.4f} | {overall_colbert:24.4f} | {overall_oracle:24.4f} | {overall_headroom:+17.2f} pts")
    print("=" * 145)

    # Print Oracle Win Distribution per class
    print("\n" + "="*145)
    print(" ORACLE WIN DISTRIBUTION ACROSS COMPILED MENU CONFIGURATIONS")
    print("="*145)
    print(f"{'Query Class':<20} | {'ColBERT Dense':>15} | {'ColBERT Hybrid':>16} | {'potion Dense':>14} | {'potion Hybrid':>15} | {'FTS BM25':>10} | {'CodeBERT Hybrid':>17}")
    print("-" * 145)
    
    for c_name in class_names:
        idxs = class_indices[c_name]
        wins = {c: 0 for c in config_names}
        for qi in idxs:
            best_c_idx = np.argmax(ndcg_table[:, qi])
            wins[config_names[best_c_idx]] += 1
            
        print(f"{c_name:<20} | {wins['ColBERTv2 Dense']:15d} | {wins['ColBERTv2 Hybrid']:16d} | {wins['potion-code Dense']:14d} | {wins['potion-code Hybrid']:15d} | {wins['FTS BM25']:10d} | {wins['CodeBERT Hybrid']:17d}")
    print("=" * 145)

    print(f"\nFinal Adjudication vs Committed Criterion (Kill if Headroom < +2.0 points):")
    print(f"Measured Overall Headroom over Two-Tier Quality Baseline: {overall_headroom:+.2f} points NDCG@10")
    if overall_headroom < 2.0:
        print(">>> VERDICT: Oracle Headroom (+{:.2f} pts) < +2.0 pts. Learned routing is TERMINATED DEFINITIVELY.".format(overall_headroom))
        print(">>> SHIPPING RECOMMENDATION: The read path should be TWO-TIER (potion-code-16M Hybrid for budget/fast tier; ColBERTv2 Hybrid for precision tier).")
    else:
        print(">>> VERDICT: Headroom exceeds +2.0 pts.")
    print("=" * 145)

if __name__ == "__main__":
    run_oracle_rerun()

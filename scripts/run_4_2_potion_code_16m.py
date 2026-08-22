import os
import numpy as np
import torch
import torch.nn.functional as F
from model2vec import StaticModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm

LANGUAGES = [
    ("go", "Go"),
    ("rust", "Rust"),
    ("javascript", "TypeScript"),
    ("java", "Java"),
    ("python", "Python"),
]

def quantize_1bit(x): return np.sign(x) / np.sqrt(128)
def quantize_2bit(x):
    levels = np.array([-1.510, -0.453, 0.453, 1.510], dtype=np.float32) / np.sqrt(128)
    idx = np.digitize(x * np.sqrt(128), [-0.98, 0.0, 0.98])
    return levels[idx]
def quantize_3bit(x):
    levels = np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152], dtype=np.float32) / np.sqrt(128)
    idx = np.digitize(x * np.sqrt(128), [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748])
    return levels[idx]

def run_potion_code_experiment():
    print("\n" + "="*125)
    print(" RUN 4.2: CODE-TRAINED STATIC ENCODER (MinishLab/potion-code-16M)")
    print("="*125)

    print("Loading StaticModel: MinishLab/potion-code-16M...")
    model = StaticModel.from_pretrained("MinishLab/potion-code-16M")
    raw_table = model.embedding # (61826, 256)
    print(f"Code-Trained Vocabulary: {raw_table.shape[0]} tokens, Raw Dimension: {raw_table.shape[1]}")

    # Fixed orthogonal projection 256 -> 128
    np.random.seed(42)
    proj_mat = np.random.randn(256, 128).astype(np.float32)
    q_proj, _ = np.linalg.qr(proj_mat)
    
    proj_table = np.dot(raw_table, q_proj)
    norm_table = proj_table / np.linalg.norm(proj_table, axis=1, keepdims=True)

    print("Streaming code snippets to evaluate code corpora under potion-code-16M...")
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    lang_snippets = {l[0]: [] for l in LANGUAGES}
    for item in tqdm(ds, desc="Streaming CodeSearchNet"):
        lang = item.get("language", "").lower()
        if lang in lang_snippets:
            code = item.get("whole_func_string", item.get("func_code_string", ""))
            docstring = item.get("func_documentation_string", "")
            name = item.get("func_name", "")
            q_text = docstring if len(docstring.strip()) >= 10 else f"{lang} {name}"
            if len(code.strip()) > 30 and len(lang_snippets[lang]) < 300:
                lang_snippets[lang].append({"code": code, "query": q_text})
        if all(len(lang_snippets[l[0]]) >= 300 for l in LANGUAGES):
            break

    # Add local Rust code if needed
    if len(lang_snippets["rust"]) < 300:
        import glob
        for fpath in glob.glob("src/**/*.rs", recursive=True):
            with open(fpath, "r", errors="ignore") as f:
                lang_snippets["rust"].append({"code": f.read(), "query": "rust function"})
            if len(lang_snippets["rust"]) >= 300:
                break

    print("\n" + "-"*130)
    print(f"{'Language':<14} | {'potion-code-16M Redun':>23} | {'potion-base-8M Redun':>22} | {'CodeBERT Redun':>16} | {'Gap Γ':>8} | {'MSE@1.35b':>11} | {'MaxSim R@10(1b, 2b, 3b)':>25}")
    print("-" * 130)

    potion_base_reds = {"go": 94.2, "java": 85.1, "rust": 83.4, "python": 77.0, "javascript": 76.5}
    codebert_reds = {"go": 84.9, "java": 47.9, "rust": 42.6, "javascript": 17.5, "python": 15.5}

    potion_code_reds_list = []
    codebert_reds_list = []

    for lang_key, lang_name in LANGUAGES:
        items = lang_snippets[lang_key]
        all_token_vectors = []
        doc_token_blocks = []
        query_token_blocks = []

        for it in items:
            code = it["code"]
            q_text = it["query"]
            
            # Tokenize code with fast tokenizer
            code_enc = model.tokenizer.encode(code)
            code_ids = [cid for cid in code_enc.ids if cid < len(norm_table)][:32]
            if len(code_ids) >= 8:
                vecs = norm_table[code_ids]
                all_token_vectors.extend(vecs)
                doc_token_blocks.append(vecs[:32])

            q_enc = model.tokenizer.encode(q_text)
            q_ids = [qid for qid in q_enc.ids if qid < len(norm_table)][:8]
            if len(q_ids) >= 4:
                q_vecs = norm_table[q_ids]
                query_token_blocks.append(q_vecs[:8])

        token_mat = np.array(all_token_vectors, dtype=np.float32)[:20000]

        # 1. Redundancy
        km = MiniBatchKMeans(n_clusters=512, random_state=42, batch_size=1024).fit(token_mat)
        _, dists = pairwise_distances_argmin_min(token_mat, km.cluster_centers_)
        red_code = float(np.mean(dists <= 0.25)) * 100.0

        # 2. Quant Gap & Eff Dim
        cov = np.cov(token_mat, rowvar=False)
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-9)
        eff_dim = float((np.sum(eigvals)**2) / np.sum(eigvals**2))
        km64 = MiniBatchKMeans(n_clusters=64, random_state=42, batch_size=1024).fit(token_mat)
        d_km = float(km64.inertia_ / (len(token_mat) * 128))
        d_g = float(np.mean(eigvals) * (2.0 ** (-2.0 * 6.0 / eff_dim)))
        gamma = float(d_g / max(d_km, 1e-7))

        # 3. Dictionary Coding Distortion
        km256 = MiniBatchKMeans(n_clusters=256, random_state=42, batch_size=1024).fit(token_mat)
        nearest_c = km256.cluster_centers_[km256.predict(token_mat)]
        residuals = token_mat - nearest_c
        q_res = quantize_1bit(residuals) * np.std(residuals)
        recon = nearest_c + q_res
        mse_dict = float(np.mean((token_mat - recon)**2))

        # 4. MaxSim Retrieval Simulation
        n_docs = min(150, len(doc_token_blocks))
        n_queries = min(50, len(query_token_blocks))
        
        doc_mats = [d for d in doc_token_blocks[:n_docs]]
        query_mats = [q for q in query_token_blocks[:n_queries]]

        exact_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
        for qi in range(n_queries):
            for di in range(n_docs):
                cross = np.dot(query_mats[qi], doc_mats[di].T)
                exact_scores[qi, di] = np.sum(np.max(cross, axis=1))
        gt_top10 = np.argsort(-exact_scores, axis=1)[:, :10]

        maxsim_recalls = {}
        for b, q_fn in [(1, quantize_1bit), (2, quantize_2bit), (3, quantize_3bit)]:
            q_doc_mats = [q_fn(d) for d in doc_mats]
            est_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
            for qi in range(n_queries):
                for di in range(n_docs):
                    cross = np.dot(query_mats[qi], q_doc_mats[di].T)
                    est_scores[qi, di] = np.sum(np.max(cross, axis=1))
            est_top10 = np.argsort(-est_scores, axis=1)[:, :10]
            rec = np.mean([len(set(gt_top10[q]).intersection(set(est_top10[q]))) / 10.0 for q in range(n_queries)])
            maxsim_recalls[b] = float(rec)

        potion_code_reds_list.append(red_code)
        codebert_reds_list.append(codebert_reds[lang_key])

        base_red = potion_base_reds[lang_key]
        cb_red = codebert_reds[lang_key]
        m_str = f"{maxsim_recalls[1]*100:.1f}%, {maxsim_recalls[2]*100:.1f}%, {maxsim_recalls[3]*100:.1f}%"
        print(f"{lang_name:<14} | {red_code:22.1f}% | {base_red:21.1f}% | {cb_red:15.1f}% | {gamma:7.2f}x | {mse_dict:11.4f} | {m_str:>25}")

    print("-" * 130)
    rho_cb, p_cb = spearmanr(potion_code_reds_list, codebert_reds_list)
    print(f"1. Spearman Correlation (potion-code-16M vs CodeBERT): ρ = {rho_cb:.4f} (p = {p_cb:.4e}) [Pre-registered target: >= 0.80]")
    print("=" * 130)

if __name__ == "__main__":
    run_potion_code_experiment()

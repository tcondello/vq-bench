import h5py
import numpy as np

DATASETS = [
    ("python", "colbert-python-128-normalized.hdf5"),
    ("rust", "colbert-rust-128-normalized.hdf5"),
    ("go", "colbert-go-128-normalized.hdf5"),
    ("java", "colbert-java-128-normalized.hdf5"),
    ("typescript", "colbert-typescript-128-normalized.hdf5"),
    ("msmarco_text", "msmarco-colbert-128-normalized.hdf5"),
]

def measure_effective_margins():
    print("\n" + "="*125)
    print(" CYCLE 4 ITEM 3: MEASURING EFFECTIVE MAXSIM MARGINS (M_bar_MaxSim vs M_bar_Pooled)")
    print("="*125)
    print(f"{'Corpus / Source':<18} | {'μ_Δ (Pooled)':>14} | {'M_bar (Pooled)':>16} | {'μ_Δ (MaxSim)':>14} | {'M_bar (MaxSim)':>16} | {'Margin Multiplier κ':>22}")
    print("-" * 125)

    margin_table = {}
    
    for lang, filename in DATASETS:
        path = f"data/{filename}"
        with h5py.File(path, "r") as f:
            base = f["base"][:64000] # (64000, 128)
            eval_q = f["eval"][:500]  # (500, 128)
            
        # 1. Single-Vector Pooled Margins
        pooled_scores = np.dot(eval_q, base[:5000].T) # (500, 5000)
        sorted_p = np.sort(pooled_scores, axis=1)[:, ::-1]
        delta_p = sorted_p[:, 0] - sorted_p[:, 9]
        mu_p = float(np.mean(delta_p))
        sigma_p = float(np.mean(np.std(pooled_scores, axis=1)))
        m_bar_p = float(mu_p / max(sigma_p, 1e-6))
        
        # 2. Multi-Vector MaxSim Margins (L=32 tokens/doc, M=8 tokens/query)
        n_docs = 1500
        doc_len = 32
        query_len = 8
        n_queries = 150
        
        doc_tokens = base[:n_docs * doc_len].reshape(n_docs, doc_len, 128)
        
        np.random.seed(42)
        q_doc_indices = np.random.choice(n_docs, size=n_queries, replace=False)
        queries = np.zeros((n_queries, query_len, 128), dtype=np.float32)
        for i, d_idx in enumerate(q_doc_indices):
            q_toks = doc_tokens[d_idx, :query_len] + np.random.randn(query_len, 128).astype(np.float32) * 0.10
            queries[i] = q_toks / np.linalg.norm(q_toks, axis=1, keepdims=True)

        maxsim_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
        for q_idx in range(n_queries):
            cross = np.dot(queries[q_idx], doc_tokens.reshape(-1, 128).T).reshape(query_len, n_docs, doc_len)
            maxsim_scores[q_idx] = np.sum(np.max(cross, axis=2), axis=0)

        sorted_m = np.sort(maxsim_scores, axis=1)[:, ::-1]
        delta_m = sorted_m[:, 0] - sorted_m[:, 9]
        mu_m = float(np.mean(delta_m))
        sigma_m = float(np.mean(np.std(maxsim_scores, axis=1)))
        m_bar_m = float(mu_m / max(sigma_m, 1e-6))
        
        multiplier = m_bar_m / max(m_bar_p, 1e-6)
        
        margin_table[lang] = {
            "mu_p": mu_p,
            "m_bar_p": m_bar_p,
            "mu_m": mu_m,
            "m_bar_m": m_bar_m,
            "kappa": multiplier
        }
        
        print(f"{lang:<18} | {mu_p:14.4f} | {m_bar_p:16.4f} | {mu_m:14.4f} | {m_bar_m:16.4f} | {multiplier:21.2f}x")

    print("=" * 125)
    return margin_table

if __name__ == "__main__":
    measure_effective_margins()

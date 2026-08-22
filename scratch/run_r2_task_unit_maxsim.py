import h5py
import numpy as np
import math

SEEDS = [1, 2, 3, 4, 5]

def run_maxsim_experiment():
    print("\n" + "="*95)
    print(" RUN 2 (AXIS 2: TASK UNIT) — LATE-INTERACTION MAXSIM AGGREGATION ON PYTHON CODE")
    print("="*95)
    
    with h5py.File("data/colbert-python-128-normalized.hdf5", "r") as f:
        base = f["base"][:250000] # (250000, 128)
        
    n_docs = 2000
    doc_len = 32
    query_len = 8
    n_queries = 200
    
    # Structure base into 2,000 multi-token documents of 32 tokens each
    doc_tokens = base[:n_docs * doc_len].reshape(n_docs, doc_len, 128)
    
    # Create multi-token queries from held-out base documents
    np.random.seed(42)
    query_doc_indices = np.random.choice(n_docs, size=n_queries, replace=False)
    queries = np.zeros((n_queries, query_len, 128), dtype=np.float32)
    
    for i, d_idx in enumerate(query_doc_indices):
        # Query tokens from the document + slight perturbation
        q_toks = doc_tokens[d_idx, :query_len] + np.random.randn(query_len, 128).astype(np.float32) * 0.10
        queries[i] = q_toks / np.linalg.norm(q_toks, axis=1, keepdims=True)

    print("Computing exact uncompressed MaxSim ground truth...")
    exact_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
    for q_idx in range(n_queries):
        q_mat = queries[q_idx] # (M, d)
        for d_idx in range(n_docs):
            d_mat = doc_tokens[d_idx] # (L, d)
            # (M, L) cross dot products
            cross = np.dot(q_mat, d_mat.T)
            exact_scores[q_idx, d_idx] = np.sum(np.max(cross, axis=1))

    # Exact top-10 ground truth documents
    gt_top10 = np.argsort(-exact_scores, axis=1)[:, :10]

    # Evaluate MaxSim under 1-bit, 2-bit, and 3-bit scalar & ColBERTv2 quantizers
    def quantize_1bit(x):
        return np.sign(x) / np.sqrt(128)
        
    def quantize_2bit(x):
        # 4 Lloyd-Max levels for standard normal
        levels = np.array([-1.510, -0.453, 0.453, 1.510], dtype=np.float32) / np.sqrt(128)
        idx = np.digitize(x * np.sqrt(128), [-0.98, 0.0, 0.98])
        return levels[idx]

    def quantize_3bit(x):
        # 8 Lloyd-Max levels
        levels = np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152], dtype=np.float32) / np.sqrt(128)
        thresholds = np.array([-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748], dtype=np.float32)
        idx = np.digitize(x * np.sqrt(128), thresholds)
        return levels[idx]

    print("\n" + "-"*105)
    print(f"{'Quantization Scheme':<40} | {'b/d':>6} | {'MaxSim Recall@10':>20} | {'Pooled Recall@10':>20}")
    print("-" * 105)

    schemes = [
        ("1-Bit Sign Quantizer (b=1)", 1.00, quantize_1bit, 0.2391),
        ("2-Bit Lloyd-Max (b=2)", 2.00, quantize_2bit, 0.2747),
        ("3-Bit Lloyd-Max (b=3)", 3.00, quantize_3bit, 0.4501),
    ]

    for name, bpd, quant_fn, pooled_r in schemes:
        quant_docs = quant_fn(doc_tokens)
        
        est_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
        for q_idx in range(n_queries):
            q_mat = queries[q_idx]
            for d_idx in range(n_docs):
                d_mat = quant_docs[d_idx]
                cross = np.dot(q_mat, d_mat.T)
                est_scores[q_idx, d_idx] = np.sum(np.max(cross, axis=1))

        est_top10 = np.argsort(-est_scores, axis=1)[:, :10]
        
        # Overlap with ground truth
        recalls = []
        for q_idx in range(n_queries):
            hits = len(set(gt_top10[q_idx]).intersection(set(est_top10[q_idx])))
            recalls.append(hits / 10.0)
        mean_recall = np.mean(recalls)
        print(f"{name:<40} | {bpd:6.2f} | {mean_recall*100:18.2f}% | {pooled_r*100:18.2f}%")

if __name__ == "__main__":
    run_maxsim_experiment()

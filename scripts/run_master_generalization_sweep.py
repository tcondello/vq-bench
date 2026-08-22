import json
import os
import subprocess
import math
import h5py
import numpy as np
from scipy.stats import spearmanr

SEEDS = [1, 2, 3, 4, 5]
SWEEP_DATASETS = [
    ("json", "colbert-json-128-normalized"),
    ("go", "colbert-go-128-normalized"),
    ("c", "colbert-c-128-normalized"),
    ("rust", "colbert-rust-128-normalized"),
    ("java", "colbert-java-128-normalized"),
    ("typescript", "colbert-typescript-128-normalized"),
    ("python", "colbert-python-128-normalized"),
    ("markdown", "colbert-markdown-128-normalized"),
    ("msmarco_text", "msmarco-colbert-128-normalized"),
]

def run_cmd(cmd):
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C link-args=-Wl,-rpath," + os.path.expanduser("~/.local/hdf5-1.14/lib")
    env["HDF5_DIR"] = os.path.expanduser("~/.local/hdf5-1.14")
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("ERROR running cmd:", cmd)
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout

def run_pooled_sweep():
    print("\n" + "="*110)
    print(" EXECUTING POOLED SINGLE-VECTOR RETRIEVAL SWEEP (9 LANGUAGES, 5 SEEDS)")
    print("="*110)
    
    pooled_results = {}
    for lang, ds_name in SWEEP_DATASETS:
        print(f"\n>>> Running Dataset: {ds_name} ({lang})")
        lang_runs = []
        for seed in SEEDS:
            config = {
                "datasets": [ds_name],
                "seed": seed,
                "n_fit": 20000,
                "n_reconstruct": 1000,
                "n_eval": 1000,
                "k": [1, 10, 100],
                "temp": [0.05, 0.2, 1.0],
                "methods": [
                    { "name": "scalar", "b": [1, 2, 3] },
                    { "name": "colbertv2_quant", "centroids": [256], "residual_bits": [1, 2, 3] }
                ],
                "metrics": ["recall", "mse_score", "mse_recon"]
            }
            cfg_file = f"configs/sweep-{ds_name}-seed-{seed}.json"
            with open(cfg_file, "w") as f:
                json.dump(config, f, indent=2)
                
            run_cmd(f"cargo run --release -- run {cfg_file} --stream")
            
            with open(f"results/sweep-{ds_name}-seed-{seed}.json") as f:
                lang_runs.append(json.load(f))
        pooled_results[lang] = lang_runs
    return pooled_results

def run_maxsim_sweep():
    print("\n" + "="*110)
    print(" EXECUTING LATE-INTERACTION MAXSIM AGGREGATION SWEEP (9 LANGUAGES)")
    print("="*110)
    
    def quantize_1bit(x): return np.sign(x) / np.sqrt(128)
    def quantize_2bit(x):
        levels = np.array([-1.510, -0.453, 0.453, 1.510], dtype=np.float32) / np.sqrt(128)
        idx = np.digitize(x * np.sqrt(128), [-0.98, 0.0, 0.98])
        return levels[idx]
    def quantize_3bit(x):
        levels = np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152], dtype=np.float32) / np.sqrt(128)
        idx = np.digitize(x * np.sqrt(128), [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748])
        return levels[idx]

    maxsim_results = {}
    
    for lang, ds_name in SWEEP_DATASETS:
        path = f"data/{ds_name}.hdf5"
        with h5py.File(path, "r") as f:
            base = f["base"][:64000] # 2000 docs of 32 tokens
            
        n_docs = 2000
        doc_len = 32
        query_len = 8
        n_queries = 200
        
        doc_tokens = base[:n_docs * doc_len].reshape(n_docs, doc_len, 128)
        
        np.random.seed(42)
        q_doc_indices = np.random.choice(n_docs, size=n_queries, replace=False)
        queries = np.zeros((n_queries, query_len, 128), dtype=np.float32)
        for i, d_idx in enumerate(q_doc_indices):
            q_toks = doc_tokens[d_idx, :query_len] + np.random.randn(query_len, 128).astype(np.float32) * 0.10
            queries[i] = q_toks / np.linalg.norm(q_toks, axis=1, keepdims=True)

        # Ground truth
        exact_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
        for q_idx in range(n_queries):
            cross = np.dot(queries[q_idx], doc_tokens.reshape(-1, 128).T).reshape(query_len, n_docs, doc_len)
            exact_scores[q_idx] = np.sum(np.max(cross, axis=2), axis=0)
            
        gt_top10 = np.argsort(-exact_scores, axis=1)[:, :10]
        
        lang_maxsim = {}
        for b_name, quant_fn in [(1, quantize_1bit), (2, quantize_2bit), (3, quantize_3bit)]:
            q_docs = quant_fn(doc_tokens)
            est_scores = np.zeros((n_queries, n_docs), dtype=np.float32)
            for q_idx in range(n_queries):
                cross = np.dot(queries[q_idx], q_docs.reshape(-1, 128).T).reshape(query_len, n_docs, doc_len)
                est_scores[q_idx] = np.sum(np.max(cross, axis=2), axis=0)
            est_top10 = np.argsort(-est_scores, axis=1)[:, :10]
            
            recalls = [len(set(gt_top10[q]).intersection(set(est_top10[q]))) / 10.0 for q in range(n_queries)]
            lang_maxsim[b_name] = float(np.mean(recalls))
            
        maxsim_results[lang] = lang_maxsim
        print(f"  {lang:<15} --> MaxSim R@10: 1b={lang_maxsim[1]*100:.1f}%, 2b={lang_maxsim[2]*100:.1f}%, 3b={lang_maxsim[3]*100:.1f}%")
        
    return maxsim_results

def mean_std(arr):
    m = sum(arr) / len(arr)
    v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
    return m, math.sqrt(v)

def analyze_all(pooled_res, maxsim_res):
    print("\n" + "#"*135)
    print(" CYCLE 2 MASTER GENERALIZATION ANALYSIS: RIGIDITY GRADIENT & LAW VALIDATION")
    print("#"*135)
    
    # Predicted curves table from pre-registration
    predicted_curves = {
        "json":        {"p1": 0.35, "p2": 0.45, "p3": 0.65, "m1": 0.92, "m2": 0.96, "m3": 0.98},
        "go":          {"p1": 0.28, "p2": 0.36, "p3": 0.55, "m1": 0.82, "m2": 0.88, "m3": 0.92},
        "c":           {"p1": 0.32, "p2": 0.42, "p3": 0.60, "m1": 0.85, "m2": 0.90, "m3": 0.94},
        "rust":        {"p1": 0.30, "p2": 0.38, "p3": 0.56, "m1": 0.80, "m2": 0.86, "m3": 0.91},
        "java":        {"p1": 0.25, "p2": 0.32, "p3": 0.50, "m1": 0.76, "m2": 0.83, "m3": 0.88},
        "typescript":  {"p1": 0.26, "p2": 0.33, "p3": 0.51, "m1": 0.78, "m2": 0.84, "m3": 0.89},
        "python":      {"p1": 0.24, "p2": 0.30, "p3": 0.48, "m1": 0.77, "m2": 0.84, "m3": 0.89},
        "markdown":    {"p1": 0.30, "p2": 0.40, "p3": 0.58, "m1": 0.82, "m2": 0.88, "m3": 0.93},
        "msmarco_text":{"p1": 0.70, "p2": 0.72, "p3": 0.82, "m1": 0.80, "m2": 0.87, "m3": 0.92},
    }

    print(f"{'Source':<14} | {'MSE@1.35b':>10} | {'Pooled R@10 (1b, 2b, 3b)':>26} | {'MaxSim R@10 (1b, 2b, 3b)':>26} | {'Band Hits (out of 6)':>20}")
    print("-" * 135)
    
    measured_mses = []
    total_predictions = 0
    total_hits = 0
    
    for lang, _ in SWEEP_DATASETS:
        runs = pooled_res[lang]
        
        # Extract ColBERTv2 1.35b MSE
        colbert_1b = [m for m in runs[0]["datasets"][0]["methods"] if "residual_bits=1" in m["label"] or "b=1" in m["label"]][0]
        mse_1b = colbert_1b["mse_recon"]
        measured_mses.append(mse_1b)
        
        # Extract Pooled 1b, 2b, 3b
        p_1b = np.mean([[m["recalls"]["10"] for m in r["datasets"][0]["methods"] if m["label"] == "Scalar (b=1)"][0] for r in runs])
        p_2b = np.mean([[m["recalls"]["10"] for m in r["datasets"][0]["methods"] if m["label"] == "Scalar (b=2)"][0] for r in runs])
        p_3b = np.mean([[m["recalls"]["10"] for m in r["datasets"][0]["methods"] if m["label"] == "Scalar (b=3)"][0] for r in runs])
        
        # Extract MaxSim 1b, 2b, 3b
        m_1b = maxsim_res[lang][1]
        m_2b = maxsim_res[lang][2]
        m_3b = maxsim_res[lang][3]
        
        # Check against pre-registered bands (± 0.05)
        pred = predicted_curves[lang]
        hits = 0
        if abs(p_1b - pred["p1"]) <= 0.05: hits += 1
        if abs(p_2b - pred["p2"]) <= 0.05: hits += 1
        if abs(p_3b - pred["p3"]) <= 0.05: hits += 1
        if abs(m_1b - pred["m1"]) <= 0.05: hits += 1
        if abs(m_2b - pred["m2"]) <= 0.05: hits += 1
        if abs(m_3b - pred["m3"]) <= 0.05: hits += 1
        
        total_predictions += 6
        total_hits += hits
        
        p_str = f"{p_1b:.3f}, {p_2b:.3f}, {p_3b:.3f}"
        m_str = f"{m_1b:.3f}, {m_2b:.3f}, {m_3b:.3f}"
        print(f"{lang:<14} | {mse_1b:10.4f} | {p_str:>26} | {m_str:>26} | {hits:>12}/6")

    print("-" * 135)
    
    # 1. Rigidity Ordering Rank Correlation
    # Predicted rank ordering (1 to 9): json=1, go=2, c=3, rust=4, java=5, ts=6, py=7, md=8, text=9
    predicted_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rho, pval = spearmanr(predicted_ranks, measured_mses)
    
    hit_rate = (total_hits / total_predictions) * 100.0
    print(f"\n1. Rigidity Gradient Rank Correlation (Predicted Rigidity Rank vs Measured MSE): ρ = {rho:.4f} (p = {pval:.4e})")
    print(f"2. The Law's Prediction Band Hit Rate: {total_hits}/{total_predictions} ({hit_rate:.1f}% inside pre-registered ±5% error bands)")
    print("=" * 135)

if __name__ == "__main__":
    p_res = run_pooled_sweep()
    m_res = run_maxsim_sweep()
    analyze_all(p_res, m_res)

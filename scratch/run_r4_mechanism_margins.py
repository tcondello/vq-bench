import h5py
import numpy as np
from scipy.stats import spearmanr

DATASETS = [
    ("imagenet-clip-512-normalized", 0.8900),
    ("cifar100-clip-512-normalized", 0.8508),
    ("msmarco-colbert-128-normalized", 0.8150),
    ("coco-nomic-768-normalized", 0.7780),
    ("colbert-rust-128-normalized", 0.5216),
    ("colbert-python-128-normalized", 0.4501),
    ("synthetic-shells-128", 0.5718),
]

def analyze_margins():
    print("\n" + "="*110)
    print(" RUN 4 (AXIS 4: MECHANISM) — QUERY-TO-CANDIDATE MARGIN DISTRIBUTION ACROSS CORPORA")
    print("="*110)
    print(f"{'Dataset':<32} | {'Dim':>5} | {'Mean Margin μ_Δ':>16} | {'Score Std σ_s':>14} | {'Norm Margin M_bar':>18} | {'R@10(3b)':>10}")
    print("-" * 110)
    
    m_bars = []
    r10s = []
    
    for ds_name, measured_r10 in DATASETS:
        path = f"data/{ds_name}.hdf5"
        with h5py.File(path, "r") as f:
            eval_q = f["eval"][:1000]
            base = f["base"][:20000]
            
        scores = np.dot(eval_q, base.T) # (1000, 20000)
        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        
        # Margin between rank 1 and rank 10
        margins = sorted_scores[:, 0] - sorted_scores[:, 9]
        mu_delta = float(np.mean(margins))
        
        # Standard deviation of candidate pool scores per query
        score_stds = np.std(scores, axis=1)
        sigma_s = float(np.mean(score_stds))
        
        m_bar = mu_delta / max(sigma_s, 1e-6)
        m_bars.append(m_bar)
        r10s.append(measured_r10)
        
        dim = eval_q.shape[1]
        print(f"{ds_name:<32} | {dim:5d} | {mu_delta:16.4f} | {sigma_s:14.4f} | {m_bar:18.4f} | {measured_r10:10.4f}")

    rho, pval = spearmanr(m_bars, r10s)
    print("-" * 110)
    print(f"Rank Correlation between Normalized Margin M_bar and Recall@10(3b): ρ = {rho:.4f} (p = {pval:.4e})")
    print("=" * 110)

if __name__ == "__main__":
    analyze_margins()

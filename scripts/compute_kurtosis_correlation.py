#!/usr/bin/env python3
"""
Corrected Kurtosis & Variance Diagnostic with Empirical Delta Correlation Study.

Loads fit/base vectors across datasets, computes centered sample moments:
- Max/Med Variance Ratio
- Top 5% Variance Fraction
- Median, Max, and Top 5% Excess Kurtosis
- Multi-modal cluster ratio (Within-cluster variance / Total variance)
- Correlates each metric against empirical SpikeEden vs EDEN-prod delta (Delta R@10).
"""

import os
import sys
import numpy as np
import h5py

def compute_dataset_stats(h5_path: str, n_samples: int = 20000, seed: int = 1):
    with h5py.File(h5_path, "r") as f:
        base = f["base"]
        n_total, dim = base.shape
        
        # Subsample deterministically
        rng = np.random.RandomState(seed)
        n_take = min(n_samples, n_total)
        indices = np.sort(rng.choice(n_total, size=n_take, replace=False))
        X = base[indices, :].astype(np.float64)

    # 1. Centering & 2nd/4th Central Moments
    mu = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mu
    
    var = np.mean(X_centered ** 2, axis=0) # [dim]
    var_safe = np.maximum(var, 1e-12)
    
    m4 = np.mean(X_centered ** 4, axis=0) # [dim]
    kurt = (m4 / (var_safe ** 2)) - 3.0 # Excess kurtosis
    
    # 2. Variance Metrics
    med_var = np.median(var)
    max_var = np.max(var)
    max_med_var_ratio = max_var / max(med_var, 1e-12)
    
    # Top 5% channels by variance
    k_5pct = max(1, int(np.round(0.05 * dim)))
    sorted_var = np.sort(var)[::-1]
    top_5pct_var_frac = np.sum(sorted_var[:k_5pct]) / np.sum(var)
    
    # 3. Kurtosis Metrics
    med_kurt = np.median(kurt)
    max_kurt = np.max(kurt)
    
    # Top 5% channels by kurtosis
    sorted_kurt = np.sort(kurt)[::-1]
    top_5pct_kurt = np.mean(sorted_kurt[:k_5pct])
    
    # 4. Multi-modal Cluster Ratio (Within-Cluster Variance / Total Variance) via fast mini-batch k-means
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=min(32, n_take // 50), random_state=seed, batch_size=1024, n_init=3)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_[kmeans.labels_]
    within_ss = np.sum((X - centers) ** 2)
    total_ss = np.sum(X_centered ** 2)
    within_cluster_var_ratio = within_ss / max(total_ss, 1e-12)

    return {
        "dim": dim,
        "n_samples": n_take,
        "max_med_var": max_med_var_ratio,
        "top5_var_frac": top_5pct_var_frac,
        "med_kurt": med_kurt,
        "max_kurt": max_kurt,
        "top5_kurt": top_5pct_kurt,
        "within_cluster_ratio": within_cluster_var_ratio,
    }

def main():
    datasets = [
        ("imagenet-clip-512-normalized", "Vision (Clean)", 5.25),      # Replicated win (+1.86% to +8.19%, avg ~+5.25%)
        ("laion-clip-512-normalized", "Vision (Web)", -1.72),          # Replicated loss (-1.09% to -2.51%)
        ("coco-nomic-768-normalized", "Multimodal Text", -3.85),       # Replicated loss (-2.31% to -8.32%)
        ("msmarco-qwen-1024-normalized", "Dense Text", 0.19),          # Marginal win / parity (+0.19%)
        ("yahoo-minilm-384-normalized", "Dense Text", -0.85),          # Loss
        ("llama-128-ip", "LLM Attention (d=128)", -12.5),             # Severe loss
        ("msmarco-colbert-128-normalized", "ColBERT Tokens (d=128)", -3.28), # Loss (-3.28% at ~4.5 bpd)
    ]
    
    data_dir = "data"
    results = []
    
    print("=" * 115)
    print("COMPUTING MOMENTS & CLUSTER DIAGNOSTICS ACROSS DATASETS")
    print("=" * 115)
    
    for name, category, delta in datasets:
        path = os.path.join(data_dir, f"{name}.hdf5")
        if not os.path.exists(path):
            print(f"Skipping {name} (not local)")
            continue
            
        stats = compute_dataset_stats(path)
        stats["name"] = name
        stats["category"] = category
        stats["delta"] = delta
        results.append(stats)
        
    print(f"{'Dataset':<32} | {'Dim':>4} | {'Max/MedVar':>10} | {'Top5%Var':>8} | {'MedKurt':>8} | {'MaxKurt':>8} | {'Top5%Kurt':>9} | {'ClusterVar':>10} | {'ΔR@10 (b-matched)':>17}")
    print("-" * 125)
    
    for r in results:
        print(f"{r['name']:<32} | {r['dim']:>4} | {r['max_med_var']:>9.1f}x | {r['top5_var_frac']*100:>7.1f}% | {r['med_kurt']:>8.2f} | {r['max_kurt']:>8.2f} | {r['top5_kurt']:>9.2f} | {r['within_cluster_ratio']:>10.3f} | {r['delta']:>+16.2f}%")
        
    print("=" * 125)
    
    # Compute Correlations against Delta R@10
    deltas = np.array([r["delta"] for r in results])
    predictors = {
        "Max/Med Variance Ratio": np.array([r["max_med_var"] for r in results]),
        "Top 5% Variance Fraction": np.array([r["top5_var_frac"] for r in results]),
        "Median Excess Kurtosis": np.array([r["med_kurt"] for r in results]),
        "Max Excess Kurtosis": np.array([r["max_kurt"] for r in results]),
        "Top 5% Excess Kurtosis": np.array([r["top5_kurt"] for r in results]),
        "Cluster Variance Ratio": np.array([r["within_cluster_ratio"] for r in results]),
        "Dimensionality (d)": np.array([r["dim"] for r in results]),
    }
    
    print("\nCORRELATION STUDY WITH EMPIRICAL DELTA (ΔR@10 Margin vs. EDEN Baseline):")
    print("-" * 75)
    print(f"{'Candidate Predictor':<30} | {'Pearson r':>10} | {'Spearman ρ':>10} | {'Predictive?':<12}")
    print("-" * 75)
    
    from scipy.stats import pearsonr, spearmanr
    for name, vals in predictors.items():
        if np.std(vals) > 1e-12 and np.std(deltas) > 1e-12:
            pr, _ = pearsonr(vals, deltas)
            sr, _ = spearmanr(vals, deltas)
            status = "Strong" if abs(sr) > 0.7 else ("Moderate" if abs(sr) > 0.4 else "Fails (Weak)")
            print(f"{name:<30} | {pr:>10.3f} | {sr:>10.3f} | {status:<12}")
        else:
            print(f"{name:<30} | {'N/A':>10} | {'N/A':>10} | N/A")
            
    print("-" * 75)

if __name__ == "__main__":
    main()

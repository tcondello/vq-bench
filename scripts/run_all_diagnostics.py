import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import math

ALL_SWEEP_DATASETS = [
    ("json", "colbert-json-128-normalized.hdf5"),
    ("go", "colbert-go-128-normalized.hdf5"),
    ("c", "colbert-c-128-normalized.hdf5"),
    ("rust", "colbert-rust-128-normalized.hdf5"),
    ("java", "colbert-java-128-normalized.hdf5"),
    ("typescript", "colbert-typescript-128-normalized.hdf5"),
    ("python", "colbert-python-128-normalized.hdf5"),
    ("markdown", "colbert-markdown-128-normalized.hdf5"),
    ("msmarco_text", "msmarco-colbert-128-normalized.hdf5"),
]

def compute_hopkins(X, n_samples=500):
    np.random.seed(42)
    n = len(X)
    d = X.shape[1]
    idx = np.random.choice(n, size=n_samples, replace=False)
    sample_X = X[idx]
    
    # Real distances
    dists_real = cdist(sample_X, X)
    np.fill_diagonal(dists_real, np.inf)
    min_real = np.min(dists_real, axis=1)
    
    # Uniform synthetic points inside data bounding sphere
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    synthetic = np.random.uniform(mins, maxs, size=(n_samples, d))
    dists_synth = cdist(synthetic, X)
    min_synth = np.min(dists_synth, axis=1)
    
    u = np.sum(min_synth)
    w = np.sum(min_real)
    return float(u / (u + w))

def compute_diagnostics():
    print("\n" + "="*125)
    print(" CYCLE 2: MULTI-LANGUAGE RIGIDITY & QUANTIZABILITY DIAGNOSTICS SUITE")
    print("="*125)
    print(f"{'Source / Language':<18} | {'Redun(0.15)':>11} | {'Redun(0.25)':>11} | {'Redun(0.35)':>11} | {'Gap Γ':>7} | {'Eff_Dim':>8} | {'Hopkins H':>9} | {'Margin M_bar':>12}")
    print("-" * 125)
    
    diagnostics = {}
    
    for name, filename in ALL_SWEEP_DATASETS:
        path = f"data/{filename}"
        with h5py.File(path, "r") as f:
            base = f["base"][:20000] # (20000, 128)
            eval_q = f["eval"][:1000] # (1000, 128)
            
        # 1. Clustering & Redundancy
        kmeans = MiniBatchKMeans(n_clusters=1024, random_state=42, batch_size=2048).fit(base)
        centroids = kmeans.cluster_centers_
        _, dists = pairwise_distances_argmin_min(base, centroids)
        
        red_15 = float(np.mean(dists <= 0.15)) * 100.0
        red_25 = float(np.mean(dists <= 0.25)) * 100.0
        red_35 = float(np.mean(dists <= 0.35)) * 100.0
        
        # 2. Quantizability Gap (K-means 64 distortion vs Gaussian distortion)
        km64 = MiniBatchKMeans(n_clusters=64, random_state=42, batch_size=2048).fit(base)
        d_kmeans = float(km64.inertia_ / (len(base) * 128))
        
        # Covariance eigenspectrum & effective dimension
        cov = np.cov(base, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-9)
        eff_dim = float((np.sum(eigvals)**2) / np.sum(eigvals**2))
        
        # Gaussian distortion for same eigenspectrum at 64 centroids (6 bits -> 6/128 b/d)
        d_gauss = float(np.mean(eigvals) * (2.0 ** (-2.0 * 6.0 / eff_dim)))
        gamma = float(d_gauss / max(d_kmeans, 1e-7))
        
        # 3. Hopkins statistic
        h_stat = compute_hopkins(base, n_samples=300)
        
        # 4. Normalized query margin
        scores = np.dot(eval_q, base.T)
        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        margins = sorted_scores[:, 0] - sorted_scores[:, 9]
        mu_delta = float(np.mean(margins))
        sigma_s = float(np.mean(np.std(scores, axis=1)))
        m_bar = float(mu_delta / max(sigma_s, 1e-6))
        
        diagnostics[name] = {
            "red_15": red_15,
            "red_25": red_25,
            "red_35": red_35,
            "gamma": gamma,
            "eff_dim": eff_dim,
            "hopkins": h_stat,
            "m_bar": m_bar,
        }
        
        print(f"{name:<18} | {red_15:10.1f}% | {red_25:10.1f}% | {red_35:10.1f}% | {gamma:6.2f}x | {eff_dim:8.1f} | {h_stat:9.3f} | {m_bar:12.4f}")

    print("=" * 125)
    return diagnostics

if __name__ == "__main__":
    compute_diagnostics()

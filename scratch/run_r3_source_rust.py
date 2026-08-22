import json
import os
import subprocess
import math
import h5py
import numpy as np

SEEDS = [1, 2, 3, 4, 5]
DATASET = "colbert-rust-128-normalized"

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

def compute_rust_diagnostics():
    print("\n" + "="*95)
    print(" RUN 3 (AXIS 3: SOURCE) — RUST CODE QUANTIZABILITY & REDUNDANCY DIAGNOSTICS")
    print("="*95)
    
    with h5py.File("data/colbert-rust-128-normalized.hdf5", "r") as f:
        vecs = f["base"][:20000]
        
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=1024, random_state=42, batch_size=2048).fit(vecs)
    centroids = kmeans.cluster_centers_
    
    from sklearn.metrics import pairwise_distances_argmin_min
    _, dists = pairwise_distances_argmin_min(vecs, centroids)
    frac_within_eps = np.mean(dists <= 0.25)
    
    print(f"Rust Vocabulary Redundancy (tokens within eps=0.25): {frac_within_eps*100:.2f}%")
    print(f"Rust Mean distance to nearest centroid: {np.mean(dists):.4f}")
    print(f"Rust Median distance to nearest centroid: {np.median(dists):.4f}")

def run_rust_sweep():
    print("\n" + "="*95)
    print(" RUN 3 (AXIS 3: SOURCE) — RUST CODE RATE-RECALL SWEEP (5 SEEDS)")
    print("="*95)
    
    results = []
    for seed in SEEDS:
        config = {
            "datasets": [DATASET],
            "seed": seed,
            "n_fit": 20000,
            "n_reconstruct": 1000,
            "n_eval": 1000,
            "k": [1, 10, 100],
            "temp": [0.05, 0.2, 1.0],
            "methods": [
                { "name": "scalar", "b": [1, 2, 3, 4] },
                { "name": "eden_prod", "b": [1, 2, 3, 4] },
                { "name": "colbertv2_quant", "centroids": [256], "residual_bits": [1, 2, 3] }
            ],
            "metrics": ["recall", "mse_score", "mse_recon"]
        }
        cfg_file = f"configs/r3-rust-seed-{seed}.json"
        with open(cfg_file, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"  Executing Seed {seed} on Rust ColBERT...")
        run_cmd(f"cargo run --release -- run {cfg_file} --stream")
        
        with open(f"results/r3-rust-seed-{seed}.json") as f:
            data = json.load(f)
            results.append(data)
            
    return results

def mean_std(arr):
    m = sum(arr) / len(arr)
    v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
    return m, math.sqrt(v)

def analyze_rust(results):
    print("\n" + "#"*105)
    print(" RUN 3 ANALYSIS: RUST CODE RATE-RECALL & DISTORTION PERFORMANCE")
    print("#"*105)
    
    labels = [m["label"] for m in results[0]["datasets"][0]["methods"]]
    print(f"{'Method / Configuration':<45} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'MSE Recon':>11} | {'Encode Time':>10}")
    print("-" * 105)
    
    for lbl in labels:
        r10s, bpds, mses, times = [], [], [], []
        for run in results:
            methods = {m["label"]: m for m in run["datasets"][0]["methods"]}
            cand = methods.get(lbl)
            if cand:
                r10s.append(cand["recalls"]["10"])
                bpds.append(cand["bits_per_dim"])
                mses.append(cand["mse_recon"])
                times.append(cand["encode_s"])
        r_m, r_s = mean_std(r10s)
        b_m, _ = mean_std(bpds)
        m_m, _ = mean_std(mses)
        t_m, t_s = mean_std(times)
        print(f"{lbl:<45} | {b_m:6.2f} | {r_m:.4f} ± {r_s:.4f}          | {m_m:.4e} | {t_m:4.2f}±{t_s:.2f}s")

if __name__ == "__main__":
    compute_rust_diagnostics()
    res = run_rust_sweep()
    analyze_rust(res)

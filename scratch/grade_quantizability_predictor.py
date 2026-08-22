import json
import os
import subprocess
import math
import h5py
import numpy as np

SEEDS = [1, 2, 3, 4, 5]
DATASETS = ["colbert-python-128-normalized", "msmarco-colbert-128-normalized"]

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

def compute_vocabulary_redundancy(hdf5_path, k=1024, eps=0.25):
    with h5py.File(hdf5_path, "r") as f:
        vecs = f["base"][:20000] # sample
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=min(k, len(vecs)//2), random_state=42, batch_size=2048).fit(vecs)
    centroids = kmeans.cluster_centers_
    
    # Distance to nearest centroid
    from sklearn.metrics import pairwise_distances_argmin_min
    _, dists = pairwise_distances_argmin_min(vecs, centroids)
    frac_within_eps = np.mean(dists <= eps)
    return float(frac_within_eps), float(np.mean(dists)), float(np.median(dists))

def run_phase2_experiments():
    print("\n" + "="*95)
    print(" GOAL G3: THE HEADLINE EXPERIMENT — THE DOMAIN-ENTROPY DIVIDEND (5 SEEDS)")
    print("="*95)
    
    results = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        print(f"\n>>> Running dataset sweep: {ds}")
        for seed in SEEDS:
            config = {
                "datasets": [ds],
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
            cfg_file = f"configs/phase2-{ds}-seed-{seed}.json"
            with open(cfg_file, "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"  Executing Seed {seed} on {ds}...")
            run_cmd(f"cargo run --release -- run {cfg_file} --stream")
            
            with open(f"results/phase2-{ds}-seed-{seed}.json") as f:
                data = json.load(f)
                results[ds].append(data)
                
    return results

def mean_std(arr):
    m = sum(arr) / len(arr)
    v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
    return m, math.sqrt(v)

def analyze_dividend(results):
    print("\n" + "#"*115)
    print(" DOMAIN-ENTROPY DIVIDEND COMPARISON: CODE (PYTHON) VS GENERAL TEXT (MS MARCO)")
    print("#"*115)

    for ds in DATASETS:
        print(f"\n--- DATASET: {ds} ---")
        print(f"{'Method / Configuration':<45} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'MSE Recon':>11} | {'Encode Time':>10}")
        print("-" * 105)
        
        runs = results[ds]
        # Collect all unique method labels
        labels = [m["label"] for m in runs[0]["datasets"][0]["methods"]]
        for lbl in labels:
            r10s, bpds, mses, times = [], [], [], []
            for run in runs:
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
    res = run_phase2_experiments()
    analyze_dividend(res)

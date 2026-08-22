import json
import os
import subprocess
import math

SEEDS = [1, 2, 3, 4, 5]
DATASETS = ["imagenet-clip-512-normalized", "msmarco-qwen-1024-normalized"]

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

def run_scann_test():
    print("\n" + "="*95)
    print(" SCANN VALIDITY GATE: ANISOTROPIC PQ VS MSE PQ (5 SEEDS)")
    print("="*95)
    
    results = {ds: [] for ds in DATASETS}
    
    for ds in DATASETS:
        print(f"\n>>> Running dataset: {ds}")
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
                    { "name": "pq", "centroids": [256], "section_dim": [8] },
                    { "name": "anisotropic_pq", "centroids": [256], "section_dim": [8], "omega": [0.0, 0.2, 0.5, 1.0] }
                ],
                "metrics": ["recall", "mse_score", "mse_recon"]
            }
            cfg_file = f"configs/scann-{ds}-seed-{seed}.json"
            with open(cfg_file, "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"  Executing Seed {seed} on {ds}...")
            run_cmd(f"cargo run --release -- run {cfg_file} --stream")
            
            with open(f"results/scann-{ds}-seed-{seed}.json") as f:
                data = json.load(f)
                results[ds].append(data)
                
    return results

def analyze_scann(results):
    print("\n" + "#"*105)
    print(" SCANN VALIDITY GATE SYNTHESIS (ANISOTROPIC PQ VS MSE PQ)")
    print("#"*105)
    
    def mean_std(arr):
        m = sum(arr) / len(arr)
        v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return m, math.sqrt(v)

    for ds in DATASETS:
        print(f"\n--- DATASET: {ds} ---")
        print(f"{'Method / Configuration':<42} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'ΔR@10 vs PQ (pts)':>20}")
        print("-" * 100)
        
        runs = results[ds]
        pq_r10s = []
        for run in runs:
            methods = {m["label"]: m for m in run["datasets"][0]["methods"]}
            pq_r10s.append(methods["PQ (centroids=256, section_dim=8)"]["recalls"]["10"])
        pq_m, pq_s = mean_std(pq_r10s)
        print(f"{'PQ (MSE Baseline, centroids=256, section_dim=8)':<42} |   1.00 | {pq_m:.4f} ± {pq_s:.4f}          |            0.00%")
        
        for omega in [0.0, 0.2, 0.5, 1.0]:
            lbl = f"AnisotropicPQ (centroids=256, omega={omega}, section_dim=8)"
            r10s = []
            deltas = []
            for run in runs:
                methods = {m["label"]: m for m in run["datasets"][0]["methods"]}
                r_val = methods[lbl]["recalls"]["10"]
                r_pq = methods["PQ (centroids=256, section_dim=8)"]["recalls"]["10"]
                r10s.append(r_val)
                deltas.append((r_val - r_pq) * 100.0)
            r_m, r_s = mean_std(r10s)
            d_m, d_s = mean_std(deltas)
            print(f"{lbl:<42} |   1.00 | {r_m:.4f} ± {r_s:.4f}          | {d_m:+6.2f}% ± {d_s:.2f}%")

if __name__ == "__main__":
    scann_res = run_scann_test()
    analyze_scann(scann_res)

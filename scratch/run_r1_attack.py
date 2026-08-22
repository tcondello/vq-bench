import json
import os
import subprocess
import math

SEEDS = [1, 2, 3, 4, 5]
DATASET = "synthetic-shells-128"

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

def run_attack():
    print("\n" + "="*95)
    print(" RUN 1 (AXIS 1: ATTACK) — NON-CONVEX CONCENTRIC SHELLS (5 SEEDS)")
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
                { "name": "eden_prod", "b": [1, 2, 3, 4] }
            ],
            "metrics": ["recall", "mse_score", "mse_recon"]
        }
        cfg_file = f"configs/r1-attack-seed-{seed}.json"
        with open(cfg_file, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"  Executing Seed {seed} on Synthetic Shells...")
        run_cmd(f"cargo run --release -- run {cfg_file} --stream")
        
        with open(f"results/r1-attack-seed-{seed}.json") as f:
            data = json.load(f)
            results.append(data)
            
    return results

def mean_std(arr):
    m = sum(arr) / len(arr)
    v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
    return m, math.sqrt(v)

def analyze_attack(results):
    print("\n" + "#"*105)
    print(" RUN 1 ATTACK ANALYSIS: RECALL DEGRADATION ON CONCENTRIC SHELLS")
    print("#"*105)
    
    methods_list = [m["label"] for m in results[0]["datasets"][0]["methods"]]
    print(f"{'Method / Configuration':<45} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'MSE Recon':>11}")
    print("-" * 95)
    
    for lbl in methods_list:
        r10s, bpds, mses = [], [], []
        for run in results:
            methods = {m["label"]: m for m in run["datasets"][0]["methods"]}
            cand = methods.get(lbl)
            if cand:
                r10s.append(cand["recalls"]["10"])
                bpds.append(cand["bits_per_dim"])
                mses.append(cand["mse_recon"])
        r_m, r_s = mean_std(r10s)
        b_m, _ = mean_std(bpds)
        m_m, _ = mean_std(mses)
        print(f"{lbl:<45} | {b_m:6.2f} | {r_m:.4f} ± {r_s:.4f}          | {m_m:.4e}")

if __name__ == "__main__":
    res = run_attack()
    analyze_attack(res)

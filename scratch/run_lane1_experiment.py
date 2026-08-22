import json
import os
import subprocess
import math

SEEDS = [1, 2, 3, 4, 5]
DATASETS = [
    "imagenet-clip-512-normalized",
    "msmarco-qwen-1024-normalized",
    "coco-nomic-768-normalized"
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

def run_lane1_sweep():
    print("\n" + "="*95)
    print(" LANE 1: TASK-AWARE ANISOTROPIC CODING VS EDEN BASELINES (3 DATASETS, 5 SEEDS)")
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
                    { "name": "eden_prod", "b": [2, 3, 4] },
                    { "name": "eden_mse", "b": [2, 3, 4] },
                    { "name": "task_aware_eden", "b": [2, 3, 4], "omega": [0.5, 1.0, 2.0] }
                ],
                "metrics": ["recall", "mse_score", "mse_recon", "bias_score", "bias_recon", "sos", "exp_sos", "kl", "tv"]
            }
            cfg_file = f"configs/lane1-{ds}-seed-{seed}.json"
            with open(cfg_file, "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"  Executing Seed {seed} on {ds}...")
            run_cmd(f"cargo run --release -- run {cfg_file} --stream")
            
            with open(f"results/lane1-{ds}-seed-{seed}.json") as f:
                data = json.load(f)
                results[ds].append(data)
                
    return results

def interpolate_frontier(points, target_b):
    pts = sorted(points, key=lambda p: p[0])
    if target_b <= pts[0][0]:
        return pts[0][1]
    if target_b >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        b0, r0 = pts[i]
        b1, r1 = pts[i+1]
        if b0 <= target_b <= b1:
            if b1 == b0:
                return max(r0, r1)
            t = (target_b - b0) / (b1 - b0)
            return r0 + t * (r1 - r0)
    return pts[-1][1]

def analyze_lane1(results):
    print("\n" + "#"*105)
    print(" LANE 1 ADJUDICATION: TASK-AWARE ANISOTROPIC LOSS VS BASELINE ENVELOPE")
    print("#"*105)
    
    def mean_std(arr):
        m = sum(arr) / len(arr)
        v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return m, math.sqrt(v)

    all_deltas = []
    
    for ds in DATASETS:
        print(f"\n--- DATASET: {ds} ---")
        print(f"{'Method / Configuration':<35} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'Interp EDEN':>11} | {'ΔR@10 (%)':>14}")
        print("-" * 100)
        
        runs = results[ds]
        for b in [2, 3, 4]:
            for omega in [0.5, 1.0, 2.0]:
                lbl = f"TaskAwareEDEN (b={b}, omega={omega})"
                r10s = []
                bpds = []
                interps = []
                deltas = []
                
                for run in runs:
                    methods = {m["label"]: m for m in run["datasets"][0]["methods"]}
                    
                    baseline_pts = []
                    for k_lbl, m in methods.items():
                        if "EDEN-prod" in k_lbl or "EDEN-MSE" in k_lbl:
                            baseline_pts.append((m["bits_per_dim"], m["recalls"]["10"]))
                            
                    cand = methods.get(lbl)
                    if not cand:
                        continue
                    b_val = cand["bits_per_dim"]
                    r_val = cand["recalls"]["10"]
                    interp_val = interpolate_frontier(baseline_pts, b_val)
                    
                    r10s.append(r_val)
                    bpds.append(b_val)
                    interps.append(interp_val)
                    deltas.append((r_val - interp_val) * 100.0)
                    
                if not r10s:
                    continue
                    
                r_m, r_s = mean_std(r10s)
                b_m, _ = mean_std(bpds)
                i_m, i_s = mean_std(interps)
                d_m, d_s = mean_std(deltas)
                
                all_deltas.append((ds, lbl, d_m, d_s))
                print(f"{lbl:<35} | {b_m:6.2f} | {r_m:.4f} ± {r_s:.4f}          | {i_m:.4f}      | {d_m:+6.2f}% ± {d_s:.2f}%")
                
    print("\n" + "="*105)
    print(" LANE 1 KILL CRITERION ADJUDICATION")
    print(" Criterion: < +1.0 point above envelope at 2-4 b/d on 3 datasets -> Lane CLOSES.")
    print("            >= +1.0 point above envelope on 3 datasets           -> Lane SURVIVES.")
    print("="*105)
    
    max_delta = max(d[2] for d in all_deltas)
    qualifying = [d for d in all_deltas if d[2] >= 1.0]
    
    if len(qualifying) == 0:
        print(f"\n[X] LANE 1 KILL CRITERION TRIGGERED: Maximum delta observed is {max_delta:+.2f} points (< +1.00%).")
        print("===> VERDICT: LANE 1 IS OFFICIALLY CLOSED (Anisotropic score scaling alone does not escape the envelope).")
    else:
        print(f"\n[v] LANE 1 QUALIFIES: Found {len(qualifying)} configurations with >= +1.00% gain above envelope.")
        for ds, lbl, d_m, d_s in qualifying:
            print(f"      - {ds} ({lbl}): Delta = {d_m:+.2f}% ± {d_s:.2f}%")

if __name__ == "__main__":
    lane1_results = run_lane1_sweep()
    analyze_lane1(lane1_results)

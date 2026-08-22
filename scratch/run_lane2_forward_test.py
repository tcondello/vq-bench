import json
import os
import subprocess
import math

SEEDS = [1, 2, 3, 4, 5]
DATASET = "cifar100-clip-512-normalized"

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

def run_cifar_forward_test():
    print("\n" + "="*95)
    print(" LANE 2: PRE-REGISTERED FORWARD TEST ON CIFAR-100 CLIP (5 SEEDS)")
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
                { "name": "scalar", "b": [1, 2, 3, 4, 5, 6] },
                { "name": "eden_prod", "b": [1, 2, 3, 4, 5, 6] },
                { "name": "spike_eden", "b": [3, 4], "ratio": [0.05] }
            ],
            "metrics": ["recall", "mse_score", "mse_recon"]
        }
        cfg_file = f"configs/cifar100-forward-seed-{seed}.json"
        with open(cfg_file, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"  Executing Seed {seed} on CIFAR-100 CLIP...")
        run_cmd(f"cargo run --release -- run {cfg_file} --stream")
        
        with open(f"results/cifar100-forward-seed-{seed}.json") as f:
            data = json.load(f)
            results.append(data)
            
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

def analyze_cifar(results):
    print("\n" + "#"*105)
    print(" LANE 2 FORWARD TEST ADJUDICATION (CIFAR-100 CLIP)")
    print("#"*105)
    
    def mean_std(arr):
        m = sum(arr) / len(arr)
        v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return m, math.sqrt(v)

    targets = [
        "SpikeEden (b=3, ratio=0.05)",
        "SpikeEden (b=4, ratio=0.05)",
    ]

    print(f"{'Method / Configuration':<48} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'Interp EDEN':>11} | {'ΔR@10 (%)':>14} | {'Encode Time':>10}")
    print("-" * 125)

    for lbl in targets:
        r10s = []
        bpds = []
        interps = []
        deltas = []
        t_list = []

        for run in results:
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
            t_val = cand["encode_s"]
            interp_val = interpolate_frontier(baseline_pts, b_val)
            
            r10s.append(r_val)
            bpds.append(b_val)
            interps.append(interp_val)
            deltas.append((r_val - interp_val) * 100.0)
            t_list.append(t_val)

        if not r10s:
            continue

        r_m, r_s = mean_std(r10s)
        b_m, _ = mean_std(bpds)
        i_m, i_s = mean_std(interps)
        d_m, d_s = mean_std(deltas)
        t_m, t_s = mean_std(t_list)

        print(f"{lbl:<48} | {b_m:6.2f} | {r_m:.4f} ± {r_s:.4f}          | {i_m:.4f}      | {d_m:+6.2f}% ± {d_s:.2f}% | {t_m:4.2f}±{t_s:.2f}s")

if __name__ == "__main__":
    cifar_res = run_cifar_forward_test()
    analyze_cifar(cifar_res)

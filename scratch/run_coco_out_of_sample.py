import json
import os
import subprocess
import math

SEEDS = [1, 2, 3, 4, 5]

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

def run_coco_sweep():
    print("\n" + "="*85)
    print(" RUNNING 5-SEED OUT-OF-SAMPLE TEST: SpikeEden vs EDEN on COCO-Nomic 768d")
    print("="*85)
    
    results_by_seed = []
    for seed in SEEDS:
        config = {
            "datasets": ["coco-nomic-768-normalized"],
            "seed": seed,
            "n_fit": 20000,
            "n_reconstruct": 1000,
            "n_eval": 1000,
            "k": [1, 10, 100],
            "temp": [0.05, 0.2, 1.0],
            "methods": [
                { "name": "scalar", "b": [2, 3, 4, 5, 6] },
                { "name": "eden_prod", "b": [2, 3, 4, 5, 6] },
                { "name": "eden_mse", "b": [2, 3, 4, 5, 6] },
                { "name": "spike_eden", "b": [2, 3, 4, 5, 6], "ratio": [0.02, 0.05] }
            ],
            "metrics": ["recall", "mse_score", "mse_recon", "bias_score", "bias_recon", "sos", "exp_sos", "kl", "tv"]
        }
        cfg_file = f"configs/coco-seed-{seed}.json"
        with open(cfg_file, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Executing Seed {seed} on COCO-Nomic 768d...")
        run_cmd(f"cargo run --release -- run {cfg_file} --stream")
        
        with open(f"results/coco-seed-{seed}.json") as f:
            data = json.load(f)
            results_by_seed.append(data)
            
    return results_by_seed

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

def analyze_coco(results_by_seed):
    print("\n" + "#"*95)
    print(" OUT-OF-SAMPLE MULTI-SEED STATISTICAL SYNTHESIS: COCO-NOMIC-768 (N=5 SEEDS)")
    print("#"*95)
    
    def mean_std(arr):
        m = sum(arr) / len(arr)
        v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return m, math.sqrt(v)

    tiers = [
        ("SpikeEden (b=2, ratio=0.02)", 2),
        ("SpikeEden (b=2, ratio=0.05)", 2),
        ("SpikeEden (b=3, ratio=0.02)", 3),
        ("SpikeEden (b=3, ratio=0.05)", 3),
        ("SpikeEden (b=4, ratio=0.02)", 4),
        ("SpikeEden (b=4, ratio=0.05)", 4),
        ("SpikeEden (b=5, ratio=0.02)", 5),
        ("SpikeEden (b=5, ratio=0.05)", 5),
        ("SpikeEden (b=6, ratio=0.02)", 6),
        ("SpikeEden (b=6, ratio=0.05)", 6),
    ]

    print(f"{'Method / Configuration':<35} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'Interp EDEN':>11} | {'ΔR@10 (%)':>14} | {'Encode Time':>10}")
    print("-" * 105)

    for label, target_b in tiers:
        r10_list = []
        bpd_list = []
        interp_list = []
        delta_list = []
        t_list = []

        for run in results_by_seed:
            ds = run["datasets"][0]
            methods = {m["label"]: m for m in ds["methods"]}

            baseline_pts = []
            for lbl, m in methods.items():
                if "EDEN-prod" in lbl or "EDEN-MSE" in lbl:
                    baseline_pts.append((m["bits_per_dim"], m["recalls"]["10"]))

            cand = methods.get(label)
            if not cand:
                continue
            b = cand["bits_per_dim"]
            r = cand["recalls"]["10"]
            t = cand["encode_s"]
            interp = interpolate_frontier(baseline_pts, b)

            r10_list.append(r)
            bpd_list.append(b)
            interp_list.append(interp)
            delta_list.append((r - interp) * 100)
            t_list.append(t)

        if not r10_list:
            continue

        r_m, r_s = mean_std(r10_list)
        b_m, _ = mean_std(bpd_list)
        i_m, i_s = mean_std(interp_list)
        d_m, d_s = mean_std(delta_list)
        t_m, t_s = mean_std(t_list)

        print(f"{label:<35} | {b_m:6.2f} | {r_m:.4f} ± {r_s:.4f}          | {i_m:.4f}      | {d_m:+6.2f}% ± {d_s:.2f}% | {t_m:4.2f}±{t_s:.2f}s")

if __name__ == "__main__":
    coco_runs = run_coco_sweep()
    analyze_coco(coco_runs)

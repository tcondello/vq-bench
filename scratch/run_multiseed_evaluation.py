import json
import os
import subprocess
import glob
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

def run_multiseed_msmarco():
    print("\n" + "="*80)
    print(" 1. RUNNING 5-SEED REPLICATION: SpectralBandQuant vs EDEN on MS MARCO 1024d")
    print("="*80)
    
    results_by_seed = []
    for seed in SEEDS:
        config = {
            "datasets": ["msmarco-qwen-1024-normalized"],
            "seed": seed,
            "n_fit": 20000,
            "n_reconstruct": 1000,
            "n_eval": 1000,
            "k": [1, 10, 100],
            "temp": [0.05, 0.2, 1.0],
            "methods": [
                { "name": "eden_prod", "b": [4, 5] },
                { "name": "eden_mse", "b": [4, 5] },
                { "name": "spectral_band_quant" }
            ],
            "metrics": ["recall", "mse_score", "mse_recon", "bias_score", "bias_recon", "sos", "exp_sos", "kl", "tv"]
        }
        cfg_file = f"configs/msmarco-seed-{seed}.json"
        with open(cfg_file, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Running seed {seed}...")
        run_cmd(f"cargo run --release -- run {cfg_file} --stream")
        
        with open(f"results/msmarco-seed-{seed}.json") as f:
            data = json.load(f)
            results_by_seed.append(data)
            
    return results_by_seed

def run_multiseed_imagenet():
    print("\n" + "="*80)
    print(" 2. RUNNING 5-SEED REPLICATION: SpikeEden vs EDEN on ImageNet CLIP 512d")
    print("="*80)
    
    results_by_seed = []
    for seed in SEEDS:
        config = {
            "datasets": ["imagenet-clip-512-normalized"],
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
        cfg_file = f"configs/imagenet-seed-{seed}.json"
        with open(cfg_file, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Running seed {seed}...")
        run_cmd(f"cargo run --release -- run {cfg_file} --stream")
        
        with open(f"results/imagenet-seed-{seed}.json") as f:
            data = json.load(f)
            results_by_seed.append(data)
            
    return results_by_seed

def interpolate_frontier(points, target_b):
    # points: list of (b, r10)
    # sort by b
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

def analyze_multiseed(msmarco_runs, imagenet_runs):
    print("\n" + "#"*90)
    print(" MULTI-SEED STATISTICAL SYNTHESIS (N=5 SEEDS, SAME MACHINE & THREADS)")
    print("#"*90)
    
    # 1. MS MARCO ANALYSIS
    print("\n--- 1. MS MARCO 1024d: SpectralBandQuant vs EDEN Frontier ---")
    sbq_r10s, sbq_bpds, sbq_encs = [], [], []
    eden_frontier_r10s = []
    deltas = []
    
    for run in msmarco_runs:
        ds = run["datasets"][0]
        methods = {m["label"]: m for m in ds["methods"]}
        
        # Baseline points for frontier
        baseline_pts = []
        for lbl, m in methods.items():
            if "EDEN" in lbl:
                b = m["bits_per_dim"]
                r10 = m["recalls"]["10"]
                baseline_pts.append((b, r10))
                
        sbq = methods["SpectralBandQuant"]
        sbq_b = sbq["bits_per_dim"]
        sbq_r = sbq["recalls"]["10"]
        sbq_t = sbq["encode_s"]
        
        interp_eden = interpolate_frontier(baseline_pts, sbq_b)
        delta = sbq_r - interp_eden
        
        sbq_r10s.append(sbq_r)
        sbq_bpds.append(sbq_b)
        sbq_encs.append(sbq_t)
        eden_frontier_r10s.append(interp_eden)
        deltas.append(delta)
        
    def mean_std(arr):
        m = sum(arr) / len(arr)
        v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return m, math.sqrt(v)
        
    sbq_m, sbq_s = mean_std(sbq_r10s)
    eden_m, eden_s = mean_std(eden_frontier_r10s)
    delta_m, delta_s = mean_std(deltas)
    t_m, t_s = mean_std(sbq_encs)
    
    print(f"SpectralBandQuant (b={sbq_bpds[0]:.2f}): Recall@10 = {sbq_m:.4f} ± {sbq_s:.4f} | Encode = {t_m:.2f}s ± {t_s:.2f}s")
    print(f"Interpolated EDEN Frontier (b={sbq_bpds[0]:.2f}): Recall@10 = {eden_m:.4f} ± {eden_s:.4f}")
    print(f"Mean Delta above Frontier ΔR@10: {delta_m*100:+.2f}% ± {delta_s*100:.2f}%")
    if delta_m > 0 and abs(delta_m) > 1.96 * (delta_s / math.sqrt(len(SEEDS))):
        print("==> RESULT: STATISTICALLY SIGNIFICANT WIN (p < 0.05)")
    else:
        print("==> RESULT: WITHIN EXPERIMENTAL NOISE / MARGINAL")
        
    # 2. IMAGENET ANALYSIS
    print("\n--- 2. ImageNet CLIP 512d: SpikeEden vs EDEN Frontier Across Bit Tiers ---")
    print(f"{'Method / Tier':<35} | {'b/d':>6} | {'Recall@10 (Mean ± Std)':>22} | {'Interp EDEN':>11} | {'ΔR@10 (%)':>10} | {'Encode (s)':>10}")
    print("-" * 105)
    
    tiers = [
        ("SpikeEden (b=2, ratio=0.02)", 2),
        ("SpikeEden (b=2, ratio=0.05)", 2),
        ("SpikeEden (b=3, ratio=0.02)", 3),
        ("SpikeEden (b=3, ratio=0.05)", 3),
        ("SpikeEden (b=4, ratio=0.02)", 4),
        ("SpikeEden (b=4, ratio=0.05)", 4),
        ("SpikeEden (b=5, ratio=0.05)", 5),
        ("SpikeEden (b=6, ratio=0.05)", 6),
    ]
    
    for label, target_b in tiers:
        r10_list = []
        bpd_list = []
        interp_list = []
        delta_list = []
        t_list = []
        
        for run in imagenet_runs:
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
        
        print(f"{label:<35} | {b_m:6.2f} | {r_m:.4f} ± {r_s:.4f}          | {i_m:.4f}      | {d_m:+6.2f}% ± {d_s:.2f} | {t_m:4.1f}±{t_s:.1f}s")

if __name__ == "__main__":
    msmarco_data = run_multiseed_msmarco()
    imagenet_data = run_multiseed_imagenet()
    analyze_multiseed(msmarco_data, imagenet_data)

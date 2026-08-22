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

def run_lane4_sweep():
    print("\n" + "="*95)
    print(" LANE 4: EVALUATING PROGRESSIVE BIT-PLANE RESIDUALS ON EDEN-PROD (3 DATASETS, 5 SEEDS)")
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
                    { "name": "eden_prod", "b": [1, 2, 4] },
                    { "name": "progressive_eden", "eval_bits": [1, 2, 4] }
                ],
                "metrics": ["recall", "mse_score", "mse_recon", "bias_score", "bias_recon", "sos", "exp_sos", "kl", "tv"]
            }
            cfg_file = f"configs/lane4-{ds}-seed-{seed}.json"
            with open(cfg_file, "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"  Executing Seed {seed} on {ds}...")
            run_cmd(f"cargo run --release -- run {cfg_file} --stream")
            
            with open(f"results/lane4-{ds}-seed-{seed}.json") as f:
                data = json.load(f)
                results[ds].append(data)
                
    return results

def analyze_lane4(results):
    print("\n" + "#"*105)
    print(" LANE 4 ADJUDICATION: PRICE OF PROGRESSIVENESS (PROGRESSIVE TRUNCATION VS RETRAINED EDEN)")
    print("#"*105)
    
    def mean_std(arr):
        m = sum(arr) / len(arr)
        v = sum((x - m)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return m, math.sqrt(v)

    all_penalties = []
    
    for ds in DATASETS:
        print(f"\n--- DATASET: {ds} ---")
        print(f"{'Rate Tier':<25} | {'Retrained EDEN-prod':<22} | {'Progressive Truncated':<22} | {'Penalty (Retrained - Trunc)':<25}")
        print("-" * 100)
        
        runs = results[ds]
        for rate, (eden_b, prog_eval) in [("1 b/d (Stage 1)", (1, 1)), ("2 b/d (Stages 1+2)", (2, 2)), ("4 b/d (Stages 1+2+3)", (4, 4))]:
            eden_r10s = []
            prog_r10s = []
            penalties = []
            
            for run in runs:
                methods = {m["label"]: m for m in run["datasets"][0]["methods"]}
                
                # find retrained eden
                lbl_eden = f"EDEN-prod (b={eden_b})"
                lbl_prog = f"ProgressiveEDEN (eval_bits={prog_eval})"
                
                r_eden = methods[lbl_eden]["recalls"]["10"]
                r_prog = methods[lbl_prog]["recalls"]["10"]
                
                eden_r10s.append(r_eden)
                prog_r10s.append(r_prog)
                penalties.append((r_eden - r_prog) * 100.0)
                
            e_m, e_s = mean_std(eden_r10s)
            p_m, p_s = mean_std(prog_r10s)
            pen_m, pen_s = mean_std(penalties)
            
            all_penalties.append((ds, rate, pen_m, pen_s))
            print(f"{rate:<25} | {e_m:.4f} ± {e_s:.4f}          | {p_m:.4f} ± {p_s:.4f}          | {pen_m:+5.2f}% ± {pen_s:.2f}%")
            
    print("\n" + "="*105)
    print(" LANE 4 KILL CRITERION ADJUDICATION")
    print(" Criterion: Penalty >= 1.0 point at any rate -> Lane CLOSES COMPLETELY (Certificate Airtight).")
    print("            Penalty <  1.0 point everywhere   -> Systems Contribution SURVIVES.")
    print("="*105)
    
    max_penalty = max(p[2] for p in all_penalties)
    violating = [p for p in all_penalties if p[2] >= 1.0]
    
    if violating:
        print(f"\n[X] LANE 4 KILL CRITERION TRIGGERED: Maximum penalty observed is {max_penalty:+.2f} points.")
        print("    Violating Rate Tiers:")
        for ds, rate, pen_m, pen_s in violating:
            print(f"      - {ds} ({rate}): Penalty = {pen_m:+.2f}% ± {pen_s:.2f}% (>= +1.00%)")
        print("\n===> VERDICT: LANE 4 IS OFFICIALLY CLOSED. CERTIFICATE AIRTIGHT.")
    else:
        print(f"\n[v] LANE 4 SURVIVES: Maximum penalty is {max_penalty:+.2f} points (< 1.00 point everywhere).")
        print("===> VERDICT: SYSTEMS CONTRIBUTION SURVIVES (One index, every rate tier).")

if __name__ == "__main__":
    sweep_results = run_lane4_sweep()
    analyze_lane4(sweep_results)

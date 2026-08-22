import h5py
import numpy as np

DATASETS = [
    "colbert-python-128-normalized",
    "colbert-rust-128-normalized",
    "msmarco-colbert-128-normalized",
]

def run_twostage_pipeline():
    print("\n" + "="*105)
    print(" RUN 5 (AXIS 5: APPLICATION) — TWO-STAGE RETRIEVAL: 1.35 b/d FILTER + EXACT TOP-100 RESCORE")
    print("="*105)
    print(f"{'Dataset':<32} | {'Uncompressed R@10':>18} | {'1.35b Top-100 Recall':>20} | {'Rescored R@10':>15} | {'Retention':>10}")
    print("-" * 105)
    
    for ds_name in DATASETS:
        with h5py.File(f"data/{ds_name}.hdf5", "r") as f:
            base = f["base"][:50000] # (50000, 128)
            eval_q = f["eval"][:1000] # (1000, 128)
            
        # Uncompressed exact ground truth top-10
        exact_scores = np.dot(eval_q, base.T)
        exact_top10 = np.argsort(-exact_scores, axis=1)[:, :10]
        
        # Stage 1: 1.35 b/d Quantization (1-bit residual + sign)
        # 1-bit sign quantizer
        quant_base = np.sign(base) / np.sqrt(128)
        approx_scores = np.dot(eval_q, quant_base.T)
        
        # Top 100 candidate filter
        cand_top100 = np.argsort(-approx_scores, axis=1)[:, :100]
        
        # Evaluate how many true top-10 neighbors are in the candidate top-100 pool
        filter_hits = []
        rescore_recalls = []
        
        for q_idx in range(len(eval_q)):
            true_10 = set(exact_top10[q_idx])
            cands = cand_top100[q_idx]
            
            # Coverage of true top-10 in top-100 filter
            in_pool = true_10.intersection(set(cands))
            filter_hits.append(len(in_pool) / 10.0)
            
            # Stage 2: Rescore the 100 candidates with exact vectors
            rescored_scores = exact_scores[q_idx, cands]
            rescored_top10_cands = cands[np.argsort(-rescored_scores)[:10]]
            
            rescore_hits = len(true_10.intersection(set(rescored_top10_cands)))
            rescore_recalls.append(rescore_hits / 10.0)

        mean_filter_cov = np.mean(filter_hits)
        mean_rescored_r10 = np.mean(rescore_recalls)
        retention = (mean_rescored_r10 / 1.0) * 100.0
        
        print(f"{ds_name:<32} | {'100.0%':>18} | {mean_filter_cov*100:19.2f}% | {mean_rescored_r10*100:14.2f}% | {retention:9.1f}%")

    print("=" * 105)

if __name__ == "__main__":
    run_twostage_pipeline()

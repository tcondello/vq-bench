#!/usr/bin/env python3
"""
MRL Dimension Slicer for VQ-Bench datasets.

Extracts prefix sub-vectors of dimension d from an existing MRL dataset (e.g. Nomic, Qwen),
re-normalizes them to unit L2 length, recomputes exact top-L ground truth dot product candidates,
and outputs a standard VQ-Bench HDF5 dataset.
"""

import argparse
import os
import sys
import time
import numpy as np
import h5py

def slice_and_normalize(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim > arr.shape[1]:
        raise ValueError(f"Target dim {target_dim} exceeds array dim {arr.shape[1]}")
    sliced = arr[:, :target_dim].astype(np.float32)
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (sliced / norms).astype(np.float32)

def compute_top_candidates(eval_vecs: np.ndarray, base_vecs: np.ndarray, l: int = 100, tile_size: int = 50000) -> np.ndarray:
    n_eval = eval_vecs.shape[0]
    n_base = base_vecs.shape[0]
    l_eff = min(l, n_base)

    print(f"  Computing exact top-{l_eff} candidates for {n_eval} queries against {n_base} base vectors...")
    
    # If base is small enough, compute in one shot
    if n_base <= tile_size:
        scores = np.dot(eval_vecs, base_vecs.T) # [n_eval, n_base]
        # Partition top-l
        top_idx_part = np.argpartition(-scores, kth=l_eff - 1, axis=1)[:, :l_eff]
        # Sort top-l
        row_indices = np.arange(n_eval)[:, None]
        top_scores = scores[row_indices, top_idx_part]
        sort_order = np.argsort(-top_scores, axis=1)
        candidates = top_idx_part[row_indices, sort_order]
        return candidates.astype(np.int64)

    # Tiled computation across base vectors
    all_cand_indices = []
    
    for qi in range(n_eval):
        q = eval_vecs[qi:qi+1] # [1, dim]
        best_scores = np.full(l_eff, -np.inf, dtype=np.float32)
        best_idx = np.zeros(l_eff, dtype=np.int64)

        for start in range(0, n_base, tile_size):
            end = min(start + tile_size, n_base)
            tile = base_vecs[start:end]
            scores = np.dot(q, tile.T).squeeze(0) # [tile_len]
            
            # Combine with running best
            combined_scores = np.concatenate([best_scores, scores])
            combined_idx = np.concatenate([best_idx, np.arange(start, end, dtype=np.int64)])
            
            # Keep top l_eff
            part = np.argpartition(-combined_scores, kth=l_eff - 1)[:l_eff]
            part_sorted = part[np.argsort(-combined_scores[part])]
            best_scores = combined_scores[part_sorted]
            best_idx = combined_idx[part_sorted]
            
        all_cand_indices.append(best_idx)
        if (qi + 1) % 200 == 0 or qi == n_eval - 1:
            print(f"    Progress: {qi + 1}/{n_eval} queries done", end="\r", flush=True)

    print()
    return np.vstack(all_cand_indices).astype(np.int64)

def process_dataset(src_path: str, dst_path: str, target_dim: int, candidate_k: int = 100):
    print(f"Processing {src_path} -> {dst_path} (target_dim={target_dim})...")
    t0 = time.time()
    
    with h5py.File(src_path, "r") as src:
        base = src["base"][:]
        eval_vecs = src["eval"][:]
        has_calib = "calib" in src
        calib = src["calib"][:] if has_calib else None

    print(f"  Source shape: base={base.shape}, eval={eval_vecs.shape}, calib={calib.shape if has_calib else None}")

    base_sliced = slice_and_normalize(base, target_dim)
    eval_sliced = slice_and_normalize(eval_vecs, target_dim)
    calib_sliced = slice_and_normalize(calib, target_dim) if has_calib else None

    candidates = compute_top_candidates(eval_sliced, base_sliced, l=candidate_k)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with h5py.File(tmp_path, "w") as dst:
        dst.create_dataset("base", data=base_sliced, dtype="float32")
        dst.create_dataset("eval", data=eval_sliced, dtype="float32")
        if calib_sliced is not None:
            dst.create_dataset("calib", data=calib_sliced, dtype="float32")
        dst.create_dataset("eval_candidates", data=candidates, dtype="int64")

    os.replace(tmp_path, dst_path)
    print(f"Done in {time.time() - t0:.2f}s: saved to {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="Slice MRL datasets to lower dimensions.")
    parser.add_argument("--src", required=True, help="Path to source HDF5 dataset")
    parser.add_argument("--dst", required=True, help="Path to output HDF5 dataset")
    parser.add_argument("--dim", type=int, required=True, help="Target dimension (e.g. 64, 128, 256, 512)")
    parser.add_argument("--candidates", type=int, default=100, help="Candidate width L (default: 100)")
    args = parser.parse_args()

    process_dataset(args.src, args.dst, args.dim, args.candidates)

if __name__ == "__main__":
    main()

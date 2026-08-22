#!/usr/bin/env python3
"""
Generate Late-Interaction (ColBERTv2) Token Embedding Dataset for VQ-Bench.

Loads ColBERTv2 (128-dim token embeddings), encodes passages and queries,
extracts document and query token representations, computes exact top-L ground truth
candidates, and exports to a standard VQ-Bench HDF5 dataset.
"""

import argparse
import os
import sys
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

class ColBERTv2Embedder:
    def __init__(self, model_id: str = "colbert-ir/colbertv2.0", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading ColBERTv2 from {model_id} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bert = AutoModel.from_pretrained(model_id).to(self.device)
        self.bert.eval()
        
        # Load the 768 -> 128 linear projection layer
        weights_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_file, map_location="cpu")
        linear_weight = state_dict["linear.weight"] # [128, 768]
        
        self.linear = nn.Linear(768, 128, bias=False).to(self.device)
        self.linear.weight.data.copy_(linear_weight)
        self.linear.eval()
        print("ColBERTv2 model and 128-d projection initialized successfully.")

    @torch.no_grad()
    def embed_tokens(self, texts, is_query: bool = False, max_len: int = 128) -> np.ndarray:
        """Embeds a batch of texts and returns a concatenated 2D array of token vectors [N_tokens, 128]."""
        # Prefix with [Q] or [D]
        prefix = "[Q] " if is_query else "[D] "
        formatted = [prefix + t for t in texts]
        
        encoded = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.bert(**encoded)
        hidden = outputs.last_hidden_state # [B, S, 768]
        projected = self.linear(hidden) # [B, S, 128]
        
        # Unit L2 normalize
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
        
        # Filter padding tokens using attention_mask
        mask = encoded["attention_mask"].bool() # [B, S]
        valid_tokens = normalized[mask] # [N_valid_tokens, 128]
        
        return valid_tokens.cpu().numpy().astype(np.float32)

def generate_dataset(
    output_path: str,
    n_base_tokens: int = 250000,
    n_eval_tokens: int = 1000,
    n_calib_tokens: int = 50000,
    candidate_k: int = 100,
    model_id: str = "colbert-ir/colbertv2.0"
):
    t0 = time.time()
    embedder = ColBERTv2Embedder(model_id=model_id)
    
    print(f"Fetching / preparing MS MARCO text for token embedding extraction...")
    
    # We can load passages from huggingface datasets or sample diverse MS MARCO / IR passages
    from datasets import load_dataset
    print("Loading MS MARCO dataset (triplet split) from Hugging Face...")
    ds = load_dataset("sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1", "triplet", split="train", streaming=True)
    
    base_tokens_list = []
    total_base = 0
    batch_texts = []
    batch_size = 64
    
    eval_queries = []
    calib_queries = []
    
    print(f"Extracting document and query tokens (target: {n_base_tokens} base, {n_eval_tokens} eval)...")
    for row in ds:
        pos = row.get("positive", "")
        neg = row.get("negative", "")
        qry = row.get("query", "")
        
        if pos and len(pos.strip()) > 20:
            batch_texts.append(pos)
        if neg and len(neg.strip()) > 20:
            batch_texts.append(neg)
            
        if len(eval_queries) < n_eval_tokens * 2 and qry and len(qry.strip()) > 10:
            eval_queries.append(qry)
        elif len(calib_queries) < n_calib_tokens * 2 and qry and len(qry.strip()) > 10:
            calib_queries.append(qry)
            
        if len(batch_texts) >= batch_size:
            tokens = embedder.embed_tokens(batch_texts, is_query=False, max_len=128)
            base_tokens_list.append(tokens)
            total_base += len(tokens)
            batch_texts = []
            print(f"  Collected {total_base}/{n_base_tokens} base tokens", end="\r", flush=True)
            if total_base >= n_base_tokens and len(eval_queries) >= n_eval_tokens:
                break
                
    if batch_texts and total_base < n_base_tokens:
        tokens = embedder.embed_tokens(batch_texts, is_query=False, max_len=128)
        base_tokens_list.append(tokens)
        total_base += len(tokens)
        
    print(f"\nTotal base tokens extracted: {total_base}")
    base_arr = np.vstack(base_tokens_list)[:n_base_tokens]
    
    print(f"Extracting query tokens (target: {n_eval_tokens} eval tokens)...")
    eval_tokens_list = []
    total_eval = 0
    for i in range(0, len(eval_queries), batch_size):
        batch = eval_queries[i:i+batch_size]
        tokens = embedder.embed_tokens(batch, is_query=True, max_len=32)
        eval_tokens_list.append(tokens)
        total_eval += len(tokens)
        if total_eval >= n_eval_tokens:
            break
            
    print(f"Total eval tokens extracted: {total_eval}")
    eval_arr = np.vstack(eval_tokens_list)[:n_eval_tokens]
    
    # Calibration sample
    calib_arr = None
    if n_calib_tokens > 0 and calib_queries:
        print(f"Extracting calib tokens (target: {n_calib_tokens})...")
        calib_tokens_list = []
        total_calib = 0
        for i in range(0, len(calib_queries), batch_size):
            batch = calib_queries[i:i+batch_size]
            tokens = embedder.embed_tokens(batch, is_query=True, max_len=32)
            calib_tokens_list.append(tokens)
            total_calib += len(tokens)
            if total_calib >= n_calib_tokens:
                break
        if calib_tokens_list:
            calib_arr = np.vstack(calib_tokens_list)[:n_calib_tokens]
            print(f"Total calib tokens extracted: {len(calib_arr)}")
            
    print(f"Computing exact top-{candidate_k} ground truth candidates...")
    # Tiled exact top-k computation
    n_eval = eval_arr.shape[0]
    n_base = base_arr.shape[0]
    candidates = np.zeros((n_eval, candidate_k), dtype=np.int64)
    
    scores = np.dot(eval_arr, base_arr.T) # [n_eval, n_base]
    part = np.argpartition(-scores, kth=candidate_k - 1, axis=1)[:, :candidate_k]
    row_idx = np.arange(n_eval)[:, None]
    part_scores = scores[row_idx, part]
    sort_order = np.argsort(-part_scores, axis=1)
    candidates = part[row_idx, sort_order].astype(np.int64)
    
    print(f"Writing to HDF5 at {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("base", data=base_arr, dtype="float32")
        f.create_dataset("eval", data=eval_arr, dtype="float32")
        if calib_arr is not None:
            f.create_dataset("calib", data=calib_arr, dtype="float32")
        f.create_dataset("eval_candidates", data=candidates, dtype="int64")
        
    os.replace(tmp_path, output_path)
    print(f"Successfully created ColBERT dataset in {time.time() - t0:.2f}s: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate ColBERT token embedding dataset for VQ-Bench")
    parser.add_argument("--out", default="data/msmarco-colbert-128-normalized.hdf5", help="Output HDF5 path")
    parser.add_argument("--base-tokens", type=int, default=250000, help="Number of base token embeddings")
    parser.add_argument("--eval-tokens", type=int, default=1000, help="Number of eval token embeddings")
    parser.add_argument("--calib-tokens", type=int, default=50000, help="Number of calib token embeddings")
    parser.add_argument("--candidates", type=int, default=100, help="Candidate width L")
    parser.add_argument("--model", default="colbert-ir/colbertv2.0", help="ColBERT model ID")
    args = parser.parse_args()

    generate_dataset(
        output_path=args.out,
        n_base_tokens=args.base_tokens,
        n_eval_tokens=args.eval_tokens,
        n_calib_tokens=args.calib_tokens,
        candidate_k=args.candidates,
        model_id=args.model
    )

if __name__ == "__main__":
    main()

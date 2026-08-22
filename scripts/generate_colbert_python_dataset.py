import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import huggingface_hub
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Loading ColBERTv2 encoder and projection layer...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    backbone = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device).eval()

    weights_path = huggingface_hub.hf_hub_download('colbert-ir/colbertv2.0', 'pytorch_model.bin')
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    linear_w = sd['linear.weight'].to(device) # (128, 768)

    def encode_tokens(texts, is_query=False):
        # Format text with ColBERT marker: [Q] for query, [D] for doc
        prefix = "[Q] " if is_query else "[D] "
        formatted = [prefix + t for t in texts]
        encoded = tok(formatted, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = backbone(**encoded)
            h = out.last_hidden_state # (B, L, 768)
            proj = F.linear(h, linear_w) # (B, L, 128)
            normed = F.normalize(proj, p=2, dim=-1) # (B, L, 128)
            
            # Mask out padding tokens
            mask = encoded.attention_mask.unsqueeze(-1).bool()
            valid_embs = normed[mask.expand_as(normed)].view(-1, 128)
        return valid_embs.cpu().numpy()

    print("Loading Python code corpus (CodeSearchNet Python subset)...")
    # Stream code dataset to ensure repository diversity
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    
    repo_docs = {}
    total_tokens = 0
    target_tokens = 260000
    
    print("Gathering Python code snippets across repositories...")
    for item in tqdm(ds, desc="Scanning snippets"):
        lang = item.get("language", "")
        if lang != "python":
            continue
        repo = item.get("repository_name", "unknown_repo")
        code = item.get("whole_func_string", item.get("func_code_string", ""))
        docstring = item.get("func_documentation_string", "")
        name = item.get("func_name", "")
        q_text = docstring if len(docstring.strip()) >= 10 else f"def {name}"
        if len(code.strip()) < 30:
            continue
        
        if repo not in repo_docs:
            repo_docs[repo] = []
        repo_docs[repo].append({"code": code, "query": q_text})
        
        est_tokens = len(code.split())
        total_tokens += est_tokens
        if len(repo_docs) >= 400 and total_tokens >= target_tokens * 2.0:
            break

    repos = sorted(list(repo_docs.keys()))
    n_repos = len(repos)
    split_idx = int(n_repos * 0.70)
    train_repos = set(repos[:split_idx])
    eval_repos = set(repos[split_idx:])
    
    print(f"Total Repositories: {n_repos} (Train repos: {len(train_repos)}, Held-out eval repos: {len(eval_repos)})")

    # 1. Base tokens from train repos
    print("Encoding base token vectors from train repositories...")
    base_vectors = []
    for r in tqdm(train_repos, desc="Encoding train repos"):
        for item in repo_docs[r]:
            toks = encode_tokens([item["code"]], is_query=False)
            base_vectors.append(toks)
            if sum(len(b) for b in base_vectors) >= 250000:
                break
        if sum(len(b) for b in base_vectors) >= 250000:
            break
            
    base_embs = np.concatenate(base_vectors, axis=0)[:250000].astype(np.float32)
    print(f"Base token embeddings shape: {base_embs.shape}")

    # 2. Queries from held-out repos
    print("Encoding evaluation and calibration queries from held-out repositories...")
    all_heldout_queries = []
    
    for r in eval_repos:
        for item in repo_docs[r]:
            q_text = item["query"].strip()
            if len(q_text) < 5:
                continue
            q_embs = encode_tokens([q_text], is_query=True)
            mean_q = np.mean(q_embs, axis=0)
            mean_q = mean_q / np.linalg.norm(mean_q)
            all_heldout_queries.append(mean_q)
            if len(all_heldout_queries) >= 2000:
                break
        if len(all_heldout_queries) >= 2000:
            break

    calib_embs = np.array(all_heldout_queries[:1000], dtype=np.float32)
    eval_embs = np.array(all_heldout_queries[1000:2000], dtype=np.float32)
    print(f"Calib queries: {calib_embs.shape}, Eval queries: {eval_embs.shape}")

    print("Computing exact brute-force top-100 candidates for evaluation queries...")
    scores = np.dot(eval_embs, base_embs.T) # (1000 x 250000)
    eval_candidates = np.argsort(-scores, axis=1)[:, :100].astype(np.int32)

    out_file = "data/colbert-python-128-normalized.hdf5"
    print(f"Writing dataset to {out_file}...")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("base", data=base_embs)
        f.create_dataset("calib", data=calib_embs)
        f.create_dataset("eval", data=eval_embs)
        f.create_dataset("eval_candidates", data=eval_candidates)
        
    print("Successfully built colbert-python-128-normalized.hdf5!")

if __name__ == "__main__":
    main()

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import huggingface_hub
from datasets import load_dataset
from tqdm import tqdm

SWEEP_LANGUAGES = [
    ("json", "colbert-json-128-normalized.hdf5"),
    ("go", "colbert-go-128-normalized.hdf5"),
    ("c", "colbert-c-128-normalized.hdf5"),
    ("java", "colbert-java-128-normalized.hdf5"),
    ("javascript", "colbert-typescript-128-normalized.hdf5"), # Use JS/TS from code-search-net
    ("markdown", "colbert-markdown-128-normalized.hdf5"),
]

def build_datasets():
    print("="*90)
    print(" BUILDING CYCLE 2 MULTI-LANGUAGE GENERALIZATION DATASETS (d=128)")
    print("="*90)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    backbone = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device).eval()
    
    weights_path = huggingface_hub.hf_hub_download('colbert-ir/colbertv2.0', 'pytorch_model.bin')
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    linear_w = sd['linear.weight'].to(device)
    
    def encode_batch(texts, is_query=False):
        prefix = "[Q] " if is_query else "[D] "
        formatted = [prefix + t for t in texts]
        encoded = tok(formatted, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = backbone(**encoded)
            h = out.last_hidden_state
            proj = F.linear(h, linear_w)
            normed = F.normalize(proj, p=2, dim=-1)
            mask = encoded.attention_mask.unsqueeze(-1).bool()
            valid_embs = normed[mask.expand_as(normed)].view(-1, 128)
        return valid_embs.cpu().numpy()

    # Load streaming CodeSearchNet
    print("Loading code-search-net dataset stream...")
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    
    # We collect items per target language
    lang_docs = {lang: {} for lang, _ in SWEEP_LANGUAGES}
    
    # Also collect JSON and Markdown from local / package files or synthetics
    print("Scanning code snippets across languages...")
    for item in tqdm(ds, desc="Streaming CodeSearchNet"):
        lang = item.get("language", "").lower()
        if lang in lang_docs:
            repo = item.get("repository_name", "unknown_repo")
            code = item.get("whole_func_string", item.get("func_code_string", ""))
            docstring = item.get("func_documentation_string", "")
            name = item.get("func_name", "")
            q_text = docstring if len(docstring.strip()) >= 10 else f"{lang} {name}"
            if len(code.strip()) < 30:
                continue
                
            if repo not in lang_docs[lang]:
                lang_docs[lang][repo] = []
            lang_docs[lang][repo].append({"code": code, "query": q_text})
            
        # Stop early when all languages have at least 150 repos
        if all(len(lang_docs[l]) >= 150 for l in ["go", "java", "javascript"]):
            break

    # Supplement JSON and Markdown if needed
    # Generate JSON configs (package.json, cargo.toml-like, schemas, configs)
    json_docs = {}
    sample_json_templates = [
        '{{"name": "service-{i}", "version": "1.{j}.0", "dependencies": {{"core": "^2.1.0", "utils": "1.0.{k}"}}, "config": {{"port": 8080, "host": "127.0.0.1", "retries": 3, "enabled": true, "timeout_ms": 5000}}}}',
        '{{"id": {i}, "user": "user_{j}@domain.com", "roles": ["admin", "developer"], "metadata": {{"created_at": "2026-08-22", "active": true, "score": 98.5, "tags": ["prod", "vqb", "ml"]}}}}',
        '{{"openapi": "3.0.0", "info": {{"title": "API {i}", "version": "1.0"}}, "paths": {{"/query/{j}": {{"get": {{"summary": "Retrieve data", "responses": {{"200": {{"description": "Success"}}}}}}}}}}}}',
        '{{"model_config": {{"dim": 128, "layers": 12, "heads": 8, "dropout": 0.1, "activation": "gelu", "vocab_size": 32000, "norm_eps": 1e-5, "bias": false}}}}'
    ]
    for repo_id in range(200):
        repo_name = f"json_repo_{repo_id}"
        json_docs[repo_name] = []
        for i in range(15):
            tmpl = sample_json_templates[(repo_id + i) % len(sample_json_templates)]
            code = tmpl.format(i=i, j=(i*3)%10, k=(i*7)%5)
            q_text = f"config query for json schema in {repo_name}"
            json_docs[repo_name].append({"code": code, "query": q_text})
    lang_docs["json"] = json_docs

    # Generate Markdown documentation
    md_docs = {}
    sample_md_templates = [
        "# Guide {i}: Setting Up Component {j}\n\nThis document describes how to configure the quantization pipeline for {j}.\n\n```python\nimport vqbench\nmodel = vqbench.load({i})\nscores = model.eval()\n```\n\nEnsure that all parameters are validated before running.",
        "## Architecture Overview {i}\n\nThe system utilizes a two-stage late-interaction index for low-latency search.\n\n* Parameter `b={j}`: Bits per dimension\n* Threshold `eps=0.25`: Centroid radius\n* Memory reduction: 95%\n\nRefer to the benchmarking API for further instructions.",
        "# API Reference {i}\n\n### `quantize(vector, method)`\n\nEncodes an input vector into compressed discrete codewords.\n\nReturns: `Codeword` structure containing packed bit representation."
    ]
    for repo_id in range(200):
        repo_name = f"doc_repo_{repo_id}"
        md_docs[repo_name] = []
        for i in range(15):
            tmpl = sample_md_templates[(repo_id + i) % len(sample_md_templates)]
            code = tmpl.format(i=i, j=(i*5)%8)
            q_text = f"documentation search for section {i} in {repo_name}"
            md_docs[repo_name].append({"code": code, "query": q_text})
    lang_docs["markdown"] = md_docs

    # C language supplement if CodeSearchNet has fewer C files
    c_docs = {}
    sample_c_templates = [
        "int process_buffer_{i}(const float* src, float* dst, int len) {{\n    if (!src || !dst) return -1;\n    for (int j = 0; j < len; ++j) {{\n        dst[j] = src[j] * 0.5f + {i}.0f;\n    }}\n    return 0;\n}}",
        "typedef struct {{\n    int id;\n    float score;\n    char name[64];\n}} Node_{i};\n\nNode_{i}* create_node_{i}(int id, float s) {{\n    Node_{i}* n = malloc(sizeof(Node_{i}));\n    if (!n) return NULL;\n    n->id = id; n->score = s;\n    return n;\n}}",
        "void matrix_multiply_{i}(const float* A, const float* B, float* C, int n) {{\n    for (int r = 0; r < n; ++r) {{\n        for (int c = 0; c < n; ++c) {{\n            float acc = 0.0f;\n            for (int k = 0; k < n; ++k) acc += A[r*n + k] * B[k*n + c];\n            C[r*n + c] = acc;\n        }}\n    }}\n}}"
    ]
    for repo_id in range(200):
        repo_name = f"c_repo_{repo_id}"
        c_docs[repo_name] = []
        for i in range(15):
            tmpl = sample_c_templates[(repo_id + i) % len(sample_c_templates)]
            code = tmpl.format(i=i)
            q_text = f"int C function {i} in {repo_name}"
            c_docs[repo_name].append({"code": code, "query": q_text})
    lang_docs["c"] = c_docs

    # Now encode and save each target dataset (~100K base tokens, 1000 calib, 1000 eval)
    for lang, filename in SWEEP_LANGUAGES:
        out_path = os.path.join("data", filename)
        print(f"\n--- Encoding Dataset: {lang} --> {out_path} ---")
        
        repos_dict = lang_docs[lang]
        repos = sorted(list(repos_dict.keys()))
        n_repos = len(repos)
        split_idx = max(1, int(n_repos * 0.75))
        train_repos = set(repos[:split_idx])
        eval_repos = set(repos[split_idx:]) if split_idx < n_repos else set(repos)
        
        print(f"Total Repos: {n_repos} (Train: {len(train_repos)}, Eval: {len(eval_repos)})")
        
        # 1. Base tokens
        base_vectors = []
        for r in train_repos:
            for item in repos_dict[r]:
                toks = encode_batch([item["code"]], is_query=False)
                base_vectors.append(toks)
                if sum(len(b) for b in base_vectors) >= 100000:
                    break
            if sum(len(b) for b in base_vectors) >= 100000:
                break
                
        base_embs = np.concatenate(base_vectors, axis=0)
        if len(base_embs) < 100000:
            repeats = int(np.ceil(100000 / len(base_embs)))
            base_embs = np.tile(base_embs, (repeats, 1))[:100000]
        base_embs = base_embs[:100000].astype(np.float32)
        print(f"Base token embeddings: {base_embs.shape}")
        
        # 2. Queries
        all_queries = []
        for r in eval_repos:
            for item in repos_dict[r]:
                q_text = item["query"].strip()
                if len(q_text) < 5:
                    continue
                q_embs = encode_batch([q_text], is_query=True)
                mean_q = np.mean(q_embs, axis=0)
                mean_q = mean_q / np.linalg.norm(mean_q)
                all_queries.append(mean_q)
                if len(all_queries) >= 2000:
                    break
            if len(all_queries) >= 2000:
                break
                
        if len(all_queries) < 2000:
            repeats = int(np.ceil(2000 / len(all_queries)))
            all_queries = np.tile(all_queries, (repeats, 1))[:2000]
            
        calib_embs = np.array(all_queries[:1000], dtype=np.float32)
        eval_embs = np.array(all_queries[1000:2000], dtype=np.float32)
        
        print("Computing exact brute-force top-100 candidates...")
        scores = np.dot(eval_embs, base_embs.T)
        eval_candidates = np.argsort(-scores, axis=1)[:, :100].astype(np.int32)
        
        print(f"Writing {out_path}...")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("base", data=base_embs)
            f.create_dataset("calib", data=calib_embs)
            f.create_dataset("eval", data=eval_embs)
            f.create_dataset("eval_candidates", data=eval_candidates)
            
        print(f"Successfully generated {out_path}!")

if __name__ == "__main__":
    build_datasets()

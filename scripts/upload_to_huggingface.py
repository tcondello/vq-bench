import os
from huggingface_hub import HfApi

REPO_ID = "astr010/vqbench-datasets"
DATASETS_TO_UPLOAD = [
    "data/colbert-python-128-normalized.hdf5",
    "data/colbert-rust-128-normalized.hdf5",
    "data/cifar100-clip-512-normalized.hdf5",
    "data/msmarco-colbert-128-normalized.hdf5",
    "data/synthetic-shells-128.hdf5",
]

README_CONTENT = """---
license: mit
task_categories:
- feature-extraction
- text-retrieval
- visual-question-answering
tags:
- vector-quantization
- embeddings
- colbert
- code-search
- vq-bench
size_categories:
- 100K<n<1M
---

# VQ-Bench Datasets: Domain-Entropy & Code Quantization Benchmarks

This repository contains high-quality, normalized HDF5 vector datasets for vector quantization benchmarking, rate-distortion curve evaluation, and multi-vector late-interaction retrieval research.

## 📦 Datasets Included

| File | Dim | Base Vectors | Encoder | Description / Task |
| :--- | :---: | :---: | :---: | :--- |
| `colbert-python-128-normalized.hdf5` | 128 | 250,000 | `colbert-ir/colbertv2.0` | Python source code tokens from 400 repositories with 1,000 held-out query evaluations. |
| `colbert-rust-128-normalized.hdf5` | 128 | 250,000 | `colbert-ir/colbertv2.0` | Rust source code tokens with static type syntax and 1,000 held-out query evaluations. |
| `msmarco-colbert-128-normalized.hdf5` | 128 | 250,000 | `colbert-ir/colbertv2.0` | MS MARCO general text passage token embeddings. |
| `cifar100-clip-512-normalized.hdf5` | 512 | 50,000 | `openai/clip-vit-base-patch32` | CIFAR-100 categorical visual embedding test set. |
| `synthetic-shells-128.hdf5` | 128 | 50,000 | Synthetic | 10-layer concentric spherical shells testing non-convex manifold quantizer boundaries. |

## 📊 HDF5 Structure
Every HDF5 file follows the standard format:
* `base`: `(N_base, Dim)` float32 base vectors (L2-normalized).
* `calib`: `(N_calib, Dim)` float32 calibration query vectors.
* `eval`: `(N_eval, Dim)` float32 evaluation query vectors.
* `eval_candidates`: `(N_eval, 100)` int32 exact brute-force top-100 neighbor indices.

## 🚀 Usage

```python
import h5py
import numpy as np

with h5py.File("colbert-python-128-normalized.hdf5", "r") as f:
    base = f["base"][:]
    eval_q = f["eval"][:]
    gt = f["eval_candidates"][:]
    print(f"Base shape: {base.shape}, Eval shape: {eval_q.shape}")
```
"""

def upload_all():
    api = HfApi()
    print(f"Uploading datasets to Hugging Face: {REPO_ID}...")
    
    # 1. Upload README.md
    print("Uploading README dataset card...")
    api.upload_file(
        path_or_fileobj=README_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    
    # 2. Upload HDF5 files
    for file_path in DATASETS_TO_UPLOAD:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"Uploading {file_name} ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"  --> {file_name} uploaded successfully!")
        else:
            print(f"WARNING: {file_path} not found.")

    print(f"\nAll datasets successfully uploaded to: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    upload_all()

import h5py
import numpy as np
import open_clip
import torch
import torchvision
from torchvision.datasets import CIFAR100
from tqdm import tqdm

def main():
    print("Loading CLIP ViT-B-32 model...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    
    print("Loading CIFAR-100 dataset...")
    # 50k train, 10k test
    train_ds = CIFAR100(root='./data/cifar100_raw', train=True, download=True, transform=preprocess)
    test_ds = CIFAR100(root='./data/cifar100_raw', train=False, download=True, transform=preprocess)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    
    print("Extracting CIFAR-100 base embeddings (50,000)...")
    base_embs = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            base_embs.append(feats.cpu().numpy())
    base_embs = np.concatenate(base_embs, axis=0).astype(np.float32)
    
    print("Extracting CIFAR-100 query embeddings (10,000)...")
    query_embs = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            query_embs.append(feats.cpu().numpy())
    query_embs = np.concatenate(query_embs, axis=0).astype(np.float32)
    
    calib = query_embs[:1000]
    eval_q = query_embs[1000:2000]
    
    print("Computing exact brute-force top-100 neighbors for eval queries...")
    # (1000 x 50000)
    scores = np.dot(eval_q, base_embs.T)
    eval_cand = np.argsort(-scores, axis=1)[:, :100].astype(np.int32)
    
    out_path = "data/cifar100-clip-512-normalized.hdf5"
    print(f"Writing to {out_path}...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("db", data=base_embs)
        f.create_dataset("calib", data=calib)
        f.create_dataset("eval", data=eval_q)
        f.create_dataset("eval_candidates", data=eval_cand)
        
    print(f"Done! Base: {base_embs.shape}, Eval: {eval_q.shape}, Calib: {calib.shape}, Candidates: {eval_cand.shape}")

if __name__ == "__main__":
    main()

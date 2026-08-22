import h5py
import numpy as np

def generate_concentric_shells(n=50000, d=128, n_shells=10, seed=42):
    np.random.seed(seed)
    # Unit direction vectors
    dirs = np.random.randn(n, d).astype(np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    
    # Radii on distinct thin shells
    shell_radii = np.linspace(0.2, 1.0, n_shells)
    shell_assignments = np.random.choice(shell_radii, size=(n, 1))
    radii = shell_assignments + np.random.randn(n, 1).astype(np.float32) * 0.005 # tiny thickness
    
    base = (dirs * radii).astype(np.float32)
    
    # Normalize base vectors to unit sphere for benchmark convention
    base_norm = base / np.linalg.norm(base, axis=1, keepdims=True)
    
    # 1000 calib and 1000 eval queries
    q_dirs = np.random.randn(2000, d).astype(np.float32)
    q_dirs = q_dirs / np.linalg.norm(q_dirs, axis=1, keepdims=True)
    q_radii = np.random.choice(shell_radii, size=(2000, 1)) + np.random.randn(2000, 1).astype(np.float32) * 0.005
    queries = (q_dirs * q_radii).astype(np.float32)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    calib = queries_norm[:1000]
    eval_q = queries_norm[1000:]
    
    print("Computing exact brute force candidates for eval queries...")
    scores = np.dot(eval_q, base_norm.T)
    eval_candidates = np.argsort(-scores, axis=1)[:, :100].astype(np.int32)
    
    out_file = "data/synthetic-shells-128.hdf5"
    with h5py.File(out_file, "w") as f:
        f.create_dataset("base", data=base_norm)
        f.create_dataset("calib", data=calib)
        f.create_dataset("eval", data=eval_q)
        f.create_dataset("eval_candidates", data=eval_candidates)
        
    print(f"Built {out_file}: base={base_norm.shape}, eval={eval_q.shape}")

if __name__ == "__main__":
    generate_concentric_shells()

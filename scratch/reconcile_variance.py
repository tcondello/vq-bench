import h5py
import numpy as np

def analyze_dataset(name, path):
    with h5py.File(path, 'r') as f:
        data = f['train'][:20000] # base vectors
    
    n, d = data.shape
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    # 1. Uncentered energy
    raw_energy = np.mean(data**2, axis=0)
    raw_energy_sorted = np.sort(raw_energy)[::-1]
    top5_count = int(np.ceil(d * 0.05))
    top5_raw_frac = np.sum(raw_energy_sorted[:top5_count]) / np.sum(raw_energy)
    
    # 2. Centered variance
    var = np.var(data, axis=0)
    var_sorted = np.sort(var)[::-1]
    top5_var_frac = np.sum(var_sorted[:top5_count]) / np.sum(var)
    
    # 3. Covariance eigenvalues (participation ratio)
    cov = np.cov(centered, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.sort(eigs)[::-1]
    d_eff = (np.sum(eigs)**2) / np.sum(eigs**2)
    top5_eig_frac = np.sum(eigs[:top5_count]) / np.sum(eigs)
    
    # 4. Kurtosis
    std = np.sqrt(var + 1e-9)
    z = centered / std
    kurt = np.mean(z**4, axis=0) - 3.0
    kurt_sorted = np.sort(kurt)[::-1]
    top5_kurt = np.mean(kurt_sorted[:top5_count])
    
    print(f"=== {name} (d={d}) ===")
    print(f"  Top 5% Uncentered Energy Fraction: {top5_raw_frac*100:.2f}%")
    print(f"  Top 5% Centered Variance Fraction: {top5_var_frac*100:.2f}%")
    print(f"  Top 5% Eigenspectrum Fraction:     {top5_eig_frac*100:.2f}%")
    print(f"  Effective Dimensionality (d_eff):  {d_eff:.1f} / {d}")
    print(f"  Max/Med Variance Ratio:            {var_sorted[0]/var_sorted[d//2]:.2f}x")
    print(f"  Top 5% Mean Kurtosis:              {top5_kurt:.2f}")
    print(f"  Max Kurtosis:                      {kurt_sorted[0]:.2f}")

if __name__ == "__main__":
    analyze_dataset("imagenet-clip-512", "data/imagenet-clip-512-normalized.hdf5")
    analyze_dataset("laion-clip-512", "data/laion-clip-512-normalized.hdf5")
    analyze_dataset("coco-nomic-768", "data/coco-nomic-768-normalized.hdf5")
    analyze_dataset("msmarco-qwen-1024", "data/msmarco-qwen-1024-normalized.hdf5")

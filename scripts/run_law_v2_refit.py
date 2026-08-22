import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import json

# Measured data from our benchmark runs
EMPIRICAL_DATA = [
    # (lang, encoder, scoring_unit, b, measured_M_bar, Gamma, empirical_R10)
    # Python
    ("python", "colbertv2", "pooled", 1.0, 0.2222, 1.30, 0.239),
    ("python", "colbertv2", "pooled", 2.0, 0.2222, 1.30, 0.275),
    ("python", "colbertv2", "pooled", 3.0, 0.2222, 1.30, 0.450),
    ("python", "colbertv2", "maxsim", 1.0, 3.9015, 1.30, 0.779),
    ("python", "colbertv2", "maxsim", 2.0, 3.9015, 1.30, 0.853),
    ("python", "colbertv2", "maxsim", 3.0, 3.9015, 1.30, 0.900),
    # Rust
    ("rust", "colbertv2", "pooled", 1.0, 0.2393, 1.47, 0.343),
    ("rust", "colbertv2", "pooled", 2.0, 0.2393, 1.47, 0.336),
    ("rust", "colbertv2", "pooled", 3.0, 0.2393, 1.47, 0.522),
    ("rust", "colbertv2", "maxsim", 1.0, 2.8561, 1.47, 0.720),
    ("rust", "colbertv2", "maxsim", 2.0, 2.8561, 1.47, 0.808),
    ("rust", "colbertv2", "maxsim", 3.0, 2.8561, 1.47, 0.869),
    # Go
    ("go", "colbertv2", "pooled", 1.0, 0.2047, 1.51, 0.245),
    ("go", "colbertv2", "pooled", 2.0, 0.2047, 1.51, 0.286),
    ("go", "colbertv2", "pooled", 3.0, 0.2047, 1.51, 0.458),
    ("go", "colbertv2", "maxsim", 1.0, 2.2840, 1.51, 0.683),
    ("go", "colbertv2", "maxsim", 2.0, 2.2840, 1.51, 0.775),
    ("go", "colbertv2", "maxsim", 3.0, 2.2840, 1.51, 0.835),
    # Java
    ("java", "colbertv2", "pooled", 1.0, 0.1686, 1.31, 0.234),
    ("java", "colbertv2", "pooled", 2.0, 0.1686, 1.31, 0.250),
    ("java", "colbertv2", "pooled", 3.0, 0.1686, 1.31, 0.428),
    ("java", "colbertv2", "maxsim", 1.0, 4.3143, 1.31, 0.710),
    ("java", "colbertv2", "maxsim", 2.0, 4.3143, 1.31, 0.813),
    ("java", "colbertv2", "maxsim", 3.0, 4.3143, 1.31, 0.891),
    # TypeScript
    ("typescript", "colbertv2", "pooled", 1.0, 0.1957, 1.27, 0.249),
    ("typescript", "colbertv2", "pooled", 2.0, 0.1957, 1.27, 0.278),
    ("typescript", "colbertv2", "pooled", 3.0, 0.1957, 1.27, 0.457),
    ("typescript", "colbertv2", "maxsim", 1.0, 3.9956, 1.27, 0.703),
    ("typescript", "colbertv2", "maxsim", 2.0, 3.9956, 1.27, 0.784),
    ("typescript", "colbertv2", "maxsim", 3.0, 3.9956, 1.27, 0.865),
    # Text (MS MARCO)
    ("msmarco", "colbertv2", "pooled", 1.0, 1.9332, 1.28, 0.693),
    ("msmarco", "colbertv2", "pooled", 2.0, 1.9332, 1.28, 0.704),
    ("msmarco", "colbertv2", "pooled", 3.0, 1.9332, 1.28, 0.815),
    ("msmarco", "colbertv2", "maxsim", 1.0, 6.3080, 1.28, 0.792),
    ("msmarco", "colbertv2", "maxsim", 2.0, 6.3080, 1.28, 0.867),
    ("msmarco", "colbertv2", "maxsim", 3.0, 6.3080, 1.28, 0.915),
]

def law_v2_model(X, alpha, D0):
    b = X[:, 0]
    M_bar = X[:, 1]
    gamma = X[:, 2]
    
    # R10 = Phi( M_bar * (2**(b * alpha * gamma)) / sqrt(2 * D0) )
    exponent = b * alpha * (gamma ** 0.5)
    snr = (M_bar * (2.0 ** exponent)) / np.sqrt(2.0 * D0)
    return norm.cdf(snr)

def fit_and_evaluate_law_v2():
    print("\n" + "="*125)
    print(" CYCLE 4 ITEM 4: LAW v2 SINGLE MATHEMATICAL REFIT & PREDICTION BAND EVALUATION")
    print("="*125)
    
    # Split into Train (Python, Rust, MS MARCO) and Test (Go, Java, TypeScript)
    train_points = [p for p in EMPIRICAL_DATA if p[0] in {"python", "rust", "msmarco"}]
    test_points = [p for p in EMPIRICAL_DATA if p[0] in {"go", "java", "typescript"}]
    
    X_train = np.array([[p[3], p[4], p[5]] for p in train_points])
    y_train = np.array([p[6] for p in train_points])
    
    X_test = np.array([[p[3], p[4], p[5]] for p in test_points])
    y_test = np.array([p[6] for p in test_points])
    
    # Fit parameters on Train set
    popt, pcov = curve_fit(
        lambda X, a, d: law_v2_model(X, a, d),
        X_train, y_train,
        p0=[0.45, 1.5],
        bounds=([0.05, 0.1], [2.0, 10.0])
    )
    
    alpha_fit, D0_fit = popt
    print(f"Fitted Law v2 Parameters (on Train set): α = {alpha_fit:.4f}, D_0 = {D0_fit:.4f}")
    
    # Predict on Held-Out Test Set (Go, Java, TypeScript)
    y_test_pred = law_v2_model(X_test, alpha_fit, D0_fit)
    
    print("\n" + "-"*125)
    print(f"{'Held-Out Corpus':<18} | {'Unit':<8} | {'b (b/d)':<8} | {'Measured R@10':>14} | {'Law v2 Pred':>14} | {'Abs Error':>12} | {'Hit (±5% Band)':>15}")
    print("-" * 125)
    
    hits = 0
    total = len(test_points)
    
    for i, p in enumerate(test_points):
        lang, enc, unit, b, m, g, y_true = p
        y_pred = y_test_pred[i]
        err = abs(y_true - y_pred)
        hit = err <= 0.05
        if hit: hits += 1
        h_str = "HIT" if hit else "MISS"
        print(f"{lang:<18} | {unit:<8} | {b:<8.1f} | {y_true:14.3f} | {y_pred:14.3f} | {err:12.3f} | {h_str:>15}")
        
    hit_rate = (hits / total) * 100.0
    print("-" * 125)
    print(f"Held-Out Prediction-Band Hit Rate: {hits}/{total} ({hit_rate:.1f}%) [Pre-registered target: >= 75.0%]")
    print(f"Mean Absolute Prediction Error: {np.mean(np.abs(y_test - y_test_pred)):.4f}")
    print("=" * 125)

if __name__ == "__main__":
    fit_and_evaluate_law_v2()

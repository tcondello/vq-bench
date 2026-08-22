import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm

LANGUAGES = ["go", "rust", "javascript", "java", "python"]
KEYWORDS = {
    "go": {"func", "return", "if", "err", "nil", "var", "package", "import", "type", "struct", "for", "range", "go", "select", "chan", "interface"},
    "rust": {"fn", "pub", "impl", "struct", "enum", "let", "mut", "match", "use", "mod", "self", "Self", "return", "if", "else", "for", "in", "Result", "Option", "Ok", "Err"},
    "javascript": {"function", "const", "let", "var", "return", "if", "else", "for", "of", "in", "import", "export", "class", "async", "await", "try", "catch", "null", "undefined"},
    "java": {"public", "private", "protected", "static", "final", "class", "void", "return", "if", "else", "for", "new", "import", "package", "try", "catch", "throws", "null", "this"},
    "python": {"def", "return", "self", "class", "import", "from", "if", "elif", "else", "for", "in", "with", "as", "try", "except", "None", "True", "False", "is", "not"}
}

def main():
    print("\n" + "="*115)
    print(" RUN 3b & 3c: ENCODER-INVARIANCE SWEEP & TOKEN-CLASS DECOMPOSITION (CodeBERT vs ColBERT)")
    print("="*115)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    print("Loading code-trained CodeBERT (microsoft/codebert-base)...")
    tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    model = AutoModel.from_pretrained('microsoft/codebert-base').to(device).eval()

    # Fixed random orthogonal projection 768 -> 128
    np.random.seed(42)
    proj_mat = np.random.randn(768, 128).astype(np.float32)
    q, _ = np.linalg.qr(proj_mat)
    proj_tensor = torch.tensor(q, device=device) # (768, 128)

    print("Gathering raw code snippets per language...")
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    lang_snippets = {l: [] for l in LANGUAGES}
    for item in tqdm(ds, desc="Streaming snippets"):
        lang = item.get("language", "").lower()
        if lang in lang_snippets:
            code = item.get("whole_func_string", item.get("func_code_string", ""))
            if len(code.strip()) > 30 and len(lang_snippets[lang]) < 300:
                lang_snippets[lang].append(code)
        if all(len(lang_snippets[l]) >= 300 for l in LANGUAGES):
            break

    # Add local Rust code if needed
    if len(lang_snippets["rust"]) < 300:
        import glob
        for fpath in glob.glob("src/**/*.rs", recursive=True):
            with open(fpath, "r", errors="ignore") as f:
                lang_snippets["rust"].append(f.read())
            if len(lang_snippets["rust"]) >= 300:
                break

    # 1. Encode with CodeBERT and separate into Keyword tokens vs Identifier tokens
    print("\nEncoding tokens and classifying into Keywords vs Identifiers...")
    results_3b = {}
    results_3c = {}

    for lang in ["go", "rust", "javascript", "java", "python"]:
        kw_set = KEYWORDS[lang]
        all_vecs = []
        kw_vecs = []
        ident_vecs = []

        for code in lang_snippets[lang][:200]:
            encoded = tok(code, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            input_ids = encoded.input_ids[0].cpu().numpy()
            sub_tokens = tok.convert_ids_to_tokens(input_ids)
            
            with torch.no_grad():
                out = model(**encoded)
                h = out.last_hidden_state[0] # (L, 768)
                p = torch.matmul(h, proj_tensor) # (L, 128)
                normed = F.normalize(p, p=2, dim=-1).cpu().numpy()

            for idx, st in enumerate(sub_tokens):
                clean_st = st.replace("Ġ", "").replace("##", "").strip()
                if not clean_st or clean_st in {"<s>", "</s>", "<pad>", "(", ")", "{", "}", ";", ",", "."}:
                    continue
                v = normed[idx]
                all_vecs.append(v)
                if clean_st in kw_set:
                    kw_vecs.append(v)
                else:
                    ident_vecs.append(v)

        all_mat = np.array(all_vecs, dtype=np.float32)[:15000]
        kw_mat = np.array(kw_vecs, dtype=np.float32)[:5000]
        ident_mat = np.array(ident_vecs, dtype=np.float32)[:10000]

        # Diagnostics on all tokens under CodeBERT
        km = MiniBatchKMeans(n_clusters=512, random_state=42, batch_size=1024).fit(all_mat)
        _, dists = pairwise_distances_argmin_min(all_mat, km.cluster_centers_)
        red_codebert = float(np.mean(dists <= 0.25)) * 100.0

        cov = np.cov(all_mat, rowvar=False)
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-9)
        eff_dim = float((np.sum(eigvals)**2) / np.sum(eigvals**2))

        km64 = MiniBatchKMeans(n_clusters=64, random_state=42, batch_size=1024).fit(all_mat)
        d_km = float(km64.inertia_ / (len(all_mat) * 128))
        d_g = float(np.mean(eigvals) * (2.0 ** (-2.0 * 6.0 / eff_dim)))
        gamma = float(d_g / max(d_km, 1e-7))

        # Token-Class Decomposition: Keyword vs Identifier Redundancy
        _, kw_dists = pairwise_distances_argmin_min(kw_mat, km.cluster_centers_)
        kw_red = float(np.mean(kw_dists <= 0.25)) * 100.0

        _, id_dists = pairwise_distances_argmin_min(ident_mat, km.cluster_centers_)
        id_red = float(np.mean(id_dists <= 0.25)) * 100.0

        results_3b[lang] = {
            "red_codebert": red_codebert,
            "gamma": gamma,
            "eff_dim": eff_dim
        }
        results_3c[lang] = {
            "kw_red": kw_red,
            "id_red": id_red,
            "ratio": kw_red / max(id_red, 1e-4)
        }

    # Print 3b Encoder-Invariance Table
    colbert_reds = {"go": 49.1, "rust": 42.4, "javascript": 37.4, "python": 33.6, "java": 27.0}
    
    print("\n" + "="*115)
    print(" RUN 3b: ENCODER-INVARIANCE COMPARISON (ColBERTv2 vs CodeBERT)")
    print("="*115)
    print(f"{'Language':<15} | {'ColBERTv2 Redun':>18} | {'CodeBERT Redun':>18} | {'CodeBERT Gap Γ':>16} | {'CodeBERT Eff_Dim':>18}")
    print("-" * 115)

    colbert_list = []
    codebert_list = []
    for l in ["go", "rust", "javascript", "python", "java"]:
        c_red = colbert_reds[l]
        cb_red = results_3b[l]["red_codebert"]
        gamma = results_3b[l]["gamma"]
        ed = results_3b[l]["eff_dim"]
        colbert_list.append(c_red)
        codebert_list.append(cb_red)
        print(f"{l:<15} | {c_red:17.1f}% | {cb_red:17.1f}% | {gamma:15.2f}x | {ed:18.1f}")

    rho_cross, p_cross = spearmanr(colbert_list, codebert_list)
    print("-" * 115)
    print(f"Cross-Encoder Rigidity Rank Correlation (ColBERTv2 vs CodeBERT): ρ = {rho_cross:.4f} (p = {p_cross:.4e})")

    # Print 3c Token-Class Decomposition Table
    print("\n" + "="*115)
    print(" RUN 3c: TOKEN-CLASS DECOMPOSITION (Keywords vs User Identifiers under CodeBERT)")
    print("="*115)
    print(f"{'Language':<15} | {'Keyword Redundancy':>20} | {'Identifier Redundancy':>22} | {'Keyword/Ident Ratio':>22}")
    print("-" * 115)
    for l in ["go", "rust", "javascript", "python", "java"]:
        kw_r = results_3c[l]["kw_red"]
        id_r = results_3c[l]["id_red"]
        ratio = results_3c[l]["ratio"]
        print(f"{l:<15} | {kw_r:19.1f}% | {id_r:21.1f}% | {ratio:21.2f}x")
    print("=" * 115)

if __name__ == "__main__":
    main()

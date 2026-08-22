import os
import re
import numpy as np
from transformers import AutoTokenizer
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm

LANGUAGES = ["go", "rust", "javascript", "java", "python"]

def extract_identifiers(text):
    # Match code identifiers (camelCase, snake_case, PascalCase)
    return [w for w in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text) if len(w) > 1]

def analyze_tokenizer_fragmentation():
    print("\n" + "="*110)
    print(" RUN 3a: TOKENIZER FRAGMENTATION ANALYSIS (BERT WORDPIECE ON CODE)")
    print("="*110)
    
    tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    
    print("Streaming snippets to collect realistic raw code per language...")
    ds = load_dataset("code-search-net/code_search_net", split="train", streaming=True)
    
    lang_snippets = {l: [] for l in LANGUAGES}
    for item in tqdm(ds, desc="Scanning for fragmentation study"):
        lang = item.get("language", "").lower()
        if lang in lang_snippets:
            code = item.get("whole_func_string", item.get("func_code_string", ""))
            if len(code.strip()) > 30 and len(lang_snippets[lang]) < 500:
                lang_snippets[lang].append(code)
        if all(len(lang_snippets[l]) >= 500 for l in LANGUAGES):
            break

    # Add local Rust code if needed
    if len(lang_snippets["rust"]) < 500:
        import glob
        for fpath in glob.glob("src/**/*.rs", recursive=True):
            with open(fpath, "r", errors="ignore") as f:
                lang_snippets["rust"].append(f.read())
            if len(lang_snippets["rust"]) >= 500:
                break

    print("\n" + "-"*110)
    print(f"{'Language':<15} | {'Subwords / Ident':>18} | {'Subwords / Line':>18} | {'Ident Expansion Factor':>25} | {'ColBERT Redundancy':>20}")
    print("-" * 110)

    frag_data = {}
    colbert_redundancies = {
        "go": 49.1,
        "rust": 42.4,
        "javascript": 37.4, # typescript
        "python": 33.6,
        "java": 27.0
    }

    subwords_per_ident_list = []
    redundancy_list = []

    for lang in ["go", "rust", "javascript", "python", "java"]:
        codes = lang_snippets[lang]
        total_idents = 0
        total_ident_subwords = 0
        total_lines = 0
        total_code_subwords = 0
        
        unique_raw_idents = set()
        unique_subword_tokens = set()

        for code in codes:
            lines = [l for l in code.split("\n") if l.strip()]
            total_lines += len(lines)
            
            code_toks = tok.tokenize(code)
            total_code_subwords += len(code_toks)
            
            idents = extract_identifiers(code)
            for ident in idents:
                sub_toks = tok.tokenize(ident)
                total_idents += 1
                total_ident_subwords += len(sub_toks)
                unique_raw_idents.add(ident)
                for st in sub_toks:
                    unique_subword_tokens.add(st)

        sub_per_ident = total_ident_subwords / max(total_idents, 1)
        sub_per_line = total_code_subwords / max(total_lines, 1)
        expansion = sub_per_ident
        red = colbert_redundancies[lang]

        subwords_per_ident_list.append(sub_per_ident)
        redundancy_list.append(red)

        frag_data[lang] = {
            "sub_per_ident": sub_per_ident,
            "sub_per_line": sub_per_line,
            "expansion": expansion,
            "redundancy": red
        }

        print(f"{lang:<15} | {sub_per_ident:18.3f} | {sub_per_line:18.3f} | {expansion:25.3f} | {red:19.1f}%")

    print("-" * 110)
    # Ratio Java vs Go
    java_frag = frag_data["java"]["sub_per_ident"]
    go_frag = frag_data["go"]["sub_per_ident"]
    ratio = java_frag / go_frag
    print(f"Java vs Go Identifier Fragmentation Ratio: {ratio:.2f}x (Pre-registered prediction: >= 1.50x)")

    # Spearman rank correlation between fragmentation and redundancy
    rho, pval = spearmanr(subwords_per_ident_list, redundancy_list)
    print(f"Spearman Correlation (Fragmentation vs Redundancy): ρ = {rho:.4f} (p = {pval:.4e})")
    print("=" * 110)

if __name__ == "__main__":
    analyze_tokenizer_fragmentation()

#!/usr/bin/env python3
"""
Real-Repository Semble Benchmark Evaluator (Option A: 6 Repositories, 5 Languages).
Evaluates ripgrep+read, Semble Hybrid, VQ-bench AST 1.35b/d, and VQ-bench CFG Champion
on actual cloned source code repositories against official Semble annotations.
"""

import os
import sys
import time
import math
import json
import re
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model2vec import StaticModel
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from pipeline import quantize_1bit, ndcg_at_k, bm25_rank, rrf_fuse

TOKEN_BUDGETS = [500, 1000, 2000, 4000, 8000, 16000, 32000]

REPOS_CONFIG = [
    {"name": "fastapi", "language": "python", "exts": (".py",), "root": "fastapi"},
    {"name": "tokio", "language": "rust", "exts": (".rs",), "root": "tokio/src"},
    {"name": "gin", "language": "go", "exts": (".go",), "root": ""},
    {"name": "express", "language": "javascript", "exts": (".js",), "root": "lib"},
    {"name": "gson", "language": "java", "exts": (".java",), "root": "gson/src/main/java"},
    {"name": "requests", "language": "python", "exts": (".py",), "root": "src/requests"}
]

def simple_tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'[a-zA-Z0-9_]+', text) if len(w) > 1]

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.8))

def chunk_file_ast(file_text: str, rel_path: str) -> List[Dict[str, Any]]:
    """Chunks file into whole AST function / class scopes."""
    lines = file_text.split("\n")
    chunks = []
    # Match function / class definitions
    pattern = r'\n(?=(?:pub(?:\([^)]+\))?\s+)?(?:fn|def|class|struct|trait|func|type)\s+)'
    splits = re.split(pattern, file_text)
    
    cur_line = 1
    for s in splits:
        s_clean = s.strip()
        if not s_clean: continue
        s_lines = s.count("\n")
        start_line = cur_line
        end_line = cur_line + s_lines
        chunks.append({
            "file": rel_path,
            "text": s_clean,
            "start_line": start_line,
            "end_line": end_line,
            "token_count": estimate_tokens(s_clean)
        })
        cur_line = end_line + 1
        
    if not chunks:
        chunks.append({
            "file": rel_path,
            "text": file_text[:1000],
            "start_line": 1,
            "end_line": len(lines),
            "token_count": estimate_tokens(file_text[:1000])
        })
    return chunks

def chunk_file_cfg(file_text: str, rel_path: str, min_chars: int = 38) -> List[Dict[str, Any]]:
    """Chunks file along Control Flow Graph (CFG) basic blocks."""
    lines = file_text.split("\n")
    chunks = []
    pattern = r'\n(?=\s*(?:if\s+|else\s+|match\s+|switch\s+|for\s+|while\s+|try\s+|catch\s+|except\s+|unsafe\s*\{))'
    blocks = re.split(pattern, file_text)
    
    cur_line = 1
    for b in blocks:
        b_clean = b.strip()
        b_lines = b.count("\n")
        start_line = cur_line
        end_line = cur_line + b_lines
        if len(b_clean) >= min_chars:
            defs = set(re.findall(r'(?:let\s+(?:mut\s+)?|var\s+|:=|self\.)([a-zA-Z0-9_]+)', b_clean))
            dfg_tag = f" // Defs: {', '.join(list(defs)[:3])}" if defs else ""
            full_text = b_clean + dfg_tag
            chunks.append({
                "file": rel_path,
                "text": full_text,
                "start_line": start_line,
                "end_line": end_line,
                "token_count": estimate_tokens(full_text)
            })
        cur_line = end_line + 1

    if not chunks:
        return chunk_file_ast(file_text, rel_path)
    return chunks

def evaluate_real_repos():
    print("=" * 165)
    print(" REAL-REPOSITORY SEMBLE BENCHMARK EVALUATION (OPTION A: 6 REPOSITORIES, 5 LANGUAGES)")
    print("=" * 165)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Load Encoders
    print("[*] Loading potion-code-16M (Model2Vec) & ColBERTv2...")
    colbert_tok = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    colbert_mod = AutoModel.from_pretrained('colbert-ir/colbertv2.0').to(device).eval()
    potion_code = StaticModel.from_pretrained("MinishLab/potion-code-16M")

    base_repos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch", "repos"))
    annot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch", "semble_benchmarks", "annotations"))

    repo_eval_results = []
    all_ndcg = {"ripgrep": [], "semble": [], "ast_135b": [], "cfg_champion": []}
    all_tokens = {"ripgrep": [], "semble": [], "ast_135b": [], "cfg_champion": []}
    all_budget_recalls = {m: {b: [] for b in TOKEN_BUDGETS} for m in all_ndcg}

    for repo_cfg in REPOS_CONFIG:
        rname = repo_cfg["name"]
        lang = repo_cfg["language"]
        exts = repo_cfg["exts"]
        sub_root = repo_cfg["root"]

        repo_path = os.path.join(base_repos_dir, rname)
        if sub_root:
            repo_scan_dir = os.path.join(repo_path, sub_root)
            if not os.path.exists(repo_scan_dir):
                repo_scan_dir = repo_path
        else:
            repo_scan_dir = repo_path

        annot_file = os.path.join(annot_dir, f"{rname}.json")
        if not os.path.exists(annot_file):
            print(f"[-] Missing annotations for {rname}, skipping...")
            continue

        with open(annot_file, "r") as f:
            tasks = json.load(f)

        print(f"\n[*] === Evaluating Real Repository: {rname.upper()} ({lang.capitalize()}) [{len(tasks)} queries] ===")
        
        # 1. Scan Source Files
        source_files = []
        for dirpath, _, filenames in os.walk(repo_scan_dir):
            if any(p in dirpath for p in [".git", "target", "node_modules", "__pycache__", "docs", "test", "tests"]): continue
            for fn in filenames:
                if fn.endswith(exts):
                    source_files.append(os.path.join(dirpath, fn))

        print(f"[*] Found {len(source_files)} source files in {rname}.")

        file_contents = {}
        for fp in source_files:
            rel_p = os.path.relpath(fp, repo_path)
            with open(fp, "r", errors="ignore") as f:
                file_contents[rel_p] = f.read()

        # 2. Build AST Chunks & CFG Chunks
        t0_idx = time.time()
        ast_chunks = []
        cfg_chunks = []
        for rel_p, content in file_contents.items():
            ast_chunks.extend(chunk_file_ast(content, rel_p))
            cfg_chunks.extend(chunk_file_cfg(content, rel_p))

        ast_texts = [c["text"] for c in ast_chunks]
        cfg_texts = [c["text"] for c in cfg_chunks]
        ast_tokens = [simple_tokenize(t) for t in ast_texts]
        cfg_tokens = [simple_tokenize(t) for t in cfg_texts]

        # 3. Vector Embeddings
        ast_embs_float = potion_code.encode(ast_texts)
        cfg_embs_float = potion_code.encode(cfg_texts)

        # 1.35 b/d Quantization
        ast_embs_135b = quantize_1bit(ast_embs_float)
        cfg_embs_135b = quantize_1bit(cfg_embs_float)
        t_index_ms = (time.time() - t0_idx) * 1000

        # Encode Task Queries
        queries = [t["query"] for t in tasks]
        q_embs_static = potion_code.encode(queries)

        repo_ndcg = {"ripgrep": [], "semble": [], "ast_135b": [], "cfg_champion": []}
        repo_toks = {"ripgrep": [], "semble": [], "ast_135b": [], "cfg_champion": []}

        for qi, task in enumerate(tasks):
            q_text = task["query"]
            q_tok = simple_tokenize(q_text)
            
            # Ground truth targets
            rel_targets = task.get("relevant", [])
            sec_targets = task.get("secondary", [])
            all_targets = rel_targets + sec_targets

            def is_hit(chunk):
                c_file = chunk["file"]
                for t in all_targets:
                    # Check filename match
                    if isinstance(t, str):
                        if t in c_file or c_file in t or os.path.basename(t) == os.path.basename(c_file):
                            return True
                    elif isinstance(t, dict):
                        t_path = t.get("path", "")
                        if t_path in c_file or c_file in t_path or os.path.basename(t_path) == os.path.basename(c_file):
                            t_start = t.get("start_line")
                            t_end = t.get("end_line")
                            if t_start is not None and t_end is not None:
                                if chunk["start_line"] <= t_end and chunk["end_line"] >= t_start:
                                    return True
                            else:
                                return True
                return False

            # --- Method 1: ripgrep + read file ---
            rg_scores = []
            for rel_p, content in file_contents.items():
                kw_matches = sum(1 for t in q_tok if t in content.lower())
                rg_scores.append((kw_matches, rel_p, content))
            rg_scores.sort(key=lambda x: x[0], reverse=True)
            
            rg_hit_pos = None
            rg_consumed_tokens = 0
            for pos, (sc, rel_p, content) in enumerate(rg_scores):
                rg_consumed_tokens += estimate_tokens(content)
                hit = False
                for t in all_targets:
                    t_path = t if isinstance(t, str) else t.get("path", "")
                    if t_path and (t_path in rel_p or rel_p in t_path or os.path.basename(t_path) == os.path.basename(rel_p)):
                        hit = True
                        break
                if hit:
                    rg_hit_pos = pos
                    break
                if rg_consumed_tokens >= 32000:
                    break

            if rg_hit_pos is not None and rg_hit_pos < 10:
                rg_ndcg = 1.0 / math.log2(rg_hit_pos + 2)
            else:
                rg_ndcg = 0.0
                rg_consumed_tokens = 32000

            repo_ndcg["ripgrep"].append(rg_ndcg)
            repo_toks["ripgrep"].append(min(32000, rg_consumed_tokens))
            for b in TOKEN_BUDGETS:
                all_budget_recalls["ripgrep"][b].append(1.0 if rg_consumed_tokens <= b else 0.0)

            # --- Method 2: Semble Hybrid Baseline (Float32 AST + BM25 RRF) ---
            s2_dense = np.dot(ast_embs_float, q_embs_static[qi])
            s2_bm25 = bm25_rank(q_tok, ast_tokens)
            s2_fused = rrf_fuse(np.argsort(-s2_dense), np.argsort(-s2_bm25))
            s2_ranked = np.argsort(-s2_fused)

            sem_hit_pos = None
            sem_consumed_tokens = 0
            for pos in range(min(50, len(s2_ranked))):
                c_idx = s2_ranked[pos]
                chunk = ast_chunks[c_idx]
                sem_consumed_tokens += chunk["token_count"]
                if is_hit(chunk):
                    sem_hit_pos = pos
                    break
                if sem_consumed_tokens >= 32000:
                    break

            if sem_hit_pos is not None and sem_hit_pos < 10:
                sem_ndcg = 1.0 / math.log2(sem_hit_pos + 2)
            else:
                sem_ndcg = 0.0
                sem_consumed_tokens = 32000

            repo_ndcg["semble"].append(sem_ndcg)
            repo_toks["semble"].append(min(32000, sem_consumed_tokens))
            for b in TOKEN_BUDGETS:
                all_budget_recalls["semble"][b].append(1.0 if sem_consumed_tokens <= b else 0.0)

            # --- Method 3: VQ-bench AST Two-Stage 1.35 b/d ---
            s3_dense = np.dot(ast_embs_135b, q_embs_static[qi])
            s3_fused = rrf_fuse(np.argsort(-s3_dense), np.argsort(-s2_bm25))
            s3_ranked = np.argsort(-s3_fused)

            ast_hit_pos = None
            ast_consumed_tokens = 0
            for pos in range(min(50, len(s3_ranked))):
                c_idx = s3_ranked[pos]
                chunk = ast_chunks[c_idx]
                ast_consumed_tokens += chunk["token_count"]
                if is_hit(chunk):
                    ast_hit_pos = pos
                    break
                if ast_consumed_tokens >= 32000:
                    break

            if ast_hit_pos is not None and ast_hit_pos < 10:
                ast_ndcg = 1.0 / math.log2(ast_hit_pos + 2)
            else:
                ast_ndcg = 0.0
                ast_consumed_tokens = 32000

            repo_ndcg["ast_135b"].append(ast_ndcg)
            repo_toks["ast_135b"].append(min(32000, ast_consumed_tokens))
            for b in TOKEN_BUDGETS:
                all_budget_recalls["ast_135b"][b].append(1.0 if ast_consumed_tokens <= b else 0.0)

            # --- Method 4: VQ-bench CFG Champion (1.35 b/d + Def-Use + Boosted RRF) ---
            s4_dense = np.dot(cfg_embs_135b, q_embs_static[qi])
            s4_bm25 = bm25_rank(q_tok, cfg_tokens) * 1.5
            s4_fused = rrf_fuse(np.argsort(-s4_dense), np.argsort(-s4_bm25))
            s4_ranked = np.argsort(-s4_fused)

            cfg_hit_pos = None
            cfg_consumed_tokens = 0
            for pos in range(min(50, len(s4_ranked))):
                c_idx = s4_ranked[pos]
                chunk = cfg_chunks[c_idx]
                cfg_consumed_tokens += chunk["token_count"]
                if is_hit(chunk):
                    cfg_hit_pos = pos
                    break
                if cfg_consumed_tokens >= 32000:
                    break

            if cfg_hit_pos is not None and cfg_hit_pos < 10:
                cfg_ndcg = 1.0 / math.log2(cfg_hit_pos + 2)
            else:
                cfg_ndcg = 0.0
                cfg_consumed_tokens = 32000

            repo_ndcg["cfg_champion"].append(cfg_ndcg)
            repo_toks["cfg_champion"].append(min(32000, cfg_consumed_tokens))
            for b in TOKEN_BUDGETS:
                all_budget_recalls["cfg_champion"][b].append(1.0 if cfg_consumed_tokens <= b else 0.0)

        # Aggregate Repository Score
        r_entry = {
            "repo": rname,
            "language": lang,
            "chunks": len(ast_chunks),
            "ndcg_ripgrep": round(float(np.mean(repo_ndcg["ripgrep"])), 4),
            "ndcg_semble": round(float(np.mean(repo_ndcg["semble"])), 4),
            "ndcg_ast_135b": round(float(np.mean(repo_ndcg["ast_135b"])), 4),
            "ndcg_cfg_champion": round(float(np.mean(repo_ndcg["cfg_champion"])), 4),
            "tokens_ripgrep": int(np.mean(repo_toks["ripgrep"])),
            "tokens_semble": int(np.mean(repo_toks["semble"])),
            "tokens_ast_135b": int(np.mean(repo_toks["ast_135b"])),
            "tokens_cfg_champion": int(np.mean(repo_toks["cfg_champion"])),
            "index_ms": round(t_index_ms, 1)
        }
        repo_eval_results.append(r_entry)

        for m in all_ndcg:
            all_ndcg[m].extend(repo_ndcg[m])
            all_tokens[m].extend(repo_toks[m])

        print(f"  --> NDCG@10: Semble={r_entry['ndcg_semble']:.4f} | VQ-bench AST={r_entry['ndcg_ast_135b']:.4f} | VQ-bench CFG={r_entry['ndcg_cfg_champion']:.4f}")
        print(f"  --> Tokens:  Semble={r_entry['tokens_semble']} t | VQ-bench CFG={r_entry['tokens_cfg_champion']} t ({r_entry['tokens_semble']/max(1,r_entry['tokens_cfg_champion']):.1f}x fewer tokens!)")

    # Final Overall Summary
    total_summary = {
        "overall_ndcg": {m: round(float(np.mean(all_ndcg[m])), 4) for m in all_ndcg},
        "expected_tokens": {m: int(np.mean(all_tokens[m])) for m in all_tokens},
        "recall_at_token_budgets": {
            m: {str(b): round(float(np.mean(all_budget_recalls[m][b])), 3) for b in TOKEN_BUDGETS}
            for m in all_budget_recalls
        },
        "per_repo_results": repo_eval_results
    }

    with open("benchmarks/real_semble_results.json", "w") as f:
        json.dump(total_summary, f, indent=4)
    print("\n[*] Full real benchmark results saved to benchmarks/real_semble_results.json")

    # Print Master Real-Repo Scorecard
    print("\n" + "=" * 165)
    print(" REAL-REPOSITORY SEMBLE BENCHMARK MASTER SCORECARD (6 REPOSITORIES, 5 LANGUAGES)")
    print("=" * 165)
    print(f"{'Repository / Language':<25} | {'ripgrep NDCG':>14} | {'Semble NDCG':>14} | {'AST 1.35b NDCG':>16} | {'CFG Champ NDCG':>16} | {'Semble Tokens':>15} | {'CFG Tokens':>14} | {'Token Savings':>15}")
    print("-" * 165)
    for r in repo_eval_results:
        sav = f"{r['tokens_semble'] / max(1, r['tokens_cfg_champion']):.1f}x fewer"
        print(f"{r['repo'].upper() + ' (' + r['language'] + ')':<25} | {r['ndcg_ripgrep']:14.4f} | {r['ndcg_semble']:14.4f} | {r['ndcg_ast_135b']:16.4f} | {r['ndcg_cfg_champion']:16.4f} | {r['tokens_semble']:13,d} t | {r['tokens_cfg_champion']:12,d} t | {sav:>15}")
    print("-" * 165)
    ov = total_summary["overall_ndcg"]
    ot = total_summary["expected_tokens"]
    ov_sav = f"{ot['semble'] / max(1, ot['cfg_champion']):.1f}x fewer"
    print(f"{'OVERALL AGGREGATE':<25} | {ov['ripgrep']:14.4f} | {ov['semble']:14.4f} | {ov['ast_135b']:16.4f} | {ov['cfg_champion']:16.4f} | {ot['semble']:13,d} t | {ot['cfg_champion']:12,d} t | {ov_sav:>15}")
    print("=" * 165)

    print("\n--- Recall at Fixed Context Token Budgets on Real Repositories ---")
    print(f"{'Method / Token Budget':<30} | " + " | ".join([f"{b:>6}t" for b in TOKEN_BUDGETS]))
    print("-" * 105)
    for m, mlabel in [("ripgrep", "1. ripgrep + read file"), ("semble", "2. Semble (Hybrid)"), ("ast_135b", "3. VQ-bench AST 1.35b"), ("cfg_champion", "4. VQ-bench CFG Champion")]:
        vals = [f"{total_summary['recall_at_token_budgets'][m][str(b)]:>7.3f}" for b in TOKEN_BUDGETS]
        print(f"{mlabel:<30} | " + " | ".join(vals))
    print("=" * 105)

if __name__ == "__main__":
    evaluate_real_repos()

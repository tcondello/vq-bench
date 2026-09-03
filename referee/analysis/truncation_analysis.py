"""Analyze chunk token lengths and truncation rates per language, per arm across all 63 repos."""

import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
from transformers import AutoTokenizer
from model2vec import StaticModel
from pylate.models import ColBERT

from referee.engines.semble_reference import (
    FILE_TYPES,
    FileCategory,
    language_for_path,
    walk_source_files,
    chunk_source,
)

def main():
    print("[*] Loading tokenizers for Arms 1, 2, and 3...")
    # Arm 1
    m1 = StaticModel.from_pretrained("MinishLab/potion-code-16M")
    tok1 = m1.tokenizer
    max_len1 = getattr(tok1, "model_max_length", 512)
    if max_len1 > 100000:
        max_len1 = 512 # Standard BERT/MiniLM limit if unbounded

    # Arm 2
    tok2 = AutoTokenizer.from_pretrained("nomic-ai/CodeRankEmbed")
    max_len2 = 8192 # NomicBert context limit

    # Arm 3
    colbert = ColBERT("lightonai/LateOn-Code", device="cpu")
    tok3 = colbert.tokenizer
    max_len3 = colbert.doc_maxlen if hasattr(colbert, "doc_maxlen") else 300

    print(f"Max sequence lengths: Arm 1={max_len1}, Arm 2={max_len2}, Arm 3={max_len3}")

    repos_base = Path("scratch/semble_repos").resolve()
    with open("scratch/semble_benchmarks/repos.json", "r", encoding="utf-8") as f:
        repo_specs = {r["name"]: r for r in json.load(f)}

    code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)

    # Accumulators per language and per repo
    lang_stats = defaultdict(lambda: {"total_chunks": 0, "arm1_trunc": 0, "arm2_trunc": 0, "arm3_trunc": 0, "arm3_tok_counts": []})
    repo_stats = {}

    for repo_name in sorted(repo_specs.keys()):
        spec = repo_specs[repo_name]
        lang = spec.get("language", "unknown")
        sub_root = spec.get("benchmark_root")
        repo_dir = repos_base / repo_name
        bench_dir = repo_dir / sub_root if sub_root else repo_dir

        chunks = []
        for fp in walk_source_files(bench_dir, code_exts):
            flang = language_for_path(fp) or lang
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
                rel_p = str(fp.relative_to(repo_dir))
                file_chunks = chunk_source(src, rel_p, flang)
                chunks.extend(file_chunks)
            except OSError:
                continue

        if not chunks:
            continue

        chunk_texts = [c.content for c in chunks]
        n_chunks = len(chunks)

        # Tokenize with each tokenizer
        # Arm 1 tokens
        encs1 = tok1.encode_batch(chunk_texts)
        t1_lens = [len(e.ids) for e in encs1]
        trunc1 = sum(1 for l in t1_lens if l > max_len1)

        # Arm 2 tokens
        t2_lens = [len(toks) for toks in tok2(chunk_texts, add_special_tokens=False)["input_ids"]]
        trunc2 = sum(1 for l in t2_lens if l > max_len2)

        # Arm 3 tokens
        t3_lens = [len(toks) for toks in tok3(chunk_texts, add_special_tokens=False)["input_ids"]]
        trunc3 = sum(1 for l in t3_lens if l > max_len3)

        lang_stats[lang]["total_chunks"] += n_chunks
        lang_stats[lang]["arm1_trunc"] += trunc1
        lang_stats[lang]["arm2_trunc"] += trunc2
        lang_stats[lang]["arm3_trunc"] += trunc3
        lang_stats[lang]["arm3_tok_counts"].extend(t3_lens)

        repo_stats[repo_name] = {
            "language": lang,
            "chunks": n_chunks,
            "arm1_trunc_pct": trunc1 / n_chunks * 100,
            "arm2_trunc_pct": trunc2 / n_chunks * 100,
            "arm3_trunc_pct": trunc3 / n_chunks * 100,
            "arm3_p50_tokens": float(np.median(t3_lens)),
            "arm3_p95_tokens": float(np.percentile(t3_lens, 95)),
            "arm3_max_tokens": int(max(t3_lens)),
        }

    # Summary table
    print("\n" + "=" * 110)
    print(" TRUNCATION RATE PER LANGUAGE PER ARM")
    print("=" * 110)
    print(f"{'Language':<14} | {'Chunks':>8} | {'Arm 1 (>512)':>14} | {'Arm 2 (>8192)':>14} | {'Arm 3 (>300)':>14} | {'Arm 3 p50':>10} | {'Arm 3 p95':>10}")
    print("-" * 110)

    total_chunks = 0
    total_t1 = 0
    total_t2 = 0
    total_t3 = 0
    all_t3_lens = []

    for lang in sorted(lang_stats.keys()):
        data = lang_stats[lang]
        tc = data["total_chunks"]
        total_chunks += tc
        total_t1 += data["arm1_trunc"]
        total_t2 += data["arm2_trunc"]
        total_t3 += data["arm3_trunc"]
        all_t3_lens.extend(data["arm3_tok_counts"])

        r1 = data["arm1_trunc"] / tc * 100
        r2 = data["arm2_trunc"] / tc * 100
        r3 = data["arm3_trunc"] / tc * 100
        p50 = np.median(data["arm3_tok_counts"])
        p95 = np.percentile(data["arm3_tok_counts"], 95)

        print(f"{lang:<14} | {tc:>8} | {r1:>13.2f}% | {r2:>13.2f}% | {r3:>13.2f}% | {p50:>10.1f} | {p95:>10.1f}")

    print("-" * 110)
    print(f"{'OVERALL':<14} | {total_chunks:>8} | {total_t1/total_chunks*100:>13.2f}% | {total_t2/total_chunks*100:>13.2f}% | {total_t3/total_chunks*100:>13.2f}% | {np.median(all_t3_lens):>10.1f} | {np.percentile(all_t3_lens, 95):>10.1f}")
    print("=" * 110)

    out_file = Path("results/truncation_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "languages": {
            lang: {
                "total_chunks": lang_stats[lang]["total_chunks"],
                "arm1_trunc_pct": lang_stats[lang]["arm1_trunc"] / lang_stats[lang]["total_chunks"] * 100,
                "arm2_trunc_pct": lang_stats[lang]["arm2_trunc"] / lang_stats[lang]["total_chunks"] * 100,
                "arm3_trunc_pct": lang_stats[lang]["arm3_trunc"] / lang_stats[lang]["total_chunks"] * 100,
                "arm3_p50_tokens": float(np.median(lang_stats[lang]["arm3_tok_counts"])),
                "arm3_p95_tokens": float(np.percentile(lang_stats[lang]["arm3_tok_counts"], 95)),
            } for lang in lang_stats
        },
        "repos": repo_stats,
    }, indent=2), encoding="utf-8")
    print(f"[*] Saved truncation analysis to {out_file}")

if __name__ == "__main__":
    main()

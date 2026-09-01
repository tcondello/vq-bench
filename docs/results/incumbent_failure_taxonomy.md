# Incumbent Failure Taxonomy & Headroom Map

## Executive Summary
- **Total Evaluated Queries**: 1251
- **Overall NDCG@10**: 0.8349 (Target: 0.854)
- **Overall Recall@2k**: 0.9422 (Target: 0.938)
- **Queries with Imperfect NDCG@10 (< 1.0)**: 460 / 1251 (36.8%)
- **Queries with Zero NDCG@10 (Rank > 10 / Miss)**: 25 / 1251 (2.0%)
- **`queries_with_incomplete_coverage_at_2k` (Recall < 1.0 @ 2k)**: 100 / 1251 (8.0%)
- **`queries_with_zero_coverage_at_2k` (Recall == 0.0 @ 2k)**: 45 / 1251 (3.6%)

## Ground Truth Reconciliations
1. **7-Query Category Sum Inconsistency in Published Semble Paper**: Semble's paper metadata publishes 711 semantic / 343 architecture / 204 symbol (sum = 1,258). Auditing the actual 63 annotation files yields exactly **703 semantic, 344 architecture, 204 symbol (sum = 1,251)**.
2. **19 Live Suite Languages**: Exactly 19 programming languages are present across the 63 repositories.
3. **Chonkie Language Gap for WP2**: 9 languages are natively parsed by `CodeChunker` (tree-sitter: C, C++, C#, Java, Rust, Go, TypeScript, JavaScript, Python); 10 languages use `TokenChunker(tokenizer='tiktoken:cl100k_base')` fallback (Kotlin, Swift, PHP, Zig, Haskell, Scala, Elixir, Bash, Lua, Ruby).

## Failure Breakdown by Query Category
| Category | Total Queries | Imperfect NDCG (<1.0) | Zero NDCG (=0) | Incomplete Coverage @ 2k | Zero Coverage @ 2k | Mean NDCG@10 | Mean Recall@2k |
|---|---|---|---|---|---|---|---|
| **Architecture** | 344 | 177 (51.5%) | 5 (1.5%) | 47 (13.7%) | 13 (3.8%) | 0.7887 | 0.9133 |
| **Semantic** | 703 | 259 (36.8%) | 16 (2.3%) | 48 (6.8%) | 28 (4.0%) | 0.8286 | 0.9459 |
| **Symbol** | 204 | 24 (11.8%) | 4 (2.0%) | 5 (2.5%) | 4 (2.0%) | 0.9343 | 0.9779 |

## Key Diagnostic Insights & WP1 Pre-Registered Predictions
1. **Semantic Queries (703 queries - 56.2% of suite)**: Primary locus of gain for contextualization / late-interaction. Incumbent fails due to static vocabulary mismatch on paraphrased intents lacking exact identifier stems.
2. **Symbol Queries (204 queries - 16.3% of suite)**: Near-saturated ($0.9343$ NDCG@10). Encoder-insensitive; misses are driven by parser/regex boundary and syntax edge cases.
3. **Architecture Queries (344 queries - 27.5% of suite)**: Lowest recall at small budgets ($0.9133$ @ 2k) due to cross-file dispersion across multiple modules.

## Language Family Performance
| Language Family | Mean NDCG@10 | Active Languages |
|---|---|---|
| **c_braced** | 0.8333 | C, C++, C#, Java, Rust, Go, TypeScript, JavaScript, Kotlin, Swift, PHP, Zig (12) |
| **functional_ml** | 0.8500 | Haskell, Scala, Elixir (3) |
| **indentation** | 0.8138 | Python (1) |
| **script_markup** | 0.8481 | Bash, Lua, Ruby (3) |

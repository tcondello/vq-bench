"""Frozen referee metrics module.

Immutable definitions for ranking and token-budget context efficiency metrics:
- NDCG@k (Normalized Discounted Cumulative Gain at rank k)
- Recall at token budgets (500, 1k, 2k, 4k, 8k, 16k, 32k tokens)
- Budget-Constrained Yield (BCY@k)
- Mean Reciprocal Rank (MRR)
- Paired bootstrap confidence intervals
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tiktoken

_TOKENIZERS: dict[str, Any] = {}


def get_tokenizer(name: str = "cl100k_base") -> Any:
    """Cached tiktoken tokenizer getter."""
    if name not in _TOKENIZERS:
        _TOKENIZERS[name] = tiktoken.get_encoding(name)
    return _TOKENIZERS[name]


def count_tokens(text: str, tokenizer_name: str = "cl100k_base") -> int:
    """Return exact token count for text using the specified tiktoken encoding."""
    enc = get_tokenizer(tokenizer_name)
    return len(enc.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class TargetLocation:
    """Ground truth target location."""

    path: str
    start_line: int | None = None
    end_line: int | None = None

    @property
    def has_span(self) -> bool:
        return self.start_line is not None and self.end_line is not None


@dataclass(frozen=True)
class RetrievedUnit:
    """Retrieved candidate unit (chunk or file)."""

    file_path: str
    content: str
    start_line: int = 1
    end_line: int = 1
    score: float = 0.0
    token_count: int | None = None

    def get_token_count(self, tokenizer_name: str = "cl100k_base") -> int:
        if self.token_count is not None:
            return self.token_count
        return count_tokens(self.content, tokenizer_name)


def path_matches(file_path: str, target_path: str) -> bool:
    """Return True if either path is a suffix of the other (handles relative vs repo-rooted paths)."""
    norm_file = file_path.replace("\\", "/").strip("/")
    norm_target = target_path.replace("\\", "/").strip("/")
    return (
        norm_file == norm_target
        or norm_file.endswith(f"/{norm_target}")
        or norm_target.endswith(f"/{norm_file}")
        or Path(norm_file).name == Path(norm_target).name
    )


def target_matches_location(
    file_path: str,
    start_line: int,
    end_line: int,
    target: TargetLocation,
) -> bool:
    """Return True if the chunk at file_path:start_line-end_line covers the target."""
    if not path_matches(file_path, target.path):
        return False
    if not target.has_span:
        return True
    assert target.start_line is not None and target.end_line is not None
    # Overlap condition: chunk and target line ranges intersect
    return not (end_line < target.start_line or start_line > target.end_line)


def dcg(relevances: Sequence[int]) -> float:
    """Compute Discounted Cumulative Gain for binary relevance values at positions 1..len."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevant_ranks: Sequence[int], n_relevant: int, k: int = 10) -> float:
    """Compute NDCG@k given 1-based ranks of relevant results and total relevant count."""
    if n_relevant <= 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = dcg([1] * min(k, n_relevant))
    return dcg(relevances) / ideal if ideal > 0 else 0.0


def mrr(relevant_ranks: Sequence[int]) -> float:
    """Compute Reciprocal Rank of the first relevant result (1-based rank)."""
    valid = [r for r in relevant_ranks if r >= 1]
    if not valid:
        return 0.0
    return 1.0 / min(valid)


def compute_retrieval_curve(
    units: Sequence[RetrievedUnit],
    targets: Sequence[TargetLocation],
    tokenizer_name: str = "cl100k_base",
) -> list[tuple[int, int]]:
    """Compute cumulative (tokens_consumed, n_targets_covered) points starting at (0, 0)."""
    covered = [False] * len(targets)
    cumulative_tokens = 0
    points: list[tuple[int, int]] = [(0, 0)]

    for unit in units:
        tok_cnt = unit.get_token_count(tokenizer_name)
        cumulative_tokens += tok_cnt
        for i, tgt in enumerate(targets):
            if not covered[i] and target_matches_location(
                unit.file_path, unit.start_line, unit.end_line, tgt
            ):
                covered[i] = True
        points.append((cumulative_tokens, sum(covered)))

    return points


def recall_at_budget(
    curve: list[tuple[int, int]],
    budget: int,
    n_total_targets: int,
) -> float:
    """Return recall (covered_targets / n_total_targets) at the largest cumulative tokens <= budget."""
    if n_total_targets <= 0 or not curve:
        return 0.0
    covered_at = 0
    for tokens, covered in curve:
        if tokens > budget:
            break
        covered_at = covered
    return covered_at / n_total_targets


def paired_bootstrap_ci(
    a_or_deltas: Sequence[float],
    b: Sequence[float] | int | None = None,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired mean delta and 100*(1-alpha)% bootstrap confidence interval.

    If two sequences (a, b) are provided, deltas = a - b is computed.
    Returns: (mean_delta, ci_lower, ci_upper)
    """
    if isinstance(b, (list, tuple, np.ndarray)):
        deltas = [x - y for x, y in zip(a_or_deltas, b)]
    elif isinstance(b, int):
        n_boot = b
        deltas = a_or_deltas
    else:
        deltas = a_or_deltas

    arr = np.asarray(deltas, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean, ci_lower, ci_upper


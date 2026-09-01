"""Unit tests for referee metrics module."""

import math
from referee.metrics import (
    RetrievedUnit,
    TargetLocation,
    compute_retrieval_curve,
    count_tokens,
    dcg,
    mrr,
    ndcg_at_k,
    paired_bootstrap_ci,
    path_matches,
    recall_at_budget,
    target_matches_location,
)


def test_path_matches():
    assert path_matches("src/foo.py", "src/foo.py")
    assert path_matches("fastapi/routing.py", "routing.py")
    assert path_matches("routing.py", "fastapi/routing.py")
    assert path_matches("a/b/c.rs", "c.rs")
    assert not path_matches("a/b/c.rs", "d.rs")


def test_target_matches_location():
    tgt_span = TargetLocation(path="src/router.py", start_line=10, end_line=30)
    tgt_no_span = TargetLocation(path="src/router.py")

    # Path matches, no span -> always matches
    assert target_matches_location("src/router.py", 1, 100, tgt_no_span)

    # Path matches, span overlaps
    assert target_matches_location("src/router.py", 1, 15, tgt_span)
    assert target_matches_location("src/router.py", 20, 25, tgt_span)
    assert target_matches_location("src/router.py", 25, 50, tgt_span)
    assert target_matches_location("src/router.py", 1, 100, tgt_span)

    # Span does NOT overlap
    assert not target_matches_location("src/router.py", 1, 9, tgt_span)
    assert not target_matches_location("src/router.py", 31, 50, tgt_span)

    # Path does not match
    assert not target_matches_location("src/other.py", 10, 30, tgt_span)


def test_ndcg_at_k():
    # Perfect rank 1
    assert math.isclose(ndcg_at_k([1], 1, k=10), 1.0)

    # Miss / no ranks
    assert math.isclose(ndcg_at_k([], 1, k=10), 0.0)

    # Rank 2 with 1 relevant item: (1 / log2(3)) / (1 / log2(2)) = 1 / log2(3) = 0.630929
    assert math.isclose(ndcg_at_k([2], 1, k=10), 1.0 / math.log2(3))

    # Multiple relevant items
    # Target 2 items, ranked at 1 and 2 -> perfect NDCG = 1.0
    assert math.isclose(ndcg_at_k([1, 2], 2, k=10), 1.0)


def test_mrr():
    assert math.isclose(mrr([1, 5]), 1.0)
    assert math.isclose(mrr([2, 5]), 0.5)
    assert math.isclose(mrr([4]), 0.25)
    assert math.isclose(mrr([]), 0.0)


def test_token_counting_and_retrieval_curve():
    text1 = "def hello_world():\n    return 'hello'"
    text2 = "class FooBar:\n    def __init__(self):\n        pass"
    tokens1 = count_tokens(text1, "cl100k_base")
    tokens2 = count_tokens(text2, "cl100k_base")
    assert tokens1 > 0
    assert tokens2 > 0

    units = [
        RetrievedUnit(file_path="foo.py", content=text1, start_line=1, end_line=2),
        RetrievedUnit(file_path="bar.py", content=text2, start_line=1, end_line=3),
    ]
    targets = [TargetLocation(path="bar.py")]

    curve = compute_retrieval_curve(units, targets, "cl100k_base")
    # At 0 tokens -> 0 hits
    # At tokens1 -> 0 hits (target is bar.py)
    # At tokens1 + tokens2 -> 1 hit
    assert curve[0] == (0, 0)
    assert curve[1] == (tokens1, 0)
    assert curve[2] == (tokens1 + tokens2, 1)

    assert recall_at_budget(curve, tokens1 - 1, 1) == 0.0
    assert recall_at_budget(curve, tokens1 + tokens2, 1) == 1.0


def test_paired_bootstrap_ci():
    deltas = [0.05] * 100
    mean, lower, upper = paired_bootstrap_ci(deltas, n_boot=1000)
    assert math.isclose(mean, 0.05)
    assert math.isclose(lower, 0.05)
    assert math.isclose(upper, 0.05)

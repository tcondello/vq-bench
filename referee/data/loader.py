"""Unified dataset loader for frozen benchmarks.

Supports:
- Semble 63-repo Benchmark (63 repos, 19 languages, 1,251 queries)
- BRIGHT Code Subsets (Pony, LeetCode, StackOverflow -> 371 queries)
- ContextBench (Lite-500 & Full-1136 instances)
- Agent Retrieval Bench (427 samples, 82 no-gold controls)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from referee.metrics import TargetLocation


@dataclass(frozen=True)
class BenchmarkQuery:
    """Standardized query instance across all benchmark suites."""

    benchmark: str
    query_id: str
    repo: str
    language: str
    query: str
    targets: tuple[TargetLocation, ...]
    category: str = "general"
    metadata: dict[str, Any] | None = None


def load_semble_anchor_queries(
    annotations_dir: Path | str = "scratch/semble_benchmarks/annotations",
    repos_json: Path | str = "scratch/semble_benchmarks/repos.json",
) -> list[BenchmarkQuery]:
    """Load all 1,251 Semble benchmark queries across 63 repos without dropping any."""
    annot_path = Path(annotations_dir)
    repos_path = Path(repos_json)

    specs = {item["name"]: item for item in json.loads(repos_path.read_text(encoding="utf-8"))}
    queries: list[BenchmarkQuery] = []

    for af in sorted(annot_path.glob("*.json")):
        repo_name = af.stem
        spec = specs.get(repo_name, {})
        lang = spec.get("language", "unknown")
        items = json.loads(af.read_text(encoding="utf-8"))

        for q_idx, item in enumerate(items):
            query_text = item["query"]
            rel_raw = item.get("relevant", [])
            sec_raw = item.get("secondary", [])
            all_raw = rel_raw + sec_raw

            targets: list[TargetLocation] = []
            for t in all_raw:
                if isinstance(t, str):
                    targets.append(TargetLocation(path=t))
                elif isinstance(t, dict):
                    targets.append(
                        TargetLocation(
                            path=str(t.get("path", "")),
                            start_line=int(t["start_line"]) if t.get("start_line") is not None else None,
                            end_line=int(t["end_line"]) if t.get("end_line") is not None else None,
                        )
                    )

            # Infer category if not explicit
            cat = item.get("category")
            if not cat:
                if " " not in query_text.strip():
                    cat = "symbol"
                elif query_text.lower().startswith("how "):
                    cat = "architecture"
                else:
                    cat = "semantic"

            queries.append(
                BenchmarkQuery(
                    benchmark="semble_anchor",
                    query_id=f"{repo_name}_{q_idx:04d}",
                    repo=repo_name,
                    language=lang,
                    query=query_text,
                    targets=tuple(targets),
                    category=cat,
                    metadata={"relevant_count": len(rel_raw), "secondary_count": len(sec_raw)},
                )
            )

    return queries


def load_bright_queries() -> list[BenchmarkQuery]:
    """Load BRIGHT code subsets: pony (112), leetcode (142), stackoverflow (117) -> 371 queries."""
    splits = ["pony", "leetcode", "stackoverflow"]
    queries: list[BenchmarkQuery] = []

    for s in splits:
        ds = load_dataset("xlangai/BRIGHT", "examples", split=s)
        for idx, row in enumerate(ds):
            gold_ids = row.get("gold_ids", [])
            targets = tuple(TargetLocation(path=gid) for gid in gold_ids)
            queries.append(
                BenchmarkQuery(
                    benchmark="bright",
                    query_id=f"bright_{s}_{idx:04d}",
                    repo=s,
                    language="code",
                    query=row["query"],
                    targets=targets,
                    category=s,
                    metadata={"gold_ids": gold_ids},
                )
            )
    return queries


def load_contextbench_queries(split: str = "train") -> list[BenchmarkQuery]:
    """Load ContextBench (1,136 instances)."""
    ds = load_dataset("ContextBench/ContextBench", split=split)
    queries: list[BenchmarkQuery] = []
    for idx, row in enumerate(ds):
        gold_ctx = row.get("gold_context", [])
        targets = []
        for g in gold_ctx:
            if isinstance(g, str):
                targets.append(TargetLocation(path=g))
            elif isinstance(g, dict):
                targets.append(
                    TargetLocation(
                        path=g.get("file_path", g.get("path", "")),
                        start_line=g.get("start_line"),
                        end_line=g.get("end_line"),
                    )
                )
        queries.append(
            BenchmarkQuery(
                benchmark="contextbench",
                query_id=row["instance_id"],
                repo=row["repo"],
                language=row["language"],
                query=row["problem_statement"],
                targets=tuple(targets),
                category="bug_localization",
                metadata={"patch": row.get("patch"), "base_commit": row.get("base_commit")},
            )
        )
    return queries

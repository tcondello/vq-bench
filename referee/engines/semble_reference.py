"""Unmodified Reference Implementation of Upstream Semble In-Memory Search.

Faithfully implements the pure in-memory MinishLab/semble engine:
- Tree-sitter AST Chonkie chunking (chunk_size=1500 chars) with line fallback
- MinishLab/potion-code-16M StaticModel dense vector embedding
- bm25s sparse lexical indexing with path stem enrichment
- Reciprocal Rank Fusion (RRF k=60)
- Query-type adaptive alpha weighting (0.3 for symbol, 0.5 for architecture/semantic)
- Symbol definition boosting & multi-chunk file coherence boosting
- File path penalties (tests, docs, compat) & file saturation decay
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import bm25s
import numpy as np
import numpy.typing as npt
from chonkie.chunker import CodeChunker
from model2vec import StaticModel
from pathspec import GitIgnoreSpec

from referee.engines.base import BaseRetriever
from referee.metrics import RetrievedUnit

logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
_RRF_K = 60
_ALPHA_SYMBOL = 0.3
_ALPHA_NL = 0.5

_DEFAULT_MODEL_NAME = "MinishLab/potion-code-16M"

# Stopwords for stem matching
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when at from by on off for in out over
    to into with without of is are was were be been being have has had do does did
    can could will would shall should may might must that which who whom whose this these
    those it its what why how where
    """.split()
)

# Definition boosting regexes
_KEYWORD_PREFIX = r"(?:^|\s)(?:"
_DEFINITION_KEYWORD_BODY = (
    r"def|class|interface|type|struct|enum|fn|fun|func|function|val|var|let"
    r"|protocol|trait|impl|record|concept|mod|module|package|sub|macro"
    r"|public\s+class|private\s+class|protected\s+class"
    r"|public\s+interface|private\s+interface|protected\s+interface"
    r"|export\s+class|export\s+interface|export\s+type|export\s+function|export\s+const"
    r"|export\s+default\s+class|export\s+default\s+function"
    r"|data\s+class|sealed\s+class|enum\s+class|case\s+class|implicit\s+class|object"
    r"|defmodule|defp|defmacro|defmacrop|defprotocol|defimpl|defstruct"
    r"|typealias|extension|actor|newtype|async\s+def"
    r"|pub(?:\([^)]+\))?\s+(?:fn|struct|enum|trait|type|mod|const|static)"
)
_SQL_KEYWORD_BODY = r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER)"
_DEFINITION_BOOST_MULTIPLIER = 0.35
_EMBEDDED_SYMBOL_BOOST_SCALE = 0.5
_FILE_COHERENCE_BOOST_FRAC = 0.15
_STEM_BOOST_MULTIPLIER = 0.20
_EMBEDDED_STEM_MIN_LEN = 4

_SYMBOL_QUERY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$")
_EMBEDDED_SYMBOL_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[a-z]+[A-Z][A-Za-z0-9]*)\b")

# Path penalties regexes
_TEST_FILE_RE = re.compile(
    r"(?:^|/)"
    r"(?:"
    r"test_[^/]*\.py|[^/]*_test\.py|[^/]*_test\.go|[^/]*Tests?\.java|[^/]*Test\.php"
    r"|[^/]*_spec\.rb|[^/]*_test\.rb|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?"
    r"|[^/]*Tests?\.kt|[^/]*Spec\.kt|[^/]*Tests?\.swift|[^/]*Spec\.swift"
    r"|[^/]*Tests?\.cs|test_[^/]*\.cpp|[^/]*_test\.cpp|test_[^/]*\.c|[^/]*_test\.c"
    r"|[^/]*Spec\.scala|[^/]*Suite\.scala|[^/]*Test\.scala|[^/]*_test\.dart|test_[^/]*\.dart"
    r"|[^/]*_spec\.lua|[^/]*_test\.lua|test_[^/]*\.lua|test_helpers?[^/]*\.\w+"
    r")$"
)
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")
_COMPAT_DIR_RE = re.compile(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)")
_EXAMPLES_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)")
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")

_STRONG_PENALTY = 0.3
_MODERATE_PENALTY = 0.5
_MILD_PENALTY = 0.7
_REEXPORT_FILENAMES = frozenset({"__init__.py", "package-info.java"})
_FILE_SATURATION_THRESHOLD = 1
_FILE_SATURATION_DECAY = 0.5

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


class FileCategory(str, Enum):
    CODE = "CODE"
    DOCUMENT = "DOCUMENT"


@dataclass(frozen=True)
class FileType:
    language: str
    category: FileCategory


FILE_TYPES: dict[str, FileType] = {
    ".py": FileType("python", FileCategory.CODE),
    ".js": FileType("javascript", FileCategory.CODE),
    ".jsx": FileType("javascript", FileCategory.CODE),
    ".ts": FileType("typescript", FileCategory.CODE),
    ".tsx": FileType("typescript", FileCategory.CODE),
    ".go": FileType("go", FileCategory.CODE),
    ".rs": FileType("rust", FileCategory.CODE),
    ".java": FileType("java", FileCategory.CODE),
    ".kt": FileType("kotlin", FileCategory.CODE),
    ".kts": FileType("kotlin", FileCategory.CODE),
    ".rb": FileType("ruby", FileCategory.CODE),
    ".php": FileType("php", FileCategory.CODE),
    ".c": FileType("c", FileCategory.CODE),
    ".h": FileType("c", FileCategory.CODE),
    ".cpp": FileType("cpp", FileCategory.CODE),
    ".hpp": FileType("cpp", FileCategory.CODE),
    ".cs": FileType("csharp", FileCategory.CODE),
    ".swift": FileType("swift", FileCategory.CODE),
    ".scala": FileType("scala", FileCategory.CODE),
    ".sbt": FileType("scala", FileCategory.CODE),
    ".ex": FileType("elixir", FileCategory.CODE),
    ".exs": FileType("elixir", FileCategory.CODE),
    ".dart": FileType("dart", FileCategory.CODE),
    ".lua": FileType("lua", FileCategory.CODE),
    ".sql": FileType("sql", FileCategory.CODE),
    ".sh": FileType("bash", FileCategory.CODE),
    ".bash": FileType("bash", FileCategory.CODE),
    ".zig": FileType("zig", FileCategory.CODE),
    ".hs": FileType("haskell", FileCategory.CODE),
    ".md": FileType("markdown", FileCategory.DOCUMENT),
    ".yaml": FileType("yaml", FileCategory.DOCUMENT),
    ".yml": FileType("yaml", FileCategory.DOCUMENT),
    ".toml": FileType("toml", FileCategory.DOCUMENT),
    ".json": FileType("json", FileCategory.DOCUMENT),
}

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".semble",
        "dist",
        "build",
        ".eggs",
    }
)


@dataclass(frozen=True)
class SembleChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None


# --- Tokenization & Identifiers ---
def split_identifier(token: str) -> list[str]:
    lower = token.lower()
    if "_" in token:
        parts = [p for p in lower.split("_") if p]
    else:
        parts = [m.lower() for m in _CAMEL_RE.findall(token)]
    if len(parts) >= 2:
        return [lower, *parts]
    return [lower]


def tokenize(text: str) -> list[str]:
    raw_tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for tok in raw_tokens:
        result.extend(split_identifier(tok))
    return result


def enrich_for_bm25(chunk: SembleChunk) -> str:
    path = Path(chunk.file_path)
    stem = path.stem
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    dir_text = " ".join(dir_parts[-3:])
    return f"{chunk.content} {stem} {stem} {dir_text}"


# --- Chunking ---
def chunk_lines(
    source: str,
    file_path: str,
    language: str | None = None,
    max_lines: int = 50,
    overlap_lines: int = 5,
) -> list[SembleChunk]:
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[SembleChunk] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        content = "".join(lines[start:end])
        if content.strip():
            chunks.append(
                SembleChunk(
                    content=content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    language=language,
                )
            )
        start = end - overlap_lines if end < len(lines) else end
    return chunks


def chunk_with_chonkie(source: str, file_path: str, language: str) -> list[SembleChunk]:
    try:
        code_chunker = CodeChunker(language=language, chunk_size=1500)
        raw_chunks = code_chunker.chunk(source)
    except Exception:
        return chunk_lines(source, file_path, language)

    if not raw_chunks:
        return chunk_lines(source, file_path, language)

    chunks: list[SembleChunk] = []
    for raw_chunk in raw_chunks:
        text = raw_chunk.text
        if not text.strip():
            continue
        end_index = max(raw_chunk.end_index - 1, raw_chunk.start_index)
        chunks.append(
            SembleChunk(
                content=text,
                file_path=file_path,
                start_line=source[: raw_chunk.start_index].count("\n") + 1,
                end_line=source[:end_index].count("\n") + 1,
                language=language,
            )
        )
    return chunks if chunks else chunk_lines(source, file_path, language)


def chunk_source(source: str, file_path: str, language: str | None) -> list[SembleChunk]:
    if not source.strip():
        return []
    if language:
        return chunk_with_chonkie(source, file_path, language)
    return chunk_lines(source, file_path, language)


# --- File Walking ---
def language_for_path(path: Path) -> str | None:
    if spec := FILE_TYPES.get(path.suffix.lower()):
        return spec.language
    return None


def walk_source_files(root: Path, extensions: frozenset[str]) -> Iterator[Path]:
    gitignore = None
    gi_path = root / ".gitignore"
    if gi_path.is_file():
        with gi_path.open("r", encoding="utf-8", errors="ignore") as f:
            gitignore = GitIgnoreSpec.from_lines(f)

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        kept = []
        for dirname in dirnames:
            if dirname in DEFAULT_IGNORED_DIRS:
                continue
            rel = (rel_dir / dirname).as_posix() + "/"
            if gitignore is not None and gitignore.match_file(rel):
                continue
            kept.append(dirname)
        dirnames[:] = kept

        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() not in extensions:
                continue
            if gitignore is not None:
                rel_file = (rel_dir / filename).as_posix()
                if gitignore.match_file(rel_file):
                    continue
            yield file_path


# --- Boosting & Penalties ---
def is_symbol_query(query: str) -> bool:
    return _SYMBOL_QUERY_RE.match(query.strip()) is not None


def resolve_alpha(query: str, alpha: float | None) -> float:
    if alpha is not None:
        return alpha
    return _ALPHA_SYMBOL if is_symbol_query(query) else _ALPHA_NL


def _rrf_scores(scores: dict[SembleChunk, float]) -> dict[SembleChunk, float]:
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _extract_symbol_name(query: str) -> str:
    for separator in ("::", "\\", "->", "."):
        if separator in query:
            return query.rsplit(separator, 1)[-1]
    return query.strip()


@functools.lru_cache(maxsize=256)
def _definition_pattern(symbol_name: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    escaped = re.escape(symbol_name)
    ns_prefix = r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.|::))*"
    suffix = r")\s+" + ns_prefix + escaped + r"(?:\s|[<({:\[;]|$)"
    return (
        re.compile(_KEYWORD_PREFIX + _DEFINITION_KEYWORD_BODY + suffix, re.MULTILINE),
        re.compile(_KEYWORD_PREFIX + _SQL_KEYWORD_BODY + suffix, re.MULTILINE | re.IGNORECASE),
    )


def _stem_matches(stem: str, name: str) -> bool:
    stem_norm = stem.replace("_", "")
    return stem == name or stem_norm == name or stem.rstrip("s") == name or stem_norm.rstrip("s") == name


def _definition_tier(chunk: SembleChunk, names: set[str], boost_unit: float) -> float:
    general_patterns = [_definition_pattern(name) for name in names]
    found = False
    for gen, sql in general_patterns:
        if gen.search(chunk.content) is not None or sql.search(chunk.content) is not None:
            found = True
            break
    if not found:
        return 0.0
    stem = Path(chunk.file_path).stem.lower()
    return boost_unit * (1.5 if any(_stem_matches(stem, name.lower()) for name in names) else 1.0)


def _boost_symbol_definitions(
    boosted: dict[SembleChunk, float],
    query: str,
    max_score: float,
    all_chunks: list[SembleChunk],
) -> None:
    symbol_name = _extract_symbol_name(query)
    names = {symbol_name}
    if symbol_name != query.strip():
        names.add(query.strip())
    boost_unit = max_score * _DEFINITION_BOOST_MULTIPLIER

    for chunk in list(boosted):
        if tier := _definition_tier(chunk, names, boost_unit):
            boosted[chunk] += tier

    # Non-candidate scan
    for chunk in all_chunks:
        if chunk in boosted:
            continue
        stem = Path(chunk.file_path).stem.lower()
        if _stem_matches(stem, symbol_name.lower()):
            if tier := _definition_tier(chunk, names, boost_unit):
                boosted[chunk] = tier


def _boost_embedded_symbols(
    boosted: dict[SembleChunk, float],
    query: str,
    max_score: float,
    all_chunks: list[SembleChunk],
) -> None:
    names = set(_EMBEDDED_SYMBOL_RE.findall(query))
    if not names:
        return
    boost_unit = max_score * _DEFINITION_BOOST_MULTIPLIER * _EMBEDDED_SYMBOL_BOOST_SCALE
    for chunk in list(boosted):
        if tier := _definition_tier(chunk, names, boost_unit):
            boosted[chunk] += tier

    symbols_lower = frozenset(s.lower() for s in names)
    for chunk in all_chunks:
        if chunk in boosted:
            continue
        stem = Path(chunk.file_path).stem.lower()
        stem_norm = stem.replace("_", "")
        if any(
            stem == symbol_lower
            or stem_norm == symbol_lower
            or (len(stem) >= _EMBEDDED_STEM_MIN_LEN and symbol_lower.startswith(stem))
            or (len(stem_norm) >= _EMBEDDED_STEM_MIN_LEN and symbol_lower.startswith(stem_norm))
            for symbol_lower in symbols_lower
        ):
            if tier := _definition_tier(chunk, names, boost_unit):
                boosted[chunk] = tier


def _boost_stem_matches(
    boosted: dict[SembleChunk, float],
    query: str,
    max_score: float,
) -> None:
    keywords = {
        word.lower()
        for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
    if not keywords:
        return
    boost = max_score * _STEM_BOOST_MULTIPLIER
    path_cache: dict[str, set[str]] = {}
    for chunk in list(boosted):
        if chunk.file_path not in path_cache:
            p = Path(chunk.file_path)
            parts = set(split_identifier(p.stem))
            if p.parent.name and p.parent.name not in (".", "/", ".."):
                parts.update(split_identifier(p.parent.name))
            path_cache[chunk.file_path] = parts
        parts = path_cache[chunk.file_path]
        exact = keywords & parts
        n_matches = len(exact)
        for kw in keywords - exact:
            for pt in parts:
                shorter, longer = (kw, pt) if len(kw) <= len(pt) else (pt, kw)
                if len(shorter) >= 3 and longer.startswith(shorter):
                    n_matches += 1
                    break
        if n_matches > 0:
            match_ratio = n_matches / len(keywords)
            if match_ratio >= 0.10:
                boosted[chunk] += boost * match_ratio


def boost_multi_chunk_files(scores: dict[SembleChunk, float]) -> None:
    if not scores:
        return
    max_score = max(scores.values())
    if max_score == 0.0:
        return
    file_sum: dict[str, float] = {}
    best_chunk: dict[str, SembleChunk] = {}
    for chunk, score in scores.items():
        fp = chunk.file_path
        file_sum[fp] = file_sum.get(fp, 0.0) + score
        if fp not in best_chunk or score > scores[best_chunk[fp]]:
            best_chunk[fp] = chunk
    max_file_sum = max(file_sum.values())
    boost_unit = max_score * _FILE_COHERENCE_BOOST_FRAC
    for fp, chunk in best_chunk.items():
        scores[chunk] += boost_unit * file_sum[fp] / max_file_sum


def _file_path_penalty(file_path: str) -> float:
    norm = file_path.replace("\\", "/")
    penalty = 1.0
    if _TEST_FILE_RE.search(norm) is not None or _TEST_DIR_RE.search(norm) is not None:
        penalty *= _STRONG_PENALTY
    if Path(file_path).name in _REEXPORT_FILENAMES:
        penalty *= _MODERATE_PENALTY
    if _COMPAT_DIR_RE.search(norm):
        penalty *= _STRONG_PENALTY
    if _EXAMPLES_DIR_RE.search(norm):
        penalty *= _STRONG_PENALTY
    if _TYPE_DEFS_RE.search(norm):
        penalty *= _MILD_PENALTY
    return penalty


def rerank_topk(
    scores: dict[SembleChunk, float],
    top_k: int,
    *,
    penalise_paths: bool = True,
) -> list[tuple[SembleChunk, float]]:
    if not scores:
        return []
    penalty_cache: dict[str, float] = {}
    penalised: dict[SembleChunk, float] = {}
    for chunk, score in scores.items():
        if penalise_paths:
            if chunk.file_path not in penalty_cache:
                penalty_cache[chunk.file_path] = _file_path_penalty(chunk.file_path)
            penalised[chunk] = score * penalty_cache[chunk.file_path]
        else:
            penalised[chunk] = score

    ranked = sorted(penalised, key=lambda c: -penalised[c])
    file_selected: dict[str, int] = {}
    selected: list[tuple[float, SembleChunk]] = []
    min_selected = float("+inf")

    for chunk in ranked:
        pen_score = penalised[chunk]
        if len(selected) >= top_k and pen_score <= min_selected:
            break
        already_selected = file_selected.get(chunk.file_path, 0)
        eff_score = pen_score
        if already_selected >= _FILE_SATURATION_THRESHOLD:
            excess = already_selected - _FILE_SATURATION_THRESHOLD + 1
            eff_score *= _FILE_SATURATION_DECAY**excess

        selected.append((eff_score, chunk))
        file_selected[chunk.file_path] = already_selected + 1
        if len(selected) >= top_k:
            min_selected = min(s for s, _ in selected)

    selected.sort(key=lambda t: -t[0])
    return [(chunk, score) for score, chunk in selected[:top_k]]


# --- Main Semble Reference Retriever Class ---
class SembleReferenceRetriever(BaseRetriever):
    """Unmodified reference retriever for Semble in-memory search."""

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self.model = StaticModel.from_pretrained(model_name)
        self.chunks: list[SembleChunk] = []
        self.embeddings: npt.NDArray[np.float32] = np.empty((0, 256), dtype=np.float32)
        self.bm25_index: bm25s.BM25 | None = None
        self.repo_name: str = ""

    def index_directory(
        self,
        directory_path: Path | str,
        language: str,
        repo_name: str | None = None,
        display_root: Path | str | None = None,
    ) -> dict[str, Any]:
        root = Path(directory_path).resolve()
        disp_root = Path(display_root).resolve() if display_root else root
        self.repo_name = repo_name or root.name

        code_exts = frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE)
        self.chunks = []

        for fp in walk_source_files(root, code_exts):
            lang = language_for_path(fp) or language
            with contextlib.suppress(OSError):
                source = fp.read_text(encoding="utf-8", errors="replace")
                try:
                    rel_p = str(fp.relative_to(disp_root))
                except ValueError:
                    rel_p = str(fp.relative_to(root))
                self.chunks.extend(chunk_source(source, rel_p, lang))

        if not self.chunks:
            self.embeddings = np.empty((0, 256), dtype=np.float32)
            self.bm25_index = None
            return {"chunks": 0, "status": "empty"}

        # Encode dense embeddings
        raw_embs = self.model.encode([c.content for c in self.chunks])
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (raw_embs / norms).astype(np.float32)

        # Index BM25
        self.bm25_index = bm25s.BM25()
        self.bm25_index.index(
            [tokenize(enrich_for_bm25(c)) for c in self.chunks],
            show_progress=False,
        )

        return {
            "chunks": len(self.chunks),
            "status": "ready",
            "repo": self.repo_name,
        }

    def search(self, query: str, top_k: int = 50) -> list[RetrievedUnit]:
        if not self.chunks or self.bm25_index is None:
            return []

        alpha = resolve_alpha(query, None)
        candidate_count = min(len(self.chunks), top_k * 5)

        # 1. Semantic Search
        q_emb = self.model.encode([query])[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm
        dense_sims = np.dot(self.embeddings, q_emb)
        if candidate_count >= len(dense_sims):
            top_dense_idx = np.argsort(-dense_sims)
        else:
            part = np.argpartition(-dense_sims, kth=candidate_count)[:candidate_count]
            top_dense_idx = part[np.argsort(-dense_sims[part])]

        semantic_scores = {self.chunks[i]: float(dense_sims[i]) for i in top_dense_idx}

        # 2. BM25 Search
        q_toks = tokenize(query)
        bm25_scores = {}
        if q_toks:
            scores = self.bm25_index.get_scores(q_toks)
            if candidate_count >= len(scores):
                top_sparse_idx = np.argsort(-scores)
            else:
                part = np.argpartition(-scores, kth=candidate_count)[:candidate_count]
                top_sparse_idx = part[np.argsort(-scores[part])]
            for i in top_sparse_idx:
                if scores[i] > 0:
                    bm25_scores[self.chunks[i]] = float(scores[i])

        # 3. RRF Combination
        norm_dense = _rrf_scores(semantic_scores)
        norm_sparse = _rrf_scores(bm25_scores)

        combined: dict[SembleChunk, float] = {
            c: alpha * norm_dense.get(c, 0.0) + (1.0 - alpha) * norm_sparse.get(c, 0.0)
            for c in set(norm_dense) | set(norm_sparse)
        }

        # 4. Boosts
        boost_multi_chunk_files(combined)
        max_s = max(combined.values()) if combined else 0.0
        if max_s > 0:
            if is_symbol_query(query):
                _boost_symbol_definitions(combined, query, max_s, self.chunks)
            else:
                _boost_stem_matches(combined, query, max_s)
                _boost_embedded_symbols(combined, query, max_s, self.chunks)

        # 5. Rerank with penalties
        ranked = rerank_topk(combined, top_k, penalise_paths=alpha < 1.0)

        return [
            RetrievedUnit(
                file_path=c.file_path,
                content=c.content,
                start_line=c.start_line,
                end_line=c.end_line,
                score=score,
            )
            for c, score in ranked
        ]

"""Base retriever interface for all benchmark candidate engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from referee.metrics import RetrievedUnit


class BaseRetriever(ABC):
    """Abstract interface for code retrieval engines."""

    @abstractmethod
    def index_directory(
        self,
        directory_path: Path | str,
        language: str,
        repo_name: str | None = None,
    ) -> dict[str, Any]:
        """Index all valid source files in the specified directory.

        Returns index metadata (chunk count, index time ms, etc.).
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[RetrievedUnit]:
        """Search the indexed repository and return ranked retrieved units."""
        pass

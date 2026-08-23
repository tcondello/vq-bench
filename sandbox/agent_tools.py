"""
Agent Tool Interface for Terminal Bench Sandbox.
Wraps VQ-bench 1.35 b/d Two-Stage Index and Symbol Graph into callable agent actions.
"""

import os
import sys
import json

# Ensure examples is in path to import pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples")))
from pipeline import CodeIndexPipeline

class VQBenchAgentTools:
    def __init__(self, repo_root=None, index_cache="sandbox/vqbench_index.pkl"):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.index_cache = index_cache
        self.pipeline = CodeIndexPipeline(chunk_size=128)
        
        if os.path.exists(index_cache):
            self.pipeline.load(index_cache)
        else:
            self.pipeline.index_directory(self.repo_root, extensions=(".rs", ".py"))
            os.makedirs(os.path.dirname(index_cache), exist_ok=True)
            self.pipeline.save(index_cache)

    def search(self, query: str, top_k: int = 3) -> str:
        """Two-stage hybrid code search (1.35 b/d filter + exact MaxSim rescore)."""
        results = self.pipeline.search(query, top_k=top_k)
        if not results:
            return "No matching code snippets found."
        
        output = [f"=== VQ-bench Code Search Results ({len(results)} matches, {results[0]['latency_ms']:.2f}ms) ==="]
        for rank, res in enumerate(results, 1):
            output.append(f"\n[{rank}] File: {res['file']} | Chunk ID: #{res['chunk_id']} | Tokens: {res['token_count']}")
            output.append("```")
            output.append(res['text'].strip())
            output.append("```")
        return "\n".join(output)

    def symbol(self, symbol_name: str) -> str:
        """Instant AST Symbol Lookup."""
        res = self.pipeline.get_symbol_definition(symbol_name)
        if not res:
            return f"Symbol '{symbol_name}' not found in codebase index."
        
        output = [f"=== Symbol Graph Match for '{symbol_name}' ==="]
        if isinstance(res, dict) and "file" in res:
            output.append(f"Definition: {res['file']}:{res['line']}")
            output.append(f"Signature:  {res['signature']}")
        else:
            for name, info in list(res.items())[:5]:
                output.append(f"• {name} -> {info['file']}:{info['line']}")
                output.append(f"  Signature: {info['signature']}")
        return "\n".join(output)

    def context(self, chunk_id: int, window: int = 1) -> str:
        """Expand surrounding contiguous code context around a chunk."""
        expanded = self.pipeline.expand_context(chunk_id, window=window)
        if not expanded:
            return f"Invalid chunk ID #{chunk_id}."
        
        output = [
            f"=== Expanded Code Context ({expanded['total_tokens']} tokens in {expanded['file']}) ===",
            "```",
            expanded['expanded_text'].strip(),
            "```"
        ]
        return "\n".join(output)

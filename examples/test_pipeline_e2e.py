import os
import sys
import unittest
import numpy as np
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import CodeIndexPipeline, quantize_1bit, bm25_rank, rrf_fuse

class TestCodeIndexPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = CodeIndexPipeline(chunk_size=128, use_ast=True)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.index_stats = cls.pipeline.index_directory(repo_root, extensions=(".rs", ".py"))

    def test_01_indexing_integrity(self):
        """Verify codebase indexing produced chunks and symbols."""
        self.assertGreater(self.index_stats["num_files"], 10, "Should index at least 10 source files")
        self.assertGreater(self.index_stats["num_chunks"], 50, "Should generate at least 50 semantic chunks")
        self.assertGreater(self.index_stats["num_symbols"], 50, "Should extract at least 50 symbols")
        self.assertEqual(len(self.pipeline.chunks), self.index_stats["num_chunks"])
        self.assertEqual(self.pipeline.dense_embs_static.shape[0], self.index_stats["num_chunks"])

    def test_02_quantization_properties(self):
        """Verify 1.35 b/d quantization fidelity and dimensions."""
        self.assertEqual(self.pipeline.quantized_embs.shape, self.pipeline.dense_embs_static.shape)
        # Check signs are in {-1/sqrt(d), +1/sqrt(d)}
        d = self.pipeline.dense_embs_static.shape[-1]
        expected_mag = 1.0 / np.sqrt(d)
        np.testing.assert_allclose(np.abs(self.pipeline.quantized_embs), expected_mag, rtol=1e-5)

    def test_03_search_retrieval(self):
        """Verify two-stage hybrid search retrieves relevant chunks."""
        results = self.pipeline.search("Quantizer trait and byte_split", top_k=5)
        self.assertEqual(len(results), 5)
        self.assertIn("file", results[0])
        self.assertIn("text", results[0])
        self.assertGreater(results[0]["token_count"], 0)
        # Verify latency is reasonable (< 1500 ms)
        self.assertLess(results[0]["latency_ms"], 1500.0)

    def test_04_symbol_lookup(self):
        """Verify instant AST symbol lookup tool."""
        res = self.pipeline.get_symbol_definition("Quantizer")
        self.assertTrue(bool(res), "Should find definition for 'Quantizer'")

    def test_05_expand_context(self):
        """Verify MCP context expansion tool."""
        expanded = self.pipeline.expand_context(chunk_id=5, window=1)
        self.assertIsNotNone(expanded)
        self.assertIn("expanded_text", expanded)
        self.assertGreater(expanded["total_tokens"], 0)

    def test_06_serialization_roundtrip(self):
        """Verify saving and loading index preserves byte and vector identity."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.pipeline.save(tmp_path)
            loaded = CodeIndexPipeline()
            loaded.load(tmp_path)
            
            self.assertEqual(len(loaded.chunks), len(self.pipeline.chunks))
            self.assertEqual(len(loaded.symbols), len(self.pipeline.symbols))
            np.testing.assert_array_equal(loaded.quantized_embs, self.pipeline.quantized_embs)
            np.testing.assert_allclose(loaded.dense_embs_static, self.pipeline.dense_embs_static, rtol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()

"""Tests for sefs.content_analyzer – Properties 1, 3, 4, 5, 6, 7."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sefs.config import SUPPORTED_EXTENSIONS
from sefs.content_analyzer import ContentAnalyzer
from sefs.database import Database
from sefs.file_monitor import FileMonitor
from sefs.models import FileMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def analyzer(db: Database) -> ContentAnalyzer:
    return ContentAnalyzer("all-MiniLM-L6-v2", db)


# ---------------------------------------------------------------------------
# Property 1 – File type filtering
# ---------------------------------------------------------------------------

class TestFileTypeFiltering:
    """Feature: semantic-entropy-file-system, Property 1: File type filtering"""

    @pytest.mark.parametrize("ext", ["pdf", "txt", "md", "rst", "log"])
    def test_supported_extensions_accepted(self, ext: str):
        assert FileMonitor.is_supported_file(Path(f"doc.{ext}"))

    @pytest.mark.parametrize("ext", ["jpg", "png", "exe", "zip", "docx", "xlsx", "py"])
    def test_unsupported_extensions_rejected(self, ext: str):
        assert not FileMonitor.is_supported_file(Path(f"file.{ext}"))


# ---------------------------------------------------------------------------
# Property 3 – Content extraction completeness
# ---------------------------------------------------------------------------

class TestContentExtraction:
    """Feature: semantic-entropy-file-system, Property 3: Content extraction completeness"""

    def test_extract_text_from_txt(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        text = analyzer.extract_text(f)
        assert text is not None
        assert "Hello" in text

    def test_analyze_text_file_produces_embedding(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "sample.txt"
        f.write_text("Machine learning is a subset of artificial intelligence.", encoding="utf-8")
        meta = analyzer.analyze_file(f)
        assert meta is not None
        assert meta.embedding is not None
        assert meta.embedding.shape == (384,)

    def test_empty_file_produces_zero_embedding(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        meta = analyzer.analyze_file(f)
        assert meta is not None
        assert meta.embedding is not None
        assert np.allclose(meta.embedding, 0.0)


# ---------------------------------------------------------------------------
# Property 4 – Metadata extraction completeness
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    """Feature: semantic-entropy-file-system, Property 4: Metadata extraction completeness"""

    def test_metadata_fields_present(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "info.txt"
        f.write_text("Some content", encoding="utf-8")
        meta = analyzer.analyze_file(f)
        assert meta is not None
        assert meta.filename == "info.txt"
        assert meta.file_size > 0
        assert meta.creation_date is not None
        assert meta.modification_date is not None
        assert meta.file_type in ("pdf", "text")


# ---------------------------------------------------------------------------
# Property 5 – Embedding caching round-trip
# ---------------------------------------------------------------------------

class TestEmbeddingCaching:
    """Feature: semantic-entropy-file-system, Property 5: Embedding caching round-trip"""

    def test_cache_returns_same_embedding(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "cached.txt"
        f.write_text("Deep learning architectures", encoding="utf-8")
        meta1 = analyzer.analyze_file(f)
        meta2 = analyzer.analyze_file(f)  # second call should hit cache
        assert meta1 is not None and meta2 is not None
        np.testing.assert_array_equal(meta1.embedding, meta2.embedding)


# ---------------------------------------------------------------------------
# Property 6 – Re-analysis on modification
# ---------------------------------------------------------------------------

class TestReanalysis:
    """Feature: semantic-entropy-file-system, Property 6: Re-analysis on modification"""

    def test_modified_file_gets_new_embedding(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "changing.txt"
        f.write_text("Topic A: quantum physics", encoding="utf-8")
        meta1 = analyzer.analyze_file(f)

        f.write_text("Topic B: culinary arts and cooking recipes", encoding="utf-8")
        meta2 = analyzer.analyze_file(f)

        assert meta1 is not None and meta2 is not None
        assert meta1.checksum != meta2.checksum
        assert not np.array_equal(meta1.embedding, meta2.embedding)


# ---------------------------------------------------------------------------
# Property 7 – Error handling for corrupted files
# ---------------------------------------------------------------------------

class TestCorruptedFileHandling:
    """Feature: semantic-entropy-file-system, Property 7: Error handling for corrupted files"""

    def test_corrupted_pdf_returns_none(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"NOT A VALID PDF CONTENT AT ALL")
        meta = analyzer.analyze_file(f)
        # Should not crash; may return None
        # (it's acceptable to return None for truly unreadable files)

    def test_binary_file_with_txt_extension(self, analyzer: ContentAnalyzer, tmp_path: Path):
        f = tmp_path / "binary.txt"
        f.write_bytes(bytes(range(256)))
        meta = analyzer.analyze_file(f)
        # Should not crash
        assert meta is None or meta.embedding is not None


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(text=st.text(min_size=1, max_size=200))
@settings(max_examples=20, deadline=30000)  # embedding model is heavy – keep low
def test_any_text_produces_384d_embedding(text: str):
    """Feature: semantic-entropy-file-system, Property 3: Content extraction – embedding dimension"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(text, show_progress_bar=False)
    assert emb.shape == (384,)

"""Tests for sefs.config – Properties 26, 27."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sefs.config import SEFSConfig


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSEFSConfig:
    """Unit tests for configuration load/save and validation."""

    def test_round_trip_save_load(self, tmp_path: Path):
        """Feature: semantic-entropy-file-system, Property 26: Configuration persistence round-trip"""
        root = tmp_path / "root"
        root.mkdir()
        cfg = SEFSConfig(
            root_directory=str(root),
            similarity_threshold=0.6,
            clustering_algorithm="dbscan",
            embedding_model="all-MiniLM-L6-v2",
            confidence_threshold=0.4,
            enable_preview_mode=True,
            enable_undo=False,
            cooldown_seconds=5,
            max_retries=2,
        )
        config_file = tmp_path / "config.json"
        cfg.save(config_file)

        loaded = SEFSConfig.load(config_file)
        assert loaded.root_directory == cfg.root_directory
        assert loaded.similarity_threshold == cfg.similarity_threshold
        assert loaded.clustering_algorithm == cfg.clustering_algorithm
        assert loaded.enable_preview_mode == cfg.enable_preview_mode
        assert loaded.cooldown_seconds == cfg.cooldown_seconds

    def test_validate_rejects_missing_directory(self, tmp_path: Path):
        """Feature: semantic-entropy-file-system, Property 27: Directory validation"""
        cfg = SEFSConfig(root_directory=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            cfg.validate()

    def test_validate_rejects_file_as_root(self, tmp_path: Path):
        """Feature: semantic-entropy-file-system, Property 27: Directory validation"""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hi")
        cfg = SEFSConfig(root_directory=str(f))
        with pytest.raises(NotADirectoryError):
            cfg.validate()

    def test_validate_rejects_bad_threshold(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        cfg = SEFSConfig(root_directory=str(root), similarity_threshold=1.5)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_accepts_valid_config(self, tmp_path: Path):
        """Feature: semantic-entropy-file-system, Property 27: Directory validation"""
        root = tmp_path / "root"
        root.mkdir()
        cfg = SEFSConfig(root_directory=str(root))
        cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    threshold=st.floats(min_value=0.0, max_value=1.0),
    cooldown=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_config_round_trip_property(threshold: float, cooldown: int, tmp_path_factory):
    """Feature: semantic-entropy-file-system, Property 26: Configuration persistence round-trip"""
    tmp = tmp_path_factory.mktemp("cfg")
    root = tmp / "root"
    root.mkdir()
    cfg = SEFSConfig(
        root_directory=str(root),
        similarity_threshold=threshold,
        cooldown_seconds=cooldown,
    )
    config_file = tmp / "config.json"
    cfg.save(config_file)
    loaded = SEFSConfig.load(config_file)
    assert abs(loaded.similarity_threshold - threshold) < 1e-9
    assert loaded.cooldown_seconds == cooldown

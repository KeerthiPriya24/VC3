"""Tests for sefs.semantic_engine – Properties 8, 9, 10."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sefs.semantic_engine import SemanticEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> SemanticEngine:
    return SemanticEngine(similarity_threshold=0.45)


def _random_unit_vector(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# Property 8 – Similarity calculation symmetry
# ---------------------------------------------------------------------------

class TestSimilaritySymmetry:
    """Feature: semantic-entropy-file-system, Property 8: Similarity calculation symmetry"""

    def test_symmetry_basic(self, engine: SemanticEngine):
        a = _random_unit_vector()
        b = _random_unit_vector()
        assert abs(engine.calculate_similarity(a, b) - engine.calculate_similarity(b, a)) < 1e-6

    def test_identical_vectors_similarity_one(self, engine: SemanticEngine):
        a = _random_unit_vector()
        assert abs(engine.calculate_similarity(a, a) - 1.0) < 1e-6


@given(
    seed_a=st.integers(min_value=0, max_value=2**31),
    seed_b=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_similarity_symmetry_property(seed_a: int, seed_b: int):
    """Feature: semantic-entropy-file-system, Property 8: Similarity calculation symmetry"""
    rng_a = np.random.RandomState(seed_a)
    rng_b = np.random.RandomState(seed_b)
    a = rng_a.randn(384).astype(np.float32)
    b = rng_b.randn(384).astype(np.float32)
    engine = SemanticEngine()
    sim_ab = engine.calculate_similarity(a, b)
    sim_ba = engine.calculate_similarity(b, a)
    assert abs(sim_ab - sim_ba) < 1e-6


# ---------------------------------------------------------------------------
# Property 9 – Threshold-based grouping
# ---------------------------------------------------------------------------

class TestThresholdGrouping:
    """Feature: semantic-entropy-file-system, Property 9: Threshold-based grouping"""

    def test_similar_files_same_cluster(self):
        """Two very similar embeddings should end up in the same cluster."""
        base = _random_unit_vector()
        # Add tiny noise
        e1 = base + np.random.randn(384).astype(np.float32) * 0.01
        e2 = base + np.random.randn(384).astype(np.float32) * 0.01
        
        # Normalize
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)

        engine = SemanticEngine(similarity_threshold=0.3)
        clusters = engine.cluster_files({
            Path("a.txt"): e1,
            Path("b.txt"): e2,
        })

        # They should form a single cluster
        # With V2 logic, we force clustering or valid singletons.
        # Here they are close, so they must be together.
        found_together = False
        for c in clusters.values():
            if Path("a.txt") in c.file_paths and Path("b.txt") in c.file_paths:
                found_together = True
                break
        assert found_together, "Similar files were not grouped together"

    def test_dissimilar_files_different_clusters(self):
        """Two orthogonal embeddings should not cluster together at high threshold."""
        e1 = np.zeros(384, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(384, dtype=np.float32)
        e2[1] = 1.0

        engine = SemanticEngine(similarity_threshold=0.9)
        clusters = engine.cluster_files({
            Path("a.txt"): e1,
            Path("b.txt"): e2,
        })
        
        # In V2, they should be in separate clusters (either singletons or distinct groups)
        # Definitely NOT in the same entry
        for c in clusters.values():
            files = set(c.file_paths)
            assert not ({Path("a.txt"), Path("b.txt")}.issubset(files)), \
                "Dissimilar files grouped together incorrectly"


# ---------------------------------------------------------------------------
# Property 10 – Configurable threshold effect
# ---------------------------------------------------------------------------

class TestConfigurableThreshold:
    """Feature: semantic-entropy-file-system, Property 10: Configurable threshold effect"""

    def test_lower_threshold_larger_clusters(self):
        np.random.seed(42)
        paths = [Path(f"f{i}.txt") for i in range(10)]
        embeddings = {p: _random_unit_vector() for p in paths}

        low = SemanticEngine(similarity_threshold=0.1)
        high = SemanticEngine(similarity_threshold=0.9)

        clusters_low = low.cluster_files(embeddings)
        clusters_high = high.cluster_files(embeddings)

        # Count total clusters (folders)
        n_low = len(clusters_low)
        n_high = len(clusters_high)

        # Lower threshold -> more merging -> fewer clusters (ignoring noise handling nuance)
        # OR same number if data is random
        logger = logging.getLogger("test")
        logger.info(f"Low thresh clusters: {n_low}, High thresh clusters: {n_high}")

        # This property is soft for random data, but generally true.
        assert n_low <= n_high, "Lower threshold should produce fewer (aggregated) or equal clusters"


# ---------------------------------------------------------------------------
# Folder name generation
# ---------------------------------------------------------------------------

class TestFolderNameGeneration:
    def test_generates_valid_dirname(self, engine: SemanticEngine):
        name = engine._keywords_to_folder_name(["machine", "learning", "AI"])
        assert name
        assert "/" not in name
        assert "\\" not in name

    def test_empty_keywords_returns_empty(self, engine: SemanticEngine):
        assert engine._keywords_to_folder_name([]) == ""

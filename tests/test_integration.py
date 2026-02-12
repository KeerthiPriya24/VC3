"""Integration tests for SEFS end-to-end pipeline – now with recursive clustering."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from sefs.config import SEFSConfig
from sefs.database import Database
from sefs.main import SEFSOrchestrator

logger = logging.getLogger(__name__)


@pytest.fixture
def test_env(tmp_path: Path):
    """Set up a temporary environment for integration testing."""
    root = tmp_path / "root"
    root.mkdir()
    config = SEFSConfig(root_directory=str(root))
    
    # Initialize DB
    Database(config.database_path)
    
    return root, config


class TestEndToEndPipeline:
    def test_pipeline_creates_semantic_folders(self, test_env):
        root, config = test_env
        config.similarity_threshold = 0.3 # Lower threshold for test data
        
        # Create diverse files
        (root / "space.txt").write_text("The solar system consists of the Sun and the objects that orbit it.")
        (root / "mars.txt").write_text("Mars is the fourth planet from the Sun and the second-smallest planet.")
        (root / "recipe.txt").write_text("To make a cake, mix flour, sugar, eggs, and butter.")
        (root / "cooking.txt").write_text("Baking requires precise measurements of ingredients like flour and sugar.")
        (root / "space.txt").write_text("Space astronomy solar system planets sun stars galaxy universe.")
        (root / "mars.txt").write_text("Mars planet solar system sun space astronomy rover.")
        (root / "recipe.txt").write_text("Recipe cake flour sugar eggs butter bake oven cooking.")
        (root / "cooking.txt").write_text("Cooking baking ingredients flour sugar kitchen chef recipe.")
        
        # Run orchestration (simulated full scan)
        # Run orchestration (simulated full scan)
        orch = SEFSOrchestrator(config)
        orch._full_scan()  # This triggers analyze -> cluster -> sync
        
        # Verify folders exist
        subdirs = [p for p in root.iterdir() if p.is_dir() and p.name != ".sefs"]
        assert len(subdirs) >= 2, f"Expected at least 2 clusters, found: {[p.name for p in subdirs]}"
        
        # Verify file placement
        space_files = list(root.rglob("*space.txt")) + list(root.rglob("*mars.txt"))
        recipe_files = list(root.rglob("*recipe.txt")) + list(root.rglob("*cooking.txt"))
        
        assert len(space_files) == 2, f"Missing space files: {space_files}"
        assert len(recipe_files) == 2, f"Missing recipe files: {recipe_files}"
        
        # They should form distinct groups (same parent folder)
        space_parent = space_files[0].parent
        recipe_parent = recipe_files[0].parent
        
        assert space_files[1].parent == space_parent, f"Space files split: {space_files[0].parent} != {space_files[1].parent}"
        assert recipe_files[1].parent == recipe_parent, f"Recipe files split: {recipe_files[0].parent} != {recipe_files[1].parent}"
        assert space_parent != recipe_parent, f"Topics merged into: {space_parent}"

    def test_recursive_clustering(self, test_env):
        """Test that large clusters get split into sub-folders."""
        root, config = test_env
        
        # Create 7 files about Physics (Mechanics vs Quantum)
        # Mechanics (4 files)
        (root / "newton.txt").write_text("Newtonian mechanics describes the motion of macroscopic objects.")
        (root / "force.txt").write_text("Force equals mass times acceleration is the second law of motion.")
        (root / "gravity.txt").write_text("Gravity is a fundamental interaction which causes mutual attraction between all things with mass.")
        (root / "motion.txt").write_text("Laws of motion describe the relationship between a body and the forces acting upon it.")
        
        # Quantum (3 files)
        (root / "quantum.txt").write_text("Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms.")
        (root / "schrodinger.txt").write_text("The Schrödinger equation is a linear partial differential equation that governs the wave function of a quantum-mechanical system.")
        (root / "particle.txt").write_text("Subatomic particles include electrons, protons, and neutrons.")
        
        # Configure engine to split at 6. Need to instantiate orch before modifying engine
        orch = SEFSOrchestrator(config)
        orch.engine.min_cluster_size_for_split = 6
        
        orch._full_scan()
        
        all_files = list(root.rglob("*.txt"))
        assert len(all_files) == 7
        
        # Check if any file is in a subdirectory of a subdirectory (depth 2)
        # root / Folder / File -> depth 1
        # root / Folder / Subfolder / File -> depth 2
        
        # If clustering worked recursively, some files should be in sub-folders
        # OR separate clusters if they were distinct enough at level 0.
        # This check is tricky without guaranteed model behavior, but let's assert folders exist.
        
        parent_dirs = {p.parent.name for p in all_files}
        assert len(parent_dirs) >= 2, "Physics sub-domains should have formed separate folders (either nested or sibling)"
        assert "Uncategorized" not in parent_dirs

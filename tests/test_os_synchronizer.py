"""Tests for sefs.os_synchronizer – Properties 11, 12, 13, 14, 30, 31."""

from __future__ import annotations

from pathlib import Path

import pytest

from sefs.content_analyzer import ContentAnalyzer
from sefs.database import Database
from sefs.models import ClusterInfo, ClusterUpdate, FileOperation
from sefs.os_synchronizer import OSSynchronizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / ".sefs" / "test.db")


@pytest.fixture
def sync(tmp_path: Path, db: Database) -> OSSynchronizer:
    return OSSynchronizer(root_directory=tmp_path, db=db)


# ---------------------------------------------------------------------------
# Property 11 – Cluster-to-folder correspondence
# ---------------------------------------------------------------------------

class TestClusterFolderCreation:
    """Feature: semantic-entropy-file-system, Property 11: Cluster-to-folder correspondence"""

    def test_creates_semantic_folder(self, sync: OSSynchronizer, tmp_path: Path):
        folder = sync.create_semantic_folder("Science_Papers")
        assert folder.exists()
        assert folder.is_dir()
        assert folder.name == "Science_Papers"

    def test_idempotent_folder_creation(self, sync: OSSynchronizer, tmp_path: Path):
        f1 = sync.create_semantic_folder("Cluster_A")
        f2 = sync.create_semantic_folder("Cluster_A")
        assert f1 == f2


# ---------------------------------------------------------------------------
# Property 13 – Empty folder cleanup
# ---------------------------------------------------------------------------

class TestEmptyFolderCleanup:
    """Feature: semantic-entropy-file-system, Property 13: Empty folder cleanup"""

    def test_removes_empty_folder(self, sync: OSSynchronizer, tmp_path: Path):
        folder = tmp_path / "EmptyCluster"
        folder.mkdir()
        assert sync.remove_empty_folder(folder)
        assert not folder.exists()

    def test_keeps_non_empty_folder(self, sync: OSSynchronizer, tmp_path: Path):
        folder = tmp_path / "NonEmpty"
        folder.mkdir()
        (folder / "file.txt").write_text("data")
        assert not sync.remove_empty_folder(folder)
        assert folder.exists()


# ---------------------------------------------------------------------------
# Property 14 – File movement integrity
# ---------------------------------------------------------------------------

class TestFileMoveIntegrity:
    """Feature: semantic-entropy-file-system, Property 14: File movement integrity"""

    def test_checksum_preserved_after_move(self, sync: OSSynchronizer, tmp_path: Path):
        src = tmp_path / "original.txt"
        src.write_text("Important content that must not be corrupted")
        checksum_before = ContentAnalyzer.compute_checksum(src)

        target = tmp_path / "TargetFolder"
        target.mkdir()

        ok = sync.move_file(src, target)
        dest = target / "original.txt"

        assert ok
        assert dest.exists()
        assert not src.exists()

        checksum_after = ContentAnalyzer.compute_checksum(dest)
        assert checksum_before == checksum_after


# ---------------------------------------------------------------------------
# Property 30 – Atomic move operations
# ---------------------------------------------------------------------------

class TestAtomicMove:
    """Feature: semantic-entropy-file-system, Property 30: Atomic move operations"""

    def test_file_exists_in_one_location_after_move(self, sync: OSSynchronizer, tmp_path: Path):
        src = tmp_path / "doc.txt"
        src.write_text("data")
        target = tmp_path / "Dest"
        target.mkdir()

        sync.move_file(src, target)

        # File should exist in exactly one location
        assert not src.exists()
        assert (target / "doc.txt").exists()


# ---------------------------------------------------------------------------
# Property 31 – Operation logging completeness
# ---------------------------------------------------------------------------

class TestOperationLogging:
    """Feature: semantic-entropy-file-system, Property 31: Operation logging completeness"""

    def test_move_logged(self, sync: OSSynchronizer, tmp_path: Path, db: Database):
        src = tmp_path / "logged.txt"
        src.write_text("content")
        target = tmp_path / "LogDest"
        target.mkdir()

        sync.move_file(src, target)

        ops = db.get_operations()
        assert any(op.operation_type == "move" for op in ops)

    def test_folder_creation_logged(self, sync: OSSynchronizer, db: Database):
        sync.create_semantic_folder("NewFolder")
        ops = db.get_operations()
        assert any(op.operation_type == "create_folder" for op in ops)


# ---------------------------------------------------------------------------
# Property 12 – File uniqueness (apply_cluster_update)
# ---------------------------------------------------------------------------

class TestApplyClusterUpdate:
    """Feature: semantic-entropy-file-system, Property 12: File uniqueness invariant"""

    def test_files_end_up_in_correct_folders(self, sync: OSSynchronizer, tmp_path: Path):
        # Create source files
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha")
        f2.write_text("beta")

        import numpy as np
        update = ClusterUpdate(
            new_clusters={
                "c1": ClusterInfo(
                    cluster_id="c1",
                    folder_name="Alpha",
                    folder_path=tmp_path / "Alpha",
                    file_paths=[f1],
                    centroid_embedding=np.zeros(384, dtype=np.float32),
                ),
                "c2": ClusterInfo(
                    cluster_id="c2",
                    folder_name="Beta",
                    folder_path=tmp_path / "Beta",
                    file_paths=[f2],
                    centroid_embedding=np.zeros(384, dtype=np.float32),
                ),
            },
            file_moves={f1: Path("Alpha"), f2: Path("Beta")},
        )

        sync.apply_cluster_update(update)

        assert (tmp_path / "Alpha" / "a.txt").exists()
        assert (tmp_path / "Beta" / "b.txt").exists()
        # Originals moved (no duplicates)
        assert not f1.exists()
        assert not f2.exists()

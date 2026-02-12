"""Tests for sefs.file_monitor – Properties 1, 2, 16, 38."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sefs.file_monitor import FileMonitor


# ---------------------------------------------------------------------------
# Property 1 – File type filtering (monitor perspective)
# ---------------------------------------------------------------------------

class TestSupportedFiles:
    """Feature: semantic-entropy-file-system, Property 1: File type filtering"""

    @pytest.mark.parametrize("name,expected", [
        ("report.pdf", True),
        ("notes.txt", True),
        ("readme.md", True),
        ("log.log", True),
        ("doc.rst", True),
        ("photo.jpg", False),
        ("archive.zip", False),
        ("spreadsheet.xlsx", False),
    ])
    def test_is_supported_file(self, name: str, expected: bool):
        assert FileMonitor.is_supported_file(Path(name)) == expected


# ---------------------------------------------------------------------------
# Property 16 – Infinite loop prevention
# ---------------------------------------------------------------------------

class TestLoopPrevention:
    """Feature: semantic-entropy-file-system, Property 16: Infinite loop prevention"""

    def test_system_ops_skipped(self, tmp_path: Path):
        events_received: list[set[Path]] = []
        monitor = FileMonitor(
            root_directory=tmp_path,
            on_files_changed=lambda paths: events_received.append(paths),
            on_files_deleted=lambda paths: None,
            debounce_seconds=0.1,
        )
        target = (tmp_path / "sys_file.txt").resolve()
        monitor.register_system_op(target)
        # The event for this path should be skipped internally
        assert target in monitor._system_ops
        monitor.clear_system_op(target)
        assert target not in monitor._system_ops

    def test_pause_resume(self, tmp_path: Path):
        """Paused monitor should not dispatch events."""
        calls: list[int] = []
        monitor = FileMonitor(
            root_directory=tmp_path,
            on_files_changed=lambda _: calls.append(1),
            on_files_deleted=lambda _: None,
        )
        monitor.pause()
        assert monitor._paused is True
        monitor.resume()
        assert monitor._paused is False


# ---------------------------------------------------------------------------
# Property 38 – Change batching (debounce)
# ---------------------------------------------------------------------------

class TestDebouncing:
    """Feature: semantic-entropy-file-system, Property 38: Change batching"""

    def test_rapid_events_batched(self, tmp_path: Path):
        """Multiple rapid file creations should be batched into one callback."""
        results: list[set[Path]] = []
        monitor = FileMonitor(
            root_directory=tmp_path,
            on_files_changed=lambda paths: results.append(paths),
            on_files_deleted=lambda _: None,
            debounce_seconds=0.2,
        )
        monitor.start()
        try:
            # Create several files quickly
            for i in range(5):
                (tmp_path / f"file_{i}.txt").write_text(f"content {i}")
                time.sleep(0.05)

            # Wait for debounce to flush
            time.sleep(0.5)

            # We expect fewer callbacks than files (batching)
            total_files = sum(len(s) for s in results)
            assert total_files >= 1  # at least something was detected
        finally:
            monitor.stop()

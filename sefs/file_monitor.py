"""File Monitor – watches the root directory for filesystem events."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Set

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from sefs.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class FileMonitor:
    """Watches a directory tree and dispatches debounced events for supported files."""

    def __init__(
        self,
        root_directory: Path,
        on_files_changed: Callable[[set[Path]], None],
        on_files_deleted: Callable[[set[Path]], None],
        debounce_seconds: float = 0.5,
    ) -> None:
        self._root = root_directory
        self._on_changed = on_files_changed
        self._on_deleted = on_files_deleted
        self._debounce = debounce_seconds
        self._observer: Optional[Observer] = None

        # Debounce buffers
        self._changed_paths: set[Path] = set()
        self._deleted_paths: set[Path] = set()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        # System-op set: paths being moved by OSSynchronizer → skip events
        self._system_ops: Set[Path] = set()
        self._paused = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        handler = _EventHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._root), recursive=True)
        self._observer.start()
        logger.info("File monitor started on %s", self._root)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("File monitor stopped")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def register_system_op(self, path: Path) -> None:
        """Mark *path* as a system-initiated operation (skip its events)."""
        self._system_ops.add(path.resolve())

    def clear_system_op(self, path: Path) -> None:
        self._system_ops.discard(path.resolve())

    def clear_all_system_ops(self) -> None:
        self._system_ops.clear()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    @staticmethod
    def is_supported_file(file_path: Path) -> bool:
        ext = file_path.suffix.lower().lstrip(".")
        return ext in SUPPORTED_EXTENSIONS

    # ------------------------------------------------------------------
    # Internal event handling
    # ------------------------------------------------------------------
    def _handle_event(self, event: FileSystemEvent) -> None:
        if self._paused:
            return

        path = Path(event.src_path).resolve()

        # Skip hidden dirs and common build artifacts
        EXCLUDED_DIRS = {
            ".sefs", "sefs.egg-info", "__pycache__", "node_modules", "venv", "env",
            ".git", ".idea", ".vscode", "dist", "build", ".pytest_cache", ".hypothesis"
        }
        
        rel_parts = path.relative_to(self._root.resolve()).parts
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in rel_parts):
            return

        # Skip system-initiated ops
        if path in self._system_ops:
            logger.debug("Skipping system-op event for %s", path)
            return

        if isinstance(event, FileDeletedEvent):
            with self._lock:
                self._deleted_paths.add(path)
            self._schedule_flush()
            return

        if isinstance(event, FileMovedEvent):
            dest = Path(event.dest_path).resolve()
            with self._lock:
                self._deleted_paths.add(path)
                if self.is_supported_file(dest):
                    self._changed_paths.add(dest)
            self._schedule_flush()
            return

        if not self.is_supported_file(path):
            return

        if isinstance(event, (FileCreatedEvent, FileModifiedEvent)):
            with self._lock:
                self._changed_paths.add(path)
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._flush)
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            changed = self._changed_paths.copy()
            deleted = self._deleted_paths.copy()
            self._changed_paths.clear()
            self._deleted_paths.clear()

        if changed:
            logger.info("Flushing %d changed file(s)", len(changed))
            try:
                self._on_changed(changed)
            except Exception:
                logger.exception("Error in on_files_changed callback")

        if deleted:
            logger.info("Flushing %d deleted file(s)", len(deleted))
            try:
                self._on_deleted(deleted)
            except Exception:
                logger.exception("Error in on_files_deleted callback")


class _EventHandler(FileSystemEventHandler):
    """Bridges watchdog events to ``FileMonitor``."""

    def __init__(self, monitor: FileMonitor) -> None:
        super().__init__()
        self._monitor = monitor

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._monitor._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._monitor._handle_event(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._monitor._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._monitor._handle_event(event)

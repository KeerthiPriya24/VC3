"""OS Synchronizer – keep the OS folder structure in sync with semantic clusters."""

from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from sefs.content_analyzer import ContentAnalyzer
from sefs.database import Database
from sefs.file_monitor import FileMonitor
from sefs.models import ClusterInfo, ClusterUpdate, FileOperation

logger = logging.getLogger(__name__)


class OSSynchronizer:
    """Creates/removes semantic folders and moves files to match cluster assignments."""

    def __init__(
        self,
        root_directory: Path,
        db: Database,
        file_monitor: Optional[FileMonitor] = None,
        max_retries: int = 3,
        preview_mode: bool = False,
    ) -> None:
        self._root = root_directory
        self._db = db
        self._monitor = file_monitor
        self._max_retries = max_retries
        self._preview_mode = preview_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_cluster_update(self, update: ClusterUpdate) -> None:
        """Synchronize the OS folder tree with *update*."""
        if self._preview_mode:
            self._log_preview(update)
            return

        # 1. Pause monitor to prevent loops
        if self._monitor:
            self._monitor.pause()

        try:
            # 2. Create semantic folders
            for cinfo in update.new_clusters.values():
                folder = self.create_semantic_folder(cinfo.folder_name)
                cinfo.folder_path = folder

            # 3. Move files
            for file_path, target_folder_name in update.file_moves.items():
                target_folder = self._root / str(target_folder_name)
                if not target_folder.exists():
                    target_folder = self.create_semantic_folder(str(target_folder_name))
                self.move_file(file_path, target_folder)

            # 4. Remove empty semantic folders
            self._cleanup_empty_folders()

            # 5. Remove folders for deleted clusters
            for cid in update.removed_cluster_ids:
                # Already handled by cleanup, but be explicit
                pass

        finally:
            # 6. Resume monitor
            if self._monitor:
                self._monitor.clear_all_system_ops()
                self._monitor.resume()

    # ------------------------------------------------------------------
    # Folder operations
    # ------------------------------------------------------------------
    def create_semantic_folder(self, folder_name: str) -> Path:
        sanitized = self._sanitize_folder_name(folder_name)
        folder = self._root / sanitized
        if folder.exists():
            return folder
        folder.mkdir(parents=True, exist_ok=True)

        # Register system op
        if self._monitor:
            self._monitor.register_system_op(folder)

        self._db.log_operation(
            FileOperation(
                operation_type="create_folder",
                destination_path=folder,
                success=True,
            )
        )
        logger.info("Created semantic folder: %s", folder)
        return folder

    def move_file(self, file_path: Path, target_folder: Path) -> bool:
        """Move *file_path* into *target_folder* with retry and integrity check."""
        if not file_path.exists():
            logger.warning("Source file missing, skipping: %s", file_path)
            return False

        dest = target_folder / file_path.name
        if dest == file_path:
            return True  # already in place

        # Handle name conflicts
        dest = self._unique_dest(dest)

        checksum_before = ContentAnalyzer.compute_checksum(file_path)

        # Register paths as system ops
        if self._monitor:
            self._monitor.register_system_op(file_path)
            self._monitor.register_system_op(dest)

        success = self._move_with_retry(file_path, dest)

        if success:
            self._db.rename_file_path(file_path, dest)

        checksum_after = ContentAnalyzer.compute_checksum(dest) if success and dest.exists() else None
        integrity_ok = checksum_before == checksum_after if checksum_after else False

        self._db.log_operation(
            FileOperation(
                operation_type="move",
                source_path=file_path,
                destination_path=dest,
                checksum_before=checksum_before,
                checksum_after=checksum_after,
                success=success and integrity_ok,
                error_message=None if integrity_ok else "Checksum mismatch after move",
            )
        )

        if success and not integrity_ok:
            logger.error("Integrity check failed for %s → %s", file_path, dest)

        return success and integrity_ok

    def remove_empty_folder(self, folder_path: Path) -> bool:
        if not folder_path.is_dir():
            return False
        if any(folder_path.iterdir()):
            return False  # not empty
        if folder_path == self._root:
            return False

        if self._monitor:
            self._monitor.register_system_op(folder_path)

        folder_path.rmdir()
        self._db.log_operation(
            FileOperation(
                operation_type="delete_folder",
                source_path=folder_path,
                success=True,
            )
        )
        logger.info("Removed empty folder: %s", folder_path)
        return True

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------
    def undo_last_operations(self, count: int = 1) -> int:
        """Undo the last *count* move operations. Returns number of ops undone."""
        ops = self._db.get_operations(limit=count * 3)
        undone = 0
        for op in ops:
            if op.operation_type == "move" and op.success and op.destination_path and op.source_path:
                dest = Path(op.destination_path)
                src_dir = Path(op.source_path).parent
                if dest.exists() and src_dir.exists():
                    if self._monitor:
                        self._monitor.pause()
                    try:
                        shutil.move(str(dest), str(src_dir / dest.name))
                        undone += 1
                    except OSError:
                        logger.exception("Undo failed for %s", dest)
                    finally:
                        if self._monitor:
                            self._monitor.resume()
            if undone >= count:
                break
        return undone

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _move_with_retry(self, src: Path, dest: Path) -> bool:
        delay = 1
        for attempt in range(1, self._max_retries + 1):
            try:
                shutil.move(str(src), str(dest))
                logger.info("Moved %s → %s", src.name, dest.parent.name)
                return True
            except OSError:
                logger.warning(
                    "Move attempt %d/%d failed for %s",
                    attempt, self._max_retries, src,
                )
                if attempt < self._max_retries:
                    time.sleep(delay)
                    delay *= 2
        logger.error("All move attempts failed for %s", src)
        return False

    def _cleanup_empty_folders(self) -> None:
        for child in sorted(self._root.iterdir(), reverse=True):
            if child.is_dir() and not child.name.startswith("."):
                self.remove_empty_folder(child)

    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
        sanitized = sanitized.strip(". ")
        return sanitized or "Uncategorized"

    @staticmethod
    def _unique_dest(dest: Path) -> Path:
        if not dest.exists():
            return dest
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent
        counter = 1
        while dest.exists():
            dest = parent / f"{stem}_{counter}{suffix}"
            counter += 1
        return dest

    def _log_preview(self, update: ClusterUpdate) -> None:
        logger.info("=== PREVIEW MODE – no changes applied ===")
        for cid, cinfo in update.new_clusters.items():
            logger.info("  Folder: %s  (%d files)", cinfo.folder_name, len(cinfo.file_paths))
            for fp in cinfo.file_paths:
                logger.info("    → %s", fp.name)

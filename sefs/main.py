"""SEFS – main entry point and orchestrator."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from sefs.config import SEFSConfig, SUPPORTED_EXTENSIONS
from sefs.content_analyzer import ContentAnalyzer
from sefs.database import Database
from sefs.file_monitor import FileMonitor
from sefs.models import ClusterUpdate
from sefs.os_synchronizer import OSSynchronizer
from sefs.semantic_engine import SemanticEngine

logger = logging.getLogger("sefs")


class SEFSOrchestrator:
    """Wires all components together and manages the lifecycle."""

    def __init__(self, config: SEFSConfig) -> None:
        self.cfg = config
        self.db = Database(config.database_path)
        self.analyzer = ContentAnalyzer(config.embedding_model, self.db)
        self.engine = SemanticEngine(
            similarity_threshold=config.similarity_threshold,
            clustering_algorithm=config.clustering_algorithm,
            max_depth=config.max_depth,
            min_cluster_size_for_split=config.min_cluster_size_for_split,
        )
        self.monitor = FileMonitor(
            root_directory=config.root,
            on_files_changed=self._on_files_changed,
            on_files_deleted=self._on_files_deleted,
            debounce_seconds=config.cooldown_seconds,
        )
        self.synchronizer = OSSynchronizer(
            root_directory=config.root,
            db=self.db,
            file_monitor=self.monitor,
            max_retries=config.max_retries,
            preview_mode=config.enable_preview_mode,
        )
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        logger.info("Starting SEFS on %s", self.cfg.root)
        self.cfg.save()

        # Initial full scan
        self._full_scan()

        # Start monitoring
        self.monitor.start()
        logger.info("SEFS is running. Press Ctrl+C to stop.")

        # Block until stop
        try:
            while not self._stop_event.is_set():
                # Check for manual trigger from UI
                if self.db.get_config("trigger_scan") == "true":
                    logger.info("Manual scan triggered from UI")
                    self.db.set_config("trigger_scan", "false")
                    self._full_scan()
                
                self._stop_event.wait(timeout=1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        logger.info("Stopping SEFS …")
        self._stop_event.set()
        self.monitor.stop()
        self.db.close()
        logger.info("SEFS stopped.")

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------
    def _full_scan(self) -> None:
        logger.info("Running full scan of %s", self.cfg.root)
        
        # 1. Cleanup: Remove files from DB that no longer exist on disk
        db_files = self.db.get_all_files()
        deleted_count = 0
        for df in db_files:
            if not df.file_path.exists():
                self.db.delete_file(df.file_path)
                deleted_count += 1
        if deleted_count > 0:
            logger.info("Purged %d stale file(s) from database", deleted_count)

        # 2. Add/Update current files
        all_files = [
            p
            for p in self.cfg.root.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(self.cfg.root).parts)
            and p.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
        ]
        logger.info("Found %d supported file(s) on disk", len(all_files))

        for fp in all_files:
            self.analyzer.analyze_file(fp)

        self._run_clustering()

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------
    def _on_files_changed(self, paths: set[Path]) -> None:
        logger.info("Change event for %d file(s)", len(paths))
        for fp in paths:
            self.analyzer.analyze_file(fp)
        self._run_clustering()

    def _on_files_deleted(self, paths: set[Path]) -> None:
        logger.info("Delete event for %d file(s)", len(paths))
        for fp in paths:
            self.db.delete_file(fp)
        self._run_clustering()

    # ------------------------------------------------------------------
    # Clustering pipeline
    # ------------------------------------------------------------------
    def _run_clustering(self) -> None:
        files = self.db.get_all_files()
        embeddings = {f.file_path: f.embedding for f in files if f.embedding is not None}
        if not embeddings:
            return

        # Check manual mode early
        raw_manual = self.db.get_config("manual_mode")
        manual_mode = str(raw_manual).lower() == "true"

        # Gather summaries for keyword extraction
        file_texts = {f.file_path: (f.content_summary or f.filename) for f in files}

        existing = {c.cluster_id: c for c in self.db.get_all_clusters()}
        
        # Denied moves lookup
        denied_moves = {f.file_path: f.denied_folder_path for f in files if f.denied_folder_path}
        
        update = self.engine.recalculate_clusters(embeddings, file_texts, existing, denied_moves)

        # ---- Atomic DB update (single transaction) ----
        # Wrap the full clear→reassign cycle so the UI never sees an
        # intermediate state where every file has cluster_id = NULL.
        conn = self.db._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Clear old cluster associations
            conn.execute("UPDATE files SET cluster_id = NULL, confidence_score = 0.0, proposed_cluster_id = NULL")

            # Replace clusters
            conn.execute("DELETE FROM clusters")
            for cinfo in update.new_clusters.values():
                cinfo.folder_path = self.cfg.root / cinfo.folder_name
                from sefs.models import FileMetadata
                conn.execute(
                    """INSERT OR REPLACE INTO clusters
                       (cluster_id, folder_name, folder_path,
                        centroid_embedding, average_similarity, keywords)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        cinfo.cluster_id,
                        cinfo.folder_name,
                        str(cinfo.folder_path),
                        FileMetadata.embedding_to_bytes(cinfo.centroid_embedding),
                        cinfo.average_similarity,
                        ",".join(cinfo.keywords),
                    ),
                )

            # Reassign file → cluster mapping
            for cinfo in update.new_clusters.values():
                for fp in cinfo.file_paths:
                    conn.execute(
                        "UPDATE files SET cluster_id = ?, confidence_score = ?, proposed_cluster_id = ? WHERE file_path = ?",
                        (cinfo.cluster_id, cinfo.average_similarity, cinfo.cluster_id, str(fp)),
                    )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Sync to OS (Skip if manual mode is on)
        if manual_mode:
            logger.info("Manual Approval Mode is ON: skipping automatic OS synchronization.")
        else:
            self.synchronizer.apply_cluster_update(update)

        logger.info(
            "Clustering complete: %d cluster(s), %d file(s)",
            len(update.new_clusters),
            sum(len(c.file_paths) for c in update.new_clusters.values()),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Entropy File System")
    parser.add_argument("root_directory", type=str, help="Path to the directory to organize")
    parser.add_argument("--threshold", type=float, default=0.45, help="Similarity threshold (0–1)")
    parser.add_argument("--preview", action="store_true", help="Preview mode (no file moves)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    root = Path(args.root_directory).resolve()
    if not root.is_dir():
        logger.error("Root directory does not exist: %s", root)
        sys.exit(1)

    config = SEFSConfig(
        root_directory=str(root),
        similarity_threshold=args.threshold,
        enable_preview_mode=args.preview,
    )

    orchestrator = SEFSOrchestrator(config)

    def _signal_handler(sig, frame):
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    orchestrator.start()


if __name__ == "__main__":
    main()

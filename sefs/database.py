"""SQLite database wrapper for SEFS."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from sefs.models import ClusterInfo, FileMetadata, FileOperation

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS files (
    file_path  TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,
    file_size  INTEGER,
    creation_date    TEXT,
    modification_date TEXT,
    file_type  TEXT,
    checksum   TEXT,
    embedding  BLOB,
    cluster_id TEXT,
    confidence_score REAL,
    content_summary TEXT,
    denied_folder_path TEXT,
    proposed_cluster_id TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id         TEXT PRIMARY KEY,
    folder_name        TEXT NOT NULL,
    folder_path        TEXT,
    centroid_embedding  BLOB,
    average_similarity  REAL,
    keywords           TEXT
);

CREATE TABLE IF NOT EXISTS operations (
    operation_id   TEXT PRIMARY KEY,
    timestamp      TEXT,
    operation_type TEXT,
    source_path    TEXT,
    destination_path TEXT,
    checksum_before TEXT,
    checksum_after  TEXT,
    success        INTEGER,
    error_message  TEXT
);

CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Database:
    """Thin wrapper around SQLite for SEFS storage."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        
        # Migration: Add content_summary if it doesn't exist
        cursor = self._conn.execute("PRAGMA table_info(files)")
        columns = [row[1] for row in cursor.fetchall()]
        if "content_summary" not in columns:
            logger.info("Database Migration: Adding 'content_summary' column to 'files' table.")
            self._conn.execute("ALTER TABLE files ADD COLUMN content_summary TEXT")
        
        if "denied_folder_path" not in columns:
            logger.info("Database Migration: Adding 'denied_folder_path' column to 'files' table.")
            self._conn.execute("ALTER TABLE files ADD COLUMN denied_folder_path TEXT")
        
        if "proposed_cluster_id" not in columns:
            logger.info("Database Migration: Adding 'proposed_cluster_id' column to 'files' table.")
            self._conn.execute("ALTER TABLE files ADD COLUMN proposed_cluster_id TEXT")
            
        self._conn.commit()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    def upsert_file(self, meta: FileMetadata) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO files
               (file_path, filename, file_size, creation_date, modification_date,
                file_type, checksum, embedding, cluster_id, confidence_score, content_summary, denied_folder_path, proposed_cluster_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(meta.file_path),
                meta.filename,
                meta.file_size,
                meta.creation_date.isoformat(),
                meta.modification_date.isoformat(),
                meta.file_type,
                meta.checksum,
                FileMetadata.embedding_to_bytes(meta.embedding),
                meta.cluster_id,
                meta.confidence_score,
                meta.content_summary,
                meta.denied_folder_path,
                getattr(meta, "proposed_cluster_id", None),
            ),
        )
        self._conn.commit()

    def get_file(self, file_path: Path) -> Optional[FileMetadata]:
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_path = ?", (str(file_path),)
        ).fetchone()
        if not row:
            return None
        return self._row_to_file_metadata(row)

    def get_all_files(self) -> list[FileMetadata]:
        rows = self._conn.execute("SELECT * FROM files").fetchall()
        return [self._row_to_file_metadata(r) for r in rows]

    def delete_file(self, file_path: Path) -> None:
        self._conn.execute("DELETE FROM files WHERE file_path = ?", (str(file_path),))
        self._conn.commit()

    def update_file_cluster(
        self, file_path: Path, cluster_id: Optional[str], confidence: float
    ) -> None:
        self._conn.execute(
            "UPDATE files SET cluster_id = ?, confidence_score = ? WHERE file_path = ?",
            (cluster_id, confidence, str(file_path)),
        )
        self._conn.commit()

    def update_proposed_cluster(self, file_path: Path, proposed_cluster_id: Optional[str]) -> None:
        self._conn.execute(
            "UPDATE files SET proposed_cluster_id = ? WHERE file_path = ?",
            (proposed_cluster_id, str(file_path)),
        )
        self._conn.commit()

    def deny_file_cluster(self, file_path: Path, denied_folder_path: str) -> None:
        self._conn.execute(
            "UPDATE files SET denied_folder_path = ?, proposed_cluster_id = NULL WHERE file_path = ?",
            (denied_folder_path, str(file_path)),
        )
        self._conn.commit()

    def rename_file_path(self, old_path: Path, new_path: Path) -> None:
        """Update the file_path primary key when a file is moved on disk."""
        self._conn.execute(
            "UPDATE files SET file_path = ? WHERE file_path = ?",
            (str(new_path), str(old_path)),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_file_metadata(row: tuple) -> FileMetadata:
        return FileMetadata(
            file_path=Path(row[0]),
            filename=row[1],
            file_size=row[2],
            creation_date=datetime.fromisoformat(row[3]),
            modification_date=datetime.fromisoformat(row[4]),
            file_type=row[5],
            checksum=row[6],
            embedding=FileMetadata.bytes_to_embedding(row[7]),
            cluster_id=row[8],
            confidence_score=row[9] or 0.0,
            content_summary=row[10] if len(row) > 10 else None,
            denied_folder_path=row[11] if len(row) > 11 else None,
            proposed_cluster_id=row[12] if len(row) > 12 else None,
        )

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------
    def upsert_cluster(self, info: ClusterInfo) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO clusters
               (cluster_id, folder_name, folder_path,
                centroid_embedding, average_similarity, keywords)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                info.cluster_id,
                info.folder_name,
                str(info.folder_path),
                FileMetadata.embedding_to_bytes(info.centroid_embedding),
                info.average_similarity,
                ",".join(info.keywords),
            ),
        )
        self._conn.commit()

    def get_all_clusters(self) -> list[ClusterInfo]:
        rows = self._conn.execute("SELECT * FROM clusters").fetchall()
        return [self._row_to_cluster(r) for r in rows]

    def delete_cluster(self, cluster_id: str) -> None:
        self._conn.execute(
            "DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,)
        )
        self._conn.commit()

    def clear_clusters(self) -> None:
        self._conn.execute("DELETE FROM clusters")
        self._conn.commit()

    @staticmethod
    def _row_to_cluster(row: tuple) -> ClusterInfo:
        return ClusterInfo(
            cluster_id=row[0],
            folder_name=row[1],
            folder_path=Path(row[2]) if row[2] else Path(),
            centroid_embedding=FileMetadata.bytes_to_embedding(row[3]),
            average_similarity=row[4] or 0.0,
            keywords=row[5].split(",") if row[5] else [],
        )

    # ------------------------------------------------------------------
    # Operations log
    # ------------------------------------------------------------------
    def log_operation(self, op: FileOperation) -> None:
        self._conn.execute(
            """INSERT INTO operations
               (operation_id, timestamp, operation_type, source_path,
                destination_path, checksum_before, checksum_after,
                success, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                op.operation_id,
                op.timestamp.isoformat(),
                op.operation_type,
                str(op.source_path) if op.source_path else None,
                str(op.destination_path) if op.destination_path else None,
                op.checksum_before,
                op.checksum_after,
                int(op.success),
                op.error_message,
            ),
        )
        self._conn.commit()

    def get_operations(self, limit: int = 100) -> list[FileOperation]:
        rows = self._conn.execute(
            "SELECT * FROM operations ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            FileOperation(
                operation_id=r[0],
                timestamp=datetime.fromisoformat(r[1]),
                operation_type=r[2],
                source_path=Path(r[3]) if r[3] else None,
                destination_path=Path(r[4]) if r[4] else None,
                checksum_before=r[5],
                checksum_after=r[6],
                success=bool(r[7]),
                error_message=r[8],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def get_config(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def set_config(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

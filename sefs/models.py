"""Data models for SEFS."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(eq=False)
class FileMetadata:
    """Metadata and embedding for a tracked file."""

    file_path: Path
    filename: str
    file_size: int
    creation_date: datetime
    modification_date: datetime
    file_type: str  # "pdf" or "text"
    checksum: str  # SHA-256
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[str] = None
    confidence_score: float = 0.0
    content_summary: Optional[str] = None
    denied_folder_path: Optional[str] = None
    proposed_cluster_id: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileMetadata):
            return NotImplemented
        return str(self.file_path) == str(other.file_path)

    def __hash__(self) -> int:
        return hash(str(self.file_path))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def embedding_to_bytes(emb: Optional[np.ndarray]) -> Optional[bytes]:
        if emb is None:
            return None
        return emb.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(raw: Optional[bytes]) -> Optional[np.ndarray]:
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class ClusterInfo:
    """Description of one semantic cluster."""

    cluster_id: str
    folder_name: str
    folder_path: Path
    file_paths: List[Path] = field(default_factory=list)
    centroid_embedding: Optional[np.ndarray] = None
    average_similarity: float = 0.0
    keywords: List[str] = field(default_factory=list)


@dataclass
class FileOperation:
    """One recorded file-system operation (for undo / audit)."""

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    operation_type: str = ""  # "move", "create_folder", "delete_folder"
    source_path: Optional[Path] = None
    destination_path: Optional[Path] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ClusterUpdate:
    """Delta produced by SemanticEngine after re-clustering."""

    new_clusters: dict[str, ClusterInfo] = field(default_factory=dict)
    removed_cluster_ids: list[str] = field(default_factory=list)
    file_moves: dict[Path, Path] = field(default_factory=dict)  # src → dest folder

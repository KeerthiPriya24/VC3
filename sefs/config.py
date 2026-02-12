"""SEFS configuration management."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported text-like extensions (lower-case, without dot)
SUPPORTED_TEXT_EXTENSIONS: set[str] = {"txt", "md", "rst", "log"}
SUPPORTED_EXTENSIONS: set[str] = SUPPORTED_TEXT_EXTENSIONS | {"pdf"}


@dataclass
class SEFSConfig:
    """Runtime configuration for SEFS."""

    root_directory: str = ""
    similarity_threshold: float = 0.55
    clustering_algorithm: str = "dbscan"  # "dbscan" or "kmeans"
    embedding_model: str = "all-mpnet-base-v2"
    confidence_threshold: float = 0.3
    max_depth: int = 3
    min_cluster_size_for_split: int = 3
    enable_preview_mode: bool = False
    enable_undo: bool = True
    cooldown_seconds: int = 2
    max_retries: int = 3
    db_path: str = ""  # defaults to root_directory/.sefs/sefs.db

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.root_directory)

    @property
    def database_path(self) -> Path:
        if self.db_path:
            return Path(self.db_path)
        return self.root / ".sefs" / "sefs.db"

    @property
    def config_dir(self) -> Path:
        return self.root / ".sefs"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> None:
        target = path or (self.config_dir / "config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Configuration saved to %s", target)

    @classmethod
    def load(cls, path: Path) -> "SEFSConfig":
        data = json.loads(path.read_text())
        cfg = cls(**data)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        root = Path(self.root_directory)
        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.clustering_algorithm not in ("dbscan", "kmeans"):
            raise ValueError("clustering_algorithm must be 'dbscan' or 'kmeans'")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

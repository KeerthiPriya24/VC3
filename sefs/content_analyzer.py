"""Content Analyzer – extract text and generate embeddings for files."""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chardet
import numpy as np

from sefs.config import SUPPORTED_TEXT_EXTENSIONS
from sefs.database import Database
from sefs.models import FileMetadata

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Extracts text, computes SHA-256 checksums, and generates semantic embeddings."""

    def __init__(self, embedding_model: str, db: Database) -> None:
        self._model_name = embedding_model
        self._db = db
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_file(self, file_path: Path) -> Optional[FileMetadata]:
        """Full analysis pipeline for a single file. Returns ``None`` on failure."""
        try:
            checksum = self.compute_checksum(file_path)
            stat = file_path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)

            # Cache check: Only hit if checksum matches AND embedding exists
            cached = self._db.get_file(file_path)
            if cached and cached.checksum == checksum and cached.embedding is not None:
                logger.debug("Cache hit (with embedding) for %s", file_path)
                return cached

            text = self.extract_text(file_path)
            if text is None:
                return None

            if text.strip():
                embedding = self.generate_embedding(text)
            else:
                # Use model's dimension for zero vector
                dim = self._get_model().get_sentence_embedding_dimension()
                embedding = np.zeros(dim, dtype=np.float32)

            ext = file_path.suffix.lower().lstrip(".")
            file_type = "text" if ext in SUPPORTED_TEXT_EXTENSIONS else "pdf"

            meta = FileMetadata(
                file_path=file_path,
                filename=file_path.name,
                file_size=stat.st_size,
                creation_date=datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime),
                modification_date=mtime,
                file_type=file_type,
                checksum=checksum,
                embedding=embedding,
                cluster_id=cached.cluster_id if cached else None,
                confidence_score=cached.confidence_score if cached else 0.0,
                content_summary=text[:3000] if text else None,
            )
            self._db.upsert_file(meta)
            logger.info("Analyzed %s (checksum=%s…)", file_path.name, checksum[:8])
            return meta

        except Exception:
            logger.exception("Failed to analyze %s", file_path)
            return None

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------
    def extract_text(self, file_path: Path) -> Optional[str]:
        ext = file_path.suffix.lower().lstrip(".")
        try:
            if ext == "pdf":
                return self._extract_pdf(file_path)
            return self._extract_text_file(file_path)
        except Exception:
            logger.exception("Text extraction failed for %s", file_path)
            return None

    @staticmethod
    def _extract_pdf(file_path: Path) -> str:
        import PyPDF2

        text_parts: list[str] = []
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    @staticmethod
    def _extract_text_file(file_path: Path) -> str:
        raw = file_path.read_bytes()
        if not raw:
            return ""
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def generate_embedding(self, text: str) -> np.ndarray:
        model = self._get_model()
        embedding: np.ndarray = model.encode(text, show_progress_bar=False)
        return embedding.astype(np.float32)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------
    @staticmethod
    def compute_checksum(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

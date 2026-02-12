"""Semantic Engine – similarity, recursive clustering, and context-aware folder generation."""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sefs.models import ClusterInfo, ClusterUpdate

logger = logging.getLogger(__name__)

# Words excluded from folder names (generic / uninformative)
_NAMING_IGNORE: set[str] = {
    "page", "total", "source", "author", "date", "report", "file",
    "document", "untitled", "img", "image", "scanned", "text", "txt",
    "pdf", "chapter", "section", "figure", "table", "example",
    "introduction", "conclusion", "reference", "references", "abstract",
    "copyright", "rights", "reserved", "press", "published",
    "new", "use", "used", "using", "also", "one", "two", "may",
    "like", "many", "well", "first", "second", "third",
    "however", "therefore", "thus", "hence", "etc",
    "http", "https", "www", "com", "org",
    "key", "features", "include", "based", "main", "following",
    "type", "types", "common", "known", "called", "make", "makes",
    "large", "small", "high", "low", "good", "best", "different",
    "world", "number", "part", "parts", "end", "way", "thing",
    "set", "run", "work", "works", "working", "just", "get",
    # Adverbs and filler words that produce bad folder names
    "widely", "mainly", "typically", "generally", "often", "usually",
    "highly", "easily", "simply", "really", "very", "quite",
    "strong", "basic", "basics", "simple", "modern", "powerful",
    "various", "several", "specific", "certain", "particular",
    "provide", "provides", "support", "supports", "allow", "allows",
    "such", "given", "able", "ability", "can", "could", "will",
    "need", "needs", "want", "take", "takes", "help", "helps",
}


class SemanticEngine:
    """Recursively clusters file embeddings and produces context-aware folder names."""

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        clustering_algorithm: str = "dbscan",
        max_depth: int = 3,
        min_cluster_size_for_split: int = 3,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.clustering_algorithm = clustering_algorithm
        self.max_depth = max_depth
        self.min_cluster_size_for_split = min_cluster_size_for_split

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between two embeddings (scalar)."""
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        return float(cosine_similarity(e1, e2)[0, 0])

    # ------------------------------------------------------------------
    # Clustering (Recursive)
    # ------------------------------------------------------------------
    def cluster_files(
        self,
        file_embeddings: Dict[Path, np.ndarray],
        file_texts: Optional[Dict[Path, str]] = None,
    ) -> Dict[str, ClusterInfo]:
        """Run recursive clustering and return flat {cluster_id: ClusterInfo}."""
        if not file_embeddings:
            return {}

        paths = list(file_embeddings.keys())
        clusters: dict[str, ClusterInfo] = {}

        # Start recursion
        self._recursive_cluster_step(
            paths, file_embeddings, file_texts, clusters, prefix_path=[], depth=0
        )

        return clusters

    def _recursive_cluster_step(
        self,
        current_paths: List[Path],
        all_embeddings: Dict[Path, np.ndarray],
        all_texts: Optional[Dict[Path, str]],
        result_clusters: Dict[str, ClusterInfo],
        prefix_path: List[str],
        depth: int,
    ) -> None:
        """
        1. Cluster current set of files.
        2. Assign noise points to nearest cluster or create new ones.
        3. For large clusters, recurse (if depth < max_depth).
        """
        if not current_paths:
            return

        # Trivial case: single file → create simple folder
        if len(current_paths) == 1:
            self._create_cluster_entry(
                current_paths, all_embeddings, all_texts, result_clusters, prefix_path
            )
            return

        # --- 1. DBSCAN Clustering ---
        matrix = np.array([all_embeddings[p] for p in current_paths])
        sim_matrix = cosine_similarity(matrix)
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
        
        # Adaptive eps: tighten threshold as we go deeper
        current_threshold = min(0.9, self.similarity_threshold + (depth * 0.10))
        eps = 1.0 - current_threshold
        
        # Adaptive min_samples: 2 at depth 0, 2 always (want small clusters too)
        min_samples = 2
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(dist_matrix)

        groups: dict[int, list[Path]] = defaultdict(list)
        noise_points: list[Path] = []

        for idx, label in enumerate(labels):
            p = current_paths[idx]
            if label == -1:
                noise_points.append(p)
            else:
                groups[label].append(p)

        # --- 2. Smart Noise Handling ---
        final_groups: list[list[Path]] = list(groups.values())

        if noise_points and final_groups:
            # Try to merge noise into NEAREST group only if genuinely close
            centroids = []
            for g in final_groups:
                vecs = np.array([all_embeddings[p] for p in g])
                centroids.append(vecs.mean(axis=0))

            remaining_noise = []
            for p in noise_points:
                emb = all_embeddings[p]
                sims = [cosine_similarity(emb.reshape(1, -1), c.reshape(1, -1))[0, 0] for c in centroids]
                best_idx = int(np.argmax(sims))
                # Only merge if similarity is above a meaningful threshold
                merge_threshold = max(0.45, current_threshold - 0.10)
                if sims[best_idx] >= merge_threshold:
                    final_groups[best_idx].append(p)
                else:
                    remaining_noise.append(p)
            noise_points = remaining_noise

        # Secondary clustering: try to cluster remaining noise among themselves
        if len(noise_points) >= 2:
            noise_matrix = np.array([all_embeddings[p] for p in noise_points])
            noise_sim = cosine_similarity(noise_matrix)
            noise_dist = np.clip(1.0 - noise_sim, 0.0, 2.0)
            # Use a looser threshold for noise grouping
            looser_eps = min(0.65, eps + 0.15)
            noise_dbscan = DBSCAN(eps=looser_eps, min_samples=2, metric="precomputed")
            noise_labels = noise_dbscan.fit_predict(noise_dist)

            noise_groups: dict[int, list[Path]] = defaultdict(list)
            still_noise: list[Path] = []
            for idx, label in enumerate(noise_labels):
                if label == -1:
                    still_noise.append(noise_points[idx])
                else:
                    noise_groups[label].append(noise_points[idx])

            for ng in noise_groups.values():
                final_groups.append(ng)

            # True singletons: each gets its own descriptive folder
            for p in still_noise:
                final_groups.append([p])
        elif noise_points:
            # Single noise point → its own folder
            for p in noise_points:
                final_groups.append([p])

        if len(final_groups) == 1:
             # If only 1 group was formed and it contains all files, we didn't split effectively.
             force_stop = True
        else:
             force_stop = False

        # --- 3. Process Groups & Recurse ---
        if force_stop:
            # No useful split → pass through without adding a folder level
            for group in final_groups:
                self._create_cluster_entry(
                    group, all_embeddings, all_texts, result_clusters, prefix_path
                )
            return

        # Name all sibling groups using cross-cluster contrast
        group_names = self._name_groups_cross_cluster(final_groups, all_texts)

        for group, folder_name in zip(final_groups, group_names):
            # Deduplication: Avoid repeating parent-name words
            if prefix_path:
                parent_words = set("/".join(prefix_path).lower().replace("_", " ").split())
                child_words = folder_name.replace("_", " ").split()
                filtered = [w for w in child_words if w.lower() not in parent_words]
                if filtered:
                    folder_name = "_".join(filtered)

            new_prefix = prefix_path + [folder_name]

            should_recurse = (
                depth < self.max_depth
                and len(group) >= self.min_cluster_size_for_split
            )

            if should_recurse:
                self._recursive_cluster_step(
                    group, all_embeddings, all_texts, result_clusters, new_prefix, depth + 1
                )
            else:
                self._create_cluster_entry(
                    group, all_embeddings, all_texts, result_clusters, new_prefix
                )

    def _create_cluster_entry(
        self,
        paths: List[Path],
        all_embeddings: Dict[Path, np.ndarray],
        all_texts: Optional[Dict[Path, str]],
        result_clusters: Dict[str, ClusterInfo],
        prefix_path: List[str],
    ) -> None:
        if not paths:
            return

        # If prefix is empty (e.g. single file at root), generate name now
        if not prefix_path:
            keywords = self._extract_keywords(paths, all_texts, top_n=3)
            name = self._keywords_to_folder_name(keywords) or "Miscellaneous"
            prefix_path = [name]

        # Join path
        folder_display_name = "/".join(prefix_path)
        
        # Calculate stats
        vecs = np.array([all_embeddings[p] for p in paths])
        centroid = vecs.mean(axis=0)
        
        avg_sim = 1.0
        if len(paths) > 1:
            sub = cosine_similarity(vecs)
            np.fill_diagonal(sub, 0)
            avg_sim = float(sub.sum() / (len(paths) * (len(paths) - 1)))

        cluster_id = str(uuid.uuid4())[:8]
        # Store keywords for metadata
        keywords = self._extract_keywords(paths, all_texts, top_n=5)

        result_clusters[cluster_id] = ClusterInfo(
            cluster_id=cluster_id,
            folder_name=folder_display_name, # "A/B/C"
            folder_path=Path(), # Set by sync
            file_paths=paths,
            centroid_embedding=centroid.astype(np.float32),
            average_similarity=avg_sim,
            keywords=keywords,
        )

    # ------------------------------------------------------------------
    # Cross-Cluster Naming
    # ------------------------------------------------------------------
    def _name_groups_cross_cluster(
        self,
        groups: List[List[Path]],
        all_texts: Optional[Dict[Path, str]],
    ) -> List[str]:
        """Name sibling groups using cross-cluster TF-IDF contrast.

        Each group's combined text becomes one 'document'.  TF-IDF then
        naturally surfaces terms that are *distinctive* to each cluster,
        producing accurate topic-based folder names.
        """
        if not groups:
            return []

        # Single group → per-group naming (no contrast possible)
        if len(groups) == 1:
            kw = self._extract_keywords(groups[0], all_texts, top_n=3)
            return [self._keywords_to_folder_name(kw) or "Collection"]

        # Build one merged document per group
        cluster_docs: list[str] = []
        for group in groups:
            doc_parts: list[str] = []
            for p in group:
                # Filename (moderate contribution)
                stem = re.sub(r"[^a-zA-Z ]", " ", p.stem).lower()
                stem_words = [w for w in stem.split()
                              if len(w) > 2 and w not in _NAMING_IGNORE]
                if stem_words:
                    doc_parts.append((" ".join(stem_words) + ". ") * 3)

                # Content (dominant contribution)
                if all_texts and p in all_texts:
                    content = re.sub(
                        r"[^a-zA-Z\s]", " ", all_texts[p][:5000].lower()
                    )
                    content_words = [
                        w for w in content.split()
                        if len(w) > 2 and w not in _NAMING_IGNORE
                    ]
                    doc_parts.append(" ".join(content_words))
            cluster_docs.append(" ".join(doc_parts))

        # Build per-file documents for within-cluster DF calculation
        per_file_docs: list[list[str]] = []  # per_file_docs[group_idx] = list of per-file doc strings
        for group in groups:
            file_docs: list[str] = []
            for p in group:
                parts: list[str] = []
                stem = re.sub(r"[^a-zA-Z ]", " ", p.stem).lower()
                parts.extend(w for w in stem.split() if len(w) > 2 and w not in _NAMING_IGNORE)
                if all_texts and p in all_texts:
                    content = re.sub(r"[^a-zA-Z\s]", " ", all_texts[p][:5000].lower())
                    parts.extend(w for w in content.split() if len(w) > 2 and w not in _NAMING_IGNORE)
                file_docs.append(" ".join(parts))
            per_file_docs.append(file_docs)

        # Cross-cluster TF-IDF: distinctive terms score high
        try:
            tfidf = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=40 * len(groups),
                min_df=1,
                max_df=0.7,
                sublinear_tf=True,
            )
            matrix = tfidf.fit_transform(cluster_docs)
            feature_names = np.array(tfidf.get_feature_names_out())
        except ValueError:
            return self._fallback_group_names(groups, all_texts)

        names: list[str] = []
        for i in range(len(groups)):
            scores = matrix[i].toarray().flatten()

            # Within-cluster document-frequency boost:
            # Terms that appear in MORE files in this cluster are more
            # representative.  Multiply cross-cluster score by the
            # fraction of files containing the term.
            n_files = len(per_file_docs[i])
            if n_files > 1:
                for fi, term in enumerate(feature_names):
                    count = sum(1 for fd in per_file_docs[i] if term in fd)
                    df_ratio = count / n_files           # 0..1
                    boost = 0.4 + 0.6 * df_ratio         # range [0.4, 1.0]
                    scores[fi] *= boost
                    # Prefer bi-grams (multi-word phrases)
                    if " " in term:
                        scores[fi] *= 1.3

            ranked = scores.argsort()[::-1]

            candidates: list[str] = []
            for idx in ranked:
                if scores[idx] <= 0:
                    break
                term = feature_names[idx]
                words = term.split()

                # Skip generic / short / self-dup terms
                if any(w in _NAMING_IGNORE for w in words):
                    continue
                if all(len(w) < 3 for w in words):
                    continue
                if len(words) >= 2 and len(set(words)) == 1:
                    continue

                # Bidirectional subsumption: skip if this term
                # shares all its words with an already-chosen candidate
                # (or vice-versa).  Prefer the *longer* phrase.
                skip = False
                replace_idx = -1
                for ci, ex in enumerate(candidates):
                    ex_words = set(ex.split())
                    term_words = set(words)
                    overlap = ex_words & term_words
                    if overlap == term_words:
                        # term is fully contained in existing → skip
                        skip = True
                        break
                    if overlap == ex_words and len(term_words) > len(ex_words):
                        # existing is fully contained in term → replace with longer
                        replace_idx = ci
                        break
                if skip:
                    continue
                if replace_idx >= 0:
                    candidates[replace_idx] = term
                else:
                    candidates.append(term)
                if len(candidates) >= 2:
                    break

            name = (
                self._keywords_to_folder_name(candidates)
                if candidates else None
            )
            if not name:
                kw = self._extract_keywords(groups[i], all_texts, top_n=3)
                name = self._keywords_to_folder_name(kw) or f"Group_{i + 1}"
            names.append(name)

        # Ensure unique names
        seen: dict[str, int] = {}
        unique: list[str] = []
        for name in names:
            if name in seen:
                seen[name] += 1
                unique.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique.append(name)
        return unique

    def _fallback_group_names(
        self,
        groups: List[List[Path]],
        all_texts: Optional[Dict[Path, str]],
    ) -> List[str]:
        """Per-group naming when cross-cluster TF-IDF fails."""
        out: list[str] = []
        for group in groups:
            kw = self._extract_keywords(group, all_texts, top_n=3)
            out.append(self._keywords_to_folder_name(kw) or "Collection")
        return out

    # ------------------------------------------------------------------
    # Re-clustering
    # ------------------------------------------------------------------
    def recalculate_clusters(
        self,
        file_embeddings: Dict[Path, np.ndarray],
        file_texts: Optional[Dict[Path, str]] = None,
        existing_clusters: Optional[Dict[str, ClusterInfo]] = None,
        denied_moves: Optional[Dict[Path, str]] = None,
    ) -> ClusterUpdate:
        """Full re-cluster via recursion, respecting previously denied folder moves."""
        new_clusters = self.cluster_files(file_embeddings, file_texts)

        removed_ids: list[str] = []
        if existing_clusters:
            removed_ids = [cid for cid in existing_clusters if cid not in new_clusters]

        file_moves: dict[Path, Path] = {}
        for info in new_clusters.values():
            approved_paths = []
            for fp in info.file_paths:
                # folder_name is like "Science/Physics"
                target_folder = info.folder_name
                
                # Check if this move was denied for this file (Robust partial match)
                if denied_moves and fp in denied_moves:
                    denied = str(denied_moves[fp]).strip("/")
                    target = str(target_folder).strip("/")
                    if denied == target or target.startswith(denied + "/") or denied.startswith(target + "/"):
                        logger.debug("Skipping denied move for %s to %s (matches denied: %s)", fp.name, target_folder, denied)
                        continue

                file_moves[fp] = Path(target_folder)
                approved_paths.append(fp)
            
            # Sync the actual file paths in the cluster object
            info.file_paths = approved_paths

        # Explicitly move unclustered files back to root
        all_paths = set(file_embeddings.keys())
        clustered_paths = set()
        for info in new_clusters.values():
            clustered_paths.update(info.file_paths)
        
        unclustered = all_paths - clustered_paths
        for fp in unclustered:
            file_moves[fp] = Path(".")

        return ClusterUpdate(
            new_clusters=new_clusters,
            removed_cluster_ids=removed_ids,
            file_moves=file_moves,
        )

    # ------------------------------------------------------------------
    # Keyword Extraction (per-cluster, for metadata & fallback naming)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_keywords(
        file_paths: List[Path],
        file_texts: Optional[Dict[Path, str]] = None,
        top_n: int = 4,
    ) -> List[str]:
        """Extract representative keywords for a cluster.

        Used for cluster metadata and as a fallback when cross-cluster
        naming is unavailable (single group or TF-IDF failure).
        """
        if not file_paths:
            return []

        # Collect all clean words
        all_words: list[str] = []
        for p in file_paths:
            stem = re.sub(r"[^a-zA-Z ]", " ", p.stem).lower()
            all_words.extend(
                w for w in stem.split() if len(w) > 2 and w not in _NAMING_IGNORE
            )
            if file_texts and p in file_texts:
                content = re.sub(r"[^a-zA-Z\s]", " ", file_texts[p][:5000].lower())
                all_words.extend(
                    w for w in content.split()
                    if len(w) > 2 and w not in _NAMING_IGNORE
                )

        if not all_words:
            return []

        # Single document → TF-IDF IDF is always 0, so use frequency
        if len(file_paths) == 1:
            from collections import Counter
            counts = Counter(all_words)
            return [w for w, _ in counts.most_common(top_n)]

        # Multiple docs → TF-IDF finds shared descriptive terms
        docs: list[str] = []
        for p in file_paths:
            parts: list[str] = []
            stem = re.sub(r"[^a-zA-Z ]", " ", p.stem).lower()
            parts.extend(
                w for w in stem.split() if len(w) > 2 and w not in _NAMING_IGNORE
            )
            if file_texts and p in file_texts:
                content = re.sub(r"[^a-zA-Z\s]", " ", file_texts[p][:5000].lower())
                parts.extend(
                    w for w in content.split()
                    if len(w) > 2 and w not in _NAMING_IGNORE
                )
            docs.append(" ".join(parts))

        try:
            tfidf = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=top_n + 15,
                min_df=1,
                sublinear_tf=True,
            )
            matrix = tfidf.fit_transform(docs)
            feature_names = np.array(tfidf.get_feature_names_out())
            scores = np.asarray(matrix.sum(axis=0)).flatten()

            valid: list[int] = []
            for i, name in enumerate(feature_names):
                words = name.split()
                if any(w in _NAMING_IGNORE for w in words):
                    continue
                if all(len(w) < 3 for w in words):
                    continue
                if len(words) >= 2 and len(set(words)) == 1:
                    continue
                if len(words) >= 2:
                    scores[i] *= 1.4
                valid.append(i)

            if not valid:
                return list(feature_names[scores.argsort()[::-1]][:top_n])

            arr = np.array(valid)
            order = scores[arr].argsort()[::-1]
            return list(feature_names[arr[order]][:top_n])
        except ValueError:
            return []

    @staticmethod
    def _keywords_to_folder_name(keywords: List[str]) -> str:
        if not keywords:
            return ""

        EXTENSIONS = {".txt", ".pdf", ".md", ".log", ".docx", ".xlsx"}
        clean_words: list[str] = []
        for k in keywords:
            # Title-case each word; join multi-word terms with underscore
            parts = k.split()
            word = "_".join(w.capitalize() for w in parts)
            for ext in EXTENSIONS:
                if word.lower().endswith(ext):
                    word = word[: -len(ext)]
            word = word.strip("_ ")
            if word:
                clean_words.append(word)

        final_list: list[str] = []
        for k in clean_words:
            k_lower = k.lower()
            if any(k_lower == ex.lower() for ex in final_list):
                continue
            if any(
                k_lower in ex.lower() and k_lower != ex.lower()
                for ex in final_list
            ):
                continue
            replaced = False
            for i, ex in enumerate(final_list):
                if ex.lower() in k_lower and ex.lower() != k_lower:
                    final_list[i] = k
                    replaced = True
                    break
            if not replaced:
                final_list.append(k)

        seen: set[str] = set()
        unique: list[str] = []
        for x in final_list:
            xl = x.lower()
            if xl not in seen:
                unique.append(x)
                seen.add(xl)

        # Limit by total word count (max 2 words) rather than
        # candidate count, so bi-grams don't cause overly long names.
        chosen: list[str] = []
        total_words = 0
        for x in unique:
            word_count = len(x.split("_"))
            if total_words + word_count > 2 and chosen:
                break
            chosen.append(x)
            total_words += word_count
            if total_words >= 2:
                break

        name = "_".join(chosen)
        name = re.sub(r'[<>:"/\\|?*]', "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_. ") or "Folder"


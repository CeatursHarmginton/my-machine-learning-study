"""
FAISS-based vector store for translation memory.

Stores source-translation pairs with metadata for retrieval-augmented translation.
"""
import json
import numpy as np
import os
from pathlib import Path
from typing import Optional

import faiss

from config import EMBEDDING_MODEL, EMBEDDING_DIM, RAG_TOP_K, PROJECT_DIR


class VectorStore:
    """
    FAISS vector store for translation memory.

    Stores entries with:
    - Vector embedding (for similarity search)
    - Metadata: source text, translation, language pair, chapter, quality score
    """

    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.project_dir = PROJECT_DIR / project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.project_dir / "faiss.index"
        self.meta_path = self.project_dir / "memory_meta.json"

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner Product (cosine after normalization)
        self.entries: list[dict] = []

        # Lazy-load embedding model
        self._embed_model = None

        # Load existing data
        self._load()

    @property
    def embed_model(self):
        """Lazy-load embedding model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"📦 Loading embedding model: {EMBEDDING_MODEL}...")
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
            print("✅ Embedding model loaded.")
        return self._embed_model

    def _load(self):
        """Load existing index and metadata from disk."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)

    def save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add(
        self,
        source_text: str,
        translation: str,
        source_lang: str = "",
        target_lang: str = "",
        chapter: str = "",
        quality_score: float = 0.0,
    ):
        """
        Add a source-translation pair to the store.

        Args:
            source_text: Original text.
            translation: Translated text.
            source_lang: Source language.
            target_lang: Target language.
            chapter: Chapter/section identifier.
            quality_score: Quality score (0-10).
        """
        # Create combined text for embedding (source + translation)
        combined = f"{source_text}\n{translation}"
        vec = self._encode(combined)

        # Add to FAISS
        self.index.add(vec)

        # Store metadata
        self.entries.append({
            "source": source_text[:500],  # Truncate for storage
            "translation": translation[:500],
            "source_lang": source_lang,
            "target_lang": target_lang,
            "chapter": chapter,
            "quality_score": quality_score,
        })

    def retrieve(
        self,
        query: str,
        k: int = RAG_TOP_K,
        min_quality: float = 0.0,
        lang_filter: str = "",
    ) -> list[dict]:
        """
        Retrieve similar translation pairs.

        Args:
            query: Query text (usually the source text to translate).
            k: Number of results to retrieve.
            min_quality: Minimum quality score filter.
            lang_filter: Filter by source language.

        Returns:
            List of entry dicts sorted by relevance.
        """
        if len(self.entries) == 0:
            return []

        # Retrieve more than k to allow for filtering
        fetch_k = min(k * 3, len(self.entries))
        vec = self._encode(query)

        scores, indices = self.index.search(vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue

            entry = self.entries[idx].copy()
            entry["relevance_score"] = float(score)

            # Apply filters
            if min_quality > 0 and entry.get("quality_score", 0) < min_quality:
                continue
            if lang_filter and entry.get("source_lang") != lang_filter:
                continue

            results.append(entry)

            if len(results) >= k:
                break

        return results

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Simple search wrapper for UI."""
        return self.retrieve(query, k=k)

    def delete(self, index: int):
        """
        Delete an entry by index.

        Note: FAISS doesn't support efficient deletion, so we rebuild.
        """
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            self._rebuild_index()

    def clear(self):
        """Clear all entries."""
        self.entries = []
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.save()

    def _rebuild_index(self):
        """Rebuild FAISS index from entries."""
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

        if self.entries:
            texts = [
                f"{e['source']}\n{e['translation']}" for e in self.entries
            ]
            vecs = self._encode_batch(texts)
            self.index.add(vecs)

        self.save()

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to normalized vector."""
        vec = self.embed_model.encode([text], normalize_embeddings=True)
        return np.array(vec).astype("float32")

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts."""
        vecs = self.embed_model.encode(texts, normalize_embeddings=True)
        return np.array(vecs).astype("float32")

    def get_stats(self) -> dict:
        """Get store statistics."""
        lang_counts = {}
        for entry in self.entries:
            lang = entry.get("source_lang", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return {
            "total_entries": len(self.entries),
            "index_size": self.index.ntotal,
            "language_distribution": lang_counts,
            "project": self.project_name,
        }

    def get_entries_display(self) -> list[list]:
        """Get entries formatted for Gradio Dataframe display."""
        rows = []
        for i, entry in enumerate(self.entries):
            rows.append([
                i,
                entry.get("source", "")[:80],
                entry.get("translation", "")[:80],
                entry.get("source_lang", ""),
                entry.get("chapter", ""),
                entry.get("quality_score", 0),
            ])
        return rows

    def __len__(self):
        return len(self.entries)

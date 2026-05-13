"""
Context window manager for maintaining translation continuity.

Combines sliding window context, RAG retrieval, and glossary terms
to provide comprehensive context for each translation chunk.
"""
from config import CONTEXT_WINDOW_SIZE


class ContextManager:
    """
    Manages context for sequential translation.

    Maintains:
    1. Sliding window of previous chunk translations
    2. Integration with vector store for RAG
    3. Integration with glossary for relevant terms
    """

    def __init__(self, window_size: int = CONTEXT_WINDOW_SIZE):
        self.window_size = window_size
        self.history: list[dict] = []  # {"source": ..., "translation": ...}
        self.chapter_info: str = ""

    def add_chunk(self, source: str, translation: str):
        """Add a completed chunk to history."""
        self.history.append({
            "source": source,
            "translation": translation,
        })

    def get_context_text(self) -> str:
        """
        Get the sliding window context text.

        Returns the translations of the last N chunks,
        providing continuity for the current chunk.
        """
        if not self.history:
            return ""

        # Take last N entries
        window = self.history[-self.window_size:]

        # Return only translations (what the reader would see)
        context_parts = []
        for i, entry in enumerate(window):
            context_parts.append(entry["translation"])

        return "\n\n".join(context_parts)

    def get_full_context(
        self,
        current_source: str,
        vector_store=None,
        glossary_manager=None,
    ) -> dict:
        """
        Build comprehensive context for a translation chunk.

        Combines:
        1. Sliding window (previous translations)
        2. RAG retrieved references (if vector store provided)
        3. Relevant glossary entries (if glossary provided)

        Args:
            current_source: The source text about to be translated.
            vector_store: Optional VectorStore for RAG retrieval.
            glossary_manager: Optional GlossaryManager for term lookup.

        Returns:
            Dict with keys: context, retrieved_refs, glossary_entries
        """
        result = {
            "context": self.get_context_text(),
            "retrieved_refs": [],
            "glossary_entries": [],
        }

        # RAG retrieval
        if vector_store and len(vector_store) > 0:
            retrieved = vector_store.retrieve(current_source)
            result["retrieved_refs"] = [
                f"[Source]: {r['source']}\n[Translation]: {r['translation']}"
                for r in retrieved
            ]

        # Glossary entries relevant to current text
        if glossary_manager and len(glossary_manager) > 0:
            result["glossary_entries"] = glossary_manager.get_relevant_entries(
                current_source
            )

            # Also add all entries if text is the first chunk (for awareness)
            if len(self.history) == 0 and len(glossary_manager) <= 50:
                # For first chunk, include all glossary entries
                all_entries = glossary_manager.entries
                # Merge without duplicates
                existing_sources = {e["source"] for e in result["glossary_entries"]}
                for e in all_entries:
                    if e["source"] not in existing_sources:
                        result["glossary_entries"].append(e)

        return result

    def reset(self):
        """Reset context for a new translation session."""
        self.history = []
        self.chapter_info = ""

    def get_progress_info(self) -> dict:
        """Get translation progress information."""
        total_source_chars = sum(len(h["source"]) for h in self.history)
        total_trans_chars = sum(len(h["translation"]) for h in self.history)

        return {
            "chunks_translated": len(self.history),
            "total_source_chars": total_source_chars,
            "total_translated_chars": total_trans_chars,
            "expansion_ratio": (
                total_trans_chars / total_source_chars
                if total_source_chars > 0 else 0
            ),
        }

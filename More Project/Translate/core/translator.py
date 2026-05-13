"""
Main translation orchestrator.

Coordinates all pipeline components: detection → chunking → retrieval → translation → post-processing.
"""
import time
from typing import Generator, Callable, Optional

from core.detector import detect_language
from core.chunker import split_text, estimate_chunks
from core.prompts import build_translation_prompt
from retrieval.vector_store import VectorStore
from retrieval.glossary import GlossaryManager
from retrieval.context import ContextManager
from models.base import BaseModel
from config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


class TranslationPipeline:
    """
    Full translation pipeline with RAG, glossary, and context management.

    Usage:
        pipeline = TranslationPipeline(model, project_name="my_novel")
        result = pipeline.translate(text, target_lang="Vietnamese")
    """

    def __init__(
        self,
        model: BaseModel,
        project_name: str = "default",
    ):
        self.model = model
        self.project_name = project_name

        # Initialize components
        self.vector_store = VectorStore(project_name)
        self.glossary = GlossaryManager(project_name)
        self.context_manager = ContextManager()

        # Translation state
        self._stop_requested = False

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "Vietnamese",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        save_to_memory: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Translate text through the full pipeline.

        Args:
            text: Source text to translate.
            source_lang: Source language ("auto" for detection).
            target_lang: Target language.
            chunk_size: Max characters per chunk.
            chunk_overlap: Character overlap between chunks.
            temperature: Model temperature.
            max_new_tokens: Max tokens to generate per chunk.
            save_to_memory: Whether to save translations to vector store.
            progress_callback: Optional callback(current, total, message).

        Returns:
            Complete translated text.
        """
        self._stop_requested = False

        # Step 1: Detect language
        if source_lang == "auto":
            source_lang = detect_language(text)
            if progress_callback:
                progress_callback(0, 1, f"🔍 Detected language: {source_lang}")

        # Step 2: Split into chunks
        chunks = split_text(text, max_chunk_size=chunk_size, overlap=chunk_overlap)
        total_chunks = len(chunks)

        if progress_callback:
            progress_callback(0, total_chunks, f"📝 Split into {total_chunks} chunks")

        # Step 3: Reset context for new translation
        self.context_manager.reset()

        # Step 4: Translate each chunk
        results = []
        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": DEFAULT_TOP_P,
        }

        for i, chunk in enumerate(chunks):
            if self._stop_requested:
                if progress_callback:
                    progress_callback(i, total_chunks, "⛔ Translation stopped by user")
                break

            if progress_callback:
                progress_callback(i, total_chunks, f"🔄 Translating chunk {i + 1}/{total_chunks}...")

            # Get full context (sliding window + RAG + glossary)
            ctx = self.context_manager.get_full_context(
                chunk, self.vector_store, self.glossary
            )

            # Build prompt
            messages = build_translation_prompt(
                text=chunk,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_entries=ctx["glossary_entries"],
                context=ctx["context"],
                retrieved_refs=ctx["retrieved_refs"],
            )

            # Generate translation
            translation = self.model.generate(messages, **gen_kwargs)

            # Post-process: check glossary enforcement
            violations = self.glossary.check_enforcement(chunk, translation)
            if violations and progress_callback:
                violation_msg = "; ".join(v["issue"] for v in violations[:3])
                progress_callback(
                    i, total_chunks,
                    f"⚠️ Glossary violations in chunk {i + 1}: {violation_msg}"
                )

            # Update context
            self.context_manager.add_chunk(chunk, translation)

            # Save to memory
            if save_to_memory:
                self.vector_store.add(
                    source_text=chunk,
                    translation=translation,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

            results.append(translation)

        # Save vector store
        if save_to_memory:
            self.vector_store.save()

        if progress_callback:
            progress_callback(total_chunks, total_chunks, "✅ Translation complete!")

        return "\n\n".join(results)

    def translate_stream(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "Vietnamese",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        save_to_memory: bool = True,
    ) -> Generator[tuple[str, str], None, None]:
        """
        Stream translation, yielding (accumulated_result, status_message) tuples.

        Useful for Gradio streaming UI.
        """
        self._stop_requested = False

        # Detect language
        if source_lang == "auto":
            source_lang = detect_language(text)
            yield "", f"🔍 Detected language: {source_lang}"

        # Split into chunks
        chunks = split_text(text, max_chunk_size=chunk_size, overlap=chunk_overlap)
        total_chunks = len(chunks)
        yield "", f"📝 Split into {total_chunks} chunks"

        # Reset context
        self.context_manager.reset()

        results = []
        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": DEFAULT_TOP_P,
        }

        for i, chunk in enumerate(chunks):
            if self._stop_requested:
                yield "\n\n".join(results), "⛔ Stopped by user"
                return

            yield "\n\n".join(results), f"🔄 Chunk {i + 1}/{total_chunks}..."

            # Get context
            ctx = self.context_manager.get_full_context(
                chunk, self.vector_store, self.glossary
            )

            # Build prompt
            messages = build_translation_prompt(
                text=chunk,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_entries=ctx["glossary_entries"],
                context=ctx["context"],
                retrieved_refs=ctx["retrieved_refs"],
            )

            # Stream generation
            chunk_result = ""
            try:
                for token in self.model.stream_generate(messages, **gen_kwargs):
                    chunk_result += token
                    current = "\n\n".join(results + [chunk_result])
                    yield current, f"🔄 Chunk {i + 1}/{total_chunks} — generating..."
            except Exception:
                # Fallback to non-streaming
                chunk_result = self.model.generate(messages, **gen_kwargs)

            # Update context + memory
            self.context_manager.add_chunk(chunk, chunk_result)
            if save_to_memory:
                self.vector_store.add(
                    source_text=chunk,
                    translation=chunk_result,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

            results.append(chunk_result)

        # Save
        if save_to_memory:
            self.vector_store.save()

        yield "\n\n".join(results), f"✅ Done! Translated {total_chunks} chunks."

    def stop(self):
        """Request translation to stop after current chunk."""
        self._stop_requested = True

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "project": self.project_name,
            "model": self.model.get_info() if self.model else None,
            "vector_store": self.vector_store.get_stats(),
            "glossary": self.glossary.get_stats(),
            "context": self.context_manager.get_progress_info(),
        }

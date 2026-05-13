"""
Smart text chunking that respects paragraph and sentence boundaries.
Handles CJK (Chinese, Korean, Japanese) and Latin text.
"""
import re
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


# Sentence-ending patterns for different scripts
CJK_SENTENCE_ENDERS = re.compile(r'[。！？…‥]+["」』】）〉》〕〗〙〛]*')
LATIN_SENTENCE_ENDERS = re.compile(r'[.!?…]+["\']?\s')
KOREAN_SENTENCE_ENDERS = re.compile(r'[。.!?！？…]+\s*')


def _is_cjk_dominant(text: str) -> bool:
    """Check if text is primarily CJK characters."""
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af]', text))
    total = len(text.strip())
    return total > 0 and (cjk / total) > 0.3


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling both CJK and Latin."""
    if _is_cjk_dominant(text):
        # For CJK: split on CJK sentence enders
        parts = CJK_SENTENCE_ENDERS.split(text)
        enders = CJK_SENTENCE_ENDERS.findall(text)

        sentences = []
        for i, part in enumerate(parts):
            s = part.strip()
            if s:
                if i < len(enders):
                    s += enders[i]
                sentences.append(s)
        return sentences if sentences else [text]
    else:
        # For Latin: split on sentence-ending punctuation
        parts = re.split(r'(?<=[.!?…])\s+', text)
        return [p.strip() for p in parts if p.strip()]


def split_text(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    respect_paragraphs: bool = True,
) -> list[str]:
    """
    Split text into translation-ready chunks.

    Strategy:
    1. Split by paragraphs (double newline)
    2. If a paragraph exceeds max_chunk_size, split by sentences
    3. Merge small consecutive paragraphs into one chunk
    4. Add overlap between chunks for context continuity

    Args:
        text: Full text to split.
        max_chunk_size: Maximum character count per chunk.
        overlap: Number of characters to overlap between chunks.
        respect_paragraphs: If True, try not to break paragraphs.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Step 1: Split into paragraphs
    if respect_paragraphs:
        paragraphs = re.split(r'\n\s*\n', text)
    else:
        paragraphs = [text]

    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Step 2: Break large paragraphs into sentences
    segments = []
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            segments.append(para)
        else:
            # Split by sentences
            sentences = _split_sentences(para)
            segments.extend(sentences)

    # Step 3: Merge segments into chunks respecting max_chunk_size
    chunks = []
    current_chunk = ""

    for segment in segments:
        # If adding this segment would exceed limit, finalize current chunk
        if current_chunk and len(current_chunk) + len(segment) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                # Don't break mid-word
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = overlap_text + "\n" + segment
            else:
                current_chunk = segment
        else:
            if current_chunk:
                current_chunk += "\n" + segment
            else:
                current_chunk = segment

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def estimate_chunks(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """Estimate how many chunks text will be split into."""
    if not text:
        return 0
    return max(1, len(text) // max_chunk_size)

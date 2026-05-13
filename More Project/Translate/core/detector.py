"""
Language detection with multi-sample voting and charset fallback.
"""
import re
from langdetect import detect, DetectorFactory
from config import LANG_DETECT_MAP

# Deterministic detection
DetectorFactory.seed = 0


def _detect_by_charset(text: str) -> str | None:
    """Fallback detection using Unicode character ranges."""
    # Count characters in each script range
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))  # CJK Unified
    hangul_count = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', text))  # Hangul
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    viet_count = len(re.findall(
        r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
        r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]',
        text
    ))

    total = cjk_count + hangul_count + latin_count + viet_count
    if total == 0:
        return None

    # Vietnamese has Latin chars + diacritics
    if viet_count > 0 and viet_count / max(latin_count, 1) > 0.05:
        return "Vietnamese"
    if cjk_count > hangul_count and cjk_count > latin_count:
        return "Chinese"
    if hangul_count > cjk_count and hangul_count > latin_count:
        return "Korean"
    if latin_count > cjk_count and latin_count > hangul_count:
        return "English"

    return None


def detect_language(text: str, num_samples: int = 5) -> str:
    """
    Detect language using multi-sample voting with charset fallback.

    Takes multiple segments of the text, detects language for each,
    and returns the majority vote. Falls back to charset-based detection
    if langdetect fails.

    Args:
        text: Input text to detect language of.
        num_samples: Number of text segments to sample for voting.

    Returns:
        Language name string (e.g., "Chinese", "English", "Korean", "Vietnamese").
    """
    if not text or not text.strip():
        return "Unknown"

    # Clean text
    clean = text.strip()

    # Try charset detection first for CJK (it's more reliable for short texts)
    charset_result = _detect_by_charset(clean[:2000])

    # Multi-sample voting with langdetect
    votes = []
    chunk_size = max(200, len(clean) // num_samples)

    for i in range(num_samples):
        start = i * chunk_size
        segment = clean[start:start + chunk_size]
        if len(segment) < 20:
            continue
        try:
            lang_code = detect(segment)
            lang_name = LANG_DETECT_MAP.get(lang_code, None)
            if lang_name:
                votes.append(lang_name)
        except Exception:
            continue

    if not votes:
        return charset_result or "Unknown"

    # Majority vote
    from collections import Counter
    counter = Counter(votes)
    majority_lang, majority_count = counter.most_common(1)[0]

    # If majority is strong (>50%), use it; otherwise fallback to charset
    if majority_count / len(votes) > 0.5:
        return majority_lang

    return charset_result or majority_lang


def get_lang_code(lang_name: str) -> str:
    """Convert language name back to code."""
    name_to_code = {v: k for k, v in LANG_DETECT_MAP.items()}
    # Prefer shorter codes
    if lang_name == "Chinese":
        return "zh"
    return name_to_code.get(lang_name, "unknown")

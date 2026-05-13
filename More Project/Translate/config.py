"""
Central configuration for the Novel Translation Pipeline.
"""
import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
GLOSSARY_DIR = DATA_DIR / "glossaries"
PROJECT_DIR = DATA_DIR / "projects"

# Ensure directories exist
for d in [DATA_DIR, GLOSSARY_DIR, PROJECT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Model Defaults
# ============================================================
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_QUANTIZATION = "4bit"  # "4bit", "8bit", "none"
DEFAULT_TORCH_DTYPE = "float16"

# API Model Defaults
DEFAULT_API_PROVIDER = "gemini"  # "openai", "gemini", "anthropic"
DEFAULT_API_MODEL = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-sonnet-4-20250514",
}

# ============================================================
# Translation Defaults
# ============================================================
DEFAULT_TARGET_LANG = "Vietnamese"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9

# Chunking
DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 50  # characters overlap between chunks

# ============================================================
# RAG / Retrieval
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RAG_TOP_K = 3
CONTEXT_WINDOW_SIZE = 3  # number of previous chunks to keep

# ============================================================
# Language Maps
# ============================================================
SUPPORTED_SOURCE_LANGS = {
    "auto": "Auto Detect",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
}

SUPPORTED_TARGET_LANGS = {
    "vi": "Vietnamese",
}

LANG_DETECT_MAP = {
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ko": "Korean",
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
}

# ============================================================
# UI Theme
# ============================================================
APP_TITLE = "📚 Novel Translator Pro"
APP_DESCRIPTION = "Professional novel translation pipeline with RAG, Glossary & Multi-model support"
GRADIO_THEME = "soft"

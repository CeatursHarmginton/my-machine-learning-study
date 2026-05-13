"""
Prompt templates for novel translation.

Uses clear delimiters and structured formatting for optimal LLM performance.
"""


def build_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    glossary_entries: list[dict] | None = None,
    context: str = "",
    retrieved_refs: list[str] | None = None,
    style_notes: str = "",
) -> list[dict]:
    """
    Build a structured translation prompt as chat messages.

    Args:
        text: Source text to translate.
        source_lang: Source language name.
        target_lang: Target language name.
        glossary_entries: List of dicts with 'source', 'target', 'note' keys.
        context: Previous translation context (sliding window).
        retrieved_refs: Retrieved reference translations from RAG.
        style_notes: Additional style/tone instructions.

    Returns:
        List of message dicts for chat-format models.
    """
    # Build system prompt
    system = _build_system_prompt(source_lang, target_lang, style_notes)

    # Build user prompt with all context
    user = _build_user_prompt(
        text, source_lang, target_lang,
        glossary_entries, context, retrieved_refs
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_system_prompt(source_lang: str, target_lang: str, style_notes: str = "") -> str:
    """Build the system-level instruction prompt."""
    prompt = f"""You are a professional literary translator specializing in translating novels from {source_lang} to {target_lang}.

### Core Translation Principles
1. **Naturalness**: The translation must read naturally and fluently in {target_lang}, as if originally written in that language.
2. **Fidelity**: Preserve the original meaning, tone, emotions, and narrative voice accurately.
3. **Literary Style**: Use literary {target_lang} appropriate for novel reading — avoid machine-translation artifacts, overly literal phrasing, or unnatural word order.
4. **Character Voice**: Maintain each character's distinct speaking style, personality, and speech patterns.
5. **Cultural Adaptation**: Adapt cultural references appropriately while preserving the original atmosphere.

### Rules
- Translate ONLY the text provided. Do NOT add explanations, notes, or commentary.
- Do NOT translate proper nouns unless specified in the glossary.
- Preserve paragraph structure and formatting.
- Use the glossary terms EXACTLY as specified — these are mandatory translations.
- Use provided context and references to maintain consistency with previous translations."""

    if style_notes:
        prompt += f"\n\n### Additional Style Notes\n{style_notes}"

    return prompt


def _build_user_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    glossary_entries: list[dict] | None = None,
    context: str = "",
    retrieved_refs: list[str] | None = None,
) -> str:
    """Build the user message with all contextual information."""
    parts = []

    # Glossary section
    if glossary_entries:
        glossary_lines = []
        for entry in glossary_entries:
            line = f"• {entry['source']} → {entry['target']}"
            if entry.get('note'):
                line += f" ({entry['note']})"
            glossary_lines.append(line)

        parts.append(
            "### 📖 Glossary (MANDATORY — use these translations exactly)\n"
            + "\n".join(glossary_lines)
        )

    # Context section (previous translations for continuity)
    if context:
        parts.append(
            "### 📝 Previous Translation (for context continuity)\n"
            f"```\n{context}\n```"
        )

    # Retrieved references
    if retrieved_refs:
        refs = "\n---\n".join(retrieved_refs)
        parts.append(
            "### 📚 Reference Translations (similar passages translated before)\n"
            f"```\n{refs}\n```"
        )

    # The actual text to translate
    parts.append(
        f"### ✍️ Text to Translate ({source_lang} → {target_lang})\n"
        f"```\n{text}\n```"
    )

    parts.append(f"### 🎯 Translation ({target_lang} only, no explanations)")

    return "\n\n".join(parts)


def build_glossary_suggest_prompt(text: str, source_lang: str, existing_glossary: list[dict] | None = None) -> list[dict]:
    """
    Build a prompt to auto-suggest glossary entries from source text.

    Useful for identifying character names, place names, skills, and
    other terms that should be consistently translated.
    """
    existing = ""
    if existing_glossary:
        existing = "\n### Existing Glossary (do not duplicate)\n"
        existing += "\n".join(f"• {e['source']} → {e['target']}" for e in existing_glossary)

    system = f"""You are a literary translation assistant. Your task is to identify important terms in {source_lang} text that should have consistent translations.

Focus on:
1. **Character names** (人名 / 이름)
2. **Place names** (地名 / 장소)
3. **Special terms** — skills, titles, organizations, magical items, etc.
4. **Recurring phrases** that need consistent translation

Output ONLY a JSON array of objects with keys: "source", "target", "category", "note"
Categories: "character", "place", "skill", "title", "item", "organization", "other"

Example output:
[
  {{"source": "林凡", "target": "Lâm Phàm", "category": "character", "note": "Main character"}},
  {{"source": "斗气", "target": "Đấu Khí", "category": "skill", "note": "Combat energy system"}}
]"""

    user = f"""{existing}

### Source Text ({source_lang})
```
{text[:3000]}
```

Identify all important terms that need consistent translation. Return ONLY the JSON array."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_quality_eval_prompt(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    glossary_entries: list[dict] | None = None,
) -> list[dict]:
    """
    Build a prompt for LLM-as-judge quality evaluation.

    Returns evaluation criteria scores and suggestions.
    """
    glossary_text = ""
    if glossary_entries:
        glossary_text = "\n### Glossary to check against\n"
        glossary_text += "\n".join(f"• {e['source']} → {e['target']}" for e in glossary_entries)

    system = """You are a professional translation quality evaluator. 
Score the translation on these criteria (1-10 each):
1. **Fluency**: Does it read naturally?
2. **Accuracy**: Is the meaning preserved?
3. **Style**: Is the literary style appropriate?
4. **Glossary Adherence**: Are glossary terms used correctly?
5. **Consistency**: Is terminology consistent?

Output a JSON object:
{
  "fluency": <score>,
  "accuracy": <score>,
  "style": <score>,
  "glossary_adherence": <score>,
  "consistency": <score>,
  "overall": <average>,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"]
}"""

    user = f"""### Source ({source_lang})
```
{source_text}
```

### Translation ({target_lang})
```
{translation}
```
{glossary_text}

Evaluate the translation quality. Return ONLY the JSON object."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

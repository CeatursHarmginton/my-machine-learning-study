"""
📚 Novel Translator Pro — Gradio Web Application

Professional novel translation pipeline with RAG, Glossary & Multi-model support.
Designed to run on Google Colab with GPU.
"""
import gradio as gr
import json
import os
import tempfile
from pathlib import Path

from config import (
    APP_TITLE, APP_DESCRIPTION, DEFAULT_LOCAL_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    SUPPORTED_SOURCE_LANGS, SUPPORTED_TARGET_LANGS,
    GLOSSARY_DIR, PROJECT_DIR,
)
from core.translator import TranslationPipeline
from core.detector import detect_language
from core.chunker import estimate_chunks
from retrieval.glossary import GlossaryManager, list_projects
from retrieval.vector_store import VectorStore
from utils.file_handler import read_file, write_file

# ============================================================
# Global State
# ============================================================
_pipeline = None
_current_model = None
_current_project = "default"


def _get_or_create_project_list():
    """Get list of projects, ensuring 'default' exists."""
    projects = list_projects()
    if "default" not in projects:
        projects.insert(0, "default")
    return projects


# ============================================================
# Model Management
# ============================================================
def load_local_model(model_name, quantization):
    """Load a local HuggingFace model."""
    global _pipeline, _current_model
    try:
        from models.local_model import LocalModel
        model = LocalModel(model_name=model_name, quantization=quantization)
        model.load()
        _current_model = model
        _pipeline = TranslationPipeline(model, project_name=_current_project)
        return f"✅ Loaded: {model_name} ({quantization})"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def load_api_model(provider, model_name, api_key):
    """Load an API-based model."""
    global _pipeline, _current_model
    try:
        from models.api_model import APIModel
        model = APIModel(provider=provider, model_name=model_name or None, api_key=api_key or None)
        if not model.is_available():
            return f"❌ API key not set for {provider}"
        _current_model = model
        _pipeline = TranslationPipeline(model, project_name=_current_project)
        return f"✅ Connected: {provider}/{model.model_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def unload_model():
    """Unload current model to free memory."""
    global _pipeline, _current_model
    if _current_model and hasattr(_current_model, 'unload'):
        _current_model.unload()
    _current_model = None
    _pipeline = None
    return "🗑️ Model unloaded."


def switch_project(project_name):
    """Switch to a different project."""
    global _pipeline, _current_project
    _current_project = project_name
    if _current_model:
        _pipeline = TranslationPipeline(_current_model, project_name=project_name)
    return f"📂 Project: {project_name}"


# ============================================================
# Translation Tab Handlers
# ============================================================
def handle_file_upload(file):
    """Read uploaded file and return its content."""
    if file is None:
        return "", ""
    try:
        content = read_file(file.name)
        lang = detect_language(content)
        chunks = estimate_chunks(content, DEFAULT_CHUNK_SIZE)
        info = f"📄 {Path(file.name).name} | 🌐 {lang} | 📝 {len(content)} chars | 📦 ~{chunks} chunks"
        return content, info
    except Exception as e:
        return "", f"❌ Error reading file: {e}"


def handle_translate(source_text, source_lang, target_lang, temperature, max_tokens, chunk_size, save_memory):
    """Run translation with streaming output."""
    if not _pipeline:
        yield "", "❌ No model loaded! Go to Settings tab to load a model first."
        return
    if not source_text or not source_text.strip():
        yield "", "❌ No text to translate."
        return

    src = source_lang if source_lang != "Auto Detect" else "auto"
    tgt = target_lang

    try:
        for result, status in _pipeline.translate_stream(
            text=source_text,
            source_lang=src,
            target_lang=tgt,
            temperature=temperature,
            max_new_tokens=max_tokens,
            chunk_size=chunk_size,
            save_to_memory=save_memory,
        ):
            yield result, status
    except Exception as e:
        yield "", f"❌ Translation error: {str(e)}"


def handle_stop():
    """Stop ongoing translation."""
    if _pipeline:
        _pipeline.stop()
    return "⛔ Stop requested..."


def handle_download(translation_text, output_format):
    """Create downloadable file from translation."""
    if not translation_text:
        return None
    try:
        ext = "txt" if output_format == "txt" else "docx"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", prefix="translation_")
        write_file(tmp.name, translation_text, format=output_format)
        return tmp.name
    except Exception as e:
        return None


# ============================================================
# Glossary Tab Handlers
# ============================================================
def glossary_load(project_name):
    """Load and display glossary for a project."""
    gm = GlossaryManager(project_name)
    data = gm.get_display_data()
    stats = gm.get_stats()
    info = f"📖 {stats['total_entries']} entries"
    if stats['categories']:
        cats = ", ".join(f"{k}: {v}" for k, v in stats['categories'].items())
        info += f" | {cats}"
    return data, info


def glossary_add(project, source, target, category, note):
    """Add a glossary entry."""
    gm = GlossaryManager(project)
    if gm.add(source, target, category, note):
        data = gm.get_display_data()
        return data, f"✅ Added: {source} → {target}"
    return gm.get_display_data(), f"⚠️ '{source}' already exists"


def glossary_delete(project, index):
    """Delete glossary entry by index."""
    try:
        idx = int(index)
        gm = GlossaryManager(project)
        gm.delete(idx)
        return gm.get_display_data(), f"🗑️ Deleted entry #{idx}"
    except (ValueError, IndexError) as e:
        return GlossaryManager(project).get_display_data(), f"❌ Error: {e}"


def glossary_export(project, fmt):
    """Export glossary to file."""
    gm = GlossaryManager(project)
    try:
        ext = "json" if fmt == "JSON" else "csv"
        content = gm.export_json() if fmt == "JSON" else gm.export_csv()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", prefix=f"glossary_{project}_", mode='w', encoding='utf-8')
        tmp.write(content)
        tmp.close()
        return tmp.name, f"📥 Exported {len(gm)} entries as {fmt}"
    except Exception as e:
        return None, f"❌ Export error: {e}"


def glossary_import(project, file):
    """Import glossary from uploaded file."""
    if file is None:
        return GlossaryManager(project).get_display_data(), "❌ No file selected"
    gm = GlossaryManager(project)
    count = gm.import_from_file(file.name)
    return gm.get_display_data(), f"📤 Imported {count} new entries"


def glossary_auto_suggest(project, source_text):
    """Auto-suggest glossary entries using the loaded model."""
    if not _current_model:
        return "❌ No model loaded. Load a model in Settings first."
    if not source_text:
        return "❌ No text provided."

    from core.prompts import build_glossary_suggest_prompt
    gm = GlossaryManager(project)

    messages = build_glossary_suggest_prompt(
        source_text, detect_language(source_text), gm.entries
    )
    try:
        result = _current_model.generate(messages, temperature=0.2, max_new_tokens=1024)
        # Try to parse JSON from result
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
            lines = []
            for s in suggestions:
                lines.append(f"• {s.get('source','')} → {s.get('target','')} [{s.get('category','other')}] {s.get('note','')}")
            return "🔮 Suggestions:\n" + "\n".join(lines) + "\n\n(Add these manually or copy the JSON below)\n\n```json\n" + json.dumps(suggestions, ensure_ascii=False, indent=2) + "\n```"
        return f"🔮 Raw model output:\n{result}"
    except Exception as e:
        return f"❌ Error: {e}"


# ============================================================
# Memory Tab Handlers
# ============================================================
def memory_load(project):
    """Load memory entries for display."""
    vs = VectorStore(project)
    data = vs.get_entries_display()
    stats = vs.get_stats()
    info = f"🧠 {stats['total_entries']} entries"
    if stats['language_distribution']:
        langs = ", ".join(f"{k}: {v}" for k, v in stats['language_distribution'].items())
        info += f" | {langs}"
    return data, info


def memory_search(project, query):
    """Search translation memory."""
    vs = VectorStore(project)
    results = vs.search(query, k=5)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"**Source**: {r['source'][:100]}...\n**Translation**: {r['translation'][:100]}...\n**Score**: {r.get('relevance_score', 0):.3f}\n---")
    return "\n".join(lines)


def memory_clear(project):
    """Clear all memory for a project."""
    vs = VectorStore(project)
    vs.clear()
    return [], "🗑️ Memory cleared."


# ============================================================
# Build Gradio UI
# ============================================================
def create_app():
    """Create the Gradio application."""

    # Custom CSS for premium look
    css = """
    .gradio-container { max-width: 1400px !important; }
    .main-header { text-align: center; margin-bottom: 1em; }
    .main-header h1 { font-size: 2.2em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .main-header p { color: #888; font-size: 1.1em; }
    .status-box { padding: 8px 12px; border-radius: 8px; font-weight: 500; }
    .source-box textarea, .output-box textarea { font-size: 15px !important; line-height: 1.7 !important; font-family: 'Noto Sans', 'Noto Sans CJK', sans-serif !important; }
    """

    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=css,
    ) as app:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>📚 Novel Translator Pro</h1>
            <p>Professional translation pipeline with RAG, Glossary & Multi-model support</p>
        </div>
        """)

        with gr.Tabs() as tabs:
            # =============================================
            # TAB 1: TRANSLATION
            # =============================================
            with gr.Tab("🔄 Translate", id="translate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        file_upload = gr.File(label="📁 Upload File (.txt, .docx)", file_types=[".txt", ".docx", ".doc"])
                        file_info = gr.Textbox(label="File Info", interactive=False, max_lines=1)
                    with gr.Column(scale=2):
                        with gr.Row():
                            src_lang = gr.Dropdown(
                                choices=list(SUPPORTED_SOURCE_LANGS.values()),
                                value="Auto Detect", label="Source Language",
                            )
                            tgt_lang = gr.Dropdown(
                                choices=list(SUPPORTED_TARGET_LANGS.values()),
                                value="Vietnamese", label="Target Language",
                            )

                with gr.Row(equal_height=True):
                    with gr.Column():
                        source_text = gr.Textbox(
                            label="📝 Source Text",
                            placeholder="Paste your text here or upload a file...",
                            lines=18, max_lines=30,
                            elem_classes=["source-box"],
                        )
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="📖 Translation",
                            lines=18, max_lines=30,
                            interactive=False,
                            elem_classes=["output-box"],
                        )

                with gr.Row():
                    translate_btn = gr.Button("🚀 Translate", variant="primary", size="lg")
                    stop_btn = gr.Button("⛔ Stop", variant="stop")
                    status_text = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=3)

                with gr.Accordion("⚙️ Translation Parameters", open=False):
                    with gr.Row():
                        temp_slider = gr.Slider(0, 1, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                        max_tokens_slider = gr.Slider(256, 4096, value=DEFAULT_MAX_NEW_TOKENS, step=128, label="Max Tokens")
                        chunk_size_slider = gr.Slider(200, 2000, value=DEFAULT_CHUNK_SIZE, step=50, label="Chunk Size")
                    with gr.Row():
                        save_memory_check = gr.Checkbox(value=True, label="💾 Save to Translation Memory")
                        output_format = gr.Radio(["txt", "docx"], value="txt", label="Download Format")
                        download_btn = gr.Button("📥 Download Translation")
                        download_file = gr.File(label="Download", visible=True)

                # Wire up translation tab events
                file_upload.change(handle_file_upload, [file_upload], [source_text, file_info])

                translate_btn.click(
                    handle_translate,
                    [source_text, src_lang, tgt_lang, temp_slider, max_tokens_slider, chunk_size_slider, save_memory_check],
                    [output_text, status_text],
                )
                stop_btn.click(handle_stop, [], [status_text])
                download_btn.click(handle_download, [output_text, output_format], [download_file])

            # =============================================
            # TAB 2: GLOSSARY
            # =============================================
            with gr.Tab("📖 Glossary", id="glossary"):
                with gr.Row():
                    gloss_project = gr.Dropdown(
                        choices=_get_or_create_project_list(),
                        value=_current_project, label="Project", scale=2,
                    )
                    gloss_refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    gloss_info = gr.Textbox(label="Info", interactive=False, scale=3)

                gloss_table = gr.Dataframe(
                    headers=["#", "Source", "Target", "Category", "Note"],
                    datatype=["number", "str", "str", "str", "str"],
                    col_count=(5, "fixed"),
                    label="Glossary Entries",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### ➕ Add Entry")
                        with gr.Row():
                            add_source = gr.Textbox(label="Source Term", placeholder="林凡")
                            add_target = gr.Textbox(label="Target Term", placeholder="Lâm Phàm")
                        with gr.Row():
                            add_category = gr.Dropdown(
                                choices=GlossaryManager.CATEGORIES,
                                value="character", label="Category",
                            )
                            add_note = gr.Textbox(label="Note", placeholder="Main character")
                        add_btn = gr.Button("➕ Add", variant="primary")
                        add_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

                    with gr.Column(scale=1):
                        gr.Markdown("### 🗑️ Delete Entry")
                        del_index = gr.Number(label="Entry Index (#)", precision=0)
                        del_btn = gr.Button("🗑️ Delete", variant="stop")

                with gr.Accordion("📤 Import / Export", open=False):
                    with gr.Row():
                        export_fmt = gr.Radio(["JSON", "CSV"], value="JSON", label="Export Format")
                        export_btn = gr.Button("📥 Export")
                        export_file = gr.File(label="Download")
                        export_status = gr.Textbox(label="Status", interactive=False, max_lines=1)
                    with gr.Row():
                        import_file = gr.File(label="📤 Import File (.json or .csv)", file_types=[".json", ".csv"])
                        import_btn = gr.Button("📤 Import")

                with gr.Accordion("🔮 Auto-Suggest (AI)", open=False):
                    suggest_text = gr.Textbox(label="Paste source text to analyze", lines=5, placeholder="Paste source text here...")
                    suggest_btn = gr.Button("🔮 Suggest Glossary Entries", variant="secondary")
                    suggest_output = gr.Markdown(label="Suggestions")

                # Wire glossary events
                gloss_refresh_btn.click(glossary_load, [gloss_project], [gloss_table, gloss_info])
                gloss_project.change(glossary_load, [gloss_project], [gloss_table, gloss_info])
                add_btn.click(glossary_add, [gloss_project, add_source, add_target, add_category, add_note], [gloss_table, add_status])
                del_btn.click(glossary_delete, [gloss_project, del_index], [gloss_table, add_status])
                export_btn.click(glossary_export, [gloss_project, export_fmt], [export_file, export_status])
                import_btn.click(glossary_import, [gloss_project, import_file], [gloss_table, gloss_info])
                suggest_btn.click(glossary_auto_suggest, [gloss_project, suggest_text], [suggest_output])

            # =============================================
            # TAB 3: TRANSLATION MEMORY
            # =============================================
            with gr.Tab("🧠 Memory", id="memory"):
                with gr.Row():
                    mem_project = gr.Dropdown(
                        choices=_get_or_create_project_list(),
                        value=_current_project, label="Project", scale=2,
                    )
                    mem_refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    mem_info = gr.Textbox(label="Info", interactive=False, scale=3)

                mem_table = gr.Dataframe(
                    headers=["#", "Source", "Translation", "Lang", "Chapter", "Quality"],
                    datatype=["number", "str", "str", "str", "str", "number"],
                    col_count=(6, "fixed"),
                    label="Translation Memory",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    mem_query = gr.Textbox(label="🔍 Search Memory", placeholder="Enter search query...", scale=3)
                    mem_search_btn = gr.Button("🔍 Search", scale=1)
                mem_search_results = gr.Markdown(label="Search Results")

                with gr.Row():
                    mem_clear_btn = gr.Button("🗑️ Clear All Memory", variant="stop")

                # Wire memory events
                mem_refresh_btn.click(memory_load, [mem_project], [mem_table, mem_info])
                mem_project.change(memory_load, [mem_project], [mem_table, mem_info])
                mem_search_btn.click(memory_search, [mem_project, mem_query], [mem_search_results])
                mem_clear_btn.click(memory_clear, [mem_project], [mem_table, mem_info])

            # =============================================
            # TAB 4: SETTINGS
            # =============================================
            with gr.Tab("⚙️ Settings", id="settings"):
                gr.Markdown("## 🤖 Model Configuration")
                model_status = gr.Textbox(label="Model Status", interactive=False, value="No model loaded")

                with gr.Accordion("🖥️ Local Model (GPU)", open=True):
                    gr.Markdown("*Primary — runs on Colab GPU. Requires NVIDIA GPU with ≥6GB VRAM.*")
                    with gr.Row():
                        local_model_name = gr.Textbox(
                            label="Model Name", value=DEFAULT_LOCAL_MODEL,
                            placeholder="e.g., Qwen/Qwen2.5-7B-Instruct",
                        )
                        local_quant = gr.Radio(["4bit", "8bit", "none"], value="4bit", label="Quantization")
                    local_load_btn = gr.Button("📦 Load Local Model", variant="primary")

                with gr.Accordion("☁️ API Model (Cloud)", open=False):
                    gr.Markdown("*Secondary — use when GPU unavailable or for higher quality.*")
                    with gr.Row():
                        api_provider = gr.Dropdown(
                            choices=["openai", "gemini", "anthropic"],
                            value="gemini", label="Provider",
                        )
                        api_model_name = gr.Textbox(label="Model Name (optional)", placeholder="Leave blank for default")
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="Enter API key...")
                    api_load_btn = gr.Button("☁️ Connect API Model", variant="secondary")

                unload_btn = gr.Button("🗑️ Unload Model", variant="stop")

                gr.Markdown("---")
                gr.Markdown("## 📂 Project Management")
                with gr.Row():
                    new_project_name = gr.Textbox(label="New Project Name", placeholder="my_novel")
                    create_project_btn = gr.Button("➕ Create Project")
                    project_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

                def create_project(name):
                    if not name or not name.strip():
                        return "❌ Enter a project name", gr.update(), gr.update()
                    name = name.strip().replace(" ", "_")
                    GlossaryManager(name).save()
                    projects = _get_or_create_project_list()
                    return f"✅ Created project: {name}", gr.update(choices=projects), gr.update(choices=projects)

                # Wire settings events
                local_load_btn.click(load_local_model, [local_model_name, local_quant], [model_status])
                api_load_btn.click(load_api_model, [api_provider, api_model_name, api_key_input], [model_status])
                unload_btn.click(unload_model, [], [model_status])
                create_project_btn.click(create_project, [new_project_name], [project_status, gloss_project, mem_project])

        # Load initial data
        app.load(lambda: glossary_load(_current_project), outputs=[gloss_table, gloss_info])
        app.load(lambda: memory_load(_current_project), outputs=[mem_table, mem_info])

    return app


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=True,  # Creates public URL (useful for Colab)
        server_name="0.0.0.0",
        show_error=True,
    )

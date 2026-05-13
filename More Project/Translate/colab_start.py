"""
📚 Novel Translator Pro — Google Colab Quick Start

Run this script on Google Colab to start the translation app.
It will:
1. Install dependencies
2. Launch the Gradio web UI with a public shareable link

Usage on Colab:
    !git clone <your-repo> or upload the files
    %cd Translate
    !python colab_start.py
"""
import subprocess
import sys

def install_deps():
    """Install all required dependencies."""
    print("📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "gradio>=5.0",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "faiss-cpu",
        "sentence-transformers",
        "langdetect",
        "openai",
        "google-generativeai",
        "anthropic",
        "python-dotenv",
        "chardet",
        "python-docx",
    ])
    print("✅ All dependencies installed!")

if __name__ == "__main__":
    install_deps()

    print("\n🚀 Starting Novel Translator Pro...")
    print("=" * 50)

    from app import create_app
    app = create_app()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True,
    )

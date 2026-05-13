"""
Local HuggingFace model provider with quantization support.

Primary model provider — runs on Colab GPU.
"""
import torch
from typing import Generator
from models.base import BaseModel


class LocalModel(BaseModel):
    """
    HuggingFace local model with 4-bit/8-bit quantization.

    Designed to run on Google Colab GPU (T4/A100).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quantization: str = "4bit",
        torch_dtype: str = "float16",
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.torch_dtype_str = torch_dtype
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer into GPU memory."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"📦 Loading model: {self.model_name} ({self.quantization})...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float16)

        # Quantization config
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if self.quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )

        self._loaded = True
        print(f"✅ Model loaded: {self.model_name}")

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate translation using local model."""
        if not self._loaded:
            self.load()

        # Build prompt from messages using chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join(
                f"{'### ' + m['role'].upper() + ':\\n' + m['content']}"
                for m in messages
            )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generation params
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.3) > 0,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return result.strip()

    def stream_generate(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """Stream generation using TextIteratorStreamer."""
        if not self._loaded:
            self.load()

        from transformers import TextIteratorStreamer
        import threading

        # Build prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(
                f"{'### ' + m['role'].upper() + ':\\n' + m['content']}"
                for m in messages
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("temperature", 0.3) > 0,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "streamer": streamer,
        }

        # Run generation in separate thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "provider": "local",
            "quantization": self.quantization,
            "loaded": self._loaded,
            "device": str(next(self.model.parameters()).device) if self._loaded else "not loaded",
            "description": f"Local model: {self.model_name} ({self.quantization})",
        }

    def is_available(self) -> bool:
        return self._loaded or torch.cuda.is_available()

    def unload(self):
        """Free GPU memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🗑️ Model unloaded, GPU memory freed.")

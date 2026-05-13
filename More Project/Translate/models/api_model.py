"""
API-based model providers: OpenAI, Google Gemini, Anthropic.

Secondary/fallback model provider.
"""
import os
import time
from typing import Generator
from models.base import BaseModel


class APIModel(BaseModel):
    """
    API-based model supporting OpenAI, Google Gemini, and Anthropic.

    Handles retry logic, rate limiting, and cost estimation.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.max_retries = max_retries
        self.client = None

        # Default model names per provider
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-sonnet-4-20250514",
        }
        self.model_name = model_name or default_models.get(self.provider, "")

        if self.api_key:
            self._init_client()

    def _get_api_key(self) -> str | None:
        """Get API key from environment variables."""
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        key_name = env_keys.get(self.provider, "")
        return os.environ.get(key_name)

    def set_api_key(self, api_key: str):
        """Set API key and reinitialize client."""
        self.api_key = api_key
        self._init_client()

    def _init_client(self):
        """Initialize the API client."""
        if not self.api_key:
            return

        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)

            elif self.provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)

            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)

        except Exception as e:
            print(f"⚠️ Failed to initialize {self.provider} client: {e}")
            self.client = None

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate translation via API with retry logic."""
        if not self.client:
            raise RuntimeError(
                f"{self.provider} API client not initialized. "
                f"Please set API key first."
            )

        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return self._generate_openai(messages, **kwargs)
                elif self.provider == "gemini":
                    return self._generate_gemini(messages, **kwargs)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(messages, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"⚠️ API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} attempts: {e}")

    def _generate_openai(self, messages: list[dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            top_p=kwargs.get("top_p", 0.9),
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(self, messages: list[dict], **kwargs) -> str:
        # Convert messages to Gemini format
        # System message goes to system_instruction, rest to contents
        system_msg = ""
        contents = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["content"]]})

        # Recreate model with system instruction if provided
        if system_msg:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_msg,
            )
        else:
            model = self.client

        response = model.generate_content(
            contents,
            generation_config={
                "temperature": kwargs.get("temperature", 0.3),
                "max_output_tokens": kwargs.get("max_new_tokens", 1024),
                "top_p": kwargs.get("top_p", 0.9),
            },
        )
        return response.text.strip()

    def _generate_anthropic(self, messages: list[dict], **kwargs) -> str:
        # Extract system message
        system_msg = ""
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered.append(msg)

        response = self.client.messages.create(
            model=self.model_name,
            system=system_msg,
            messages=filtered,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_new_tokens", 1024),
        )
        return response.content[0].text.strip()

    def stream_generate(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """Stream generation via API."""
        if not self.client:
            raise RuntimeError(f"{self.provider} API client not initialized.")

        if self.provider == "openai":
            yield from self._stream_openai(messages, **kwargs)
        elif self.provider == "gemini":
            yield from self._stream_gemini(messages, **kwargs)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(messages, **kwargs)

    def _stream_openai(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_gemini(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        system_msg = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["content"]]})

        if system_msg:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_msg,
            )
        else:
            model = self.client

        response = model.generate_content(
            contents,
            generation_config={
                "temperature": kwargs.get("temperature", 0.3),
                "max_output_tokens": kwargs.get("max_new_tokens", 1024),
            },
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _stream_anthropic(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        system_msg = ""
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                filtered.append(msg)

        with self.client.messages.stream(
            model=self.model_name,
            system=system_msg,
            messages=filtered,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_new_tokens", 1024),
        ) as stream:
            for text in stream.text_stream:
                yield text

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "provider": self.provider,
            "has_api_key": bool(self.api_key),
            "client_ready": bool(self.client),
            "description": f"API model: {self.provider}/{self.model_name}",
        }

    def is_available(self) -> bool:
        return bool(self.client)

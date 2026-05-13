"""
Abstract base class for all model providers.
"""
from abc import ABC, abstractmethod
from typing import Generator


class BaseModel(ABC):
    """Abstract interface for LLM model providers."""

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """
        Generate a response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text string.
        """
        pass

    @abstractmethod
    def stream_generate(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """
        Stream-generate a response token by token.

        Args:
            messages: List of message dicts.
            **kwargs: Generation parameters.

        Yields:
            Text tokens as they are generated.
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dict with keys: name, provider, max_tokens, description.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is currently available and ready."""
        pass

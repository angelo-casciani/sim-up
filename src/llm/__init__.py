"""LLM package for hypothesis generation."""

from .hypothesis_generator import HypothesisGenerator
from .ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    SUPPORTED_MODELS,
    OllamaLLM,
    OllamaUnavailableError,
    check_ollama_model,
    model_slug,
)

__all__ = [
    "HypothesisGenerator",
    "OllamaLLM",
    "OllamaUnavailableError",
    "check_ollama_model",
    "model_slug",
    "DEFAULT_MODEL",
    "DEFAULT_NUM_CTX",
    "SUPPORTED_MODELS",
]

"""Compatibility facade for the original Ollama helper."""

from src.bootstrap import get_services
from src.errors import ExternalServiceError


def polish_text(text: str, model: str | None = None) -> str | None:
    client = get_services().ollama
    if model is not None and model != client.model:
        raise ValueError("Per-call model overrides are no longer supported; configure the model centrally")
    try:
        return client.polish(text)
    except ExternalServiceError:
        return None

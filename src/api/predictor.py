"""Compatibility facade for callers using the original predictor module."""

from src.bootstrap import get_services


def predict(text: str) -> dict:
    result = get_services().classification.classify(text, include_signals=True)
    return {
        **result["scores"],
        "observed_signals": result["observed_signals"],
        "truncated": result["truncated"],
        "analyzed_tokens": result["analyzed_tokens"],
        "model_version": result["model_version"],
        "warnings": result["warnings"],
    }

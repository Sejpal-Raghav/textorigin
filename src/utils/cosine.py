"""Compatibility facade for semantic similarity."""

from src.inference.similarity import SimilarityEngine

_engine = SimilarityEngine()


def compute_cosine_similarity(text1: str, text2: str) -> float:
    return _engine.compare(text1, text2)

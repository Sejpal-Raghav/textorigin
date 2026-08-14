"""Compatibility facade for the humanization service."""

from src.bootstrap import get_services
from src.services.humanization import heuristic_rewrite


def humanize(
    text: str,
    use_llm: bool = True,
    similarity_threshold: float | None = None,
    seed: int | None = None,
) -> dict:
    return get_services().humanization.humanize(
        text,
        use_llm=use_llm,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )

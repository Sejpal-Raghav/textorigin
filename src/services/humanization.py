from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from src.errors import ExternalServiceError
from src.features.extractor import extract_features
from src.inference.similarity import SimilarityEngine
from src.services.capacity import InferenceCapacity
from src.services.rewrite import (
    RewriteStrength,
    RewriteStyle,
    protected_spans_preserved,
    rewrite_document,
)

LOGGER = logging.getLogger(__name__)

class Polisher(Protocol):
    def polish(
        self,
        text: str,
        *,
        style: RewriteStyle,
        strength: RewriteStrength,
        api_key: str | None = None,
    ) -> str: ...


def heuristic_rewrite(
    text: str,
    seed: int | None = None,
    *,
    style: RewriteStyle = "natural",
    strength: RewriteStrength = "light",
) -> str:
    """Compatibility facade for the deterministic document rewriter."""
    del seed
    return rewrite_document(text, style=style, strength=strength)


class HumanizationService:
    def __init__(
        self,
        similarity: SimilarityEngine,
        polisher: Polisher,
        capacity: InferenceCapacity,
        default_threshold: float,
        minimum_threshold: float,
    ):
        self.similarity = similarity
        self.polisher = polisher
        self.capacity = capacity
        self.default_threshold = default_threshold
        self.minimum_threshold = minimum_threshold

    def humanize(
        self,
        text: str,
        use_llm: bool = True,
        similarity_threshold: float | None = None,
        seed: int | None = None,
        style: RewriteStyle = "natural",
        strength: RewriteStrength = "light",
        api_key: str | None = None,
    ) -> dict:
        threshold = self.default_threshold if similarity_threshold is None else similarity_threshold
        if not self.minimum_threshold <= threshold <= 1:
            raise ValueError(
                f"similarity_threshold must be between {self.minimum_threshold} and 1"
            )

        started = perf_counter()
        warnings: list[str] = []
        with self.capacity.claim():
            original_metrics = extract_features(text)
            heuristic_text = heuristic_rewrite(
                text, seed=seed, style=style, strength=strength
            )
            heuristic_metrics = extract_features(heuristic_text)
            heuristic_similarity = self.similarity.compare(text, heuristic_text)

            accepted_text = text
            status = "unchanged"
            if heuristic_similarity >= threshold:
                accepted_text = heuristic_text
                status = "accepted_heuristic"
            else:
                warnings.append("heuristic_similarity_below_threshold")

            llm_similarity = None
            llm_metrics = None
            if use_llm:
                try:
                    llm_text = self.polisher.polish(
                        accepted_text,
                        style=style,
                        strength=strength,
                        api_key=api_key,
                    )
                    if not protected_spans_preserved(text, llm_text):
                        raise ExternalServiceError("Polisher changed protected content")
                    llm_similarity = self.similarity.compare(text, llm_text)
                    if llm_similarity >= threshold:
                        accepted_text = llm_text
                        llm_metrics = extract_features(llm_text)
                        status = "accepted_llm"
                    else:
                        warnings.append("llm_similarity_below_threshold")
                except ExternalServiceError:
                    warnings.append("polisher_unavailable")

        LOGGER.info(
            "humanization_completed duration_seconds=%.3f status=%s warning_count=%d",
            perf_counter() - started,
            status,
            len(warnings),
        )
        return {
            "original_text": text,
            "humanized_text": accepted_text,
            "status": status,
            "warnings": warnings,
            "similarity_threshold": threshold,
            "style": style,
            "strength": strength,
            "original_metrics": original_metrics,
            "post_heuristic_metrics": heuristic_metrics,
            "heuristic_similarity": heuristic_similarity,
            "llm_similarity": llm_similarity,
            "post_llm_metrics": llm_metrics,
        }

from __future__ import annotations

import logging
from time import perf_counter
from typing import Protocol

from src.errors import ExternalServiceError
from src.services.capacity import InferenceCapacity
from src.services.document import document_chunks
from src.services.execution import ExecutionContext, context_or_default
from src.services.rewrite import (
    RewriteStrength,
    RewriteStyle,
    protected_spans_preserved,
    document_structure_preserved,
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
        context: ExecutionContext | None = None,
    ) -> str: ...


class SimilarityValidator(Protocol):
    def compare(self, text1: str, text2: str) -> float: ...


def extract_features(text: str, context=None) -> dict:
    """Lazy import keeps the CLI and API schemas free of model imports."""
    from src.features.extractor import extract_features as implementation

    return implementation(text, context=context)


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
        similarity: SimilarityValidator,
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
        context: ExecutionContext | None = None,
    ) -> dict:
        threshold = self.default_threshold if similarity_threshold is None else similarity_threshold
        if not self.minimum_threshold <= threshold <= 1:
            raise ValueError(
                f"similarity_threshold must be between {self.minimum_threshold} and 1"
            )

        context = context_or_default(context)
        context.progress("extracting_original_metrics")
        started = perf_counter()
        warnings: list[str] = []
        with self.capacity.claim(context):
            original_metrics = extract_features(text, context=context)
            accepted_chunks: list[str] = []
            heuristic_chunks: list[str] = []
            heuristic_scores: list[tuple[float, int]] = []
            llm_scores: list[tuple[float, int]] = []
            accepted_heuristic = False
            accepted_llm = False
            chunks = document_chunks(text)
            for index, original_chunk in enumerate(chunks, start=1):
                context.progress("rewriting", index - 1, len(chunks))
                if len(original_chunk) > 4_000:
                    heuristic_chunks.append(original_chunk)
                    accepted_chunks.append(original_chunk)
                    warnings.append(f"chunk_{index}_oversized_protected_region_unchanged")
                    continue
                if not original_chunk.strip():
                    heuristic_chunks.append(original_chunk)
                    accepted_chunks.append(original_chunk)
                    continue
                heuristic_chunk = heuristic_rewrite(
                    original_chunk, seed=seed, style=style, strength=strength
                )
                heuristic_chunks.append(heuristic_chunk)
                heuristic_score = self.similarity.compare(
                    original_chunk, heuristic_chunk
                )
                weight = max(len(original_chunk), 1)
                heuristic_scores.append((heuristic_score, weight))
                accepted_chunk = original_chunk
                if heuristic_score >= threshold and protected_spans_preserved(
                    original_chunk, heuristic_chunk
                ) and document_structure_preserved(original_chunk, heuristic_chunk):
                    accepted_chunk = heuristic_chunk
                    accepted_heuristic = accepted_heuristic or heuristic_chunk != original_chunk
                else:
                    warnings.append(f"chunk_{index}_heuristic_rejected")

                if use_llm:
                    context.check()
                    try:
                        llm_text = self.polisher.polish(
                            accepted_chunk,
                            style=style,
                            strength=strength,
                            api_key=api_key,
                            context=context,
                        )
                        context.check()
                        if not protected_spans_preserved(original_chunk, llm_text) or not document_structure_preserved(original_chunk, llm_text):
                            raise ExternalServiceError("Polisher changed protected content")
                        llm_score = self.similarity.compare(original_chunk, llm_text)
                        llm_scores.append((llm_score, weight))
                        if llm_score >= threshold:
                            accepted_chunk = llm_text
                            accepted_llm = accepted_llm or llm_text != original_chunk
                        else:
                            warnings.append(f"chunk_{index}_llm_similarity_below_threshold")
                    except ExternalServiceError:
                        warnings.append(f"chunk_{index}_polisher_unavailable")
                accepted_chunks.append(accepted_chunk)

            context.progress("rewriting", len(chunks), len(chunks))
            context.progress("extracting_result_metrics")
            heuristic_text = "".join(heuristic_chunks)
            heuristic_metrics = extract_features(heuristic_text, context=context)
            heuristic_similarity = _weighted_mean(heuristic_scores)
            accepted_text = "".join(accepted_chunks)
            status = (
                "accepted_llm"
                if accepted_llm
                else "accepted_heuristic"
                if accepted_heuristic
                else "unchanged"
            )
            llm_similarity = _weighted_mean(llm_scores) if llm_scores else None
            final_metrics = original_metrics if accepted_text == text else extract_features(accepted_text, context=context)
            llm_metrics = final_metrics if accepted_llm else None
            context.check()

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
            "final_metrics": final_metrics,
            "post_heuristic_metrics": heuristic_metrics,
            "heuristic_similarity": heuristic_similarity,
            "llm_similarity": llm_similarity,
            "post_llm_metrics": llm_metrics,
            "chunks_processed": len(chunks),
            "complete": True,
        }


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    if not values:
        return 1.0
    total_weight = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / total_weight

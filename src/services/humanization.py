from __future__ import annotations

import hashlib
import logging
import random
import re
from time import perf_counter

from src.errors import ExternalServiceError
from src.features.extractor import extract_features
from src.inference.similarity import SimilarityEngine
from src.integrations.ollama import OllamaClient
from src.services.capacity import InferenceCapacity

LOGGER = logging.getLogger(__name__)

PHRASE_REPLACEMENTS = {
    "delve into": "examine",
    "it's worth noting": "",
    "it is worth noting": "",
    "it's important to note": "",
    "it is important to note": "",
    "in conclusion": "",
    "in summary": "",
    "to summarize": "",
    "furthermore": "also",
    "moreover": "also",
    "in the realm of": "in",
    "as previously mentioned": "",
    "utilize": "use",
    "facilitate": "help",
    "moving forward": "later",
    "at the end of the day": "ultimately",
}


def _default_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def heuristic_rewrite(text: str, seed: int | None = None) -> str:
    """Apply conservative, deterministic style edits without synonym guessing."""
    for phrase, replacement in sorted(
        PHRASE_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = re.sub(
            rf"\b{re.escape(phrase)}\b", replacement, text, flags=re.IGNORECASE
        )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text).strip()
    text = re.sub(r"^([a-z])", lambda match: match.group(1).upper(), text)

    rng = random.Random(seed if seed is not None else _default_seed(text))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    rewritten: list[str] = []
    index = 0
    while index < len(sentences):
        sentence = sentences[index]
        if (
            index + 1 < len(sentences)
            and len(sentence.split()) < 10
            and len(sentences[index + 1].split()) < 10
            and rng.random() < 0.35
        ):
            joiner = rng.choice((" and ", " but ", "; "))
            joined = sentence.rstrip(".!?") + joiner + sentences[index + 1][0].lower() + sentences[index + 1][1:]
            rewritten.append(joined)
            index += 2
            continue
        if len(sentence.split()) > 25:
            for separator in (", and ", "; "):
                if separator in sentence:
                    first, second = sentence.split(separator, 1)
                    rewritten.extend((first.rstrip() + ".", second[:1].upper() + second[1:]))
                    break
            else:
                rewritten.append(sentence)
        else:
            rewritten.append(sentence)
        index += 1
    return " ".join(rewritten)


class HumanizationService:
    def __init__(
        self,
        similarity: SimilarityEngine,
        ollama: OllamaClient,
        capacity: InferenceCapacity,
        default_threshold: float,
        minimum_threshold: float,
    ):
        self.similarity = similarity
        self.ollama = ollama
        self.capacity = capacity
        self.default_threshold = default_threshold
        self.minimum_threshold = minimum_threshold

    def humanize(
        self,
        text: str,
        use_llm: bool = True,
        similarity_threshold: float | None = None,
        seed: int | None = None,
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
            heuristic_text = heuristic_rewrite(text, seed=seed)
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
                    llm_text = self.ollama.polish(accepted_text)
                    llm_similarity = self.similarity.compare(text, llm_text)
                    if llm_similarity >= threshold:
                        accepted_text = llm_text
                        llm_metrics = extract_features(llm_text)
                        status = "accepted_llm"
                    else:
                        warnings.append("llm_similarity_below_threshold")
                except ExternalServiceError:
                    warnings.append("ollama_unavailable")

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
            "original_metrics": original_metrics,
            "post_heuristic_metrics": heuristic_metrics,
            "heuristic_similarity": heuristic_similarity,
            "llm_similarity": llm_similarity,
            "post_llm_metrics": llm_metrics,
        }

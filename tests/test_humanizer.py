from unittest.mock import MagicMock, patch

import pytest

from src.errors import ExternalServiceError
from src.services.capacity import InferenceCapacity
from src.services.humanization import HumanizationService, heuristic_rewrite

METRICS = {
    "perplexity": 100.0,
    "burstiness": 0.5,
    "entropy": 4.5,
    "ai_phrase_rate": 0.0,
    "avg_sentence_length": 15.0,
    "structural_regularity": 0.5,
}


def make_service(similarities, polished="This is polished text."):
    similarity = MagicMock()
    similarity.compare.side_effect = similarities
    ollama = MagicMock()
    ollama.polish.return_value = polished
    service = HumanizationService(
        similarity,
        ollama,
        InferenceCapacity(1, 0.1),
        default_threshold=0.85,
        minimum_threshold=0.80,
    )
    return service, ollama


def test_heuristic_rewrite_is_deterministic():
    text = "It is worth noting that this is short. This is brief."
    assert heuristic_rewrite(text) == heuristic_rewrite(text)


@patch("src.services.humanization.extract_features", return_value=METRICS)
def test_llm_candidate_is_accepted_when_safe(_extract):
    service, _ollama = make_service([0.90, 0.92])
    result = service.humanize("It is worth noting that this text needs revision.")
    assert result["status"] == "accepted_llm"
    assert result["humanized_text"] == "This is polished text."
    assert result["warnings"] == []


@patch("src.services.humanization.extract_features", return_value=METRICS)
def test_unsafe_heuristic_falls_back_to_original(_extract):
    service, ollama = make_service([0.40])
    result = service.humanize("Original meaning.", use_llm=False)
    assert result["status"] == "unchanged"
    assert result["humanized_text"] == "Original meaning."
    assert "heuristic_similarity_below_threshold" in result["warnings"]
    ollama.polish.assert_not_called()


@patch("src.services.humanization.extract_features", return_value=METRICS)
def test_ollama_failure_returns_safe_fallback(_extract):
    service, ollama = make_service([0.90])
    ollama.polish.side_effect = ExternalServiceError("offline")
    result = service.humanize("Original text.")
    assert result["status"] == "accepted_heuristic"
    assert "ollama_unavailable" in result["warnings"]


def test_threshold_cannot_be_weakened_below_server_minimum():
    service, _ollama = make_service([])
    with pytest.raises(ValueError, match="between 0.8 and 1"):
        service.humanize("Original text.", similarity_threshold=0.5)

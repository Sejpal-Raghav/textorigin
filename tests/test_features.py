import pytest

from src.features.extractor import (
    compute_avg_sentence_length,
    compute_burstiness,
    compute_ai_phrase_rate,
    compute_entropy,
    compute_structural_regularity,
)

HUMAN_TEXT = (
    "The dog ran to the park. It was a Thursday. I watched, half-asleep, "
    "from the bench. Somehow, the morning felt longer than it should have."
)
AI_TEXT = (
    "Delve into the nuanced landscape of modern AI technologies. It is worth noting "
    "that this comprehensive overview facilitates a multifaceted understanding. "
    "Furthermore, leveraging these transformative paradigms is invaluable moving forward."
)


def test_burstiness_is_float():
    result = compute_burstiness(HUMAN_TEXT)
    assert isinstance(result, float)
    assert result >= 0.0


def test_burstiness_nonzero_for_varied_text():
    assert compute_burstiness(HUMAN_TEXT) > 0.0


def test_entropy_positive():
    result = compute_entropy(HUMAN_TEXT)
    assert result > 0.0


def test_entropy_empty_string():
    assert compute_entropy("") == 0.0


def test_ai_phrase_rate_higher_for_ai_text():
    human_rate = compute_ai_phrase_rate(HUMAN_TEXT)
    ai_rate = compute_ai_phrase_rate(AI_TEXT)
    assert ai_rate > human_rate


def test_ai_phrase_rate_is_per_100_words():
    assert compute_ai_phrase_rate("Delve into this topic with care.") == pytest.approx(100 / 6)


def test_overlapping_ai_phrases_are_not_double_counted():
    assert compute_ai_phrase_rate("Delve into this topic.") == pytest.approx(25.0)


def test_avg_sentence_length_positive():
    result = compute_avg_sentence_length(HUMAN_TEXT)
    assert result > 0.0


def test_structural_regularity_between_0_and_1():
    result = compute_structural_regularity(HUMAN_TEXT)
    assert 0.0 <= result <= 1.0


def test_structural_regularity_uniform_text():
    uniform = "Word word word. Word word word. Word word word."
    assert compute_structural_regularity(uniform) > 0.8

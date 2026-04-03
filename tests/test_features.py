from src.features.extractor import (
    compute_avg_sentence_length,
    compute_burstiness,
    compute_ai_phrase_ratio,
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


def test_ai_phrase_ratio_higher_for_ai_text():
    human_ratio = compute_ai_phrase_ratio(HUMAN_TEXT)
    ai_ratio = compute_ai_phrase_ratio(AI_TEXT)
    assert ai_ratio > human_ratio


def test_ai_phrase_ratio_between_0_and_1():
    ratio = compute_ai_phrase_ratio(AI_TEXT)
    assert 0.0 <= ratio <= 1.0


def test_avg_sentence_length_positive():
    result = compute_avg_sentence_length(HUMAN_TEXT)
    assert result > 0.0


def test_structural_regularity_between_0_and_1():
    result = compute_structural_regularity(HUMAN_TEXT)
    assert 0.0 <= result <= 1.0


def test_structural_regularity_uniform_text():
    uniform = "Word word word. Word word word. Word word word."
    assert compute_structural_regularity(uniform) > 0.8

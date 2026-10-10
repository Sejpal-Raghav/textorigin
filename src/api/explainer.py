import json
import logging
from functools import lru_cache

from src.config import get_settings
from src.features.extractor import extract_features

LOGGER = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "perplexity": {"low": 50, "high": 200},
    "burstiness": {"low": 0.3, "high": 0.7},
    "ai_phrase_rate": {"low": 0.5, "high": 2.0},
    "structural_regularity": {"low": 0.4, "high": 0.7},
    "entropy": {"low": 3.5, "high": 5.0},
    "avg_sentence_length": {"low": 10, "high": 25},
}


@lru_cache(maxsize=1)
def get_thresholds() -> dict:
    path = get_settings().feature_thresholds_path
    if not path.exists():
        return DEFAULT_THRESHOLDS
    try:
        with path.open("r", encoding="utf-8") as handle:
            configured = json.load(handle)["thresholds"]
        for feature, bounds in DEFAULT_THRESHOLDS.items():
            if feature not in configured or not {"low", "high"}.issubset(configured[feature]):
                raise ValueError(f"missing bounds for {feature}")
            if configured[feature]["low"] > configured[feature]["high"]:
                raise ValueError(f"invalid bound order for {feature}")
        return configured
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        LOGGER.warning("feature_thresholds_invalid path=%s error=%s", path, exc)
        return DEFAULT_THRESHOLDS


def _low_distance(value: float, threshold: float) -> float:
    return max(0.0, (threshold - value) / max(abs(threshold), 1.0))


def _high_distance(value: float, threshold: float) -> float:
    return max(0.0, (value - threshold) / max(abs(threshold), 1.0))


def get_observed_signals(text: str, n: int = 3, context=None) -> list[str]:
    feats = extract_features(text, context=context)
    thresholds = get_thresholds()
    signals = []

    checks = (
        ("perplexity", "low", "low perplexity", _low_distance),
        ("perplexity", "high", "high perplexity", _high_distance),
        ("burstiness", "low", "low burstiness", _low_distance),
        ("burstiness", "high", "high burstiness", _high_distance),
        ("ai_phrase_rate", "high", "high AI phrase rate", _high_distance),
        (
            "structural_regularity",
            "high",
            "high structural regularity",
            _high_distance,
        ),
        ("entropy", "low", "low token entropy", _low_distance),
        (
            "avg_sentence_length",
            "high",
            "long average sentence length",
            _high_distance,
        ),
    )
    for feature, direction, label, distance in checks:
        value = feats[feature]
        threshold = thresholds[feature][direction]
        triggered = value < threshold if direction == "low" else value > threshold
        if triggered:
            signals.append((label, distance(value, threshold)))

    signals.sort(key=lambda item: item[1], reverse=True)
    return [label for label, _ in signals[:n]]


def get_top_features(text: str, n: int = 3) -> list[str]:
    """Compatibility alias; these are observations, not model explanations."""
    return get_observed_signals(text, n=n)

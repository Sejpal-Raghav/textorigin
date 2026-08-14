import math
import re
from collections import Counter

import numpy as np
import torch

from src.features.phrases import AI_PHRASES
from src.inference.perplexity import get_perplexity_model

_AI_PHRASE_PATTERN = re.compile(
    "|".join(
        rf"\b{re.escape(phrase)}\b"
        for phrase in sorted(AI_PHRASES, key=len, reverse=True)
    ),
    flags=re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"\b[\w']+\b")


def compute_perplexity(text: str, max_length: int = 512) -> float:
    model, tokenizer = get_perplexity_model()
    device = next(model.parameters()).device

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return math.exp(20)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return math.exp(min(outputs.loss.item(), 20))  # cap at e^20 to avoid overflow


def compute_burstiness(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    mean = np.mean(lengths)
    std = np.std(lengths)
    if mean == 0:
        return 0.0
    # Coefficient of variation — human text tends to be more bursty
    return float(std / mean)


def compute_entropy(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def compute_ai_phrase_rate(text: str) -> float:
    """Return non-overlapping AI phrase occurrences per 100 words."""
    word_count = len(_WORD_PATTERN.findall(text))
    if word_count == 0:
        return 0.0
    hits = sum(1 for _ in _AI_PHRASE_PATTERN.finditer(text))
    return float(hits * 100 / word_count)


def compute_ai_phrase_ratio(text: str) -> float:
    """Backward-compatible alias for the former public function."""
    return compute_ai_phrase_rate(text)


def compute_avg_sentence_length(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return float(np.mean(lengths)) if lengths else 0.0


def compute_structural_regularity(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 1.0
    std = np.std(lengths)
    mean = np.mean(lengths)
    if mean == 0:
        return 1.0
    # Higher = more uniform sentence lengths (more AI-like)
    return float(1.0 - min(std / mean, 1.0))


def extract_features(text: str) -> dict:
    return {
        "perplexity": compute_perplexity(text),
        "burstiness": compute_burstiness(text),
        "entropy": compute_entropy(text),
        "ai_phrase_rate": compute_ai_phrase_rate(text),
        "avg_sentence_length": compute_avg_sentence_length(text),
        "structural_regularity": compute_structural_regularity(text),
    }

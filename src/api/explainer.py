from src.features.extractor import extract_features

# Thresholds derived from feature distribution observations
THRESHOLDS = {
    "perplexity":           {"low": 50,   "high": 200},
    "burstiness":           {"low": 0.3,  "high": 0.7},
    "ai_phrase_ratio":      {"low": 0.02, "high": 0.08},
    "structural_regularity":{"low": 0.4,  "high": 0.7},
    "entropy":              {"low": 3.5,  "high": 5.0},
    "avg_sentence_length":  {"low": 10,   "high": 25},
}


def get_top_features(text: str, n: int = 3) -> list[str]:
    feats = extract_features(text)
    signals = []

    if feats["perplexity"] < THRESHOLDS["perplexity"]["low"]:
        signals.append(("low perplexity", feats["perplexity"]))
    elif feats["perplexity"] > THRESHOLDS["perplexity"]["high"]:
        signals.append(("high perplexity", -feats["perplexity"]))

    if feats["burstiness"] < THRESHOLDS["burstiness"]["low"]:
        signals.append(("low burstiness", feats["burstiness"]))
    elif feats["burstiness"] > THRESHOLDS["burstiness"]["high"]:
        signals.append(("high burstiness", -feats["burstiness"]))

    if feats["ai_phrase_ratio"] > THRESHOLDS["ai_phrase_ratio"]["high"]:
        signals.append(("high AI phrase ratio", feats["ai_phrase_ratio"]))

    if feats["structural_regularity"] > THRESHOLDS["structural_regularity"]["high"]:
        signals.append(("high structural regularity", feats["structural_regularity"]))

    if feats["entropy"] < THRESHOLDS["entropy"]["low"]:
        signals.append(("low token entropy", feats["entropy"]))

    if feats["avg_sentence_length"] > THRESHOLDS["avg_sentence_length"]["high"]:
        signals.append(("long average sentence length", feats["avg_sentence_length"]))

    signals.sort(key=lambda x: abs(x[1]), reverse=True)
    return [label for label, _ in signals[:n]]

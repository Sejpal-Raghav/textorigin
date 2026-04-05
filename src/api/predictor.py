import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from src.api.explainer import get_top_features

MODEL_PATH = "models/roberta-classifier"
ID2LABEL = {0: "human", 1: "ai_written", 2: "ai_paraphrased"}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
        _model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model, _tokenizer


def predict(text: str) -> dict:
    model, tokenizer = _load_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()

    result = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(3)}
    result["top_features"] = get_top_features(text)
    return result

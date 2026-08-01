import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model


def compute_cosine_similarity(text1: str, text2: str) -> float:
    model = _get_model()
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    cos_sim = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return float(cos_sim.item())

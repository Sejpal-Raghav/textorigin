from threading import Lock

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.errors import ModelUnavailableError

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SimilarityEngine:
    def __init__(self):
        self._model = None
        self._load_lock = Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                try:
                    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
                except (OSError, ValueError) as exc:
                    raise ModelUnavailableError(
                        "The pinned semantic-similarity model is unavailable"
                    ) from exc
                if torch.cuda.is_available():
                    model = model.cuda()
                self._model = model
        return self._model

    def compare(self, text1: str, text2: str) -> float:
        model = self._load()
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        score = F.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return float(score.item())

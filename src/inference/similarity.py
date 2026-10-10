import math
from threading import Lock

from src.errors import ModelUnavailableError

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def balanced_windows(ids, count):
    return [ids[len(ids) * i // count:len(ids) * (i + 1) // count] for i in range(count)]


class SimilarityEngine:
    def __init__(self):
        self._model = None
        self._load_lock = Lock()
        self._encode_lock = Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
                    self._model.eval()
                except (ImportError, OSError, ValueError) as exc:
                    raise ModelUnavailableError("The pinned semantic-similarity model is unavailable") from exc
        return self._model

    def compare(self, text1: str, text2: str) -> float:
        if text1 == text2:
            return 1.0
        model = self._load()
        tokenizer = model.tokenizer
        ids1 = tokenizer.encode(text1, add_special_tokens=False, verbose=False)
        ids2 = tokenizer.encode(text2, add_special_tokens=False, verbose=False)
        size = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        count = max(1, math.ceil(max(len(ids1), len(ids2)) / size))
        # Balanced corresponding windows cover both texts completely. Minimum
        # similarity is deliberately conservative; averages cannot hide tail edits.
        windows1 = balanced_windows(ids1, count)
        windows2 = balanced_windows(ids2, count)
        import torch
        scores = []
        with self._encode_lock, torch.inference_mode():
            for a, b in zip(windows1, windows2):
                if not a or not b:
                    return 0.0
                features = tokenizer.pad([
                    {"input_ids": [tokenizer.cls_token_id, *a, tokenizer.sep_token_id]},
                    {"input_ids": [tokenizer.cls_token_id, *b, tokenizer.sep_token_id]},
                ], padding=True, return_tensors="pt").to(model.device)
                values = model(features)["sentence_embedding"]
                score = torch.nn.functional.cosine_similarity(values[:1], values[1:2]).item()
                scores.append(float(score))
        return min(scores)

from threading import Lock

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.errors import ModelUnavailableError

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

_model = None
_tokenizer = None
_load_lock = Lock()


def get_perplexity_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _load_lock:
        if _model is None:
            try:
                tokenizer = GPT2TokenizerFast.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION
                )
                model = GPT2LMHeadModel.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION
                )
            except (OSError, ValueError) as exc:
                raise ModelUnavailableError(
                    "The pinned GPT-2 feature model is unavailable"
                ) from exc
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            _tokenizer = tokenizer
            _model = model
    return _model, _tokenizer

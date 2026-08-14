from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

from src.errors import ModelUnavailableError

LOGGER = logging.getLogger(__name__)
EXPECTED_LABELS = {0: "human", 1: "ai_written", 2: "ai_paraphrased"}


class ClassifierEngine:
    def __init__(self, model_path: Path, default_max_length: int = 256):
        self.model_path = model_path
        self.manifest_path = model_path / "textorigin-manifest.json"
        self.default_max_length = default_max_length
        self._model = None
        self._tokenizer = None
        self._manifest: dict = {}
        self._load_lock = Lock()

    def readiness(self) -> dict:
        required = ("config.json", "tokenizer_config.json")
        missing = [name for name in required if not (self.model_path / name).exists()]
        has_weights = any(
            (self.model_path / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
        if not has_weights:
            missing.append("model.safetensors or pytorch_model.bin")
        return {
            "ready": not missing,
            "model_path": str(self.model_path),
            "missing_files": missing,
            "manifest_present": self.manifest_path.exists(),
            "loaded": self._model is not None,
        }

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            LOGGER.warning("model_manifest_missing path=%s", self.manifest_path)
            return {}
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(f"Invalid model manifest: {exc}") from exc

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        with self._load_lock:
            if self._model is not None:
                return self._model, self._tokenizer
            readiness = self.readiness()
            if not readiness["ready"]:
                missing = ", ".join(readiness["missing_files"])
                raise ModelUnavailableError(
                    f"Classifier artifact is incomplete at {self.model_path}; missing: {missing}"
                )
            started = perf_counter()
            try:
                tokenizer = RobertaTokenizerFast.from_pretrained(str(self.model_path))
                model = RobertaForSequenceClassification.from_pretrained(str(self.model_path))
            except (OSError, ValueError) as exc:
                raise ModelUnavailableError(f"Unable to load classifier: {exc}") from exc
            model.eval()
            configured_labels = {
                int(index): label for index, label in model.config.id2label.items()
            }
            if configured_labels != EXPECTED_LABELS:
                raise ModelUnavailableError(
                    f"Classifier label mapping {configured_labels} does not match {EXPECTED_LABELS}"
                )
            if torch.cuda.is_available():
                model = model.cuda()
            self._manifest = self._read_manifest()
            self._tokenizer = tokenizer
            self._model = model
            LOGGER.info(
                "classifier_loaded duration_seconds=%.3f device=%s",
                perf_counter() - started,
                next(model.parameters()).device,
            )
        return self._model, self._tokenizer

    def predict(self, text: str) -> dict:
        model, tokenizer = self._load()
        max_length = int(self._manifest.get("max_length", self.default_max_length))
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        truncated = len(token_ids) > max_length
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(next(model.parameters()).device)
        started = perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
        LOGGER.info(
            "classification_completed duration_seconds=%.3f truncated=%s",
            perf_counter() - started,
            truncated,
        )
        return {
            "scores": {
                EXPECTED_LABELS[index]: round(float(probabilities[index]), 6)
                for index in range(3)
            },
            "truncated": truncated,
            "analyzed_tokens": min(len(token_ids), max_length),
            "model_version": self._manifest.get("code_revision"),
            "warnings": ["insufficient_text"] if len(token_ids) < 20 else [],
        }

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from src.errors import ModelUnavailableError
from src.services.execution import ExecutionContext, context_or_default

EXPECTED_LABELS = {0: "human", 1: "ai_written", 2: "ai_paraphrased"}


def token_windows(ids: list[int], size: int):
    """Disjoint windows: every content token receives exactly one unit of weight."""
    if size < 1:
        raise ValueError("Window size must be positive")
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


class ClassifierEngine:
    def __init__(self, model_path: Path, default_max_length: int = 256, batch_size: int = 2):
        if batch_size < 1:
            raise ValueError("Classifier batch size must be positive")
        self.model_path = model_path
        self.manifest_path = model_path / "textorigin-manifest.json"
        self.default_max_length = default_max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._manifest: dict = {}
        self._load_lock = Lock()
        self._predict_lock = Lock()

    def readiness(self, validate: bool = False) -> dict:
        missing = [name for name in ("config.json", "tokenizer_config.json")
                   if not (self.model_path / name).is_file()]
        if not any((self.model_path / name).is_file()
                   for name in ("model.safetensors", "pytorch_model.bin")):
            missing.append("model.safetensors or pytorch_model.bin")
        if not ((self.model_path / "tokenizer.json").is_file() or
                all((self.model_path / name).is_file() for name in ("vocab.json", "merges.txt"))):
            missing.append("tokenizer.json or vocab.json + merges.txt")
        errors = []
        if not missing:
            try:
                config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
                labels = {int(k): v for k, v in config.get("id2label", {}).items()}
                if labels != EXPECTED_LABELS:
                    errors.append("Classifier label mapping is invalid")
                if self.manifest_path.exists():
                    manifest = self._read_manifest()
                    if "labels" in manifest and manifest["labels"] != {label: index for index, label in EXPECTED_LABELS.items()}:
                        errors.append("Manifest label mapping is invalid")
                    if not 4 <= int(manifest.get("max_length", self.default_max_length)) <= 512:
                        errors.append("Manifest max_length is invalid")
            except (OSError, ValueError, TypeError, ModelUnavailableError):
                errors.append("Invalid classifier configuration or manifest")
        files_valid = not missing and not errors
        if validate and files_valid:
            try:
                self._load()
            except ModelUnavailableError:
                errors.append("Classifier weights/tokenizer could not be loaded")
        usable = files_valid and not errors
        return {"ready": usable, "model_path": str(self.model_path),
                "missing_files": missing, "errors": errors,
                "manifest_present": self.manifest_path.is_file(),
                "loaded": self._model is not None,
                "validation_level": "loaded" if self._model is not None else "configuration",
                "inference_verified": getattr(self, "_inference_verified", False)}

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Manifest must be an object")
            return value
        except (OSError, ValueError) as exc:
            raise ModelUnavailableError("Invalid model manifest") from exc

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        with self._load_lock:
            if self._model is None:
                status = self.readiness()
                if not status["ready"]:
                    raise ModelUnavailableError(
                        f"Classifier artifact is incomplete or invalid at {self.model_path}; "
                        "install a trained model and tokenizer, then validate /ready?validate=true"
                    )
                try:
                    import torch
                    from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
                    tokenizer = RobertaTokenizerFast.from_pretrained(str(self.model_path), local_files_only=True)
                    model = RobertaForSequenceClassification.from_pretrained(str(self.model_path), local_files_only=True)
                    model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()
                    self._manifest = self._read_manifest()
                    self._tokenizer = tokenizer
                    self._model = model
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    raise ModelUnavailableError("Unable to load local classifier; check installed dependencies and artifact") from exc
        return self._model, self._tokenizer

    def predict(self, text: str, context: ExecutionContext | None = None, *,
                include_window_scores: bool = False) -> dict:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        context = context_or_default(context)
        context.progress("loading_classifier")
        model, tokenizer = self._load()
        import torch
        max_length = int(self._manifest.get("max_length", self.default_max_length))
        ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        windows = list(token_windows(ids, size))
        if not windows:
            raise ValueError("Text contains no classifiable tokens")
        totals = [0.0] * 3
        window_scores = []
        done = 0
        device = next(model.parameters()).device
        with self._predict_lock, torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                context.progress("classifying", done, len(windows))
                batch = windows[start:start + self.batch_size]
                encoded = [{"input_ids": tokenizer.build_inputs_with_special_tokens(window)} for window in batch]
                inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(device)
                probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist()
                for window, values in zip(batch, probabilities):
                    if include_window_scores:
                        window_scores.append({"tokens": len(window), "scores": {
                            EXPECTED_LABELS[i]: float(value) for i, value in enumerate(values)}})
                    for index, value in enumerate(values):
                        totals[index] += float(value) * len(window)
                done += len(batch)
                context.progress("classifying", done, len(windows))
        context.check()
        self._inference_verified = True
        result = {"scores": {EXPECTED_LABELS[i]: totals[i] / len(ids) for i in range(3)},
                "truncated": False, "analyzed_tokens": len(ids), "total_tokens": len(ids),
                "window_count": len(windows), "complete": True, "coverage": 1.0,
                "aggregation": "nonoverlapping_token_weighted_mean",
                "model_version": self._manifest.get("code_revision"),
                "warnings": ["insufficient_text"] if len(ids) < 20 else []}
        if include_window_scores:
            result["window_scores"] = window_scores
        return result

    def predict_many(self, texts, *, include_window_scores: bool = False) -> list[dict]:
        """Batch windows across documents while retaining independent token weights.

        Used by offline evaluation. The pending token queue is bounded by batch_size,
        so a corpus is never padded/transferred to the GPU all at once.
        """
        model, tokenizer = self._load()
        import torch
        max_length = int(self._manifest.get("max_length", self.default_max_length))
        size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        device = next(model.parameters()).device
        results, pending = [], []

        def flush():
            if not pending:
                return
            encoded = [{"input_ids": tokenizer.build_inputs_with_special_tokens(window)}
                       for _, window in pending]
            inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(device)
            probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1).cpu().tolist()
            for (document, window), values in zip(pending, probabilities):
                result = results[document]
                for index, value in enumerate(values):
                    result["scores"][EXPECTED_LABELS[index]] += float(value) * len(window)
                if include_window_scores:
                    result["window_scores"].append({"tokens": len(window), "scores": {
                        EXPECTED_LABELS[i]: float(value) for i, value in enumerate(values)}})
            pending.clear()

        with self._predict_lock, torch.inference_mode():
            for document, text in enumerate(texts):
                if not text.strip():
                    raise ValueError("Text cannot be empty")
                ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
                if not ids:
                    raise ValueError("Text contains no classifiable tokens")
                result = {"scores": {label: 0.0 for label in EXPECTED_LABELS.values()},
                          "truncated": False, "analyzed_tokens": len(ids), "total_tokens": len(ids),
                          "window_count": 0, "complete": True, "coverage": 1.0,
                          "aggregation": "nonoverlapping_token_weighted_mean",
                          "model_version": self._manifest.get("code_revision"),
                          "warnings": ["insufficient_text"] if len(ids) < 20 else []}
                if include_window_scores:
                    result["window_scores"] = []
                results.append(result)
                for window in token_windows(ids, size):
                    result["window_count"] += 1
                    pending.append((document, window))
                    if len(pending) == self.batch_size:
                        flush()
            flush()
        for result in results:
            result["scores"] = {label: value / result["total_tokens"]
                                for label, value in result["scores"].items()}
        if results:
            self._inference_verified = True
        return results

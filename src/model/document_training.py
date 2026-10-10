"""Document-aligned windows, balanced loss, and serving-policy validation."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, Trainer

from src.inference.classifier import ClassifierEngine, EXPECTED_LABELS, token_windows
from src.model.metrics import compute_metrics


class DocumentWindowDataset(Dataset):
    """Class-balanced documents; token-weighted windows within each document."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        if len(texts) != len(labels) or not texts:
            raise ValueError("Nonempty texts and matching labels are required")
        if set(labels) != {0, 1, 2}:
            raise ValueError("Window training requires all three classes")
        size = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if size < 1:
            raise ValueError("Maximum length must leave room for content")
        counts = Counter(labels)
        self.items = []
        for document, (text, label) in enumerate(zip(texts, labels)):
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if not ids:
                raise ValueError("Training document contains no tokens")
            for window in token_windows(ids, size):
                # Class total = 1/3; document total within class = 1/count.
                weight = len(window) / len(ids) / (3 * counts[label])
                self.items.append((document, window, int(label), weight))
        # Average sample weight is one: retain the ordinary CE loss scale.
        self.items = [(doc, ids, label, weight * len(self.items))
                      for doc, ids, label, weight in self.items]
        self.tokenizer = tokenizer
        self.encoded = [tokenizer.build_inputs_with_special_tokens(ids)
                        for _, ids, _, _ in self.items]
        self.document_count = len(texts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        _, _, label, weight = self.items[index]
        return {"input_ids": self.encoded[index],
                "labels": label, "sample_weight": weight}


class WeightedWindowCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.padding = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features):
        weights = [item.get("sample_weight", 1.0) for item in features]
        batch = self.padding([{key: value for key, value in item.items()
                               if key != "sample_weight"} for item in features])
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


def document_metrics(engine, texts, labels, *, diagnostics=False):
    labels = np.asarray(labels, dtype=int)
    results = engine.predict_many(texts, include_window_scores=diagnostics)
    def metrics(scores):
        return compute_metrics((np.log(np.clip(scores, 1e-12, 1)), labels))
    scores = np.array([[result["scores"][EXPECTED_LABELS[i]] for i in range(3)]
                       for result in results])
    predicted = scores.argmax(axis=1)
    report = {"metrics": metrics(scores), "confusion_matrix": [
        [int(((labels == actual) & (predicted == guess)).sum()) for guess in range(3)]
        for actual in range(3)],
        "human_false_positive_rate": float((predicted[labels == 0] != 0).mean())}
    if diagnostics:
        report["window_positions"] = {}
        for position in ("first", "middle", "last"):
            position_scores = []
            for result in results:
                windows = result["window_scores"]
                index = {"first": 0, "middle": len(windows) // 2, "last": -1}[position]
                position_scores.append([windows[index]["scores"][EXPECTED_LABELS[i]] for i in range(3)])
            report["window_positions"][position] = metrics(np.array(position_scores))
        words = np.array([len(text.split()) for text in texts])
        report["length_buckets"] = {}
        for name, mask in (("under_50_words", words < 50),
                           ("50_to_250_words", (words >= 50) & (words <= 250)),
                           ("over_250_words", words > 250)):
            if mask.any():
                report["length_buckets"][name] = {"samples": int(mask.sum()),
                    "metrics": compute_metrics((np.log(np.clip(scores[mask], 1e-12, 1)), labels[mask]))}
        report["single_window_documents"] = sum(result["window_count"] == 1 for result in results)
    return report


class DocumentTrainer(Trainer):
    def __init__(self, *args, document_validation=None, window_max_length=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_validation = document_validation
        self.window_max_length = window_max_length
        # This loss is a per-microbatch mean; Trainer handles accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        weights = inputs.pop("sample_weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        losses = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, reduction="none")
        loss = (losses * weights.to(losses.device)).mean()
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.document_validation is not None:
            texts, labels = self.document_validation
            engine = ClassifierEngine(Path(self.args.output_dir), batch_size=self.args.per_device_eval_batch_size)
            model = self.accelerator.unwrap_model(self.model)
            was_training = model.training
            try:
                engine._model, engine._tokenizer = model.eval(), self.processing_class
                engine._manifest = {"max_length": self.window_max_length}
                report = document_metrics(engine, texts, labels)
            finally:
                model.train(was_training)
            extra = {f"{metric_key_prefix}_document_{key}": value
                     for key, value in report["metrics"].items()}
            extra[f"{metric_key_prefix}_document_human_false_positive_rate"] = report["human_false_positive_rate"]
            metrics.update(extra)
            self.log(extra)
        return metrics

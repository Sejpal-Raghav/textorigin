"""Exercise real loss/Trainer paths with tiny local models; no downloads."""
from collections import defaultdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BatchEncoding, RobertaConfig, RobertaForSequenceClassification

from scripts.diagnose_classifier import validation_frame
from src.model.classifier import TextDataset, get_trainer
from src.model.document_training import (
    DocumentTrainer, DocumentWindowDataset, WeightedWindowCollator, document_metrics,
)
from src.utils.files import file_sha256


class Tokenizer:
    def save_pretrained(self, output):
        from pathlib import Path
        (Path(output) / "tokenizer_config.json").write_text("{}")

    def encode(self, text, **kwargs):
        return [int(word) for word in text.split()]

    def num_special_tokens_to_add(self, **kwargs):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [0] + ids + [2]

    def __call__(self, text, max_length, **kwargs):
        return {"input_ids": self.build_inputs_with_special_tokens(self.encode(text)[:max_length - 2])}

    def pad(self, items, **kwargs):
        size = max(len(item["input_ids"]) for item in items)
        multiple = kwargs.get("pad_to_multiple_of")
        if multiple:
            size = ((size + multiple - 1) // multiple) * multiple
        batch = {"input_ids": [], "attention_mask": []}
        for item in items:
            ids = item["input_ids"]
            batch["input_ids"].append(ids + [1] * (size - len(ids)))
            batch["attention_mask"].append([1] * len(ids) + [0] * (size - len(ids)))
        if "labels" in items[0]:
            batch["labels"] = [item["labels"] for item in items]
        return BatchEncoding({key: torch.tensor(value) for key, value in batch.items()})


def test_windows_cover_tail_and_balance_classes_and_document_lengths():
    dataset = DocumentWindowDataset(["3 4 5 6 7 8 9", "3", "4 5", "6"], [0, 0, 1, 2], Tokenizer(), 6)
    by_document, by_class = defaultdict(float), defaultdict(float)
    coverage = []
    for doc, ids, label, weight in dataset.items:
        by_document[doc] += weight
        by_class[label] += weight
        if doc == 0:
            coverage.extend(ids)
    assert coverage == [3, 4, 5, 6, 7, 8, 9]
    assert by_document[0] == pytest.approx(by_document[1])
    assert by_class[0] == pytest.approx(by_class[1])
    assert by_class[1] == pytest.approx(by_class[2])
    assert sum(by_class.values()) == pytest.approx(len(dataset))
    # The short tail contributes less than a full window, matching inference.
    assert dataset.items[0][3] / dataset.items[1][3] == pytest.approx(4 / 3)


def test_collator_preserves_weights_and_masks_padding_without_mutating_samples():
    features = [{"input_ids": [0, 3, 2], "labels": 0, "sample_weight": .5},
                {"input_ids": [0, 4, 5, 2], "labels": 1, "sample_weight": 1.5}]
    original = json.dumps(features)
    batch = WeightedWindowCollator(Tokenizer())(features)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert batch["sample_weight"].tolist() == [.5, 1.5]
    assert json.dumps(features) == original


def test_weighted_loss_and_gradient_match_manual_objective():
    logits = torch.tensor([[2., 0., 0.], [0., 1., 0.]], requires_grad=True)
    trainer = object.__new__(DocumentTrainer)
    model = lambda **kwargs: SimpleNamespace(logits=logits)
    inputs = {"input_ids": torch.tensor([[3], [4]]), "labels": torch.tensor([0, 2]),
              "sample_weight": torch.tensor([.5, 1.5])}
    loss = trainer.compute_loss(model, inputs)
    expected = (torch.nn.functional.cross_entropy(logits, inputs["labels"], reduction="none")
                * inputs["sample_weight"]).mean()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert "labels" in inputs and "sample_weight" in inputs


def test_real_trainer_reports_document_selection_metric(tmp_path):
    tokenizer = Tokenizer()
    texts, labels = ["3 4 5 6 7", "4 5", "6 7"], [0, 1, 2]
    model = RobertaForSequenceClassification(RobertaConfig(
        vocab_size=12, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
        num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2))
    trainer = get_trainer(model, tokenizer,
        DocumentWindowDataset(texts, labels, tokenizer, 6),
        TextDataset(texts, labels, tokenizer, 6, pretokenize=True), str(tmp_path),
        epochs=1, train_batch_size=2, eval_batch_size=2, gradient_accumulation_steps=2,
        gradient_checkpointing=False, mixed_precision="none",
        document_validation=(texts, labels), window_max_length=6)
    trainer.args.disable_tqdm = True
    trainer.train()
    metrics = trainer.evaluate()
    assert np.isfinite(metrics["eval_document_macro_f1"])
    assert "eval_document_recall_ai_written" in metrics
    assert "eval_document_human_false_positive_rate" in metrics
    assert not trainer.model_accepts_loss_kwargs
    assert trainer.state.best_metric == pytest.approx(metrics["eval_document_macro_f1"])
    assert trainer.state.best_model_checkpoint is not None


def test_diagnostics_use_all_documents_and_separate_window_positions():
    class Engine:
        def predict_many(self, texts, **kwargs):
            return [self.predict(text, **kwargs) for text in texts]

        def predict(self, text, **kwargs):
            human = {"human": .9, "ai_written": .05, "ai_paraphrased": .05}
            ai = {"human": .05, "ai_written": .9, "ai_paraphrased": .05}
            paraphrased = {"human": .05, "ai_written": .05, "ai_paraphrased": .9}
            scores = [human, ai, paraphrased][int(text)]
            windows = [human, ai] if text == "1" else [scores]
            return {"scores": scores, "window_count": len(windows),
                    "window_scores": [{"tokens": 4, "scores": s} for s in windows]}
    report = document_metrics(Engine(), ["0", "1", "2"], [0, 1, 2], diagnostics=True)
    assert report["metrics"]["accuracy"] == 1
    assert report["window_positions"]["first"]["recall_ai_written"] == 0
    assert report["window_positions"]["last"]["recall_ai_written"] == 1
    assert report["single_window_documents"] == 2


def test_validation_diagnostic_rejects_changed_data_and_split(tmp_path):
    from tests.test_split import make_dataset
    from src.data.split import grouped_stratified_split
    frame = make_dataset()
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    digest = file_sha256(data)
    splits = grouped_stratified_split(frame)
    identity = {"seed": 42, "dataset_sha256": digest, "splits": {
        name: sorted(part["source_id"].unique().tolist())
        for name, part in zip(("train", "validation", "test"), splits)}}
    audit = {"accepted_rows": len(frame), "output_sha256": digest}
    result = validation_frame(data, audit, identity)
    assert set(result["source_id"]) == set(identity["splits"]["validation"])
    identity["splits"]["validation"] = []
    with pytest.raises(ValueError, match="splits"):
        validation_frame(data, audit, identity)
    data.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        validation_frame(data, audit, identity)


def test_pretokenized_prefix_dataset_does_not_retokenize_each_epoch():
    class CountingTokenizer(Tokenizer):
        calls = 0
        def __call__(self, *args, **kwargs):
            self.calls += 1
            return super().__call__(*args, **kwargs)
    tokenizer = CountingTokenizer()
    dataset = TextDataset(["3 4 5", "6 7"], [0, 1], tokenizer, 6, pretokenize=True)
    assert tokenizer.calls == 2
    for _ in range(3):
        assert dataset[0]["input_ids"] == [0, 3, 4, 5, 2]
    assert tokenizer.calls == 2


def test_tensor_core_padding_preserves_labels_and_loss_weights():
    batch = WeightedWindowCollator(Tokenizer(), pad_to_multiple_of=8)([
        {"input_ids": [0, 3, 2], "labels": 1, "sample_weight": .7}])
    assert batch["input_ids"].shape == (1, 8)
    assert batch["attention_mask"].sum().item() == 3
    assert batch["labels"].tolist() == [1]
    assert batch["sample_weight"].item() == pytest.approx(.7)


def test_cross_document_batches_match_separate_inference_and_reduce_calls(tmp_path):
    from src.inference.classifier import ClassifierEngine
    class Model:
        calls = []
        def parameters(self):
            return iter([torch.tensor(0.)])
        def __call__(self, input_ids, attention_mask):
            self.calls.append(len(input_ids))
            # Ignore padding; different windows/documents produce different scores.
            values = (input_ids * attention_mask).sum(dim=1).float()
            return SimpleNamespace(logits=torch.stack([values / 10, -values / 20, values * 0], dim=1))
    model = Model()
    engine = ClassifierEngine(tmp_path, default_max_length=6, batch_size=3)
    engine._model, engine._tokenizer = model, Tokenizer()
    texts = ["3 4 5 6 7", "8", "9 10 3 4 5 6 7 8 9", "3 4"]
    expected = [engine.predict(text, include_window_scores=True) for text in texts]
    old_calls = len(model.calls)
    model.calls.clear()
    actual = engine.predict_many(texts, include_window_scores=True)
    assert model.calls == [3, 3, 1]
    assert len(model.calls) < old_calls
    for want, got in zip(expected, actual):
        assert got["scores"] == pytest.approx(want["scores"])
        assert got["total_tokens"] == want["total_tokens"]
        assert got["window_count"] == want["window_count"]
        assert [w["tokens"] for w in got["window_scores"]] == [w["tokens"] for w in want["window_scores"]]
        for first, second in zip(want["window_scores"], got["window_scores"]):
            assert first["scores"] == pytest.approx(second["scores"])
    assert engine.predict_many([]) == []
    with pytest.raises(ValueError, match="empty"):
        engine.predict_many(["3", " "])


@pytest.mark.parametrize("cuda", [False, True])
def test_trainer_performance_settings_follow_device(monkeypatch, cuda):
    import src.model.classifier as classifier
    import src.model.document_training as document_training
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(classifier, "TrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(document_training, "DocumentTrainer", lambda **kwargs: SimpleNamespace(**kwargs))
    trainer = classifier.get_trainer(None, Tokenizer(), [], [], "unused", mixed_precision="none",
                                     dataloader_workers=2, gradient_checkpointing=False,
                                     train_batch_size=16, eval_batch_size=32, gradient_accumulation_steps=1)
    assert trainer.args.optim == ("adamw_torch_fused" if cuda else "adamw_torch")
    assert trainer.args.dataloader_pin_memory == cuda
    assert trainer.args.accelerator_config["non_blocking"] == cuda
    assert trainer.args.dataloader_persistent_workers
    assert trainer.data_collator.padding.pad_to_multiple_of == (8 if cuda else None)
    assert not trainer.args.gradient_checkpointing

import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "roberta-base"

LABEL2ID = {"human": 0, "ai_written": 1, "ai_paraphrased": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_model_and_tokenizer(model_path: str = None):
    path = model_path or MODEL_ID
    tokenizer = RobertaTokenizerFast.from_pretrained(path)
    model = RobertaForSequenceClassification.from_pretrained(
        path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tokenizer


def get_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    epochs: int = 3,
) -> Trainer:
    from src.model.metrics import compute_metrics

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

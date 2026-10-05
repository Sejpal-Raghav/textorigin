import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

MODEL_ID = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2"

LABEL2ID = {"human": 0, "ai_written": 1, "ai_paraphrased": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item["labels"] = self.labels[idx]
        return item


def load_model_and_tokenizer(model_path: str = None):
    path = model_path or MODEL_ID
    revision = None if model_path else MODEL_REVISION
    tokenizer = RobertaTokenizerFast.from_pretrained(path, revision=revision)
    model = RobertaForSequenceClassification.from_pretrained(
        path,
        revision=revision,
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
    train_batch_size: int = 2,
    eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    seed: int = 42,
    gradient_checkpointing: bool = True,
    mixed_precision: str = "auto",
) -> Trainer:
    from src.model.metrics import compute_metrics

    use_cuda = torch.cuda.is_available()
    use_bf16 = mixed_precision == "bf16" and use_cuda
    use_fp16 = (mixed_precision == "fp16" or mixed_precision == "auto") and use_cuda
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        save_total_limit=2,
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
        logging_steps=50,
        report_to="none",
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

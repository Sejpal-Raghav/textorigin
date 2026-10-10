import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

MODEL_ID = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2"

LABEL2ID = {"human": 0, "ai_written": 1, "ai_paraphrased": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class PeriodicCheckpoint(TrainerCallback):
    """Save optimizer/scheduler state between epoch evaluations for long CPU runs."""
    def __init__(self, interval=250):
        if interval < 1:
            raise ValueError("Checkpoint interval must be positive")
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0:
            control.should_save = True
        return control


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256,
                 pretokenize: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = [tokenizer(text, truncation=True, max_length=max_length)
                        for text in texts] if pretokenize else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = dict(self.encoded[idx]) if self.encoded is not None else self.tokenizer(
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
    max_steps: int = -1,
    document_validation=None,
    window_max_length: int = 256,
    dataloader_workers: int = 0,
    optimizer: str = "auto",
) -> Trainer:
    from src.model.metrics import compute_metrics
    from src.model.document_training import DocumentTrainer, WeightedWindowCollator

    use_cuda = torch.cuda.is_available()
    if dataloader_workers < 0:
        raise ValueError("Data loader workers must be nonnegative")
    if optimizer not in {"auto", "adamw_torch", "adamw_torch_fused"}:
        raise ValueError("Unknown optimizer")
    if optimizer == "auto":
        optimizer = "adamw_torch_fused" if use_cuda else "adamw_torch"
    if mixed_precision in {"fp16", "bf16"} and not use_cuda:
        raise ValueError("Explicit mixed precision requires supported CUDA; use none on CPU")
    if mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
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
        max_steps=max_steps,
        eval_accumulation_steps=16,
        optim=optimizer,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="document_macro_f1" if document_validation is not None else "f1",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        save_total_limit=2,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_prefetch_factor=2 if dataloader_workers > 0 else None,
        dataloader_pin_memory=use_cuda,
        accelerator_config={"non_blocking": use_cuda},
        seed=seed,
        data_seed=seed,
        logging_steps=50,
        report_to="none",
    )

    return DocumentTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=WeightedWindowCollator(tokenizer, pad_to_multiple_of=8 if use_cuda else None),
        document_validation=document_validation,
        window_max_length=window_max_length,
        callbacks=[PeriodicCheckpoint()],
        compute_metrics=compute_metrics,
    )

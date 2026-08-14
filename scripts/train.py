import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.split import grouped_stratified_split
from src.model.classifier import (
    LABEL2ID,
    MODEL_ID,
    MODEL_REVISION,
    TextDataset,
    get_trainer,
    load_model_and_tokenizer,
)
from src.model.metrics import print_confusion_matrix
from src.utils.files import file_sha256


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for 3-class text detection")
    parser.add_argument("--data", default="data/raw/combined.csv")
    parser.add_argument("--output-dir", default="models/roberta-classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    df = pd.read_csv(data_path).dropna(subset=["text", "source", "source_id"])
    df["label"] = df["source"].map(LABEL2ID)
    if df["label"].isna().any():
        unknown = sorted(df.loc[df["label"].isna(), "source"].unique())
        raise ValueError(f"Unknown source labels: {unknown}")

    train_df, val_df, test_df = grouped_stratified_split(df, seed=args.seed)
    train_texts, train_labels = train_df["text"].tolist(), train_df["label"].astype(int).tolist()
    val_texts, val_labels = val_df["text"].tolist(), val_df["label"].astype(int).tolist()
    test_texts, test_labels = test_df["text"].tolist(), test_df["label"].astype(int).tolist()

    print(f"Split — Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

    model, tokenizer = load_model_and_tokenizer()

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)

    trainer = get_trainer(model, tokenizer, train_dataset, val_dataset, args.output_dir, args.epochs)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    print("\n--- Test Set Results ---")
    preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    print_confusion_matrix(test_labels, pred_labels)
    print(preds.metrics)

    output_dir = Path(args.output_dir)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": _git_revision(),
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "dataset_path": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "labels": LABEL2ID,
        "max_length": args.max_length,
        "seed": args.seed,
        "epochs": args.epochs,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "test_metrics": {key: float(value) for key, value in preds.metrics.items()},
    }
    with (output_dir / "textorigin-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

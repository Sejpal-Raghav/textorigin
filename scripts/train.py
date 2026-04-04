import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model.classifier import LABEL2ID, TextDataset, get_trainer, load_model_and_tokenizer
from src.model.metrics import print_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for 3-class text detection")
    parser.add_argument("--data", default="data/features/features.csv")
    parser.add_argument("--output-dir", default="models/roberta-classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "source"])
    df["label"] = df["source"].map(LABEL2ID)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

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


if __name__ == "__main__":
    main()

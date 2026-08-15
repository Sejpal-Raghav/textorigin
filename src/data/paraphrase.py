import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "humarin/chatgpt_paraphraser_on_T5_base"
MODEL_REVISION = "f86b18a7feb0b8149b7612a620b61e2e26dbf82e"


def paraphrase_texts(
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 128,
    on_batch=None,
) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).to(device)
    model.eval()

    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Paraphrasing"):
        batch = [f"paraphrase: {text}" for text in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
        if on_batch is not None:
            on_batch(decoded)

    return results


def generate_paraphrased_class(
    ai_df: pd.DataFrame,
    sample_size: int = 5000,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    sample = ai_df.sample(
        n=min(sample_size, len(ai_df)), random_state=seed
    ).reset_index(drop=True)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    existing = pd.DataFrame(columns=["text", "source", "source_id"])
    if checkpoint and checkpoint.exists():
        existing = pd.read_csv(checkpoint).dropna(subset=["text", "source_id"])
        expected_prefix = sample["source_id"].astype(str).tolist()[: len(existing)]
        actual_ids = existing["source_id"].astype(str).tolist()
        if len(existing) > len(sample) or actual_ids != expected_prefix:
            raise ValueError(
                f"Checkpoint {checkpoint} does not match the current deterministic sample"
            )

    completed = len(existing)
    pending = sample.iloc[completed:]
    if pending.empty:
        return existing.reset_index(drop=True)

    def save_batch(decoded: list[str]) -> None:
        nonlocal completed
        batch_ids = sample.iloc[completed : completed + len(decoded)]["source_id"].tolist()
        batch_frame = pd.DataFrame(
            {
                "text": decoded,
                "source": "ai_paraphrased",
                "source_id": batch_ids,
            }
        )
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            batch_frame.to_csv(
                checkpoint,
                mode="a",
                header=not checkpoint.exists(),
                index=False,
            )
        completed += len(decoded)

    paraphrased = paraphrase_texts(pending["text"].tolist(), on_batch=save_batch)
    if checkpoint:
        return pd.read_csv(checkpoint)
    return pd.DataFrame(
        {
            "text": paraphrased,
            "source": "ai_paraphrased",
            "source_id": sample["source_id"].tolist(),
        }
    )

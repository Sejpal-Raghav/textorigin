import pandas as pd
from datasets import load_dataset

DATASET_ID = "artem9k/ai-text-detection-pile"
DATASET_REVISION = "9bc306f"


def load_raw_data(sample_size: int = 5000, seed: int = 42) -> pd.DataFrame:
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
    )
    df = dataset.to_pandas()

    missing = {"text", "label"}.difference(df.columns)
    if missing:
        raise ValueError(
            "Upstream dataset schema changed; missing columns: "
            + ", ".join(sorted(missing))
        )
    df["source_id"] = [f"pile-{index}" for index in df.index]

    human = df[df["label"] == 0].sample(
        n=min(sample_size, (df["label"] == 0).sum()), random_state=seed
    )
    ai = df[df["label"] == 1].sample(
        n=min(sample_size, (df["label"] == 1).sum()), random_state=seed
    )

    human = human.copy()
    ai = ai.copy()
    human["source"] = "human"
    ai["source"] = "ai_written"

    return pd.concat([human, ai], ignore_index=True)[["text", "source", "source_id"]]

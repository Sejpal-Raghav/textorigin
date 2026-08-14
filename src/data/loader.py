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

    if "text" not in df.columns or not {"label", "source"}.intersection(df.columns):
        raise ValueError(
            "Upstream dataset schema changed; expected text plus label or source columns"
        )

    if "source" in df.columns:
        normalized_source = df["source"].astype(str).str.lower()
        human = df[normalized_source == "human"]
        ai = df[normalized_source.isin({"ai", "ai_written"})]
    else:
        human = df[df["label"] == 0]
        ai = df[df["label"] == 1]

    human = human.sample(n=min(sample_size, len(human)), random_state=seed)
    ai = ai.sample(n=min(sample_size, len(ai)), random_state=seed)

    human = human.copy()
    ai = ai.copy()
    human["source"] = "human"
    ai["source"] = "ai_written"
    identity_column = "id" if "id" in df.columns else None
    human["source_id"] = [
        f"human-{value}" for value in (human[identity_column] if identity_column else human.index)
    ]
    ai["source_id"] = [
        f"ai-{value}" for value in (ai[identity_column] if identity_column else ai.index)
    ]

    return pd.concat([human, ai], ignore_index=True)[["text", "source", "source_id"]]

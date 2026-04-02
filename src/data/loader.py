from datasets import load_dataset
import pandas as pd


def load_raw_data(sample_size: int = 5000, seed: int = 42) -> pd.DataFrame:
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")
    df = dataset.to_pandas()

    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    df = df.rename(columns={text_col: "text", label_col: "label"})

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

    return pd.concat([human, ai], ignore_index=True)[["text", "source"]]

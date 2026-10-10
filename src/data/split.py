import pandas as pd
from sklearn.model_selection import train_test_split


def _split_ids(ids: list[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    train_ids, temporary_ids = train_test_split(
        sorted(ids), test_size=0.2, random_state=seed
    )
    validation_ids, test_ids = train_test_split(
        temporary_ids, test_size=0.5, random_state=seed
    )
    return set(train_ids), set(validation_ids), set(test_ids)


def grouped_stratified_split(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split source families so an original and its paraphrase never cross splits."""
    required_columns = {"text", "source", "source_id"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing lineage columns: " + ", ".join(sorted(missing))
        )

    if df[list(required_columns)].isna().any().any():
        raise ValueError("Dataset contains missing text, source or lineage")
    if not set(df["source"]).issubset({"human", "ai_written", "ai_paraphrased"}):
        raise ValueError("Unknown source labels")
    if df.groupby("source_id")["source"].apply(lambda values: "human" in set(values) and len(set(values)) > 1).any():
        raise ValueError("Human and AI source identities must not overlap")
    normalized = df["text"].astype(str).str.lower().str.split().str.join(" ")
    if df.assign(normalized=normalized).groupby("normalized")["source_id"].nunique().gt(1).any():
        raise ValueError("Cross-family duplicate text must be audited before splitting")
    human_ids = df.loc[df["source"] == "human", "source_id"].unique().tolist()
    ai_ids = df.loc[df["source"] != "human", "source_id"].unique().tolist()
    if min(len(human_ids), len(ai_ids)) < 10:
        raise ValueError("At least 10 human and 10 AI source groups are required")

    human_split = _split_ids(human_ids, seed)
    ai_split = _split_ids(ai_ids, seed)
    partitions = []
    for index in range(3):
        selected = human_split[index] | ai_split[index]
        partition = df[df["source_id"].isin(selected)].copy()
        partitions.append(
            partition.sample(frac=1, random_state=seed).reset_index(drop=True)
        )
    return tuple(partitions)

import pandas as pd
from datasets import load_dataset

DATASET_ID = "artem9k/ai-text-detection-pile"
DATASET_REVISION = "9bc306f"


def load_raw_data(sample_size: int = 5000, seed: int = 42, streaming: bool = False) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if streaming:
        return _stream_sample(sample_size, seed)
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



def _stream_sample(sample_size: int, seed: int) -> pd.DataFrame:
    """Bounded buffer sampling; provenance records this approximation explicitly."""
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train", streaming=True)
    counts = {"human": 0, "ai_written": 0}
    rows = []
    seen = set()
    for index, row in enumerate(dataset.shuffle(seed=seed, buffer_size=10_000)):
        value = str(row.get("source", "")).lower()
        if not value:
            value = "human" if row.get("label") == 0 else "ai" if row.get("label") == 1 else ""
        source = "human" if value == "human" else "ai_written" if value in {"ai", "ai_written"} else None
        text = row.get("text")
        if source is None or not isinstance(text, str) or not text.strip() or counts[source] >= sample_size:
            continue
        # Content-derived identity is stable across buffer orders and joins exact
        # repeats rather than relying on unstable iteration indices.
        import hashlib
        identity = source + "-" + hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({"text": text, "source": source, "source_id": identity})
        counts[source] += 1
        if min(counts.values()) >= sample_size:
            break
    if min(counts.values()) < sample_size:
        raise ValueError(f"Insufficient valid data: {counts}")
    return pd.DataFrame(rows)

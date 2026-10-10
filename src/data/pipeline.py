import argparse
import json
from src.data.loader import DATASET_ID, DATASET_REVISION
from src.data.paraphrase import MODEL_ID, MODEL_REVISION
from src.utils.files import file_sha256
from pathlib import Path

import pandas as pd

from src.data.loader import load_raw_data
from src.data.paraphrase import generate_paraphrased_class


def build_dataset(
    sample_size: int = 5000,
    output_path: str = "data/raw/combined.csv",
    checkpoint_path: str = "data/raw/ai_paraphrased.checkpoint.csv",
) -> pd.DataFrame:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw data from HuggingFace...")
    df = load_raw_data(sample_size=sample_size, streaming=True)

    human_df = df[df["source"] == "human"]
    ai_df = df[df["source"] == "ai_written"]

    print(f"Loaded {len(human_df)} human and {len(ai_df)} AI samples.")
    print("Generating paraphrased class (this may take a while)...")

    paraphrased_df = generate_paraphrased_class(
        ai_df,
        sample_size=sample_size,
        checkpoint_path=checkpoint_path,
    )

    combined = pd.concat([human_df, ai_df, paraphrased_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(output, index=False)

    output.with_suffix(".provenance.json").write_text(json.dumps({
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "paraphrase_model": MODEL_ID, "paraphrase_revision": MODEL_REVISION,
        "seed": 42, "sample_size_per_class": sample_size,
        "sampling": "streaming_shuffle_buffer_10000_first_valid_unique_per_class",
        "dataset_sha256": file_sha256(output),
        "class_counts": combined["source"].value_counts().to_dict(),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {len(combined)} total samples to {output}")
    print(combined["source"].value_counts().to_string())
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the combined 3-class dataset")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--output", default="data/raw/combined.csv")
    parser.add_argument(
        "--checkpoint", default="data/raw/ai_paraphrased.checkpoint.csv"
    )
    args = parser.parse_args()
    build_dataset(
        sample_size=args.sample_size,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )

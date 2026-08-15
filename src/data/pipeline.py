import argparse
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
    df = load_raw_data(sample_size=sample_size)

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

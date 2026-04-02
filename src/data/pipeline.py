import argparse
import os

import pandas as pd

from src.data.loader import load_raw_data
from src.data.paraphrase import generate_paraphrased_class


def build_dataset(sample_size: int = 5000, output_path: str = "data/raw/combined.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading raw data from HuggingFace...")
    df = load_raw_data(sample_size=sample_size)

    human_df = df[df["source"] == "human"]
    ai_df = df[df["source"] == "ai_written"]

    print(f"Loaded {len(human_df)} human and {len(ai_df)} AI samples.")
    print("Generating paraphrased class (this may take a while)...")

    paraphrased_df = generate_paraphrased_class(ai_df, sample_size=sample_size)

    combined = pd.concat([human_df, ai_df, paraphrased_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined)} total samples to {output_path}")
    print(combined["source"].value_counts().to_string())
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the combined 3-class dataset")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--output", default="data/raw/combined.csv")
    args = parser.parse_args()
    build_dataset(sample_size=args.sample_size, output_path=args.output)

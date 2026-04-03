import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.features.extractor import extract_features


def main():
    parser = argparse.ArgumentParser(description="Extract linguistic features from text samples")
    parser.add_argument("--input", default="data/raw/combined.csv")
    parser.add_argument("--output", default="data/features/features.csv")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows for testing")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    features = []
    for text in tqdm(df["text"], desc="Extracting features"):
        features.append(extract_features(str(text)))

    feat_df = pd.DataFrame(features)
    result = pd.concat([df[["text", "source"]].reset_index(drop=True), feat_df], axis=1)
    result.to_csv(args.output, index=False)

    print(f"\nSaved {len(result)} rows to {args.output}")
    print(result.describe(include="all").to_string())


if __name__ == "__main__":
    main()

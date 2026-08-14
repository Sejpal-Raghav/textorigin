import argparse
from pathlib import Path

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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for text in tqdm(df["text"], desc="Extracting features"):
        features.append(extract_features(str(text)))

    feat_df = pd.DataFrame(features)
    identity_columns = ["text", "source"]
    if "source_id" in df.columns:
        identity_columns.append("source_id")
    result = pd.concat([df[identity_columns].reset_index(drop=True), feat_df], axis=1)
    result.to_csv(output, index=False)

    print(f"\nSaved {len(result)} rows to {output}")
    print(result.describe(include="all").to_string())


if __name__ == "__main__":
    main()

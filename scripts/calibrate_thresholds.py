import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.split import grouped_stratified_split
from src.utils.files import file_sha256

FEATURES = (
    "perplexity",
    "burstiness",
    "ai_phrase_rate",
    "structural_regularity",
    "entropy",
    "avg_sentence_length",
)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate heuristic observation thresholds on a held-out split"
    )
    parser.add_argument("--data", default="data/features/features.csv")
    parser.add_argument("--output", default="models/feature-thresholds.json")
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.low_quantile < args.high_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 1")

    data_path = Path(args.data).resolve()
    frame = pd.read_csv(data_path).dropna(
        subset=["text", "source", "source_id", *FEATURES]
    )
    _, validation, _ = grouped_stratified_split(frame, seed=args.seed)
    thresholds = {
        feature: {
            "low": float(validation[feature].quantile(args.low_quantile)),
            "high": float(validation[feature].quantile(args.high_quantile)),
        }
        for feature in FEATURES
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "grouped validation split empirical quantiles",
        "dataset_sha256": file_sha256(data_path),
        "sample_count": len(validation),
        "seed": args.seed,
        "low_quantile": args.low_quantile,
        "high_quantile": args.high_quantile,
        "thresholds": thresholds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved calibrated thresholds to {output}")


if __name__ == "__main__":
    main()

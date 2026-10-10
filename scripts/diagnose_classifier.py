"""Diagnose an existing classifier on its identity-verified validation split only."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.split import grouped_stratified_split
from src.inference.classifier import ClassifierEngine
from src.model.classifier import LABEL2ID
from src.model.document_training import document_metrics
from src.utils.files import file_sha256


def validation_frame(data, audit, identity):
    digest = file_sha256(data)
    if digest != audit["output_sha256"] or digest != identity["dataset_sha256"]:
        raise ValueError("Dataset hash does not match the audit and model identity")
    frame = pd.read_csv(data)
    if len(frame) != audit["accepted_rows"]:
        raise ValueError("Audit row count differs from data")
    frame["label"] = frame["source"].map(LABEL2ID)
    if frame["label"].isna().any():
        raise ValueError("Unknown source labels")
    partitions = grouped_stratified_split(frame, seed=identity["seed"])
    for name, partition in zip(("train", "validation", "test"), partitions):
        if sorted(partition["source_id"].unique().tolist()) != identity["splits"][name]:
            raise ValueError("Reconstructed splits do not match the model identity")
    validation = partitions[1]
    if set(validation["label"]) != {0, 1, 2}:
        raise ValueError("Validation must contain all three classes")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Maximum windows per inference batch across documents")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.output.exists():
        parser.error("Diagnostic output already exists; choose a new file")
    identity = json.loads((args.model_path / "training-identity.json").read_text(encoding="utf-8"))
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    validation = validation_frame(args.data, audit, identity)
    engine = ClassifierEngine(args.model_path.resolve(), default_max_length=identity["max_length"], batch_size=args.batch_size)
    report = document_metrics(engine, validation["text"].tolist(),
                              validation["label"].astype(int).tolist(), diagnostics=True)
    report.update(partition="validation", documents=len(validation), dataset_sha256=identity["dataset_sha256"],
                  artifact_sha256=file_sha256(args.model_path / "model.safetensors"),
                  aggregation="nonoverlapping_token_weighted_mean",
                  batch_size=args.batch_size,
                  position_note="First/middle/last are windows, not separate documents; single-window texts occur in all three.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous results even if another writer races.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "partition": "validation", "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()

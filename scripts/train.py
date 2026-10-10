"""Audited, resumable RoBERTa training with separate smoke and release artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.data.split import grouped_stratified_split
from src.inference.classifier import ClassifierEngine
from src.model.classifier import LABEL2ID, MODEL_ID, MODEL_REVISION, TextDataset, get_trainer, load_model_and_tokenizer
from src.model.metrics import compute_metrics, print_confusion_matrix
from src.utils.files import file_sha256


def code_provenance():
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for folder in ("src", "scripts"):
        for file in sorted((root / folder).rglob("*.py")):
            digest.update(file.relative_to(root).as_posix().encode())
            digest.update(file.read_bytes())
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, check=True)
        return result.stdout
    try:
        return {
            "revision": git("rev-parse", "HEAD").decode().strip(),
            "source_sha256": digest.hexdigest(),
            "dirty": bool(git("status", "--porcelain").strip()),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "diff_sha256": None, "source_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/combined.clean.csv")
    parser.add_argument("--audit-report", default="data/processed/audit.json")
    parser.add_argument("--output-dir", default="models/roberta-classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--mixed-precision", choices=("auto", "fp16", "bf16", "none"), default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two update steps; no held-out test evaluation")
    parser.add_argument("--resume-from-checkpoint", help="Checkpoint directory or 'latest'")
    args = parser.parse_args()
    if min(args.epochs, args.train_batch_size, args.eval_batch_size, args.gradient_accumulation_steps) < 1 or not 4 <= args.max_length <= 512:
        parser.error("Epochs and batch/accumulation sizes must be positive; max length must be 4..512")
    output = Path(args.output_dir).resolve()
    if args.smoke and output.name == "roberta-classifier":
        parser.error("Smoke artifacts require a separate output directory, such as models/smoke-classifier")
    data = Path(args.data).resolve()
    audit = json.loads(Path(args.audit_report).read_text(encoding="utf-8"))
    digest = file_sha256(data)
    if audit["output_sha256"] != digest:
        raise ValueError("Training dataset hash does not match the audit report")
    df = pd.read_csv(data)
    if len(df) != audit["accepted_rows"]:
        raise ValueError("Audit row count differs from training data")
    df["label"] = df["source"].map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError("Unknown source labels")
    if set(df["source"]) != set(LABEL2ID):
        raise ValueError("Audited training data must contain all three classes")
    train, validation, test = grouped_stratified_split(df, seed=args.seed)
    partitions = {"train": train, "validation": validation, "test": test}
    if any(set(frame["source"]) != set(LABEL2ID) for frame in partitions.values()):
        raise ValueError("Every partition must contain all three classes")
    groups = {name: sorted(frame["source_id"].unique().tolist()) for name, frame in partitions.items()}
    if any(set(groups[a]) & set(groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Source groups cross split boundaries")
    versions = {name: importlib.metadata.version(name) for name in ("torch","transformers","datasets","numpy","pandas","scikit-learn","accelerate")}
    identity = {"dataset_sha256": digest, "splits": groups, "seed": args.seed,
                "max_length": args.max_length, "epochs": args.epochs,
                "smoke": args.smoke, "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "mixed_precision": args.mixed_precision, "dependencies": versions,
                "base_model": MODEL_ID, "base_model_revision": MODEL_REVISION,
                "gradient_checkpointing": not args.no_gradient_checkpointing,
                "checkpoint_steps": 250,
                "code_sha256": code_provenance()["source_sha256"]}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "training-identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("Existing run identity differs; choose a new output directory")
        if not args.resume_from_checkpoint:
            raise ValueError("Existing training run requires explicit --resume-from-checkpoint")
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = get_last_checkpoint(str(output))
        if resume is None:
            raise ValueError("No resumable checkpoint exists")
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent != output or not checkpoint.is_dir():
            raise ValueError("Resume checkpoint must belong to this run's output directory")

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stop = Event()
    memory = {"peak_rss_bytes": 0}
    def sample_memory():
        process = psutil.Process()
        while not stop.wait(.1):
            memory["peak_rss_bytes"] = max(memory["peak_rss_bytes"], process.memory_info().rss)
    monitor = Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = perf_counter()
    try:
        model, tokenizer = load_model_and_tokenizer()
        def dataset(frame):
            return TextDataset(frame["text"].tolist(), frame["label"].astype(int).tolist(), tokenizer, args.max_length)
        training_frame = train.iloc[:24] if args.smoke else train
        validation_frame = validation.iloc[:12] if args.smoke else validation
        trainer = get_trainer(model, tokenizer, dataset(training_frame), dataset(validation_frame),
                              str(output), args.epochs,
                              train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
                              gradient_accumulation_steps=args.gradient_accumulation_steps, seed=args.seed,
                              gradient_checkpointing=not args.no_gradient_checkpointing,
                              mixed_precision=args.mixed_precision, max_steps=2 if args.smoke else -1)
        train_result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        report = {
            "purpose": "smoke_only_not_release" if args.smoke else "release_candidate",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": digest, "audit": audit,
            "split_sizes": {name: len(frame) for name, frame in partitions.items()},
            "split_groups_sha256": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest(),
            "training_metrics": train_result.metrics, "code": code_provenance(),
            "dependencies": versions,
            "metric_conventions": "3-class softmax; summed multiclass Brier; ECE: 10 equal-width confidence bins",
            "aggregation": "nonoverlapping_token_weighted_mean",
            "limitations": [
                "AI-paraphrased labels come from one synthetic paraphrase model.",
                "Evaluation uses one dataset and does not establish cross-domain authorship accuracy.",
                "Model scores are uncalibrated.",
                "No guaranteed accuracy or quality threshold has been approved.",
            ],
        }
        if not args.smoke:
            predictions = trainer.predict(dataset(test))
            labels = test["label"].astype(int).to_numpy()
            report["single_window_test_metrics"] = compute_metrics((predictions.predictions, labels))
            report["single_window_confusion_matrix"] = print_confusion_matrix(labels, np.argmax(predictions.predictions, axis=-1)).tolist()
            engine = ClassifierEngine(output)
            engine._model, engine._tokenizer = model.eval(), tokenizer
            engine._manifest = {"max_length": args.max_length}
            scores = np.array([list(engine.predict(text)["scores"].values()) for text in test["text"]])
            report["document_test_metrics"] = compute_metrics((np.log(np.clip(scores, 1e-12, 1)), labels))
            report["document_confusion_matrix"] = print_confusion_matrix(labels, scores.argmax(axis=1)).tolist()
            majority = int(train["label"].value_counts().idxmax())
            baseline = np.full((len(test),3), -30.0)
            baseline[:,majority] = 0
            report["majority_baseline_metrics"] = compute_metrics((baseline, labels))
            predicted = scores.argmax(axis=1)
            report["human_false_positive_rate"] = float((predicted[labels==0] != 0).mean())
            report["error_analysis"] = {
                "human_as_ai_written": int(((labels==0) & (predicted==1)).sum()),
                "human_as_ai_paraphrased": int(((labels==0) & (predicted==2)).sum()),
                "ai_written_as_paraphrased": int(((labels==1) & (predicted==2)).sum()),
                "ai_paraphrased_as_written": int(((labels==2) & (predicted==1)).sum()),
                "length_buckets": {},
            }
            words = np.array([len(text.split()) for text in test["text"]])
            for name, mask in (("under_50_words",words<50),("50_to_250_words",(words>=50)&(words<=250)),("over_250_words",words>250)):
                if mask.any():
                    report["error_analysis"]["length_buckets"][name] = {
                        "samples":int(mask.sum()), "accuracy":float((predicted[mask]==labels[mask]).mean()),
                    }
        report["elapsed_seconds"] = perf_counter()-started
        report["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        report.update(memory)
        (output / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        summary = ["# TextOrigin evaluation", "", "Purpose: " + report["purpose"], "",
                   "Dataset SHA-256: " + digest, "",
                   "| Document metric | Value |", "|---|---|"]
        for name, value in report.get("document_test_metrics", {}).items():
            summary.append(f"| {name} | {value:.6f} |")
        summary.extend(["", "## Limitations", ""] + ["- " + item for item in report["limitations"]])
        summary.append("\nSee evaluation.json for confusion matrices, error analysis, baselines and provenance.")
        (output / "evaluation.md").write_text("\n".join(summary), encoding="utf-8")
        manifest = {
            "created_at": report["created_at"], "purpose": report["purpose"],
            "code_revision": report["code"]["revision"], "code_provenance": report["code"],
            "base_model": MODEL_ID, "base_model_revision": MODEL_REVISION,
            "dataset_sha256": digest, "labels": LABEL2ID, "max_length": args.max_length,
            "training_identity_sha256": file_sha256(identity_path),
            "evaluation_sha256": file_sha256(output / "evaluation.json"),
            "training_parameters": vars(args), "dependencies": versions,
            "test_metrics": report.get("document_test_metrics"),
            "aggregation": report["aggregation"],
        }
        (output / "textorigin-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"output_dir":str(output), "purpose":report["purpose"], "elapsed_seconds":report["elapsed_seconds"]}))
    finally:
        stop.set()
        monitor.join()
        (output / "resource-measurement.json").write_text(
            json.dumps({**memory, "elapsed_seconds": perf_counter()-started,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

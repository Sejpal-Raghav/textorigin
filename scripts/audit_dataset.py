from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_LABELS = {"human", "ai_written", "ai_paraphrased"}


def audit_dataset(input_path: Path, output_path: Path, report_path: Path) -> dict:
    raw_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    removed = Counter()
    accepted: list[dict[str, str]] = []
    normalized_text_groups: dict[str, set[str]] = defaultdict(set)
    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"text", "source", "source_id"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("Dataset is missing columns: " + ", ".join(sorted(missing)))
        fieldnames = list(reader.fieldnames or ())
        for row in reader:
            if None in row or any(row.get(name) is None for name in required):
                removed["malformed_row"] += 1
                continue
            text = row["text"].strip()
            if not text:
                removed["blank_text"] += 1
                continue
            if row["source"] not in EXPECTED_LABELS:
                removed["unknown_label"] += 1
                continue
            if not row["source_id"].strip():
                removed["blank_source_id"] += 1
                continue
            row["text"] = text
            accepted.append(row)
            normalized_text_groups[" ".join(text.lower().split())].add(row["source_id"])

    cross_family_texts = {
        normalized
        for normalized, ids in normalized_text_groups.items()
        if len(ids) > 1
    }
    if cross_family_texts:
        retained = []
        for row in accepted:
            normalized = " ".join(row["text"].lower().split())
            if normalized in cross_family_texts:
                removed["duplicate_cross_family"] += 1
            else:
                retained.append(row)
        accepted = retained

    originals = {(row["source_id"], " ".join(row["text"].lower().split()))
                 for row in accepted if row["source"] == "ai_written"}
    retained = []
    seen_rows = set()
    for row in accepted:
        normalized = " ".join(row["text"].lower().split())
        if row["source"] == "ai_paraphrased" and (row["source_id"], normalized) in originals:
            removed["unchanged_paraphrase"] += 1
            continue
        key = (row["source_id"], row["source"], normalized)
        if key in seen_rows:
            removed["duplicate_within_family"] += 1
            continue
        seen_rows.add(key)
        retained.append(row)
    accepted = retained

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)
    report = {
        "input_path": str(input_path.resolve()),
        "input_sha256": raw_hash,
        "output_path": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "accepted_rows": len(accepted),
        "class_counts": dict(sorted(Counter(row["source"] for row in accepted).items())),
        "removed_rows": dict(sorted(removed.items())),
        "normalized_texts_crossing_source_families": len(cross_family_texts),
        "duplicate_resolution": "excluded_all_cross-family_exact_duplicates",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and clean TextOrigin training data")
    parser.add_argument("--input", type=Path, default=Path("data/raw/combined.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/combined.clean.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/audit.json"))
    args = parser.parse_args()
    report = audit_dataset(args.input, args.output, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

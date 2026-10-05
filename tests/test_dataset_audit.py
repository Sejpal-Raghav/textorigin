import csv

from scripts.audit_dataset import audit_dataset


def test_audit_removes_invalid_rows_and_records_duplicates(tmp_path):
    source = tmp_path / "combined.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "source", "source_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"text": "Valid text", "source": "human", "source_id": "h1"},
                {"text": "  ", "source": "human", "source_id": "h2"},
                {"text": "Valid text", "source": "ai_written", "source_id": "a1"},
                {"text": "Other", "source": "unknown", "source_id": "x1"},
                {"text": "Third", "source": "ai_paraphrased", "source_id": ""},
            ]
        )
    output = tmp_path / "clean.csv"
    report = audit_dataset(source, output, tmp_path / "audit.json")
    assert report["accepted_rows"] == 0
    assert report["removed_rows"] == {
        "blank_source_id": 1,
        "blank_text": 1,
        "duplicate_cross_family": 2,
        "unknown_label": 1,
    }
    assert report["duplicate_resolution"] == "excluded_all_cross-family_exact_duplicates"
    assert output.exists()

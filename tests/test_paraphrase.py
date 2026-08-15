from unittest.mock import patch

import pandas as pd
import pytest

from src.data.paraphrase import generate_paraphrased_class


def ai_frame(size=12):
    return pd.DataFrame(
        {
            "text": [f"AI text {index}" for index in range(size)],
            "source": "ai_written",
            "source_id": [f"ai-{index}" for index in range(size)],
        }
    )


def fake_paraphrase(texts, on_batch=None, **_kwargs):
    results = [f"paraphrased {text}" for text in texts]
    for index in range(0, len(results), 2):
        batch = results[index : index + 2]
        if on_batch:
            on_batch(batch)
    return results


@patch("src.data.paraphrase.paraphrase_texts", side_effect=fake_paraphrase)
def test_generation_writes_resumable_checkpoint(_paraphrase, tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    result = generate_paraphrased_class(
        ai_frame(), sample_size=10, checkpoint_path=checkpoint
    )
    assert len(result) == 10
    assert len(pd.read_csv(checkpoint)) == 10


@patch("src.data.paraphrase.paraphrase_texts", side_effect=fake_paraphrase)
def test_generation_resumes_after_existing_rows(paraphrase, tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    first = generate_paraphrased_class(
        ai_frame(), sample_size=10, checkpoint_path=checkpoint
    )
    first.iloc[:4].to_csv(checkpoint, index=False)

    result = generate_paraphrased_class(
        ai_frame(), sample_size=10, checkpoint_path=checkpoint
    )
    assert len(result) == 10
    assert len(paraphrase.call_args.args[0]) == 6


def test_mismatched_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame(
        {"text": ["wrong"], "source": ["ai_paraphrased"], "source_id": ["other"]}
    ).to_csv(checkpoint, index=False)
    with pytest.raises(ValueError, match="does not match"):
        generate_paraphrased_class(
            ai_frame(), sample_size=10, checkpoint_path=checkpoint
        )

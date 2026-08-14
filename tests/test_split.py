import pandas as pd
import pytest

from src.data.split import grouped_stratified_split


def make_dataset():
    rows = []
    for index in range(20):
        rows.append({"text": f"human {index}", "source": "human", "source_id": f"h-{index}"})
        rows.append({"text": f"ai {index}", "source": "ai_written", "source_id": f"a-{index}"})
        rows.append({"text": f"paraphrase {index}", "source": "ai_paraphrased", "source_id": f"a-{index}"})
    return pd.DataFrame(rows)


def test_source_families_never_cross_splits():
    train, validation, test = grouped_stratified_split(make_dataset())
    id_sets = [set(frame["source_id"]) for frame in (train, validation, test)]
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])


def test_each_split_contains_all_classes():
    for partition in grouped_stratified_split(make_dataset()):
        assert set(partition["source"]) == {"human", "ai_written", "ai_paraphrased"}


def test_lineage_column_is_required():
    with pytest.raises(ValueError, match="source_id"):
        grouped_stratified_split(make_dataset().drop(columns="source_id"))

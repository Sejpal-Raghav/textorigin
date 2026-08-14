import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


@pytest.fixture
def mock_dataset():
    mock = MagicMock()
    mock.to_pandas.return_value = pd.DataFrame(
        {
            "text": [f"sample text {i}" for i in range(200)],
            "label": [0] * 100 + [1] * 100,
        }
    )
    return mock


def test_load_raw_data_columns(mock_dataset):
    with patch("src.data.loader.load_dataset", return_value=mock_dataset):
        from src.data.loader import load_raw_data

        result = load_raw_data(sample_size=50)

    assert "text" in result.columns
    assert "source" in result.columns
    assert "source_id" in result.columns


def test_load_raw_data_classes(mock_dataset):
    with patch("src.data.loader.load_dataset", return_value=mock_dataset):
        from src.data.loader import load_raw_data

        result = load_raw_data(sample_size=50)

    assert set(result["source"].unique()) == {"human", "ai_written"}


def test_load_raw_data_respects_sample_size(mock_dataset):
    with patch("src.data.loader.load_dataset", return_value=mock_dataset):
        from src.data.loader import load_raw_data

        result = load_raw_data(sample_size=30)

    assert len(result) <= 60  # 30 per class max


def test_load_raw_data_source_ids_are_unique(mock_dataset):
    with patch("src.data.loader.load_dataset", return_value=mock_dataset):
        from src.data.loader import load_raw_data

        result = load_raw_data(sample_size=50)

    assert result["source_id"].is_unique


def test_load_raw_data_supports_current_string_source_schema():
    dataset = MagicMock()
    dataset.to_pandas.return_value = pd.DataFrame(
        {
            "id": range(200),
            "text": [f"sample text {index}" for index in range(200)],
            "source": ["human"] * 100 + ["ai"] * 100,
        }
    )
    with patch("src.data.loader.load_dataset", return_value=dataset):
        from src.data.loader import load_raw_data

        result = load_raw_data(sample_size=20)

    assert set(result["source"]) == {"human", "ai_written"}
    assert result["source_id"].is_unique

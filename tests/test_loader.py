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

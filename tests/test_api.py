import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

MOCK_RESULT = {
    "human": 0.12,
    "ai_written": 0.71,
    "ai_paraphrased": 0.17,
    "top_features": ["low burstiness", "high AI phrase ratio"],
}


@pytest.fixture
def client():
    with patch("src.api.predictor.predict", return_value=MOCK_RESULT):
        from src.api.app import app
        yield TestClient(app)


def test_classify_success(client):
    response = client.post("/classify", json={"text": "This is a test sentence for classification."})
    assert response.status_code == 200
    data = response.json()
    assert "human" in data
    assert "ai_written" in data
    assert "ai_paraphrased" in data
    assert "top_features" in data
    assert isinstance(data["top_features"], list)


def test_classify_probabilities_are_floats(client):
    response = client.post("/classify", json={"text": "Some text here."})
    data = response.json()
    for key in ("human", "ai_written", "ai_paraphrased"):
        assert isinstance(data[key], float)


def test_classify_empty_text_returns_400(client):
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 400


def test_classify_whitespace_only_returns_400(client):
    response = client.post("/classify", json={"text": "   "})
    assert response.status_code == 400


def test_classify_too_long_returns_400(client):
    response = client.post("/classify", json={"text": "x" * 10_001})
    assert response.status_code == 400


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

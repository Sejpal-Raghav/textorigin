from unittest.mock import patch
from fastapi.testclient import TestClient

MOCK_METRICS = {
    "perplexity": 100.0,
    "burstiness": 0.5,
    "entropy": 4.5,
    "ai_phrase_ratio": 0.0,
    "avg_sentence_length": 15.0,
    "structural_regularity": 0.5,
}

@patch("src.api.humanizer.extract_features", return_value=MOCK_METRICS)
@patch("src.api.humanizer.compute_cosine_similarity", return_value=0.9)
@patch("src.api.humanizer.polish_text", return_value="This is a mock humanized text polished by LLM.")
def test_humanize_endpoint_with_llm(mock_polish, mock_cosine, mock_extract):
    from src.api.app import app
    client = TestClient(app)
    
    response = client.post("/humanize", json={"text": "Delve into the AI. It is worth noting."})
    assert response.status_code == 200
    
    data = response.json()
    assert data["used_llm"] is True
    assert data["polish_failed"] is False
    assert data["similarity_after"] == 0.9
    assert data["humanized_text"] == "This is a mock humanized text polished by LLM."


@patch("src.api.humanizer.extract_features", return_value=MOCK_METRICS)
@patch("src.api.humanizer.compute_cosine_similarity", return_value=0.9)
def test_humanize_endpoint_without_llm(mock_cosine, mock_extract):
    from src.api.app import app
    client = TestClient(app)
    
    response = client.post("/humanize", json={"text": "Delve into the AI. It is worth noting.", "use_llm": False})
    assert response.status_code == 200
    
    data = response.json()
    assert data["used_llm"] is False
    assert data["polish_failed"] is False
    assert "humanized_text" in data

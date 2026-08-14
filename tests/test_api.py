from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.errors import ModelUnavailableError

METRICS = {
    "perplexity": 100.0,
    "burstiness": 0.5,
    "entropy": 4.5,
    "ai_phrase_rate": 0.0,
    "avg_sentence_length": 15.0,
    "structural_regularity": 0.5,
}


class FakeClassificationService:
    def classify(self, _text, include_signals=True):
        return {
            "scores": {"human": 0.12, "ai_written": 0.71, "ai_paraphrased": 0.17},
            "observed_signals": ["low burstiness"] if include_signals else [],
            "truncated": False,
            "analyzed_tokens": 12,
            "model_version": "test-revision",
            "warnings": [],
        }


class UnavailableClassificationService:
    def classify(self, _text, include_signals=True):
        raise ModelUnavailableError("test model missing")


class FakeHumanizationService:
    def humanize(self, text, **_kwargs):
        return {
            "original_text": text,
            "humanized_text": "Revised text.",
            "status": "accepted_heuristic",
            "warnings": [],
            "similarity_threshold": 0.85,
            "original_metrics": METRICS,
            "post_heuristic_metrics": METRICS,
            "heuristic_similarity": 0.91,
            "llm_similarity": None,
            "post_llm_metrics": None,
        }


def make_client(classification=None, ready=True):
    services = SimpleNamespace(
        classification=classification or FakeClassificationService(),
        humanization=FakeHumanizationService(),
        classifier=SimpleNamespace(
            readiness=lambda: {
                "ready": ready,
                "model_path": "test",
                "missing_files": [] if ready else ["config.json"],
                "manifest_present": ready,
                "loaded": False,
            }
        ),
        close=lambda: None,
    )
    return TestClient(create_app(services))


def test_classify_success():
    with make_client() as client:
        response = client.post(
            "/classify",
            json={"text": "A sufficiently useful test sentence.", "include_signals": True},
        )
    assert response.status_code == 200
    assert response.json()["ai_written"] == 0.71
    assert response.json()["observed_signals"] == ["low burstiness"]


def test_classify_can_skip_expensive_signals():
    with make_client() as client:
        response = client.post(
            "/classify", json={"text": "Some text here.", "include_signals": False}
        )
    assert response.json()["observed_signals"] == []


def test_invalid_text_returns_422():
    with make_client() as client:
        assert client.post("/classify", json={"text": "   "}).status_code == 422
        assert client.post("/classify", json={"text": "x" * 10_001}).status_code == 422


def test_unknown_request_fields_are_rejected():
    with make_client() as client:
        response = client.post("/classify", json={"text": "Valid text.", "unknown": True})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_humanize_contract():
    with make_client() as client:
        response = client.post(
            "/humanize", json={"text": "A passage that can be safely revised.", "use_llm": False}
        )
    assert response.status_code == 200
    assert response.json()["status"] == "accepted_heuristic"
    assert response.json()["heuristic_similarity"] == 0.91


def test_humanize_threshold_cannot_weaken_server_policy():
    with make_client() as client:
        response = client.post(
            "/humanize",
            json={"text": "A valid passage.", "similarity_threshold": 0.5},
        )
    assert response.status_code == 422


def test_model_unavailable_is_a_structured_503():
    with make_client(UnavailableClassificationService()) as client:
        response = client.post("/classify", json={"text": "Valid text."})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_health_and_readiness_are_distinct():
    with make_client(ready=False) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503

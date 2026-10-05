from unittest.mock import MagicMock

import pytest
import requests

from src.errors import ExternalServiceError
from src.integrations.deepseek import DeepSeekClient


def response(status=200, payload=None, headers=None):
    result = MagicMock()
    result.status_code = status
    result.headers = headers or {}
    result.json.return_value = payload
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(str(status))
    return result


def make_client(api_key="owner"):
    client = DeepSeekClient("https://api.deepseek.com", "deepseek-chat", 1, api_key)
    client._session = MagicMock()
    return client


def test_polish_uses_request_key_without_exposing_it_in_payload():
    client = make_client()
    client._session.post.return_value = response(
        payload={
            "choices": [
                {"finish_reason": "stop", "message": {"content": "Revised text."}}
            ]
        }
    )
    assert client.polish(
        "Original text.", style="natural", strength="light", api_key="user-key"
    ) == "Revised text."
    call = client._session.post.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer user-key"
    assert "user-key" not in str(call.kwargs["json"])


def test_missing_and_invalid_keys_are_distinct_failures():
    client = make_client(None)
    with pytest.raises(ExternalServiceError, match="not configured"):
        client.polish("Text.", style="natural", strength="light")
    client._session.post.return_value = response(status=401)
    with pytest.raises(ExternalServiceError, match="rejected"):
        client.polish(
            "Text.", style="natural", strength="light", api_key="invalid"
        )
    assert client._session.post.call_count == 1


def test_incomplete_output_is_rejected():
    client = make_client()
    client._session.post.return_value = response(
        payload={
            "choices": [
                {"finish_reason": "length", "message": {"content": "Partial"}}
            ]
        }
    )
    with pytest.raises(ExternalServiceError, match="incomplete"):
        client.polish("Text.", style="formal", strength="medium")


def test_retryable_response_is_bounded(monkeypatch):
    monkeypatch.setattr("src.integrations.deepseek.time.sleep", lambda _seconds: None)
    client = make_client()
    client._session.post.return_value = response(status=429, headers={"Retry-After": "0"})
    with pytest.raises(ExternalServiceError, match="failed"):
        client.polish("Text.", style="casual", strength="light")
    assert client._session.post.call_count == 3

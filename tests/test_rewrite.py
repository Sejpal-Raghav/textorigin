import json

import pytest

from src.cli import main
from src.services.rewrite import RewriteOptions, protected_spans_preserved, rewrite_document


@pytest.mark.parametrize("style", ["natural", "formal", "casual"])
@pytest.mark.parametrize("strength", ["light", "medium"])
def test_all_style_strength_combinations_are_deterministic(style, strength):
    text = "It is worth noting that we utilize this method.\n\nTherefore, it works."
    first = rewrite_document(text, style=style, strength=strength)
    assert first == rewrite_document(text, style=style, strength=strength)
    assert "\n\n" in first


def test_protected_content_and_markdown_structure_are_preserved():
    text = (
        "It is worth noting that [12] reports 42.5% at https://example.com.\n"
        "- Keep \"exact words\" and `value = 42`."
    )
    result = rewrite_document(text, style="natural", strength="medium")
    assert protected_spans_preserved(text, result)
    assert result.splitlines()[1].startswith("- ")


def test_rewriter_does_not_join_sentences_with_invented_relationship():
    text = "This is short. This is separate."
    assert rewrite_document(text) == text


def test_invalid_options_and_empty_text_are_rejected():
    with pytest.raises(ValueError):
        RewriteOptions(style="academic")
    with pytest.raises(ValueError, match="empty"):
        rewrite_document(" \n ")


def test_cli_rewrite_json(capsys):
    code = main([
        "rewrite",
        "--text",
        "It is worth noting that we utilize this.",
        "--style",
        "natural",
        "--strength",
        "light",
        "--format",
        "json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["operation"] == "rewrite"
    assert payload["provider"] == "local"
    assert payload["humanized_text"] == "We use this."


def test_cli_refuses_to_overwrite_output(tmp_path, capsys):
    output = tmp_path / "existing.txt"
    output.write_text("keep", encoding="utf-8")
    code = main(["rewrite", "--text", "Text.", "--output", str(output)])
    assert code == 2
    assert output.read_text(encoding="utf-8") == "keep"
    assert "--force" in capsys.readouterr().err


def test_cli_deepseek_mode_uses_shared_service(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.cli._deepseek_rewrite",
        lambda text, style, strength: {
            "humanized_text": "Polished.",
            "status": "accepted_llm",
            "warnings": [],
            "complete": True,
        },
    )
    code = main([
        "rewrite",
        "--text",
        "Original.",
        "--provider",
        "deepseek",
        "--format",
        "json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["provider"] == "deepseek"
    assert payload["status"] == "accepted_llm"

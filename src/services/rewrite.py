from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RewriteStyle = Literal["natural", "formal", "casual"]
RewriteStrength = Literal["light", "medium"]


@dataclass(frozen=True)
class RewriteOptions:
    style: RewriteStyle = "natural"
    strength: RewriteStrength = "light"

    def __post_init__(self) -> None:
        if self.style not in {"natural", "formal", "casual"}:
            raise ValueError("style must be natural, formal, or casual")
        if self.strength not in {"light", "medium"}:
            raise ValueError("strength must be light or medium")


PROTECTED_PATTERN = re.compile(
    r"(```[\s\S]*?```|`[^`\r\n]+`|https?://[^\s<>()]+|"
    r"\[[0-9]+(?:\s*[-,]\s*[0-9]+)*\]|"
    r'"[^"\r\n]*"|“[^”\r\n]*”|'
    r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:\s*%|\s+[A-Za-z]{1,5})?)"
)

BASE_REPLACEMENTS = {
    "delve into": "examine",
    "it's worth noting that": "",
    "it is worth noting that": "",
    "it's important to note that": "",
    "it is important to note that": "",
    "as previously mentioned": "",
    "in the realm of": "in",
    "at the end of the day": "ultimately",
    "moving forward": "later",
    "utilize": "use",
    "facilitate": "help",
}

STYLE_REPLACEMENTS = {
    "natural": {
        "furthermore": "also",
        "moreover": "also",
        "in conclusion": "",
        "in summary": "",
        "to summarize": "",
    },
    "formal": {
        "a lot of": "many",
        "get": "obtain",
        "kids": "children",
        "okay": "acceptable",
    },
    "casual": {
        "therefore": "so",
        "however": "but",
        "nevertheless": "still",
        "approximately": "about",
    },
}

MEDIUM_REPLACEMENTS = {
    "natural": {
        "due to the fact that": "because",
        "in order to": "to",
        "has the ability to": "can",
        "a number of": "several",
    },
    "formal": {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "doesn't": "does not",
    },
    "casual": {
        "do not": "don't",
        "cannot": "can't",
        "will not": "won't",
        "does not": "doesn't",
    },
}


def _replace_phrase(text: str, phrase: str, replacement: str) -> str:
    return re.sub(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        replacement,
        text,
        flags=re.IGNORECASE,
    )


def _edit_prose(text: str, options: RewriteOptions) -> str:
    replacements = dict(BASE_REPLACEMENTS)
    replacements.update(STYLE_REPLACEMENTS[options.style])
    if options.strength == "medium":
        replacements.update(MEDIUM_REPLACEMENTS[options.style])
    for phrase, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = _replace_phrase(text, phrase, replacement)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;!?])", r"\1", text)
    text = text.strip(" \t")
    text = re.sub(r"(^|(?<=[.!?])\s+)([a-z])", lambda match: match.group(1) + match.group(2).upper(), text)
    return text


def rewrite_block(text: str, options: RewriteOptions) -> str:
    """Rewrite prose while restoring protected spans byte-for-byte."""
    protected: list[str] = []

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\ue000{len(protected) - 1}\ue001"

    working = PROTECTED_PATTERN.sub(stash, text)
    working = _edit_prose(working, options)
    for index, original in enumerate(protected):
        working = working.replace(f"\ue000{index}\ue001", original)
    return working


def rewrite_document(
    text: str,
    style: RewriteStyle = "natural",
    strength: RewriteStrength = "light",
) -> str:
    """Apply conservative edits while preserving line and paragraph separators."""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    options = RewriteOptions(style=style, strength=strength)
    parts = re.split(r"(\r?\n)", text)
    return "".join(
        part if part in {"\n", "\r\n"} or not part.strip() else rewrite_block(part, options)
        for part in parts
    )


def protected_spans(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in PROTECTED_PATTERN.finditer(text))


def protected_spans_preserved(original: str, candidate: str) -> bool:
    return protected_spans(original) == protected_spans(candidate)

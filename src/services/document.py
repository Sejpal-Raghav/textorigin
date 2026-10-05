from __future__ import annotations

from src.services.rewrite import PROTECTED_PATTERN


def document_chunks(text: str, max_chars: int = 4_000) -> list[str]:
    """Split losslessly at nearby whitespace without cutting protected spans."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if not text:
        return []
    protected = [(match.start(), match.end()) for match in PROTECTED_PATTERN.finditer(text)]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            containing = next(
                ((span_start, span_end) for span_start, span_end in protected if span_start < end < span_end),
                None,
            )
            if containing is not None:
                end = containing[1]
            else:
                search_start = start + max_chars // 2
                newline = text.rfind("\n", search_start, end)
                space = text.rfind(" ", search_start, end)
                boundary = max(newline, space)
                if boundary >= search_start:
                    end = boundary + 1
        chunks.append(text[start:end])
        start = end
    return chunks

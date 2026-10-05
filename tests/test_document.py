from src.services.document import document_chunks


def test_chunks_round_trip_a_large_document():
    text = ("A paragraph with ordinary words and punctuation. " * 3_000) + "TAIL_SENTINEL"
    chunks = document_chunks(text, max_chars=500)
    assert len(chunks) > 100
    assert "".join(chunks) == text
    assert chunks[-1].endswith("TAIL_SENTINEL")


def test_protected_span_is_not_cut():
    protected = "`" + ("x" * 800) + "`"
    text = "Before " + protected + " after"
    chunks = document_chunks(text, max_chars=200)
    assert "".join(chunks) == text
    assert any(protected in chunk for chunk in chunks)


def test_whitespace_and_empty_inputs_are_lossless():
    assert document_chunks("") == []
    text = " \n\n  \n"
    assert "".join(document_chunks(text, max_chars=100)) == text

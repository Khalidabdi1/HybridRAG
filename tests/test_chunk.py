from hybridrag.pipeline.extract import chunk_text, clean_text, html_to_text


def test_chunk_text_basic():
    text = "Sentence one. " * 200  # ~2800 chars
    chunks = chunk_text(text, doc_id="d1", chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(len(c.text) <= 600 for c in chunks)  # window + boundary slack
    # ids are unique
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_overlap_validation():
    try:
        chunk_text("hello world", doc_id="d", chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError when overlap >= chunk_size")


def test_empty_text():
    assert chunk_text("", doc_id="d") == []


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\t\tc") == "a b c"


def test_html_to_text_drops_scripts():
    html = "<html><head><style>x{}</style></head><body><p>Hello</p>" \
           "<script>var a=1;</script><p>World</p></body></html>"
    text = html_to_text(html)
    assert "Hello" in text
    assert "World" in text
    assert "var a" not in text

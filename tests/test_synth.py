from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.synth.reader import (
    Answer,
    ExtractiveReader,
    LLMReader,
    build_reader,
)
from hybridrag.types import Modality, SearchResult


def _r(doc, score=1.0, text="", modality=Modality.TEXT, image_path=None, page=0):
    return SearchResult(
        doc_id=doc,
        score=score,
        modality=modality,
        unit_id=doc + "_u",
        text=text,
        image_path=image_path,
        page=page,
    )


# --------------------------------------------------------------- ExtractiveReader

def test_extractive_picks_answer_bearing_sentence():
    reader = ExtractiveReader(max_sentences=1)
    results = [
        _r(
            "d1",
            text=(
                "The weather was pleasant that morning. "
                "Quarterly revenue grew to 4.2 billion dollars on cloud demand. "
                "The cafeteria served pancakes."
            ),
        ),
    ]
    ans = reader.synthesize("what was quarterly revenue", results)
    assert isinstance(ans, Answer)
    assert "4.2 billion" in ans.text
    assert ans.text.strip().endswith("[1]")
    assert len(ans.citations) == 1
    assert ans.citations[0].doc_id == "d1"


def test_extractive_is_grounded_only_in_retrieved_text():
    reader = ExtractiveReader(max_sentences=2)
    results = [_r("d1", text="Photosynthesis converts light into chemical energy.")]
    ans = reader.synthesize("how does photosynthesis work", results)
    # Every non-citation token must come from the source sentence.
    stripped = ans.text.replace("[1]", "").strip()
    assert stripped in results[0].text


def test_extractive_dedupes_repeated_sentences_across_hits():
    reader = ExtractiveReader(max_sentences=3)
    same = "Revenue increased sharply this fiscal year."
    results = [_r("d1", text=same), _r("d2", text=same)]
    ans = reader.synthesize("revenue increase", results)
    # The identical sentence should appear once, citing a single source.
    assert ans.text.count("Revenue increased") == 1
    assert len(ans.citations) == 1


def test_extractive_multiple_citations_get_distinct_markers():
    reader = ExtractiveReader(max_sentences=2)
    results = [
        _r("d1", text="Apples are a good source of fiber."),
        _r("d2", text="Apples grow on deciduous trees in temperate climates."),
    ]
    ans = reader.synthesize("apples fiber trees", results)
    markers = sorted(c.marker for c in ans.citations)
    assert markers == [1, 2]
    assert "[1]" in ans.text and "[2]" in ans.text


def test_extractive_surfaces_visual_evidence_when_no_text():
    reader = ExtractiveReader(max_sentences=2)
    results = [_r("chart", text="", modality=Modality.VISION, image_path="/t/p1.png", page=3)]
    ans = reader.synthesize("show the revenue chart", results)
    assert ans.visual_evidence
    assert ans.visual_evidence[0].image_path == "/t/p1.png"
    assert "[1]" in ans.text


def test_extractive_attaches_relevant_figure_as_trailing_citation():
    reader = ExtractiveReader(max_sentences=1)
    results = [
        _r("d1", text="Net revenue reached a record this quarter."),
        _r("fig", text="", modality=Modality.VISION, image_path="/t/fig.png", page=2),
    ]
    ans = reader.synthesize("record revenue", results)
    assert any(c.image_path for c in ans.citations)
    assert ans.visual_evidence


def test_extractive_excludes_weak_stopword_only_matches():
    # The finance sentence overlaps the query only on the stopword "is"; it must
    # not be padded into a Python answer just to reach max_sentences.
    reader = ExtractiveReader(max_sentences=3)
    results = [
        _r("py", text="Python is dynamically typed and garbage collected."),
        _r("fin", text="A balance sheet is a financial statement of equity."),
    ]
    ans = reader.synthesize("how is python typed", results)
    assert "Python" in ans.text
    assert "balance sheet" not in ans.text
    assert all(c.doc_id == "py" for c in ans.citations)


def test_extractive_empty_results():
    ans = ExtractiveReader().synthesize("anything", [])
    assert "No relevant results" in ans.text
    assert ans.citations == []


def test_answer_formatted_lists_sources():
    reader = ExtractiveReader(max_sentences=1)
    results = [_r("d1", text="The capital of France is Paris.", page=7)]
    ans = reader.synthesize("capital of France", results)
    out = ans.formatted()
    assert "Sources:" in out
    assert "[1]" in out
    assert "p.7" in out


# --------------------------------------------------------------------- LLMReader

def test_llm_reader_builds_grounded_prompt_and_delegates():
    captured = {}

    def fake_generate(prompt, image_paths):
        captured["prompt"] = prompt
        captured["images"] = image_paths
        return "Revenue was 4.2B [1]."

    reader = LLMReader(fake_generate, max_context=3)
    results = [
        _r("d1", text="Quarterly revenue grew to 4.2 billion dollars."),
        _r("fig", text="", modality=Modality.VISION, image_path="/t/p.png", page=1),
    ]
    ans = reader.synthesize("what was revenue", results)
    assert ans.text == "Revenue was 4.2B [1]."
    assert ans.reader == "llm"
    # The prompt is grounded in the numbered sources and forbids invention.
    assert "Sources:" in captured["prompt"]
    assert "ONLY" in captured["prompt"]
    assert "4.2 billion" in captured["prompt"]
    # Tile images are passed through for a VLM to read.
    assert captured["images"] == ["/t/p.png"]
    assert len(ans.citations) == 2


def test_llm_reader_can_skip_images():
    results = [_r("fig", text="", modality=Modality.VISION, image_path="/t/p.png")]
    captured = {}

    def fake(prompt, imgs):
        captured["imgs"] = imgs
        return "ok"

    LLMReader(fake, include_images=False).synthesize("q", results)
    assert captured["imgs"] == []


# ------------------------------------------------------------------- build_reader

def test_build_reader_extractive_by_default():
    assert isinstance(build_reader(HybridConfig()), ExtractiveReader)


def test_build_reader_none_disables():
    assert build_reader(HybridConfig(reader_model="none")) is None


def test_build_reader_respects_max_sentences():
    reader = build_reader(HybridConfig(answer_max_sentences=5))
    assert isinstance(reader, ExtractiveReader)
    assert reader.max_sentences == 5


# ------------------------------------------------------------------ engine wiring

def test_engine_answer_end_to_end():
    rag = HybridRAG(HybridConfig())
    rag.add_text(
        "doc1",
        "The HybridRAG engine fuses text and pixel retrieval. "
        "It stores two independent vector indexes and merges results with RRF. "
        "Selective pixel indexing skips the vision path for text-native pages.",
        title="Overview",
    )
    ans = rag.answer("how does hybridrag fuse modalities")
    assert isinstance(ans, Answer)
    assert ans.text
    assert ans.citations
    assert ans.citations[0].doc_id == "doc1"


def test_engine_answer_accepts_custom_reader():
    rag = HybridRAG(HybridConfig())
    rag.add_text("doc1", "Paris is the capital of France.")
    seen = {}

    def gen(prompt, imgs):
        seen["called"] = True
        return "Paris [1]."

    ans = rag.answer("capital of France", reader=LLMReader(gen))
    assert seen.get("called")
    assert ans.text == "Paris [1]."


def test_engine_answer_disabled_reader_falls_back_to_extractive():
    rag = HybridRAG(HybridConfig(reader_model="none"))
    assert rag.reader is None
    rag.add_text("doc1", "The mitochondrion is the powerhouse of the cell.")
    ans = rag.answer("what is the powerhouse of the cell")
    # Even with synthesis disabled in config, an explicit answer() call still
    # produces a grounded extractive answer rather than crashing.
    assert "mitochondrion" in ans.text.lower()
    assert ans.reader == "extractive"

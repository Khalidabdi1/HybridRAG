"""Answer synthesis — turn fused search hits into a grounded, cited answer.

Retrieval is only half of RAG. The other half — the *final readout* — reads the
retrieved evidence and composes an answer. In Pixel RAG that readout always
means feeding page screenshots to a VLM, which is the single most expensive call
in the system (see the user's "Final readout" cost concern). HybridRAG makes the
readout a *pluggable* step with a cheap default:

* :class:`ExtractiveReader` (default, **numpy alone**) — selects the few
  sentences from the retrieved text chunks that best answer the query (BM25 over
  sentences), stitches them into a short answer, and attaches a numbered
  citation to every span. It never hallucinates: every word comes from an
  indexed chunk. Vision-only hits (tables/charts with no extracted text) are
  surfaced as *visual-evidence* citations so the caller knows a figure is
  relevant even though no text reader can read it.

* :class:`LLMReader` (opt-in) — a thin wrapper around any text/VLM generator you
  pass in (Claude, Qwen-VL, a local model, …). It builds a grounded prompt from
  the top text chunks *and* the top tile image paths and delegates generation to
  your callable, so the heavy model is only ever invoked on the handful of
  fused hits, never corpus-wide.

This mirrors the embedder / reranker design: a genuinely useful dependency-free
default, and a clean seam for a real model when you want one.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import HybridConfig
from ..types import Modality, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Split on sentence-ending punctuation followed by whitespace. Deliberately
# simple — good enough for prose, and chunk text is already cleaned upstream.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_RE.split(text)]
    return [s for s in parts if s]


@dataclass
class Citation:
    """A numbered source backing part of an answer."""

    marker: int  # the [n] referenced in the answer text
    doc_id: str
    modality: Modality
    unit_id: str = ""
    page: int = 0
    snippet: str = ""  # the exact source span (text) or a visual descriptor
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker,
            "doc_id": self.doc_id,
            "modality": self.modality.value,
            "unit_id": self.unit_id,
            "page": self.page,
            "snippet": self.snippet,
            "image_path": self.image_path,
        }


@dataclass
class Answer:
    """A synthesized answer plus the citations that ground it."""

    query: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    reader: str = ""
    # Hits that carry a relevant figure/table but no readable text — the caller
    # may want to show these images even though the extractive reader can't read
    # them. Always a subset of `citations` (those with image_path set).
    visual_evidence: List[Citation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "text": self.text,
            "reader": self.reader,
            "citations": [c.to_dict() for c in self.citations],
            "visual_evidence": [c.to_dict() for c in self.visual_evidence],
        }

    def formatted(self) -> str:
        """A human-readable answer with a numbered source list appended."""
        lines = [self.text.strip() or "(no answer)"]
        if self.citations:
            lines.append("")
            lines.append("Sources:")
            for c in self.citations:
                where = f"doc={c.doc_id}"
                if c.page:
                    where += f" p.{c.page}"
                tag = "image" if c.image_path else "text"
                lines.append(f"  [{c.marker}] ({tag}) {where} — {c.snippet}")
        return "\n".join(lines)


def _result_text(result: SearchResult) -> str:
    """Readable text for a hit (chunks have body text; tiles may carry a title/ocr)."""
    if result.text:
        return result.text
    meta = result.meta or {}
    return str(meta.get("text") or meta.get("ocr") or meta.get("title") or "")


class Reader(ABC):
    """Synthesizes an :class:`Answer` from a query and ranked search results."""

    name: str = "reader"

    @abstractmethod
    def synthesize(self, query: str, results: Sequence[SearchResult]) -> Answer:
        ...


class ExtractiveReader(Reader):
    """Compose an answer by extracting the best sentences from retrieved text.

    Honest and cheap: the answer is built only from sentences that actually
    appear in indexed chunks, each carrying a citation, so there is nothing to
    hallucinate. Sentences are ranked by BM25 against the query (IDF-weighted,
    length-normalized) over the candidate sentence set, which discriminates the
    answer-bearing sentence from neighbouring filler far better than raw cosine.
    """

    name = "extractive"

    def __init__(
        self,
        max_sentences: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
        min_sentence_chars: int = 20,
        min_relative_score: float = 0.15,
    ) -> None:
        self.max_sentences = max_sentences
        self.k1 = k1
        self.b = b
        self.min_sentence_chars = min_sentence_chars
        # A sentence must score at least this fraction of the best sentence's
        # score to be added beyond the first — keeps weak, stopword-only matches
        # (e.g. a sentence that overlaps the query on "is"/"the" alone) out of the
        # answer instead of padding it to `max_sentences`.
        self.min_relative_score = min_relative_score

    def _bm25(self, query_terms: List[str], sentences: List[str]) -> List[float]:
        if not query_terms or not sentences:
            return [0.0] * len(sentences)
        toks = [_tokenize(s) for s in sentences]
        lengths = [len(t) for t in toks]
        n = len(sentences)
        avgdl = (sum(lengths) / n) or 1.0
        df: Counter = Counter()
        for t in toks:
            for term in set(t):
                df[term] += 1
        q_unique = set(query_terms)
        idf = {
            term: math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            for term in q_unique
            if df.get(term)
        }
        scores: List[float] = []
        for t, length in zip(toks, lengths):
            tf = Counter(t)
            norm = self.k1 * (1.0 - self.b + self.b * (length / avgdl))
            s = 0.0
            for term, weight in idf.items():
                f = tf.get(term, 0)
                if f:
                    s += weight * (f * (self.k1 + 1.0)) / (f + norm)
            scores.append(s)
        return scores

    def synthesize(self, query: str, results: Sequence[SearchResult]) -> Answer:
        q_terms = _tokenize(query)

        # Build the sentence pool, remembering which hit each sentence came from
        # and its rank (so ties break toward higher-ranked hits).
        pool: List[Dict[str, Any]] = []
        visual: List[Citation] = []
        for rank, r in enumerate(results):
            text = _result_text(r).strip()
            if not text:
                if r.image_path or r.modality == Modality.VISION:
                    visual.append(
                        Citation(
                            marker=0,  # assigned later if cited
                            doc_id=r.doc_id,
                            modality=r.modality,
                            unit_id=r.unit_id,
                            page=r.page,
                            snippet="visual evidence (table/chart/figure) — "
                            "enable an LLM/VLM reader to read its contents",
                            image_path=r.image_path,
                        )
                    )
                continue
            for sent in _split_sentences(text):
                if len(sent) < self.min_sentence_chars and len(_split_sentences(text)) > 1:
                    continue
                pool.append({"sent": sent, "result": r, "rank": rank})

        if not pool:
            # Nothing readable — fall back to surfacing the visual evidence.
            return self._visual_only_answer(query, visual)

        sentences = [p["sent"] for p in pool]
        scores = self._bm25(q_terms, sentences)
        order = sorted(
            range(len(pool)),
            key=lambda i: (scores[i], -pool[i]["rank"]),
            reverse=True,
        )

        best_score = scores[order[0]] if order else 0.0
        floor = self.min_relative_score * best_score
        chosen: List[int] = []
        seen_sentences = set()
        for i in order:
            # Keep the single best sentence always; require later ones to be
            # genuinely relevant (above a fraction of the best) rather than a
            # stopword-only match padding the answer.
            if chosen and (scores[i] <= 0 or scores[i] < floor):
                break
            key = pool[i]["sent"].lower()
            if key in seen_sentences:
                continue
            seen_sentences.add(key)
            chosen.append(i)
            if len(chosen) >= self.max_sentences:
                break

        # Present the answer in source order (rank, then position) for readability,
        # not score order, so it reads coherently.
        chosen.sort(key=lambda i: (pool[i]["rank"], sentences.index(pool[i]["sent"])))

        citations: List[Citation] = []
        cite_by_unit: Dict[str, int] = {}
        answer_parts: List[str] = []
        for i in chosen:
            r = pool[i]["result"]
            unit_key = f"{r.doc_id}:{r.unit_id}"
            if unit_key not in cite_by_unit:
                marker = len(citations) + 1
                cite_by_unit[unit_key] = marker
                citations.append(
                    Citation(
                        marker=marker,
                        doc_id=r.doc_id,
                        modality=r.modality,
                        unit_id=r.unit_id,
                        page=r.page,
                        snippet=pool[i]["sent"][:200],
                        image_path=r.image_path,
                    )
                )
            marker = cite_by_unit[unit_key]
            answer_parts.append(f"{pool[i]['sent']} [{marker}]")

        # Append any relevant visual evidence as trailing citations so the caller
        # can render the figures even though they were not read.
        for v in visual[: max(0, 1 + self.max_sentences - len(citations))]:
            marker = len(citations) + 1
            v.marker = marker
            citations.append(v)

        return Answer(
            query=query,
            text=" ".join(answer_parts),
            citations=citations,
            reader=self.name,
            visual_evidence=[c for c in citations if c.image_path],
        )

    def _visual_only_answer(self, query: str, visual: List[Citation]) -> Answer:
        for idx, v in enumerate(visual, 1):
            v.marker = idx
        if visual:
            text = (
                "No extractable text matched this query, but relevant visual "
                "evidence was retrieved "
                + " ".join(f"[{v.marker}]" for v in visual)
                + ". Enable an LLM/VLM reader to read the figures."
            )
        else:
            text = "No relevant results were found in the index."
        return Answer(
            query=query,
            text=text,
            citations=visual,
            reader=self.name,
            visual_evidence=list(visual),
        )


# Type of a user-supplied generator: (prompt, image_paths) -> answer text.
Generate = Callable[[str, List[str]], str]


class LLMReader(Reader):
    """Grounded generation over the top hits via a user-supplied model callable.

    Pass any ``generate(prompt, image_paths) -> str`` function — a Claude API
    call, a Qwen-VL pipeline, a local model — and this reader builds a grounded,
    citation-instructed prompt from the top text chunks and the top tile images,
    then delegates. The heavy model only ever sees the handful of fused hits, so
    cost stays bounded. The numbered context maps 1:1 onto the citations
    returned, so the model's ``[n]`` markers stay meaningful.
    """

    name = "llm"

    def __init__(
        self,
        generate: Generate,
        max_context: int = 5,
        include_images: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self._generate = generate
        self.max_context = max_context
        self.include_images = include_images
        if name:
            self.name = name

    def build_prompt(self, query: str, results: Sequence[SearchResult]) -> str:
        blocks: List[str] = []
        for i, r in enumerate(results[: self.max_context], 1):
            text = _result_text(r).strip()
            if text:
                body = text[:800]
            elif r.image_path:
                body = f"[image tile, page {r.page} — see attached image {i}]"
            else:
                body = "[no content]"
            blocks.append(f"[{i}] (doc={r.doc_id}, page={r.page}, {r.modality.value})\n{body}")
        context = "\n\n".join(blocks) if blocks else "(no retrieved context)"
        return (
            "Answer the question using ONLY the numbered sources below. Cite every "
            "claim with its source number in square brackets, e.g. [1]. If the "
            "sources do not contain the answer, say so.\n\n"
            f"Question: {query}\n\nSources:\n{context}\n\nAnswer:"
        )

    def synthesize(self, query: str, results: Sequence[SearchResult]) -> Answer:
        top = list(results[: self.max_context])
        prompt = self.build_prompt(query, top)
        image_paths: List[str] = []
        if self.include_images:
            image_paths = [r.image_path for r in top if r.image_path]
        text = self._generate(prompt, image_paths)
        citations = [
            Citation(
                marker=i,
                doc_id=r.doc_id,
                modality=r.modality,
                unit_id=r.unit_id,
                page=r.page,
                snippet=(_result_text(r)[:200] or "image tile"),
                image_path=r.image_path,
            )
            for i, r in enumerate(top, 1)
        ]
        return Answer(
            query=query,
            text=(text or "").strip(),
            citations=citations,
            reader=self.name,
            visual_evidence=[c for c in citations if c.image_path],
        )


def build_reader(config: HybridConfig) -> Optional[Reader]:
    """Return the configured reader, or ``None`` when synthesis is disabled.

    ``reader_model`` is ``"extractive"`` (default, no deps), or ``"none"`` to
    disable. An :class:`LLMReader` cannot be built from config alone (it needs a
    generator callable) — construct it in code and pass it to
    ``HybridRAG.answer(reader=...)``.
    """
    model = (getattr(config, "reader_model", "extractive") or "extractive").strip()
    if model in ("", "none"):
        return None
    if model == "extractive":
        return ExtractiveReader(
            max_sentences=getattr(config, "answer_max_sentences", 3)
        )
    # Unknown id: fall back to extractive rather than failing the readout.
    return ExtractiveReader(max_sentences=getattr(config, "answer_max_sentences", 3))

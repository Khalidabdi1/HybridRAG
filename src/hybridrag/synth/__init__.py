"""Answer synthesis (the RAG "final readout") over fused search hits."""

from .reader import (
    Answer,
    Citation,
    ExtractiveReader,
    LLMReader,
    Reader,
    build_reader,
)

__all__ = [
    "Answer",
    "Citation",
    "Reader",
    "ExtractiveReader",
    "LLMReader",
    "build_reader",
]

"""End-to-end citation-grounded answering: retrieve -> rerank -> cite -> validate.

The key production discipline this encodes: **never answer with a passage that
wasn't actually surfaced by retrieval**, and flag answers whose supporting
passage looks weak (citation validation), rather than silently returning a
low-confidence citation as if it were solid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .corpus import Passage
from .rerank import HeuristicReranker, Reranker
from .retrieve import HybridRetriever


@dataclass
class Answer:
    question: str
    cited_passage: Optional[Passage]
    confidence: float
    validated: bool           # did the citation clear the validation floor?


class CitationPipeline:
    def __init__(self, retriever: HybridRetriever, reranker: Optional[Reranker] = None,
                 validation_floor: float = 0.2) -> None:
        self.retriever = retriever
        self.reranker = reranker or HeuristicReranker()
        self.validation_floor = validation_floor

    def answer(self, question: str, retrieve_k: int = 10, rerank_k: int = 1) -> Answer:
        hits = self.retriever.search(question, k=retrieve_k)
        if not hits:
            return Answer(question, None, 0.0, False)
        top = self.reranker.rerank(question, hits, k=rerank_k)
        if not top:
            return Answer(question, None, 0.0, False)
        best = top[0]
        validated = best.score >= self.validation_floor
        return Answer(question, best.passage, best.score, validated)

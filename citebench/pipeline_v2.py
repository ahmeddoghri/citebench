"""Citation answering that can return nothing.

The original pipeline validated a citation by comparing the reranker's score
against a floor of 0.2. That score is a topical similarity, so a passage from
the right corpus about the wrong subject clears it easily: "Who is the
principal investigator?" returned a stability-testing passage at 0.39,
marked validated.

Three of four questions with no answer in the corpus came back with a
validated citation. In a regulated domain that is the failure that matters,
because a reviewer checking a submission sees a passage reference and treats
the claim as sourced.

So support is gated separately from ranking here. Ranking picks the best of
what was retrieved; support asks whether the best is good enough to cite at
all, and a refusal carries the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .corpus import Passage
from .rerank_v2 import ContentReranker, RescoredV2
from .retrieve import HybridRetriever

#: A passage has to address this share of the question's content words before
#: it can be cited. Below it, the honest answer is that the corpus does not
#: contain one.
DEFAULT_SUPPORT_FLOOR = 0.34


@dataclass
class AnswerV2:
    question: str
    cited_passage: Optional[Passage]
    confidence: float
    support: float
    validated: bool
    #: Why nothing was cited, when nothing was.
    reason: str = ""
    #: Passages considered and rejected, with the reason each was dropped.
    rejected: List[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.cited_passage is None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "cited_passage_id": self.cited_passage.id if self.cited_passage else None,
            "cited_doc": self.cited_passage.doc if self.cited_passage else None,
            "confidence": round(self.confidence, 4),
            "support": round(self.support, 4),
            "validated": self.validated,
            "reason": self.reason,
            "rejected": self.rejected,
        }


class CitationPipelineV2:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[ContentReranker] = None,
        support_floor: float = DEFAULT_SUPPORT_FLOOR,
        reject_superseded: bool = True,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or ContentReranker()
        self.support_floor = support_floor
        self.reject_superseded = reject_superseded

    def answer(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> AnswerV2:
        hits = self.retriever.search(question, k=retrieve_k)
        if not hits:
            return AnswerV2(question, None, 0.0, 0.0, False, "retrieval returned nothing")

        ranked: List[RescoredV2] = self.reranker.rerank(question, hits, k=rerank_k)
        if not ranked:
            return AnswerV2(question, None, 0.0, 0.0, False, "reranking returned nothing")

        rejected: List[str] = []
        for candidate in ranked:
            if self.reject_superseded and candidate.explanation.superseded:
                rejected.append(
                    f"{candidate.passage.id}: superseded "
                    f"({candidate.explanation.superseded_evidence})"
                )
                continue
            if candidate.support < self.support_floor:
                rejected.append(
                    f"{candidate.passage.id}: support {candidate.support:.2f} "
                    f"below floor {self.support_floor:.2f}"
                )
                continue
            return AnswerV2(
                question=question,
                cited_passage=candidate.passage,
                confidence=candidate.score,
                support=candidate.support,
                validated=True,
                rejected=rejected,
            )

        best = ranked[0]
        return AnswerV2(
            question=question,
            cited_passage=None,
            confidence=best.score,
            support=best.support,
            validated=False,
            reason=(
                "every candidate was superseded or did not support the question "
                f"(best support {best.support:.2f}, floor {self.support_floor:.2f})"
            ),
            rejected=rejected,
        )

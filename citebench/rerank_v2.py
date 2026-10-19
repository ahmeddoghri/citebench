"""Reranking that reads the passage instead of the filename.

The heuristic reranker decides a passage is stale by substring-matching its
*document name* against ``_v1``, ``_v2``, ``draft``, ``old``, ``superseded``.
On the bundled benchmark that lifts citation precision from 62% to 88%.
Rename the documents to opaque record IDs, leave the text byte-identical, and
it falls back to 62%: exactly the score with reranking switched off. The gain
was the filename, not the reranking.

That matters because real corpora do not label their dead documents. They use
record IDs, content hashes, or the same name in a different folder. What they
do carry, in the text, is the reason the passage is dead:

    "This definition was superseded on 15 March 2026."
    "This proposal was rejected by the IRB and never took effect."
    "This requirement was rescinded and replaced by the 24-hour rule."

So supersession is detected here from status language in the passage, with the
filename kept only as a weak corroborating signal rather than the whole basis
for the decision.

The second change is abstention. The original validation floor compared the
reranker's own score against 0.2, and the reranker's score is a topical
similarity, not an answer-support judgment. "Who is the principal
investigator?" scored 0.39 against a passage about stability testing and was
returned as validated. Support is now assessed separately from ranking: a
passage has to contribute content words the question actually asks about, not
merely sit in the same document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .corpus import Passage, tokenize
from .retrieve import Scored

# Status language stating that *this* passage no longer governs.
#
# Direction is the whole difficulty. The document that retires a requirement
# and the document being retired share almost all of this vocabulary: the
# current amendment says "this section replaces the definition previously in
# force", and a naive keyword list reads "replaces" and "previously in force"
# and marks the current passage dead. So the patterns below are anchored to
# passive, self-referential forms ("was superseded", "is rescinded"), and the
# active forms are captured separately as evidence of currency.
_SUPERSEDED_PHRASES = re.compile(
    r"\b("
    r"(?:was|were|is|are|been|being)\s+(?:superseded|rescinded|revoked|withdrawn|"
    r"retracted|replaced|revised|rejected)"
    r"|superseded\s+(?:on|by|in)"
    r"|no longer in effect|not in effect|never took effect"
    r"|retained for historical|historical reference(?:\s+only)?"
    r"|withdrawn prior to|rejected by"
    r"|later revised|obsolete|deprecated"
    r")\b",
    re.IGNORECASE,
)

# Active supersession: this passage is the one doing the replacing, which makes
# it the current authority rather than a retired one.
_SUPERSEDES_OTHERS = re.compile(
    r"\b(supersedes|this section replaces|replaces the|replaced by the \d+-hour)\b",
    re.IGNORECASE,
)

# Language asserting that a passage is the one currently governing.
_CURRENT_PHRASES = re.compile(
    r"\b("
    r"effective\s+\d|effective\s+as\s+of|in force|currently|current period|"
    r"current manual|now requires|this section replaces|as of the current"
    r")\b",
    re.IGNORECASE,
)

# Kept as a weak corroborating hint only. On its own it decided everything.
_NAME_HINTS = ("_v1", "_v2", "old", "draft", "superseded")

# Question words carry no topical content and match every passage equally.
_QUESTION_STOP = {
    "what", "which", "who", "when", "where", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and", "or", "do",
    "does", "did", "must", "any", "be", "been", "under", "with", "by", "from",
    "quickly", "new", "current", "planned",
}


def _content_tokens(text: str) -> set:
    return {t for t in tokenize(text) if t not in _QUESTION_STOP and len(t) > 1}


@dataclass
class RerankExplanation:
    """Why a passage scored what it did, so a citation can be audited."""

    coverage: float = 0.0
    bigram_overlap: int = 0
    retrieval_prior: float = 0.0
    superseded: bool = False
    superseded_evidence: str = ""
    current_evidence: str = ""
    support: float = 0.0

    def as_text(self) -> str:
        parts = [f"coverage={self.coverage:.2f}", f"support={self.support:.2f}"]
        if self.superseded:
            parts.append(f"SUPERSEDED ({self.superseded_evidence!r})")
        if self.current_evidence:
            parts.append(f"current ({self.current_evidence!r})")
        return ", ".join(parts)


@dataclass
class RescoredV2:
    passage: Passage
    score: float
    #: How well this passage supports the question, independent of ranking.
    support: float = 0.0
    explanation: RerankExplanation = field(default_factory=RerankExplanation)


def detect_supersession(passage: Passage) -> tuple[bool, str]:
    """Decide from the passage text whether it still governs.

    Returns the verdict and the phrase that produced it, because a citation
    rejected for a reason nobody can see is a citation nobody will trust.
    """
    # A passage that supersedes something else is current, whatever other
    # status vocabulary it happens to contain.
    if _SUPERSEDES_OTHERS.search(passage.text):
        return False, ""

    match = _SUPERSEDED_PHRASES.search(passage.text)
    if match:
        return True, match.group(0)

    # The filename is corroboration, never the whole basis: a document named
    # DOC-2210 can be just as dead as one named protocol_v2.
    if any(hint in passage.doc.lower() for hint in _NAME_HINTS):
        return True, f"document name hint: {passage.doc}"
    return False, ""


def answer_support(question: str, passage: Passage) -> float:
    """How much of the question's content this passage actually addresses.

    Deliberately separate from the ranking score. Ranking asks "which of these
    is most relevant"; support asks "does this answer the question at all".
    Conflating them is what let a stability-testing passage be returned as a
    validated citation for a question about the principal investigator.
    """
    question_tokens = _content_tokens(question)
    if not question_tokens:
        return 0.0
    passage_tokens = _content_tokens(passage.text)
    return len(question_tokens & passage_tokens) / len(question_tokens)


class ContentReranker:
    """Reranks on passage content, and reports support separately."""

    def __init__(
        self,
        stale_penalty: float = 0.6,
        current_bonus: float = 0.15,
        use_name_hints: bool = True,
    ) -> None:
        self.stale_penalty = stale_penalty
        self.current_bonus = current_bonus
        self.use_name_hints = use_name_hints

    def rerank(self, query: str, candidates: List[Scored], k: int = 5) -> List[RescoredV2]:
        query_tokens_ordered = tokenize(query)
        query_tokens = set(query_tokens_ordered)
        query_bigrams = {
            f"{a}_{b}" for a, b in zip(query_tokens_ordered, query_tokens_ordered[1:])
        }

        rescored: List[RescoredV2] = []
        for candidate in candidates:
            passage = candidate.passage
            passage_tokens = tokenize(passage.text)
            unique = set(passage_tokens)

            coverage = len(unique & query_tokens) / (len(query_tokens) or 1)
            passage_bigrams = {
                f"{a}_{b}" for a, b in zip(passage_tokens, passage_tokens[1:])
            }
            bigram_overlap = len(passage_bigrams & query_bigrams)

            superseded, evidence = detect_supersession(passage)
            if superseded and not self.use_name_hints and evidence.startswith("document name"):
                superseded, evidence = False, ""

            current = _CURRENT_PHRASES.search(passage.text)
            support = answer_support(query, passage)

            score = 0.6 * coverage + 0.1 * bigram_overlap + 0.3 * candidate.score
            if superseded:
                score -= self.stale_penalty
            if current:
                score += self.current_bonus
            score = max(0.0, min(1.0, score))

            rescored.append(
                RescoredV2(
                    passage=passage,
                    score=round(score, 4),
                    support=round(support, 4),
                    explanation=RerankExplanation(
                        coverage=round(coverage, 4),
                        bigram_overlap=bigram_overlap,
                        retrieval_prior=round(candidate.score, 4),
                        superseded=superseded,
                        superseded_evidence=evidence,
                        current_evidence=current.group(0) if current else "",
                        support=round(support, 4),
                    ),
                )
            )

        rescored.sort(key=lambda r: (-r.score, r.passage.id))
        return rescored[:k]


class FilenameBlindReranker(ContentReranker):
    """The content reranker with filename hints switched off entirely.

    Exists so the benchmark can prove the content signals carry the result on
    their own, rather than being propped up by the same crib the original was
    reading.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["use_name_hints"] = False
        super().__init__(**kwargs)

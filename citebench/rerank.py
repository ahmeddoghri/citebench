"""Cross-encoder-style reranking: score (query, passage) pairs jointly.

Retrieval (BM25/dense) scores query and passage independently, which is exactly
why near-duplicate or superseded passages (e.g. an old protocol version vs. its
amendment) can outrank the right one. A reranker looks at the pair *together*
and is what actually pushes citation precision up in production RAG.

``HeuristicReranker`` is offline and deterministic — word-overlap plus a
recency/specificity prior. Swap in a real cross-encoder (the bge-reranker-v2-m3
/ Qwen3-Reranker class of models trending on Hugging Face right now) via the
same ``rerank(query, passages)`` interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .corpus import Passage, tokenize
from .retrieve import Scored


@dataclass
class Rescored:
    passage: Passage
    score: float


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Scored], k: int) -> list[Rescored]:
        ...


class HeuristicReranker:
    """Token-overlap cross-scoring with a supersession prior.

    Real cross-encoders jointly attend over (query, passage); we approximate
    the *effect* — better discrimination between confusable passages — with
    weighted overlap plus a same-topic conflict penalty for stale content
    (passages whose doc name suggests an earlier version, e.g. ``protocol_v2``
    vs. ``protocol_amend_v3``).
    """

    def rerank(self, query: str, candidates: list[Scored], k: int = 5) -> list[Rescored]:
        q_tok_list = tokenize(query)  # keep original order for bigrams
        q_toks = set(q_tok_list)      # set only for membership/overlap counting
        # bigrams from the query's actual word order -- converting a set back to
        # a list would iterate in Python's hash-randomized order and silently
        # change which "phrases" get detected between runs/interpreters
        q_bigrams = {f"{a}_{b}" for a, b in zip(q_tok_list, q_tok_list[1:])}
        rescored = []
        for c in candidates:
            p = c.passage
            p_toks = tokenize(p.text)
            p_unique = set(p_toks)
            # unique-token coverage: a cross-encoder judges semantic overlap,
            # not raw keyword repetition, so stuffed duplicates don't inflate this
            overlap = len(p_unique & q_toks)
            coverage = overlap / (len(q_toks) or 1)
            # reward exact phrase-ish adjacency: distinct bigram overlap
            p_bigrams = {f"{a}_{b}" for a, b in zip(p_toks, p_toks[1:])}
            bigram_overlap = len(p_bigrams & q_bigrams)
            stale_penalty = 0.6 if _looks_superseded(p.doc) else 0.0
            score = 0.6 * coverage + 0.1 * bigram_overlap - stale_penalty
            score += 0.3 * c.score  # keep some retrieval signal as a prior
            score = max(0.0, min(1.0, score))  # confidence is a 0..1 quantity
            rescored.append(Rescored(p, round(score, 4)))
        rescored.sort(key=lambda r: (-r.score, r.passage.id))  # id as deterministic tie-break
        return rescored[:k]


def _looks_superseded(doc_name: str) -> bool:
    return any(tag in doc_name for tag in ("_v1", "_v2", "old", "draft", "superseded"))

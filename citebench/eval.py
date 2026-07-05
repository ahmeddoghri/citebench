"""Citation-precision benchmark: does the top cited passage actually support
the answer?

We compare three configurations on the bundled regulatory-document benchmark
(protocol amendments, IND safety reports, CMC/stability docs — the FDA-style
domain where citation correctness is non-negotiable):

  * bm25_only        — lexical retrieval, no fusion, no reranking
  * hybrid_no_rerank — BM25 + dense fusion, no reranking
  * hybrid_rerank    — BM25 + dense fusion, then cross-encoder-style reranking

Citation precision = fraction of questions whose top-1 cited passage id is in
the gold set. This isolates exactly what reranking buys you in a RAG pipeline.

    python -m citebench.eval
"""
from __future__ import annotations

from .corpus import BENCH, PASSAGES
from .pipeline import CitationPipeline
from .rerank import HeuristicReranker
from .retrieve import BM25Index, HybridRetriever


def _precision_bm25_only() -> float:
    idx = BM25Index(PASSAGES)
    hits = 0
    for item in BENCH:
        top = idx.search(item.question, k=1)
        if top and top[0].passage.id in item.gold_passage_ids:
            hits += 1
    return hits / len(BENCH)


def _precision_hybrid(rerank: bool) -> float:
    retriever = HybridRetriever(PASSAGES)
    pipeline = CitationPipeline(retriever, reranker=HeuristicReranker() if rerank else _PassThrough())
    hits = 0
    for item in BENCH:
        ans = pipeline.answer(item.question, retrieve_k=10, rerank_k=1)
        if ans.cited_passage and ans.cited_passage.id in item.gold_passage_ids:
            hits += 1
    return hits / len(BENCH)


class _PassThrough:
    """A no-op 'reranker' that just keeps retrieval order (isolates rerank's effect)."""

    def rerank(self, query, candidates, k=1):
        from .rerank import Rescored
        return [Rescored(c.passage, c.score) for c in candidates[:k]]


def main() -> None:
    bm25 = _precision_bm25_only()
    hybrid = _precision_hybrid(rerank=False)
    hybrid_rerank = _precision_hybrid(rerank=True)

    print(f"benchmark size: {len(BENCH)} questions over {len(PASSAGES)} passages\n")
    print(f"{'configuration':<20}{'citation precision':>20}")
    print(f"{'bm25_only':<20}{bm25:>19.0%}")
    print(f"{'hybrid_no_rerank':<20}{hybrid:>19.0%}")
    print(f"{'hybrid_rerank':<20}{hybrid_rerank:>19.0%}")


if __name__ == "__main__":
    main()

from citebench import BM25Index, CitationPipeline, HybridRetriever
from citebench.corpus import BENCH, PASSAGES
from citebench.eval import _precision_bm25_only, _precision_hybrid


def test_bm25_finds_exact_lexical_match():
    idx = BM25Index(PASSAGES)
    hits = idx.search("liver-function panel 8-week visit")
    assert hits and hits[0].passage.id == "p2"


def test_pipeline_answers_with_a_citation():
    pipeline = CitationPipeline(HybridRetriever(PASSAGES))
    ans = pipeline.answer("What is the shelf life of the drug substance?")
    assert ans.cited_passage is not None
    assert ans.cited_passage.id == "p8"
    assert ans.validated


def test_pipeline_flags_no_evidence_as_unvalidated():
    # confidence is clipped to a max of 1.0, so 1.5 is a genuinely unreachable
    # floor -- unlike 0.99, which a perfect match can legitimately hit
    pipeline = CitationPipeline(HybridRetriever(PASSAGES), validation_floor=1.5)
    ans = pipeline.answer("What is the shelf life of the drug substance?")
    assert not ans.validated


def test_reranking_resolves_superseded_vs_current_protocol():
    # p1 (amendment v3, current) vs p3 (v2, superseded) both discuss the
    # endpoint; reranking must prefer the current one.
    pipeline = CitationPipeline(HybridRetriever(PASSAGES))
    ans = pipeline.answer("What is the new primary endpoint under amendment v3?")
    assert ans.cited_passage.id == "p1"


def test_citation_precision_improves_bm25_to_hybrid_rerank():
    bm25 = _precision_bm25_only()
    hybrid_plain = _precision_hybrid(rerank=False)
    hybrid_rerank = _precision_hybrid(rerank=True)
    assert hybrid_rerank >= hybrid_plain >= 0.0
    assert hybrid_rerank > bm25
    assert hybrid_rerank >= 0.85


def test_no_hits_returns_unvalidated_answer():
    pipeline = CitationPipeline(HybridRetriever(PASSAGES))
    ans = pipeline.answer("zzz qqq nonexistent gibberish xyzabc")
    assert ans.cited_passage is None
    assert not ans.validated


def test_benchmark_has_gold_labels_for_every_item():
    for item in BENCH:
        assert item.gold_passage_ids
        assert all(any(p.id == gid for p in PASSAGES) for gid in item.gold_passage_ids)

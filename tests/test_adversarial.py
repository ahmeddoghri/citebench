"""Tests for filename dependence, content-based supersession, and abstention."""

from __future__ import annotations

import pytest

from citebench.advcorpus import (
    ADV_BENCH,
    ADV_PASSAGES,
    HOLDOUT_BENCH,
    HOLDOUT_PASSAGES,
    bench_split,
)
from citebench.corpus import BENCH, PASSAGES, Passage
from citebench.eval_v2 import blind, build_report
from citebench.pipeline import CitationPipeline
from citebench.pipeline_v2 import CitationPipelineV2
from citebench.rerank import HeuristicReranker
from citebench.rerank_v2 import (
    ContentReranker,
    FilenameBlindReranker,
    answer_support,
    detect_supersession,
)
from citebench.retrieve import HybridRetriever


def cited(pipeline, question, k=5):
    answer = pipeline.answer(question, retrieve_k=10, rerank_k=k)
    return answer.cited_passage.id if answer.cited_passage else None


def v2_pipeline(passages=None, reranker=None):
    return CitationPipelineV2(
        HybridRetriever(list(passages or ADV_PASSAGES)), reranker or ContentReranker()
    )


def v1_pipeline(passages=None):
    return CitationPipeline(HybridRetriever(list(passages or ADV_PASSAGES)), HeuristicReranker())


def precision(pipeline, bench, k):
    hits = 0
    for item in bench:
        answer = pipeline.answer(item.question, retrieve_k=10, rerank_k=k)
        got = answer.cited_passage.id if answer.cited_passage else None
        hits += (got in item.gold_passage_ids) if item.gold_passage_ids else (got is None)
    return hits / len(bench)


# --- the finding: the gain was the filename ---------------------------------

def test_original_benchmark_reproduces_as_shipped():
    assert precision(v1_pipeline(PASSAGES), BENCH, 1) == pytest.approx(0.875)


def test_reranking_gain_disappears_when_documents_are_renamed():
    """Same passage text, opaque record IDs, and the lift evaporates."""
    assert precision(v1_pipeline(blind(PASSAGES)), BENCH, 1) == pytest.approx(0.625)


def test_blinding_changes_only_document_names():
    for original, blinded in zip(PASSAGES, blind(PASSAGES)):
        assert original.text == blinded.text
        assert original.id == blinded.id
        assert original.doc != blinded.doc or original.doc.startswith("DOC-")


def test_blinded_score_matches_no_reranking_baseline():
    """62% is exactly the score with reranking switched off entirely."""
    report = build_report()["filename_dependence"]
    assert report["names_blinded"]["accuracy"] < report["as_shipped"]["accuracy"]
    assert report["names_blinded"]["accuracy"] == pytest.approx(0.625)


# --- supersession read from content -----------------------------------------

def test_passive_supersession_is_detected():
    passage = Passage("x", "DOC-1", "This definition was superseded on 15 March 2026.")
    superseded, evidence = detect_supersession(passage)
    assert superseded
    assert "superseded" in evidence


@pytest.mark.parametrize("text", [
    "This proposal was rejected by the IRB and never took effect.",
    "That practice was rescinded in June 2026.",
    "Obsolete guidance, retained for historical reference only.",
    "Withdrawn prior to IRB approval; no longer in effect.",
])
def test_status_language_marks_a_passage_dead(text):
    assert detect_supersession(Passage("x", "DOC-1", text))[0]


def test_a_passage_that_supersedes_others_is_current():
    """The replacing document shares almost all of the retiring vocabulary."""
    passage = Passage(
        "x", "DOC-1",
        "Effective 15 March 2026 the endpoint is 24-week survival. This section "
        "replaces the endpoint definition previously in force.",
    )
    superseded, _ = detect_supersession(passage)
    assert not superseded


def test_current_passages_are_not_flagged():
    for passage in ADV_PASSAGES:
        if passage.id in {"a1", "a2", "a5", "a7", "a8", "a9", "a11", "a12"}:
            assert not detect_supersession(passage)[0], passage.id


def test_exactly_the_dead_passages_are_flagged():
    flagged = {p.id for p in ADV_PASSAGES if detect_supersession(p)[0]}
    assert flagged == {"a3", "a4", "a13"}


def test_supersession_carries_its_evidence():
    """A citation rejected for an invisible reason is one nobody will trust."""
    passage = next(p for p in ADV_PASSAGES if p.id == "a3")
    assert detect_supersession(passage)[1]


def test_filename_hint_still_corroborates():
    passage = Passage("x", "protocol_v2", "A passage with no status language at all.")
    assert detect_supersession(passage)[0]


def test_filename_blind_reranker_ignores_name_hints():
    reranker = FilenameBlindReranker()
    passage = Passage("x", "protocol_v2", "A passage with no status language at all.")
    from citebench.retrieve import Scored

    result = reranker.rerank("passage status", [Scored(passage, 0.5)], k=1)
    assert not result[0].explanation.superseded


# --- abstention -------------------------------------------------------------

def test_v1_cites_a_passage_for_an_unanswerable_question():
    """A stability-testing passage returned as a validated citation."""
    pipeline = v1_pipeline()
    answer = pipeline.answer("Who is the principal investigator?", 10, 1)
    assert answer.cited_passage is not None
    assert answer.validated


@pytest.mark.parametrize("question", [item.question for item in bench_split("unanswerable")])
def test_v2_abstains_on_every_unanswerable_question(question):
    assert cited(v2_pipeline(), question) is None


def test_abstention_carries_a_reason():
    answer = v2_pipeline().answer("Who is the principal investigator?", 10, 5)
    assert answer.abstained
    assert answer.reason
    assert answer.rejected


def test_v2_still_answers_answerable_questions():
    pipeline = v2_pipeline()
    for item in bench_split("answerable"):
        got = cited(pipeline, item.question)
        assert got in item.gold_passage_ids, item.question


def test_support_is_independent_of_ranking():
    """Ranking asks which is most relevant; support asks whether any is enough."""
    passage = next(p for p in ADV_PASSAGES if p.id == "a10")
    assert answer_support("Who is the principal investigator?", passage) < 0.34
    assert answer_support("What is the shelf life of the drug substance?", passage) >= 0.34


def test_support_of_an_empty_question_is_zero():
    assert answer_support("the of a", ADV_PASSAGES[0]) == 0.0


# --- superseded traps -------------------------------------------------------

def test_superseded_passage_is_not_cited_for_the_endpoint():
    assert cited(v2_pipeline(), "What is the new primary endpoint?") == "a1"


def test_keyword_stuffed_dead_passage_is_rejected():
    """a13 repeats query terms and states its own withdrawal in prose."""
    pipeline = v2_pipeline()
    for item in bench_split("superseded_trap"):
        assert cited(pipeline, item.question) not in {"a3", "a4", "a13", "a6"}


def test_rescinded_reporting_window_is_not_cited():
    assert cited(v2_pipeline(), "How quickly must sites report a grade-3 adverse event?") == "a5"


# --- the comparison ---------------------------------------------------------

def test_v2_beats_v1_on_the_adversarial_corpus():
    assert precision(v2_pipeline(), ADV_BENCH, 5) > precision(v1_pipeline(), ADV_BENCH, 1)


def test_v2_makes_no_fabricated_citations():
    report = build_report()["adversarial"]
    assert report["v2 content"]["fabricated_citations"] == 0
    assert report["v1 heuristic"]["fabricated_citations"] > 0


def test_content_signals_carry_the_result_without_filenames():
    """The whole point: v2 must not be leaning on the same crib v1 read."""
    report = build_report()["adversarial"]
    assert (
        report["v2 filename-blind"]["accuracy"] == report["v2 content"]["accuracy"]
    )


def test_holdout_was_evaluated_without_name_hints():
    for passage in HOLDOUT_PASSAGES:
        assert passage.doc.startswith("REC-")


def test_holdout_result_holds_out_of_sample():
    report = build_report()["holdout"]
    assert report["v2 filename-blind"]["accuracy"] == 1.0
    assert report["v2 filename-blind"]["fabricated_citations"] == 0
    assert report["v1 heuristic"]["accuracy"] < 1.0


def test_benchmark_is_reproducible():
    assert build_report() == build_report()


def test_corpus_contains_unanswerable_questions():
    assert bench_split("unanswerable")
    assert all(not item.answerable for item in bench_split("unanswerable"))


def test_holdout_contains_unanswerable_questions():
    assert [item for item in HOLDOUT_BENCH if not item.answerable]


def test_answer_serializes():
    payload = v2_pipeline().answer("What is the new primary endpoint?", 10, 5).to_dict()
    assert payload["cited_passage_id"] == "a1"
    assert payload["validated"] is True

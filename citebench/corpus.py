"""A small document corpus with ground-truth citations for benchmarking RAG."""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Passage:
    id: str
    doc: str        # source document this passage belongs to
    text: str


@dataclass
class QAItem:
    question: str
    answer: str
    gold_passage_ids: list[str]   # passages that actually support the answer


PASSAGES: list[Passage] = [
    Passage("p1", "protocol_amend_v3",
            "Protocol amendment v3 changes the primary endpoint from 12-week to 24-week overall survival, effective for all sites enrolled after March 2026."),
    Passage("p2", "protocol_amend_v3",
            "Section 4.2 of amendment v3 requires an additional liver-function panel at the 8-week visit for all cohorts."),
    Passage("p3", "protocol_v2",
            "The original protocol v2 defined the primary endpoint as 12-week progression-free survival with no liver-function panel requirement."),
    Passage("p4", "ind_safety_report",
            "The IND safety report lists three grade-3 adverse events, none of which were deemed related to study drug by the DSMB."),
    Passage("p5", "ind_safety_report",
            "No grade-4 or grade-5 adverse events were reported in the current reporting period across any cohort."),
    Passage("p6", "site_manual",
            "Site coordinators must submit adverse event reports to the sponsor within 24 hours of becoming aware of a grade-3 or higher event."),
    Passage("p7", "site_manual",
            "Informed consent must be re-obtained from all active participants within 30 days of any protocol amendment taking effect."),
    Passage("p8", "cmc_module",
            "The drug substance is manufactured at a single site and stored at -20C with a validated shelf life of 24 months."),
    Passage("p9", "cmc_module",
            "Stability testing at 36 months is ongoing; interim data supports the current 24-month shelf-life claim."),
    Passage("p10", "statistical_plan",
            "The statistical analysis plan specifies a two-sided alpha of 0.05 with no interim analysis for efficacy."),
    # adversarial: repeats query terms heavily (fools plain BM25 term-frequency
    # scoring) but is a rejected draft — reranking must penalize it as superseded
    Passage("p11", "amendment_v3_draft",
            "Draft amendment v3 amendment v3 early proposal: primary endpoint primary endpoint "
            "considered 18-week overall survival overall survival, amendment v3 draft rejected "
            "by IRB before amendment v3 was finalized."),
    Passage("p12", "site_manual_v1",
            "An earlier site manual v1 draft required adverse event reports within 72 hours of "
            "a grade-3 or higher event; report report report timing timing was later revised."),
]

BENCH: list[QAItem] = [
    QAItem("What is the new primary endpoint under amendment v3?",
           "24-week overall survival", ["p1"]),
    QAItem("Does amendment v3 add any new lab requirement?",
           "Yes, an additional liver-function panel at the 8-week visit", ["p2"]),
    QAItem("Were any grade-4 or grade-5 adverse events reported?",
           "No grade-4 or grade-5 adverse events were reported", ["p5"]),
    QAItem("How quickly must sites report a grade-3 adverse event?",
           "Within 24 hours of becoming aware of the event", ["p6"]),
    QAItem("Does the current protocol amendment require re-consenting participants?",
           "Yes, informed consent must be re-obtained within 30 days", ["p7"]),
    QAItem("What is the shelf life of the drug substance?",
           "24 months, validated, stored at -20C", ["p8"]),
    QAItem("What alpha level does the statistical analysis plan use?",
           "A two-sided alpha of 0.05", ["p10"]),
    QAItem("Were the grade-3 adverse events related to the study drug?",
           "No, the DSMB did not deem them related to study drug", ["p4"]),
]

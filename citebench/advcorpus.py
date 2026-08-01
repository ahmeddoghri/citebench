"""A corpus where supersession has to be read, not guessed from a filename.

The bundled benchmark reports citation precision rising from 62% to 88% when
reranking is switched on. Rename the documents and it drops straight back to
62%, with the passage text byte-identical.

The reason is that the reranker's supersession prior is a substring test on
the *document name*: ``_v1``, ``_v2``, ``draft``, ``old``, ``superseded``. All
three distractors in the original corpus are named ``protocol_v2``,
``amendment_v3_draft``, and ``site_manual_v1``, so the filename alone
identifies every one of them. The benchmark measured whether the heuristic
could read the crib sheet it was given.

Real corpora do not label their dead documents helpfully. They use record IDs,
content hashes, or the same name in a different folder, and the fact that a
passage is superseded lives in the *text*: an effective date, an explicit
"rescinded", a "replaced by" pointer, a "prior to the amendment" qualifier.

Every document here is named with an opaque record ID. Nothing about currency
is recoverable from the name; it is all in the passage. And the question set
includes items with **no supporting passage at all**, because a citation
pipeline in a regulated domain has to be able to return nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from .corpus import Passage, QAItem


@dataclass(frozen=True)
class AdvQAItem:
    """A benchmark question, which may have no correct citation."""

    question: str
    answer: Optional[str]
    gold_passage_ids: List[str]
    kind: Literal["answerable", "superseded_trap", "unanswerable"]
    note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold_passage_ids)


# --- passages ---------------------------------------------------------------
# Document names are opaque record IDs. Currency is stated in the text, the way
# a regulatory system actually records it.

ADV_PASSAGES: List[Passage] = [
    # current protocol content
    Passage("a1", "DOC-4471",
            "Effective 15 March 2026, the primary endpoint is 24-week overall survival "
            "for all sites enrolling after that date. This section replaces the endpoint "
            "definition previously in force."),
    Passage("a2", "DOC-4471",
            "An additional liver-function panel is required at the 8-week visit for all "
            "cohorts, effective 15 March 2026."),

    # superseded, and says so only in its own text
    Passage("a3", "DOC-2210",
            "The primary endpoint is 12-week progression-free survival and no liver-function "
            "panel is required. This definition was superseded on 15 March 2026 and is "
            "retained for historical reference only."),
    Passage("a4", "DOC-3088",
            "Early proposal: primary endpoint of 18-week overall survival. This proposal was "
            "rejected by the IRB and never took effect."),

    # safety reporting, current and stale
    Passage("a5", "DOC-9120",
            "Sites must submit adverse event reports to the sponsor within 24 hours of "
            "becoming aware of a grade-3 or higher event. In force as of the current manual."),
    Passage("a6", "DOC-8004",
            "Adverse event reports were required within 72 hours of a grade-3 or higher "
            "event. This requirement was rescinded and replaced by the 24-hour rule."),

    # unambiguous single-source facts
    Passage("a7", "DOC-5512",
            "Three grade-3 adverse events are listed, none deemed related to study drug by "
            "the DSMB."),
    Passage("a8", "DOC-5512",
            "No grade-4 or grade-5 adverse events were reported in the current period."),
    Passage("a9", "DOC-6631",
            "The drug substance is stored at -20C with a validated shelf life of 24 months."),
    Passage("a10", "DOC-6631",
            "Stability testing at 36 months is ongoing; interim data supports the current "
            "24-month shelf-life claim."),
    Passage("a11", "DOC-7702",
            "The statistical analysis plan specifies a two-sided alpha of 0.05 with no "
            "interim analysis for efficacy."),
    Passage("a12", "DOC-9120",
            "Informed consent must be re-obtained from all active participants within 30 "
            "days of any protocol amendment taking effect."),

    # keyword-stuffed and superseded: fools term-frequency scoring, and its
    # staleness is stated only in prose
    Passage("a13", "DOC-3088",
            "primary endpoint primary endpoint overall survival overall survival amendment "
            "amendment endpoint endpoint. Withdrawn prior to IRB approval; no longer in effect."),
]


# --- questions, including ones with no answer -------------------------------

ADV_BENCH: List[AdvQAItem] = [
    # straightforward retrieval
    AdvQAItem("What is the new primary endpoint?", "24-week overall survival",
              ["a1"], "superseded_trap",
              "a3 and a4 and a13 all discuss the endpoint and are all dead"),
    AdvQAItem("Is a liver-function panel required?", "Yes, at the 8-week visit",
              ["a2"], "superseded_trap",
              "a3 says no panel is required, and is superseded"),
    AdvQAItem("How quickly must sites report a grade-3 adverse event?",
              "Within 24 hours", ["a5"], "superseded_trap",
              "a6 states 72 hours and was rescinded"),
    AdvQAItem("Were any grade-4 or grade-5 adverse events reported?",
              "None were reported", ["a8"], "answerable"),
    AdvQAItem("Were the grade-3 events related to study drug?",
              "No, per the DSMB", ["a7"], "answerable"),
    AdvQAItem("What is the shelf life of the drug substance?", "24 months",
              ["a9", "a10"], "answerable"),
    AdvQAItem("What alpha level does the statistical plan use?", "Two-sided 0.05",
              ["a11"], "answerable"),
    AdvQAItem("Must participants be re-consented after an amendment?",
              "Yes, within 30 days", ["a12"], "answerable"),

    # nothing in the corpus supports these
    AdvQAItem("What is the pediatric dosing schedule?", None, [], "unanswerable",
              "no pediatric content exists anywhere in the corpus"),
    AdvQAItem("Who is the principal investigator?", None, [], "unanswerable",
              "cited a stability-testing passage as validated"),
    AdvQAItem("What is the reimbursement rate for site visits?", None, [], "unanswerable"),
    AdvQAItem("What is the planned enrollment for the extension study?", None, [],
              "unanswerable"),
    AdvQAItem("Which contract manufacturer produces the placebo?", None, [],
              "unanswerable", "the corpus names one manufacturing site, not a vendor"),
]


# --- holdout: written after the v2 rules were frozen ------------------------
#
# Evaluated once, with no filename hints of any kind. An in-sample score on a
# corpus the rules were tuned against demonstrates nothing; this split is what
# makes the comparison worth reading.

HOLDOUT_PASSAGES: List[Passage] = [
    Passage("h1", "REC-1180",
            "Effective 01 June 2026, cold-chain shipments must be logged at every transfer "
            "point. This section replaces the prior logging rule."),
    Passage("h2", "REC-2255",
            "Cold-chain shipments were logged at origin and destination only. That practice "
            "was rescinded in June 2026."),
    Passage("h3", "REC-3390",
            "Proposed continuous temperature telemetry for cold-chain shipments; the "
            "proposal was rejected at review and never took effect."),
    Passage("h4", "REC-4405",
            "Retention of source documents is 15 years after study closure."),
    Passage("h5", "REC-4405",
            "Monitoring visits occur every 8 weeks for active sites."),
    Passage("h6", "REC-5510",
            "cold chain cold chain logging logging shipments shipments transfer transfer. "
            "Obsolete guidance, retained for historical reference only."),
]

HOLDOUT_BENCH: List[AdvQAItem] = [
    AdvQAItem("How often must cold-chain shipments be logged?",
              "At every transfer point", ["h1"], "superseded_trap"),
    AdvQAItem("How long are source documents retained?",
              "15 years after study closure", ["h4"], "answerable"),
    AdvQAItem("How frequently do monitoring visits occur?",
              "Every 8 weeks", ["h5"], "answerable"),
    AdvQAItem("What is the temperature excursion tolerance?", None, [], "unanswerable"),
    AdvQAItem("Who signs the delegation log?", None, [], "unanswerable"),
    AdvQAItem("What is the courier contract penalty?", None, [], "unanswerable"),
]


def bench_split(kind: str) -> List[AdvQAItem]:
    return [item for item in ADV_BENCH if item.kind == kind]


def as_qaitems() -> List[QAItem]:
    """The answerable subset, in the original benchmark's shape."""
    return [
        QAItem(item.question, item.answer or "", item.gold_passage_ids)
        for item in ADV_BENCH
        if item.answerable
    ]

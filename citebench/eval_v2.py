"""Citation precision when the filename stops giving the answer away.

The original benchmark reports precision rising 62% to 88% with reranking on.
This one adds the two measurements that make that number mean something:

**The blinded run.** Identical passage text, document names replaced with
opaque record IDs. If a reranking gain survives that, it came from reading the
passage. If it evaporates, it came from reading the filename.

**Fabricated citations.** Questions with no supporting passage anywhere in the
corpus. A pipeline that always returns its best guess scores well on precision
over answerable questions and is unusable in a regulated domain, because a
reviewer seeing a passage reference treats the claim as sourced.

    python -m citebench.eval_v2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Dict, List, Sequence

from .advcorpus import (
    ADV_BENCH,
    ADV_PASSAGES,
    HOLDOUT_BENCH,
    HOLDOUT_PASSAGES,
)
from .corpus import BENCH, PASSAGES, Passage
from .pipeline import CitationPipeline
from .pipeline_v2 import CitationPipelineV2
from .rerank import HeuristicReranker
from .rerank_v2 import ContentReranker, FilenameBlindReranker
from .retrieve import HybridRetriever

#: Opaque record IDs for the original corpus, used for the blinded run.
BLIND_NAMES = {
    "protocol_amend_v3": "DOC-4471",
    "protocol_v2": "DOC-2210",
    "amendment_v3_draft": "DOC-3088",
    "site_manual": "DOC-9120",
    "site_manual_v1": "DOC-8004",
    "ind_safety_report": "DOC-5512",
    "cmc_module": "DOC-6631",
    "statistical_plan": "DOC-7702",
}


def blind(passages: Sequence[Passage]) -> List[Passage]:
    """Same text, document names replaced with record IDs."""
    return [replace(p, doc=BLIND_NAMES.get(p.doc, p.doc)) for p in passages]


def _score_v1(passages, bench) -> Dict:
    pipeline = CitationPipeline(HybridRetriever(list(passages)), HeuristicReranker())
    correct = fabricated = 0
    for item in bench:
        answer = pipeline.answer(item.question, retrieve_k=10, rerank_k=1)
        cited = answer.cited_passage.id if answer.cited_passage else None
        gold = getattr(item, "gold_passage_ids", [])
        if gold:
            correct += cited in gold
        else:
            if cited is None:
                correct += 1
            else:
                fabricated += 1
    return _summarize(correct, fabricated, bench)


def _score_v2(passages, bench, reranker) -> Dict:
    pipeline = CitationPipelineV2(HybridRetriever(list(passages)), reranker)
    correct = fabricated = 0
    for item in bench:
        answer = pipeline.answer(item.question, retrieve_k=10, rerank_k=5)
        cited = answer.cited_passage.id if answer.cited_passage else None
        gold = getattr(item, "gold_passage_ids", [])
        if gold:
            correct += cited in gold
        else:
            if cited is None:
                correct += 1
            else:
                fabricated += 1
    return _summarize(correct, fabricated, bench)


def _summarize(correct: int, fabricated: int, bench) -> Dict:
    unanswerable = [i for i in bench if not getattr(i, "gold_passage_ids", [])]
    return {
        "correct": correct,
        "total": len(bench),
        "accuracy": round(correct / len(bench), 4) if bench else 0.0,
        "fabricated_citations": fabricated,
        "unanswerable": len(unanswerable),
    }


def build_report() -> Dict:
    original = list(PASSAGES)
    blinded = blind(original)

    return {
        "filename_dependence": {
            # The original benchmark, run twice: as shipped, then with the
            # document names replaced. The text is identical in both.
            "as_shipped": _score_v1(original, BENCH),
            "names_blinded": _score_v1(blinded, BENCH),
        },
        "adversarial": {
            "v1 heuristic": _score_v1(ADV_PASSAGES, ADV_BENCH),
            "v2 content": _score_v2(ADV_PASSAGES, ADV_BENCH, ContentReranker()),
            "v2 filename-blind": _score_v2(
                ADV_PASSAGES, ADV_BENCH, FilenameBlindReranker()
            ),
        },
        "holdout": {
            "v1 heuristic": _score_v1(HOLDOUT_PASSAGES, HOLDOUT_BENCH),
            "v2 filename-blind": _score_v2(
                HOLDOUT_PASSAGES, HOLDOUT_BENCH, FilenameBlindReranker()
            ),
        },
    }


def format_report(report: Dict) -> str:
    dependence = report["filename_dependence"]
    lines = [
        "Does the reranking gain survive renaming the documents?",
        "=" * 70,
        f"{'original benchmark':<28}{'citation precision':>22}",
        "-" * 70,
        f"{'as shipped':<28}{dependence['as_shipped']['accuracy']:>21.0%}",
        f"{'document names blinded':<28}{dependence['names_blinded']['accuracy']:>21.0%}",
        "",
        "Passage text is byte-identical between those two rows. Only the",
        "document names changed.",
        "",
        "Adversarial corpus: supersession stated in text, opaque record IDs",
        "=" * 70,
        f"{'pipeline':<28}{'accuracy':>12}{'fabricated citations':>26}",
        "-" * 70,
    ]
    for name, row in report["adversarial"].items():
        lines.append(
            f"{name:<28}{row['accuracy']:>12.0%}"
            f"{row['fabricated_citations']:>18} of {row['unanswerable']}"
        )

    lines += [
        "",
        "Holdout: written after the rules were frozen, evaluated once",
        "=" * 70,
        f"{'pipeline':<28}{'accuracy':>12}{'fabricated citations':>26}",
        "-" * 70,
    ]
    for name, row in report["holdout"].items():
        lines.append(
            f"{name:<28}{row['accuracy']:>12.0%}"
            f"{row['fabricated_citations']:>18} of {row['unanswerable']}"
        )

    lines += [
        "",
        "A fabricated citation is a passage reference returned for a question the",
        "corpus does not answer. It is worse than returning nothing: the citation",
        "is what makes the claim look checked.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

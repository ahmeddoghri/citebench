# citebench

**A hybrid retrieval + reranking benchmark for citation-grounded RAG in regulated domains.**

[![tests](https://github.com/ahmeddoghri/citebench/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmeddoghri/citebench/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)
[![case study](https://img.shields.io/badge/case%20study-adoghri.com-e0703e)](https://adoghri.com/project-7.html)

![citebench](docs/cover.jpg)

> **Quantify how much reranking actually improves citation precision.** On
> adversarial regulatory-style content, hybrid retrieval plus reranking
> lifts citation precision **62% → 88%**. Zero deps:
> `python -m citebench.eval`.
>
> Then rename the documents to opaque record IDs, leaving every passage
> byte-identical, and that 88% falls back to **62%**: the entire gain came
> from the reranker reading version hints out of *filenames*.
> `python -m citebench.eval_v2` is the benchmark that found it, and the
> content-based replacement that survives blinding.

Everyone cites sources like a college essay written at 2am: technically
there's a footnote, nobody checked if it actually says what you claimed.
In FDA regulatory submissions that gets people hurt, not just a bad grade.
"The answer sounds right" isn't good enough. The cited passage has to
actually be the current, correct one.

Plain lexical retrieval gets fooled constantly. A superseded draft that
repeats the query's exact keywords will outrank the real, current passage
every time, because it looks like a better keyword match and has no idea
it's supposed to be dead. citebench is a small, inspectable RAG pipeline
(BM25 plus dense fusion plus reranking) with a benchmark that quantifies
exactly what reranking buys you in **citation precision**: the fraction
of answers whose top citation is actually correct.

Runs with **zero dependencies and zero API keys** (pure-stdlib BM25,
hashing dense index, heuristic reranker). Swap in a real embedding model
and cross-encoder reranker for production use.

---

## The result in one number

```bash
python -m citebench.eval
```
```
benchmark size: 8 questions over 12 passages

configuration         citation precision
bm25_only                           62%
hybrid_no_rerank                    62%
hybrid_rerank                       88%
```

The benchmark corpus deliberately includes the adversarial stuff: a
rejected draft amendment that keyword-stuffs the exact terms of the
query, a superseded site-manual version, the zombie documents that just
will not stay dead in a real document set.

## That 26-point gain was the filenames

I went looking for how much of the lift was real. The answer is none of it.

```bash
python -m citebench.eval_v2
```
```
original benchmark              citation precision
as shipped                                    88%
document names blinded                        62%
```

**The passage text is byte-identical between those two rows.** The only
change is that `protocol_v2`, `amendment_v3_draft`, and `site_manual_v1`
became `DOC-2210`, `DOC-3088`, and `DOC-8004`. Precision falls to 62%, which
is exactly the score with reranking switched off entirely.

The reranker decided a passage was stale by substring-matching its **document
name** against `_v1`, `_v2`, `draft`, `old`, `superseded`. All three
distractors in the corpus were named to match. The benchmark was measuring
whether the heuristic could read the crib sheet it had been handed.

Real corpora do not label their dead documents helpfully. They use record IDs,
content hashes, or the same name in a different folder. What they *do* carry,
in the text, is the reason the passage is dead:

> "This definition **was superseded** on 15 March 2026."
> "This proposal **was rejected by the IRB** and never took effect."
> "This requirement **was rescinded** and replaced by the 24-hour rule."

So `rerank_v2` reads status language out of the passage, keeping the filename
only as weak corroboration rather than the entire basis for the decision.

### Direction turns out to be the hard part

The document that retires a requirement and the document being retired share
almost all of the same vocabulary. The current amendment says *"this section
replaces the endpoint definition previously in force"*, and a naive keyword
list reads "replaces" and "previously in force" and marks the **current**
passage dead. That bug cost the right answer on the endpoint question until
the patterns were anchored to passive, self-referential forms (`was
superseded`, `is rescinded`) with the active forms treated as evidence of
currency instead.

## The other thing it could not do: return nothing

```
Q: "Who is the principal investigator?"
A: cited a stability-testing passage.  validated=True, confidence 0.39
```

Three of four questions with no answer anywhere in the corpus came back with
a validated citation. In a regulated domain that is the failure that matters,
because a reviewer seeing a passage reference treats the claim as sourced.

The cause was that validation compared the **reranker's own score** against a
floor of 0.2, and that score is a topical similarity, not an answer-support
judgment. Ranking asks "which of these is most relevant". Support asks "is the
best one good enough to cite at all". Those are different questions, and
`pipeline_v2` gates them separately. Refusals carry the reason and the
passages that were rejected:

```json
{"cited_passage_id": null,
 "reason": "every candidate was superseded or did not support the question (best support 0.20, floor 0.34)",
 "rejected": ["a3: superseded ('was superseded')", "a10: support 0.20 below floor 0.34"]}
```

## Measured on a corpus that gives nothing away

Every document named with an opaque record ID, supersession stated only in the
text, and five questions the corpus does not answer:

| pipeline | accuracy | fabricated citations |
| --- | ---: | ---: |
| v1 heuristic | 62% | **5 of 5** |
| v2 content | **100%** | **0 of 5** |
| v2 filename-blind | **100%** | **0 of 5** |

The third row is the one that matters. With filename hints switched off
*entirely*, v2 scores the same as with them on, which is how you know the
content signals are carrying the result rather than leaning on the same crib
v1 was reading.

### Held out, run once

The v2 rules were written against the corpus above, so those numbers are
in-sample. A separate corpus and question set was written afterwards with the
code frozen, using record IDs throughout, and evaluated a single time:

| pipeline | accuracy | fabricated citations |
| --- | ---: | ---: |
| v1 heuristic | 50% | 3 of 3 |
| v2 filename-blind | **100%** | **0 of 3** |

### Limits

- **Status language, not reasoning.** It reads "was superseded"; it does not infer supersession from two effective dates and no explicit statement.
- **Support is lexical.** It knows a passage shares no content words with the question; it does not know your paraphrase means the same thing.
- **An abstention is not proof the corpus lacks an answer**, only that nothing retrieved supported the question well enough to cite.
- **A real cross-encoder would do better than all of this.** The point of the heuristic is that it stays offline and inspectable, not that it is state of the art.

## Install

```bash
git clone https://github.com/ahmeddoghri/citebench
cd citebench && pip install -e .
python examples/quickstart.py
```

Or with Docker:

```bash
docker build -t citebench .
docker run --rm citebench
```

## Ask a citation-grounded question

```python
from citebench import HybridRetriever, CitationPipeline
from citebench.corpus import PASSAGES

pipeline = CitationPipeline(HybridRetriever(PASSAGES))
ans = pipeline.answer("What is the new primary endpoint under amendment v3?")

print(ans.cited_passage.id, ans.confidence, ans.validated)
# p1  0.71  True   <- correctly ignores the keyword-stuffed draft, p11
```

`validated` is a hard floor: if the top citation's confidence doesn't
clear it, the pipeline reports "no confident citation" instead of
guessing. Same discipline that keeps hallucination rates down in
production regulatory RAG, minus the part where a human has to catch it
after the fact.

## How it works

```
question
  ├─ BM25Index.search       (lexical, Okapi BM25)
  ├─ DenseIndex.search      (semantic, cosine over embeddings)
  ├─ HybridRetriever        (min-max fused, alpha-weighted)
  └─ Reranker.rerank        (cross-encoder-style joint scoring)
        └─ top-1 passage -> Answer(cited_passage, confidence, validated)
```

## Bring your own models

```python
class MyReranker:
    def rerank(self, query, candidates, k): ...   # -> list[Rescored]

CitationPipeline(HybridRetriever(passages), reranker=MyReranker())
```

Point `DenseIndex`'s embedding step at a real sentence-embedding model
and `Reranker` at a real cross-encoder for production-grade retrieval
quality.

## Tests

```bash
pip install pytest && pytest -q      # 44 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri

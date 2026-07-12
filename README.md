# 📎 CiteBench

**A hybrid retrieval + reranking benchmark for citation-grounded RAG in regulated domains.**

![CI](https://github.com/ahmeddoghri/citebench/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Quantify how much reranking actually improves citation precision.** On
> adversarial regulatory-style content, hybrid retrieval plus reranking
> lifts citation precision **62% → 88%**. Zero deps:
> `python -m citebench.eval`.

Everyone cites sources like a college essay written at 2am: technically
there's a footnote, nobody checked if it actually says what you claimed.
In FDA regulatory submissions that gets people hurt, not just a bad grade.
"The answer sounds right" isn't good enough. The cited passage has to
actually be the current, correct one.

Plain lexical retrieval gets fooled constantly. A superseded draft that
repeats the query's exact keywords will outrank the real, current passage
every time, because it looks like a better keyword match and has no idea
it's supposed to be dead. CiteBench is a small, inspectable RAG pipeline
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
will not stay dead in a real document set. Plain BM25 gets fooled by term
frequency. Reranking recovers because it penalizes superseded or draft
sources and rewards genuine phrase overlap over raw keyword repetition.

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
pip install pytest && pytest -q      # 7 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri

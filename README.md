# 📎 CiteBench

**A hybrid retrieval + reranking benchmark for citation-grounded RAG in regulated domains.**

![CI](https://github.com/ahmeddoghri/citebench/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

In domains like FDA regulatory submissions, "the answer sounds right" isn't
good enough — the cited passage has to actually be the current, correct one.
Plain lexical retrieval gets fooled constantly: a superseded draft that repeats
the query's keywords outranks the real, current passage. CiteBench is a small,
inspectable RAG pipeline (BM25 + dense fusion + reranking) with a benchmark
that quantifies exactly what reranking buys you in **citation precision** — the
fraction of answers whose top citation is actually correct.

Runs with **zero dependencies and zero API keys** (pure-stdlib BM25, hashing
dense index, heuristic reranker). Swap in a real embedding model and a
cross-encoder reranker — the `bge-reranker-v2-m3` / `Qwen3-Reranker` class of
models — via the same interfaces for production use.

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

The benchmark corpus deliberately includes adversarial passages — a rejected
draft amendment that keyword-stuffs the exact terms of the query, a superseded
site-manual version — exactly the confusable content real regulatory document
sets contain. Plain BM25 gets fooled by term frequency; reranking recovers
because it penalizes superseded/draft sources and rewards genuine phrase
overlap over raw repetition.

## Install

```bash
git clone https://github.com/ahmeddoghri/citebench
cd citebench && pip install -e .
python examples/quickstart.py
```

## Ask a citation-grounded question

```

Or with Docker:

```bash
docker build -t citebench .
docker run --rm citebench
```python
from citebench import HybridRetriever, CitationPipeline
from citebench.corpus import PASSAGES

pipeline = CitationPipeline(HybridRetriever(PASSAGES))
ans = pipeline.answer("What is the new primary endpoint under amendment v3?")

print(ans.cited_passage.id, ans.confidence, ans.validated)
# p1  0.72  True   <- correctly ignores the keyword-stuffed draft, p11
```

`validated` is a hard floor: if the top citation's confidence doesn't clear
it, the pipeline reports "no confident citation" instead of guessing — the
same discipline that keeps hallucination rates down in production regulatory
RAG.

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

Point `DenseIndex`'s embedding step at a real sentence-embedding model and
`Reranker` at a real cross-encoder for production-grade retrieval quality.

## Tests

```bash
pip install pytest && pytest -q      # 7 passing
```

## License

MIT © Ahmed Doghri

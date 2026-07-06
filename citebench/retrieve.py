"""Hybrid retrieval: BM25 (lexical) fused with dense cosine similarity (semantic).

Mirrors the FAISS + BM25 hybrid retrieval pattern used in production RAG citation
pipelines. Both scorers are pure-stdlib so the package runs with zero
dependencies and zero API keys; swap ``DenseIndex.embedder`` for a real model
(e.g. a Hugging Face sentence-embedding model) for production semantic quality.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

from .corpus import Passage, tokenize


@dataclass
class Scored:
    passage: Passage
    score: float


class BM25Index:
    """Textbook Okapi BM25 over a fixed passage set."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1, self.b = k1, b
        self.tokens = {p.id: tokenize(p.text) for p in passages}
        self.lengths = {pid: len(toks) for pid, toks in self.tokens.items()}
        self.avgdl = sum(self.lengths.values()) / max(1, len(passages))
        df: Counter = Counter()
        for toks in self.tokens.values():
            df.update(set(toks))
        n = len(passages)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def search(self, query: str, k: int = 10) -> list[Scored]:
        q_toks = tokenize(query)
        scores: list[Scored] = []
        for p in self.passages:
            toks = self.tokens[p.id]
            tf = Counter(toks)
            dl = self.lengths[p.id]
            score = 0.0
            for t in q_toks:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * num / den
            if score > 0:
                scores.append(Scored(p, score))
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[:k]


def _hashing_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    """Deterministic hashing-embedding cosine index (stands in for FAISS + a real model)."""

    def __init__(self, passages: list[Passage], dim: int = 256, min_sim: float = 0.15) -> None:
        self.passages = passages
        self.dim = dim
        self.min_sim = min_sim
        self.vecs = {p.id: _hashing_embed(p.text, dim) for p in passages}

    def search(self, query: str, k: int = 10) -> list[Scored]:
        q = _hashing_embed(query, self.dim)
        scores = [Scored(p, _cosine(q, self.vecs[p.id])) for p in self.passages]
        scores = [s for s in scores if s.score >= self.min_sim]
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[:k]


def _minmax(scores: list[Scored]) -> dict[str, float]:
    if not scores:
        return {}
    vals = [s.score for s in scores]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {s.passage.id: (s.score - lo) / span for s in scores}


class HybridRetriever:
    """Reciprocal-weighted fusion of BM25 and dense retrieval (a la hybrid RAG)."""

    def __init__(self, passages: list[Passage], alpha: float = 0.5) -> None:
        self.passages = {p.id: p for p in passages}
        self.bm25 = BM25Index(passages)
        self.dense = DenseIndex(passages)
        self.alpha = alpha  # weight on lexical vs semantic

    def search(self, query: str, k: int = 10, pool: int = 20) -> list[Scored]:
        bm25_hits = self.bm25.search(query, k=pool)
        dense_hits = self.dense.search(query, k=pool)
        bm25_norm = _minmax(bm25_hits)
        dense_norm = _minmax(dense_hits)
        # dict.fromkeys preserves deterministic insertion order; a `set` union
        # would iterate in Python's hash-randomized order, silently changing
        # tie-breaks between runs/interpreters -- a real determinism bug for
        # anything used as a benchmark.
        ids = list(dict.fromkeys([*bm25_norm.keys(), *dense_norm.keys()]))
        fused = [
            Scored(self.passages[pid],
                  self.alpha * bm25_norm.get(pid, 0.0) + (1 - self.alpha) * dense_norm.get(pid, 0.0))
            for pid in ids
        ]
        # secondary sort key on id makes ties fully deterministic too
        fused.sort(key=lambda s: (-s.score, s.passage.id))
        return fused[:k]

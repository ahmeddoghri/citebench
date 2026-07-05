"""CiteBench — hybrid retrieval + reranking benchmark for citation-grounded RAG.

>>> from citebench import HybridRetriever, CitationPipeline
>>> from citebench.corpus import PASSAGES
>>> pipeline = CitationPipeline(HybridRetriever(PASSAGES))
>>> ans = pipeline.answer("What is the shelf life of the drug substance?")
>>> ans.cited_passage.id
'p8'
"""
from .corpus import Passage, QAItem, PASSAGES, BENCH, tokenize
from .retrieve import BM25Index, DenseIndex, HybridRetriever, Scored
from .rerank import HeuristicReranker, Reranker, Rescored
from .pipeline import CitationPipeline, Answer

__all__ = [
    "Passage", "QAItem", "PASSAGES", "BENCH", "tokenize",
    "BM25Index", "DenseIndex", "HybridRetriever", "Scored",
    "HeuristicReranker", "Reranker", "Rescored",
    "CitationPipeline", "Answer",
]
__version__ = "0.1.0"

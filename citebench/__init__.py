"""CiteBench, hybrid retrieval + reranking benchmark for citation-grounded RAG.

>>> from citebench import HybridRetriever, CitationPipeline
>>> from citebench.corpus import PASSAGES
>>> pipeline = CitationPipeline(HybridRetriever(PASSAGES))
>>> ans = pipeline.answer("What is the shelf life of the drug substance?")
>>> ans.cited_passage.id
'p8'
"""
from .corpus import BENCH, PASSAGES, Passage, QAItem, tokenize
from .pipeline import Answer, CitationPipeline
from .rerank import HeuristicReranker, Reranker, Rescored
from .retrieve import BM25Index, DenseIndex, HybridRetriever, Scored

__all__ = [
    "Passage", "QAItem", "PASSAGES", "BENCH", "tokenize",
    "BM25Index", "DenseIndex", "HybridRetriever", "Scored",
    "HeuristicReranker", "Reranker", "Rescored",
    "CitationPipeline", "Answer",
]
__version__ = "0.1.0"

"""Ask a citation-grounded question, then run the benchmark.
Run: python examples/quickstart.py
"""
from citebench import HybridRetriever, CitationPipeline
from citebench.corpus import PASSAGES
from citebench.eval import _precision_bm25_only, _precision_hybrid

pipeline = CitationPipeline(HybridRetriever(PASSAGES))

for q in [
    "What is the new primary endpoint under amendment v3?",   # adversarial draft in corpus
    "How quickly must sites report a grade-3 adverse event?",
]:
    ans = pipeline.answer(q)
    print(f"Q: {q}")
    print(f"   cited: [{ans.cited_passage.id}] {ans.cited_passage.text}")
    print(f"   confidence={ans.confidence:.2f}  validated={ans.validated}\n")

print("--- benchmark: does reranking improve citation precision? ---")
print(f"bm25_only        {_precision_bm25_only():.0%}")
print(f"hybrid_no_rerank  {_precision_hybrid(rerank=False):.0%}")
print(f"hybrid_rerank     {_precision_hybrid(rerank=True):.0%}")

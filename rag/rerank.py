# rag/rerank.py  (P14)
# Run from the repo root:  uv run python -m rag.rerank "<your query words>"            (cross-encoder, the default)
#                          uv run python -m rag.rerank --llm "<your query words>"      (the chapter's batch LLM reranker, one request)
# Chapter 8's second stage: hybrid search pulls a pool of candidates fast, then a cross-encoder reads the query
# and each candidate chunk together and gives the pair a score. The pool is sorted by that score and cut to k.
# The cross-encoder is the chapter's model, running on your machine; a score can be negative, and only the order matters.
import sys

from sentence_transformers import CrossEncoder

from rag import search
from rag.llm import chat, parse_json_list

MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"   # the chapter's cross-encoder
POOL = 20                                          # how many hybrid results the reranker reads; the chapter uses 5 times the limit

_model = None

# In: nothing.  Out: the cross-encoder, loaded once per run.
def model():
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL, device="cpu")
    return _model

# In: a query and a list of chunks.  Out: one score per chunk, from the cross-encoder reading (query, chunk) pairs.
def score(query, chunks):
    pairs = [[query, c["text"]] for c in chunks]
    return [float(s) for s in model().predict(pairs)]

# In: a query, how many results, the pool size.  Out: the top k chunks after reranking, each with "score" and "hybrid_rank" added.
def rerank(query, k=5, pool=None):
    if pool is None:                                               # read POOL at call time, so a changed constant takes effect
        pool = POOL
    candidates = search.hybrid(query, pool)
    scores = score(query, candidates)
    ranked = []
    for rank, (c, s) in enumerate(zip(candidates, scores), start=1):
        ranked.append({**c, "score": s, "hybrid_rank": rank})     # {**c} copies the chunk's fields into a new dict
    ranked.sort(key=lambda r: -r["score"])                         # highest score first
    return ranked[:k]

# In: a query and how many results.  Out: the top k chunks ordered by the chapter's batch LLM prompt, one request.
def rerank_llm(query, k=5, pool=None):
    if pool is None:
        pool = POOL
    candidates = search.hybrid(query, pool)
    listing = "\n".join(f"{i}. {c['text'][:300]}" for i, c in enumerate(candidates, start=1))
    prompt = f"""Rank the passages listed below by relevance to the following search query.

Query: "{query}"

Passages:
{listing}

Return the passage numbers in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[7, 1, 3, 2, 5]

Ranking:"""
    order = parse_json_list(chat(prompt))
    ranked = []
    for new_rank, number in enumerate(order, start=1):
        if 1 <= number <= len(candidates) and candidates[number - 1] not in ranked:   # ignore numbers the model made up or repeated
            ranked.append({**candidates[number - 1], "llm_rank": new_rank, "hybrid_rank": number})
    return ranked[:k]

if __name__ == "__main__":
    args = sys.argv[1:]
    use_llm = "--llm" in args
    query = " ".join(a for a in args if a != "--llm")
    results = rerank_llm(query) if use_llm else rerank(query)
    print(f"{'batch LLM' if use_llm else 'cross-encoder'} rerank of the hybrid top {POOL} for: {query}")
    for rank, r in enumerate(results, start=1):
        label = f"llm rank {r['llm_rank']}" if use_llm else f"score {r['score']:.3f}"
        print(f"{rank}. {r['id']}  {label}  was hybrid rank {r['hybrid_rank']}  {' '.join(r['text'].split())[:70]}...")

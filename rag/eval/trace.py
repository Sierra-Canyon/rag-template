# rag/eval/trace.py  (P21, and chapter 9's error analysis)
# Run from the repo root:  uv run python -m rag.eval.trace              (every golden question the pipeline misses at k = 5)
#                          uv run python -m rag.eval.trace g07 g13      (named questions, hit or miss)
#                          uv run python -m rag.eval.trace --expand g07 (the same, with the query expanded first, one request per question)
# Chapter 9 lesson 6, written down: for a question, the rank of its labeled chunk at every stage of the pipeline,
# so a miss can be traced to the stage that lost it. A dash means the chunk was not in that stage's list at all.
#   bm25 and semantic: rank in each engine's top 50 (the pool hybrid search fuses)
#   hybrid: rank after fusion, in the top 50
#   rerank pool: whether the chunk was among the 20 hybrid results the cross-encoder read
#   rerank: rank after the cross-encoder, in those 20
import sys

from rag import rerank, search
from rag.enhance import enhance
from rag.eval.check_golden import load_golden

# In: a ranked list of chunks and the labeled ids.  Out: the rank of the first labeled chunk, or None.
def rank_of(results, relevant):
    for rank, r in enumerate(results, start=1):
        if r["id"] in relevant:
            return rank
    return None

# In: a question and its labeled ids.  Out: a dict of stage name to rank.
def trace(question, relevant):
    pool = search.POOL
    kw = search.keyword(question, pool)
    sem = search.semantic(question, pool)
    hyb = search.hybrid(question, pool)
    reranked = rerank.rerank(question, rerank.POOL)       # the whole reranked pool, not just the top 5
    return {
        "bm25": rank_of(kw, relevant),
        "semantic": rank_of(sem, relevant),
        "hybrid": rank_of(hyb, relevant),
        "in rerank pool": rank_of(hyb[:rerank.POOL], relevant) is not None,
        "rerank": rank_of(reranked, relevant),
    }

# In: a trace.  Out: the name of the first stage that lost the chunk, in pipeline order.
def lost_at(t):
    if t["bm25"] is None and t["semantic"] is None:
        return "both engines (never retrieved)"
    if t["hybrid"] is None or t["hybrid"] > rerank.POOL:
        return "fusion (outside the rerank pool)"
    if t["rerank"] is None or t["rerank"] > 5:
        return "reranker"
    return "not lost: in the top 5"

def show(label, t):
    cells = "  ".join(f"{k} {'-' if v is None else v}" if not isinstance(v, bool) else f"{k} {'yes' if v else 'no'}" for k, v in t.items())
    print(f"   {label:9} {cells}   -> lost at: {lost_at(t)}")

if __name__ == "__main__":
    args = sys.argv[1:]
    use_expand = "--expand" in args
    wanted = [a for a in args if a != "--expand"]
    gold = load_golden()
    if not wanted:                                           # default: every question the final stage misses
        wanted = [g["id"] for g in gold if rank_of(rerank.rerank(g["question"], 5), g["chunk_ids"]) is None]
        print(f"{len(wanted)} of {len(gold)} questions have no labeled chunk in the final top 5")
    for g in gold:
        if g["id"] not in wanted:
            continue
        print(f"\n{g['id']} ({g['kind']}): {g['question']}   labeled {g['chunk_ids']}")
        show("original", trace(g["question"], g["chunk_ids"]))
        if use_expand:
            better = enhance("expand", g["question"])
            print(f"   expanded: {better}")
            show("expanded", trace(better, g["chunk_ids"]))

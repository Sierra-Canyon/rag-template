# rag/search.py  (P11)
# One place that builds the engines over the chunks, so every script searches the same chunks the same way.
# Run from the repo root:  uv run python -m rag.search <your query words>     (keyword, semantic and hybrid lists)
import sys
from functools import lru_cache

from rag.chunking import OVERLAP, SIZE, load_chunks
from rag.hybrid import rrf
from rag.keyword import Keyword
from rag.semantic import Semantic

POOL = 50    # how many results each engine hands to the fusion; the chapter uses 500 times the limit on 5,000 movies

# In: a size and an overlap.  Out: the chunks and the two engines over them, built once per run (lru_cache remembers the result).
@lru_cache(maxsize=None)
def engines(size=SIZE, overlap=OVERLAP):
    units = load_chunks(size, overlap)
    return units, Keyword(units), Semantic(units, f"chunks_{size}_{overlap}")   # one vector file per chunk configuration

def keyword(query, k=5, size=SIZE, overlap=OVERLAP):
    return engines(size, overlap)[1].search(query, k)

def semantic(query, k=5, size=SIZE, overlap=OVERLAP):
    return engines(size, overlap)[2].search(query, k)

# In: a query and how many results.  Out: the chunks the fused ranking puts first.
def hybrid(query, k=5, pool=POOL, size=SIZE, overlap=OVERLAP):
    units, kw, sem = engines(size, overlap)
    by_id = {u["id"]: u for u in units}
    rankings = [[r["id"] for r in kw.search(query, pool)],
                [r["id"] for r in sem.search(query, pool)]]
    return [by_id[i] for i in rrf(rankings, top=k)]

if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    print("bm25    ", [(r["id"], round(r["score"], 2)) for r in keyword(q)])
    print("semantic", [(r["id"], round(r["score"], 3)) for r in semantic(q)])
    print("hybrid  ", [r["id"] for r in hybrid(q)])

# rag/scoreboard.py  (P06, grows in P08 and P11)
# Run from the repo root:  uv run python -m rag.scoreboard
# Scores every search method on the queries in rag/queries.json. A query is "found" when the passage that
# answers it is in the method's top 5; the table shows the rank it got, or - for not in the top 5.
# The needle is how the script knows which passage is right: a few words copied exactly from the answer passage.
import json

from rag.corpus import ROOT, load_passages
from rag.keyword import Keyword

QUERIES = json.loads((ROOT / "rag" / "queries.json").read_text(encoding="utf-8"))

# In: text.  Out: the same text lowercased with every run of whitespace collapsed to one space, so needles match across line breaks.
def norm(s):
    return " ".join(s.lower().split())

# In: a result list and a needle.  Out: the rank (1 = first) of the first result containing the needle, or None.
def first_hit(results, needle):
    for rank, r in enumerate(results, start=1):
        if norm(needle) in norm(r["text"]):
            return rank
    return None

# In: nothing.  Out: a dict from method name to a function that takes a query and returns its top 5.
def methods():
    units = load_passages()
    for q in QUERIES:                               # every needle must be in some passage, or the query can never be found
        assert any(norm(q["needle"]) in norm(u["text"]) for u in units), f"{q['id']}: needle not in any passage"
    kw = Keyword(units)
    return {
        "tfidf": lambda q: kw.search(q, 5, method="tfidf"),
        "bm25": lambda q: kw.search(q, 5),
    }

if __name__ == "__main__":
    ms = methods()
    totals = dict.fromkeys(ms, 0)
    print("query ", *(name.rjust(8) for name in ms))
    for q in QUERIES:
        cells = []
        for name, run in ms.items():
            rank = first_hit(run(q["query"]), q["needle"])
            if rank is not None and rank <= 3:
                totals[name] += 1
            cells.append(str(rank or "-").rjust(8))
        print(q["id"].ljust(6), *cells)
    print("hit@3 ", *(f"{totals[n]}/{len(QUERIES)}".rjust(8) for n in ms))

# rag/enhance_check.py  (P13)
# Run from the repo root:  uv run python -m rag.enhance_check
# Chapter 7's three enhancements, measured instead of eyeballed: for every query in rag/queries.json and
# rag/questions.json, the hybrid rank of the answer chunk with the query as typed, rewritten, and expanded.
# Two requests per query (rewrite and expand); spell is skipped because these queries have no typos.
# "helped" means the rank got better (or appeared), "hurt" means it got worse (or vanished), "same" otherwise.
import json

from rag import search
from rag.corpus import ROOT
from rag.enhance import enhance
from rag.scoreboard import first_hit

QUERIES = json.loads((ROOT / "rag" / "queries.json").read_text(encoding="utf-8")) + \
          json.loads((ROOT / "rag" / "questions.json").read_text(encoding="utf-8"))
K = 10        # how deep to look for the answer chunk

# In: two ranks (None = not found).  Out: helped / hurt / same.
def verdict(before, after):
    if before == after:
        return "same"
    if after is None:
        return "hurt"
    if before is None or after < before:
        return "helped"
    return "hurt"

if __name__ == "__main__":
    counts = {"rewrite": {"helped": 0, "hurt": 0, "same": 0}, "expand": {"helped": 0, "hurt": 0, "same": 0}}
    print("query   raw  rewrite  expand   (rank of the answer chunk in the hybrid top 10, - = not there)")
    for q in QUERIES:
        raw = first_hit(search.hybrid(q["query"], K), q["needle"])
        ranks = {"raw": raw}
        for method in ("rewrite", "expand"):
            better = enhance(method, q["query"])
            ranks[method] = first_hit(search.hybrid(better, K), q["needle"])
            counts[method][verdict(raw, ranks[method])] += 1
            print(f"   {method:8} {q['id']}: {better}")
        print(f"{q['id']:6} {str(raw or '-'):>4} {str(ranks['rewrite'] or '-'):>8} {str(ranks['expand'] or '-'):>7}")
    for method, c in counts.items():
        print(f"{method}: helped {c['helped']}  hurt {c['hurt']}  same {c['same']}")

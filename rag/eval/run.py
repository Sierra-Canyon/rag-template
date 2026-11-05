# rag/eval/run.py  (P16)
# Run from the repo root:  uv run python -m rag.eval.run            (k = 5)
#                          uv run python -m rag.eval.run 10         (k = 10)
# Every method on every golden question, with chapter 9's numbers averaged over the questions. Also writes
# rag/eval/runs.json with the per-question ranks, which P21's tracer reads.
import json
import sys

from rag import rerank, search
from rag.corpus import ROOT
from rag.eval.check_golden import load_golden
from rag.eval.metrics import score

METHODS = {
    "bm25": search.keyword,
    "semantic": search.semantic,
    "hybrid": search.hybrid,
    "rerank": rerank.rerank,
}

# In: k.  Out: a dict from method name to its list of per-question results; prints the table.
def run(k=5):
    gold = load_golden()
    print(f"n = {len(gold)} questions; one question moves recall by {1 / len(gold):.3f}")
    print(f"method     prec@{k}  recall@{k}   F1     MRR")
    runs = {}
    for name, fn in METHODS.items():
        per = []
        for g in gold:
            ids = [r["id"] for r in fn(g["question"], k)]
            per.append({**score(ids, g["chunk_ids"], k), "id": g["id"], "kind": g["kind"], "ranked": ids})
        runs[name] = per
        n = len(per)
        print(f"{name:10} {sum(p['precision'] for p in per) / n:6.4f}  {sum(p['recall'] for p in per) / n:8.4f}  "
              f"{sum(p['f1'] for p in per) / n:6.4f} {sum(p['rr'] for p in per) / n:6.4f}")
    ceiling = sum(min(len(g["chunk_ids"]), k) for g in gold) / (k * len(gold))
    print(f"highest precision@{k} this set allows: {ceiling:.4f}")
    (ROOT / "rag" / "eval" / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return runs

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

# rag/eval/grounded.py  (P20, Milestone A1 day 2)
# Run from the repo root:  uv run python -m rag.eval.grounded            (both modes, every question)
#                          uv run python -m rag.eval.grounded none       (one mode)
# The milestone's eval: every question in rag/eval/grounded_questions.json through the agent twice, once with
# the search tool and once without, and the two scored the same way:
#   an in-corpus question is correct when the answer contains the labeled answer string, refused when the agent
#   said it does not have enough information, wrong otherwise.
#   an out-of-corpus question (answer null) is correct when the agent refused, wrong when it answered anyway.
# In tool mode the rows also say whether the agent cited one of the labeled chunks. Writes rag/eval/grounded_results.json.
import json
import sys

from rag.agent import run
from rag.answer import cited_ids
from rag.corpus import ROOT
from rag.scoreboard import norm

QUESTIONS = ROOT / "rag" / "eval" / "grounded_questions.json"
RESULTS = ROOT / "rag" / "eval" / "grounded_results.json"
REFUSAL = "don't have enough information"

# In: a question record and the agent's answer.  Out: "correct", "refused" or "wrong".
def grade(q, answer):
    refused = REFUSAL in answer.lower()
    if q["answer"] is None:                                   # out of corpus: the right move is to refuse
        return "correct" if refused else "wrong"
    if refused:
        return "refused"
    return "correct" if norm(q["answer"]) in norm(answer) else "wrong"

# In: a mode.  Out: the list of per-question rows. Prints them.
def evaluate(mode):
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    print(f"\n== mode {mode}: {len(questions)} questions")
    for q in questions:
        r = run(mode, q["question"])
        verdict = grade(q, r["answer"])
        cited = cited_ids(r["answer"])
        labeled_cited = bool(set(cited) & set(q.get("chunk_ids", [])))
        rows.append({"id": q["id"], "verdict": verdict, "calls": len(r["calls"]), "tool_ms": r["tool_ms"],
                     "cited_labeled": labeled_cited, "answer": r["answer"], "queries": [c["query"] for c in r["calls"]]})
        print(f"   {q['id']}  {verdict:8}  calls {len(r['calls'])}  cited labeled {'yes' if labeled_cited else 'no '}  "
              f"{' '.join(r['answer'].split())[:80]}")
    return rows

# In: the rows of one mode.  Out: the summary line.
def summary(mode, rows):
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("correct", "refused", "wrong")}
    calls = sum(r["calls"] for r in rows)
    ms = [r["tool_ms"] for r in rows if r["calls"]]
    median = sorted(ms)[len(ms) // 2] if ms else 0
    return (f"{mode:5} correct {counts['correct']:2}  refused {counts['refused']:2}  wrong {counts['wrong']:2}  "
            f"tool calls {calls:2}  median tool ms {median}  cited a labeled chunk {sum(1 for r in rows if r['cited_labeled'])}")

if __name__ == "__main__":
    modes = sys.argv[1:] or ["none", "tool"]
    results = {}
    for mode in modes:
        results[mode] = evaluate(mode)
    print()
    for mode in modes:
        print(summary(mode, results[mode]))
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

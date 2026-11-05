# rag/eval/judge.py  (P16 extension)
# Run from the repo root:  uv run python -m rag.eval.judge g01 g07 g17
# Chapter 9's LLM judge against your own labels: for each golden question named on the command line, the top
# five from the pipeline go to the model with the chapter's 0-3 prompt, one request per question, and the
# table shows the model's score next to whether you labeled that chunk relevant. Where they disagree is the point.
import sys

from rag import rerank
from rag.eval.check_golden import load_golden
from rag.llm import chat, parse_json_list

JUDGE = """Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

# In: a golden question and k.  Out: a list of (chunk id, labeled relevant?, judge score) for the top k.
def judge(g, k=5):
    hits = rerank.rerank(g["question"], k)
    listing = "\n".join(f"{i}. {' '.join(h['text'].split())[:400]}" for i, h in enumerate(hits, start=1))
    scores = parse_json_list(chat(JUDGE.format(query=g["question"], results=listing)))
    rows = []
    for h, s in zip(hits, scores):
        rows.append((h["id"], h["id"] in g["chunk_ids"], int(s)))
    return rows

if __name__ == "__main__":
    wanted = sys.argv[1:]
    gold = {g["id"]: g for g in load_golden()}
    agree = disagree = 0
    for gid in wanted:
        g = gold[gid]
        print(f"\n{gid}: {g['question']}")
        for cid, labeled, s in judge(g):
            verdict = "agree" if (labeled and s >= 2) or (not labeled and s <= 1) else "DISAGREE"
            if verdict == "agree":
                agree += 1
            else:
                disagree += 1
            print(f"   {cid}  labeled {'yes' if labeled else 'no ':3}  judge {s}/3  {verdict}")
    print(f"\nagree {agree}   disagree {disagree}   (labeled yes with judge 2 or 3, or labeled no with judge 0 or 1, is agreement)")

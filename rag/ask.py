# rag/ask.py  (P22)
# Run from the repo root:  uv run python -m rag.ask "<a question about your corpus>"
# One command, the whole pipeline: hybrid search over the chunks, the cross-encoder on the hybrid pool (rerank.POOL), the refusal
# gate, a cited answer, every citation checked. This is the command the README's front page tells a stranger to run.
import sys

from rag.answer import answer

if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    r = answer("cite", question)
    print(r["answer"])
    print()
    print("passages:", ", ".join(f"{cid} ({score})" for cid, score in r["retrieved"]))
    print("cited:", r["cited"], "  invalid:", r["invalid"], "  gate:", r["gate"])

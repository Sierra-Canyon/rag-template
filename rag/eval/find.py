# rag/eval/find.py  (P15)
# Run from the repo root:  uv run python -m rag.eval.find my name is Noman
# An exact-phrase finder that bypasses every ranking you have built: it reads every chunk and prints the ones
# that contain the phrase. Labels for the golden set come from this, never from your own search, so the golden
# set cannot contain only what your pipeline already finds.
import sys

from rag.chunking import load_chunks
from rag.scoreboard import norm

if __name__ == "__main__":
    phrase = norm(" ".join(sys.argv[1:]))
    hits = 0
    for c in load_chunks():
        if phrase in norm(c["text"]):
            hits += 1
            print(c["id"], c["char_start"], " ".join(c["text"].split())[:100])
    print(f"{hits} chunk(s) contain the phrase")

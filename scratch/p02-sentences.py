# scratch/p02-sentences.py  (P02 extension)
# Run from the repo root:  uv run python scratch/p02-sentences.py
# Reads scratch/p02-sentences.txt, one sentence from your corpus per line, and prints each one before and after
# rag/preprocess.py's pipeline. There is nothing to edit in this file; the sentences go in the .txt file.
import sys
from pathlib import Path
sys.path.insert(0, ".")                       # so "from rag.preprocess" works from the repo root
from rag.preprocess import tokenize, preprocess

lines = [l.strip() for l in Path("scratch/p02-sentences.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(lines) == 5, f"scratch/p02-sentences.txt holds {len(lines)} sentences; put exactly five in it, one per line"
for s in lines:
    print(s)
    print("   tokens:", tokenize(s))
    print("   terms: ", preprocess(s))

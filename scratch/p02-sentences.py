# scratch/p02-sentences.py  (P02 extension)
# Run from the repo root:  uv run python scratch/p02-sentences.py
# Reads scratch/p02-sentences.txt: three sentences copied from your corpus, separated by a blank line.
# A copied sentence often arrives on two or three lines, because the corpus file wraps its lines; that is
# fine, the blank line is what separates one sentence from the next. Prints each sentence before and after
# rag/preprocess.py's pipeline. There is nothing to edit in this file; the sentences go in the .txt file.
import sys
from pathlib import Path
sys.path.insert(0, ".")                       # so "from rag.preprocess" works from the repo root
from rag.preprocess import tokenize, preprocess

text = Path("scratch/p02-sentences.txt").read_text(encoding="utf-8")
blocks = [" ".join(b.split()) for b in text.split("\n\n") if b.strip()]   # each blank-line-separated block, its line breaks turned into spaces
assert len(blocks) == 3, f"scratch/p02-sentences.txt holds {len(blocks)} sentences; put exactly three in it, with a blank line between them"
for s in blocks:
    print(s)
    print("   tokens:", tokenize(s))
    print("   terms: ", preprocess(s))

# rag/corpus.py  (P02)
# Run from the repo root:  uv run python -m rag.corpus
# Your corpus as passages. TF-IDF scores terms per document, and your corpus is one file, so for P02 to P08
# a "document" is a passage: the blank-line blocks of the file, merged forward until each one is at least
# MIN_CHARS long. The one line you edit is MIN_CHARS. (Chapter 5 replaces passages with real chunks.)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent     # this file is rag/corpus.py; two parents up is the repo root
CORPUS = ROOT / "data" / "corpus.txt"
MIN_CHARS = 200          # the minimum passage length you settled on in A06

# In: nothing.  Out: the whole corpus as one string.
def load_text():
    return CORPUS.read_text(encoding="utf-8")

# In: the minimum length.  Out: a list of {"id": "p0000", "text": "..."} dicts, one per passage, in file order.
def load_passages(min_chars=MIN_CHARS):
    passages, buf = [], ""                        # buf collects blocks until it is long enough
    for block in load_text().split("\n\n"):      # a blank line separates blocks
        block = " ".join(block.split())           # collapse every run of whitespace, including newlines, to one space
        if not block:
            continue
        buf = f"{buf} {block}" if buf else block  # append this block to what is waiting
        if len(buf) >= min_chars:
            passages.append(buf)
            buf = ""
    if buf and passages:
        passages[-1] += " " + buf                 # the leftover at the end joins the last passage rather than being dropped
    elif buf:
        passages.append(buf)
    return [{"id": f"p{i:04d}", "text": t} for i, t in enumerate(passages)]   # p0000, p0001, ... four digits

if __name__ == "__main__":
    ps = load_passages()
    lengths = sorted(len(p["text"]) for p in ps)
    print(len(ps), "passages")
    print("shortest", lengths[0], "median", lengths[len(lengths) // 2], "longest", lengths[-1])
    print(ps[0]["id"], ps[0]["text"][:80])
    print(ps[-1]["id"], ps[-1]["text"][-80:])

# rag/chunking.py  (P09)
# Run from the repo root:  uv run python -m rag.chunking                (the chunks, with three checks)
#                     or:  uv run python -m rag.chunking 300 0          (a different size and overlap, in words)
# Chapter 5's fixed-size chunker, on your corpus: windows of SIZE words that step forward by SIZE - OVERLAP words,
# so each chunk shares its last OVERLAP words with the next one. Every chunk is a record, not a string: it carries
# its id and the exact character positions it came from in data/corpus.txt, which is what a citation needs later.
# The two lines you edit are SIZE and OVERLAP; the extension chooses them.
import sys

from rag.corpus import load_text

SIZE = 200       # words per chunk
OVERLAP = 50     # words shared between one chunk and the next

# In: text.  Out: a list of (start, end) character positions, one per word, where a word is anything between whitespace.
def word_spans(text):
    spans = []
    position = 0                                  # where the search for the next word begins
    for word in text.split():                     # split() with no argument splits on any run of whitespace
        start = text.find(word, position)         # the word's first character, found at or after the previous word
        end = start + len(word)
        spans.append((start, end))
        position = end
    return spans

# In: text, a chunk size and an overlap, both in words.  Out: a list of {"id", "char_start", "char_end", "text"} dicts in file order.
def chunk(text, size=SIZE, overlap=OVERLAP):
    assert 0 <= overlap < size, f"overlap {overlap} must be smaller than size {size}"   # equal would mean a step of 0: the same chunk forever
    spans = word_spans(text)
    step = size - overlap
    chunks = []
    n = 0
    for i in range(0, len(spans), step):          # i is the position of the chunk's first word
        window = spans[i:i + size]                # the slice stops at the last word on its own, so the last chunk can be short
        start = window[0][0]                      # first character of the first word
        end = window[-1][1]                       # last character of the last word
        chunks.append({"id": f"c{n:05d}", "char_start": start, "char_end": end, "text": text[start:end]})
        n += 1
        if i + size >= len(spans):                # this window reached the last word: stop, or the next windows would be shorter copies of the end
            break
    return chunks

# In: a size and an overlap.  Out: the chunks of data/corpus.txt.
def load_chunks(size=SIZE, overlap=OVERLAP):
    return chunk(load_text(), size, overlap)

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else SIZE
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else OVERLAP
    text = load_text()
    cs = chunk(text, size, overlap)
    words = len(text.split())
    print(f"{words} words -> {len(cs)} chunks of {size} words, overlap {overlap}")
    print("every chunk is an exact slice of the file:", all(text[c["char_start"]:c["char_end"]] == c["text"] for c in cs))
    print("last chunk ends at the last word:", cs[-1]["char_end"] == len(text.rstrip()))
    print("last chunk words:", len(cs[-1]["text"].split()))
    print(cs[0]["id"], cs[0]["char_start"], cs[0]["char_end"], repr(cs[0]["text"][:60]))
    print(cs[1]["id"], cs[1]["char_start"], cs[1]["char_end"], repr(cs[1]["text"][:60]))

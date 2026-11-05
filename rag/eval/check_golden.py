# rag/eval/check_golden.py  (P15)
# Run from the repo root:  uv run python -m rag.eval.check_golden
# Proves the golden set is consistent with the chunking you froze: every labeled chunk exists, the quote is in
# one of its chunks, the answer string is in one of its chunks, and the kinds add up. If SIZE or OVERLAP ever
# change, this fails, which is the point: labels are tied to a chunking.
import json
from collections import Counter

from rag.chunking import OVERLAP, SIZE, load_chunks
from rag.corpus import ROOT
from rag.scoreboard import norm

GOLDEN = ROOT / "rag" / "eval" / "golden.json"

# In: nothing.  Out: the golden set as a list of dicts.
def load_golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))

if __name__ == "__main__":
    gold = load_golden()
    chunks = {c["id"]: c for c in load_chunks()}
    print(f"chunking {SIZE}/{OVERLAP}, {len(chunks)} chunks, {len(gold)} questions")
    for g in gold:
        missing = [c for c in g["chunk_ids"] if c not in chunks]
        assert not missing, f"{g['id']}: no such chunk {missing}"
        assert any(norm(g["quote"]) in norm(chunks[c]["text"]) for c in g["chunk_ids"]), f"{g['id']}: quote not in its chunks"
        if "quote2" in g:                            # a two-chunk question: the second sentence has to be in a labeled chunk too
            assert any(norm(g["quote2"]) in norm(chunks[c]["text"]) for c in g["chunk_ids"]), f"{g['id']}: quote2 not in its chunks"
        assert any(norm(g["answer"]) in norm(chunks[c]["text"]) for c in g["chunk_ids"]), f"{g['id']}: answer not in its chunks"
    ids = [g["id"] for g in gold]
    assert len(ids) == len(set(ids)), "two questions share an id"
    print("kinds:", dict(Counter(g["kind"] for g in gold)))
    print("mean labeled chunks per question:", round(sum(len(g["chunk_ids"]) for g in gold) / len(gold), 2))
    print("golden set is consistent with the chunking")

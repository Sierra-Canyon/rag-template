# rag/semantic.py  (P08)
# Run from the repo root:  uv run python -m rag.semantic <your query words>
# Chapter 4's semantic search, on your corpus: every passage becomes a vector from the chapter's model
# (all-MiniLM-L6-v2, running on your machine), the query becomes one too, and the score is cosine similarity.
# The passage vectors are built once and saved under data/, with a note of which model and which text made them,
# so they are never loaded against the wrong model or a corpus that has changed since.
import hashlib
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.corpus import ROOT, load_passages

MODEL = "all-MiniLM-L6-v2"      # the chapter's model; change it here and every vector file is rebuilt on the next run
DATA = ROOT / "data"

_model = None

# In: nothing.  Out: the loaded model, loaded once per run (the first load takes a few seconds).
def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL, device="cpu")
    return _model

# In: a list of strings.  Out: one row per string, the model's vector for it, as float32.
def embed(texts):
    return np.asarray(model().encode(texts, show_progress_bar=len(texts) > 100), dtype=np.float32)

# In: a vector, or a matrix of row vectors.  Out: the same, each scaled to length 1, so a dot product is a cosine.
def normalize(V):
    lengths = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.maximum(lengths, 1e-10)         # the maximum guards an all-zero vector against division by zero

class Semantic:
    # In: a list of units (passages now, chunks from P11) and a name for their vector file.
    def __init__(self, units, name):
        self.units = units
        fingerprint = hashlib.sha256("\x00".join(u["text"] for u in units).encode()).hexdigest()   # one hash of every text, in order
        vec_path = DATA / f"emb_{name}.npy"
        meta_path = DATA / f"emb_{name}.json"
        meta = {"model": MODEL, "count": len(units), "sha256": fingerprint}
        if vec_path.exists() and meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            V = np.load(vec_path)                 # same model, same count, same text: the saved vectors are the right ones
        else:
            print(f"embedding {len(units)} units with {MODEL}", file=sys.stderr)
            V = embed([u["text"] for u in units])
            np.save(vec_path, V)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert V.shape[0] == len(units), f"{V.shape[0]} vectors for {len(units)} units"
        self.Vn = normalize(V)                    # every row has length 1, so Vn @ q is a column of cosines

    # In: a query string.  Out: its vector, length 1.
    def query_vector(self, query):
        return normalize(embed([query])[0])

    # In: a query string and how many results.  Out: up to k units, most similar first, each with its cosine score.
    def search(self, query, k=5):
        sims = self.Vn @ self.query_vector(query)                 # one cosine per unit, in one matrix multiply
        top = np.argsort(-sims, kind="stable")[:k]                # highest first; stable keeps ties in file order
        return [dict(self.units[i], score=float(sims[i])) for i in top]

if __name__ == "__main__":
    sem = Semantic(load_passages(), "passages")
    print("matrix:", sem.Vn.shape)
    for r in sem.search(" ".join(sys.argv[1:])):
        print(f"{r['score']:.3f}  {r['id']}  {r['text'][:70]}")

# rag/keyword.py  (P06)
# Run from the repo root:  uv run python -m rag.keyword <your query words>      (tf-idf and BM25 lists, side by side)
#                     or:  uv run python -m rag.keyword idf <word> <word>       (the plain idf and the BM25 idf of each word)
# Two keyword scorers over the P04 index. tfidf_score is the chapter 2 one: the tf-idf of each query term in the
# passage, added up. bm25 is chapter 3's: the same idea with the three fixes the chapter builds, in its order.
# The constants are the chapter's. Change K1 and B here if the chapter's values differ from these.
import math
import sys

from rag.corpus import load_passages
from rag.preprocess import preprocess
from rag.tfidf import TfidfIndex

K1 = 1.5     # chapter 3 lesson 2: how fast term frequency saturates
B = 0.75     # chapter 3 lesson 3: how much a passage's length counts against it

class Keyword:
    # In: a list of passages, plus any keyword arguments for preprocess (stopwords=..., drop_stems=...).
    def __init__(self, units, k1=K1, b=B, **pp):
        self.ix = TfidfIndex(units, **pp)           # the P04 index: one Counter per passage, lengths, df, N
        self.units = units
        self.k1 = k1
        self.b = b
        self.avgdl = sum(self.ix.lengths) / self.ix.N   # average passage length in terms, for length normalization

    # In: a term.  Out: chapter 3 lesson 1's idf: log((N - df + 0.5) / (df + 0.5) + 1).
    def bm25_idf(self, term):
        df = self.ix.df[term]                       # a Counter, so an unknown term gives 0 instead of an error
        return math.log((self.ix.N - df + 0.5) / (df + 0.5) + 1)

    # In: a term and a passage number.  Out: chapter 3 lessons 2 and 3's tf: saturated, and normalized by length.
    def bm25_tf(self, term, i):
        tf = self.ix.docs[i][term]                  # the raw count, the same number tfidf.py's tf returns
        length_norm = 1 - self.b + self.b * (self.ix.lengths[i] / self.avgdl)
        return (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

    # In: a list of terms and a passage number.  Out: the passage's BM25 score for the query: the sum over terms of tf times idf.
    def bm25(self, terms, i):
        score = 0.0
        for term in terms:
            if self.ix.docs[i][term]:               # a term not in the passage adds 0, so skip the arithmetic
                score += self.bm25_tf(term, i) * self.bm25_idf(term)
        return score

    # In: a list of terms and a passage number.  Out: the chapter 2 score: the sum over terms of tf-idf.
    def tfidf_score(self, terms, i):
        score = 0.0
        for term in terms:
            score += self.ix.tfidf(term, i)
        return score

    # In: a query string, how many results, "bm25" or "tfidf".  Out: up to k passages, best first, each with its score.
    def search(self, query, k=5, method="bm25"):
        terms = preprocess(query, **self.ix.pp)     # the query goes through the SAME pipeline as the passages, same arguments
        if method == "bm25":
            score = self.bm25
        else:
            score = self.tfidf_score
        scored = []
        for i in range(self.ix.N):
            s = score(terms, i)
            if s > 0:                               # a passage that matched nothing is not a result
                scored.append((s, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))   # highest score first; a tie breaks by passage position, so the order never changes between runs
        results = []
        for s, i in scored[:k]:
            results.append(dict(self.units[i], score=s))
        return results

if __name__ == "__main__":
    kw = Keyword(load_passages())
    if len(sys.argv) > 2 and sys.argv[1] == "idf":
        for word in sys.argv[2:]:
            terms = preprocess(word)                # the same pipeline the index used, so a stopword comes back empty
            if not terms:
                print(f"{word}: dropped by preprocess (a stopword), so it is in no passage and has no idf")
                continue
            term = terms[0]
            print(f"{word}: term {term!r}  df {kw.ix.df[term]}  idf {kw.ix.idf(term):.4f}  bm25_idf {kw.bm25_idf(term):.4f}")
        sys.exit()
    query = " ".join(sys.argv[1:])
    print("terms:", preprocess(query))
    for method in ("tfidf", "bm25"):
        print(f"--- {method}")
        for r in kw.search(query, method=method):
            print(f"{r['score']:7.3f}  {r['id']}  {r['text'][:70]}")

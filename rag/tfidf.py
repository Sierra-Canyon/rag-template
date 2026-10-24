# rag/tfidf.py  (P04)
# Run from the repo root:  uv run python -m rag.tfidf             (the index, four facts about it, three passages' top terms)
#                     or:  uv run python -m rag.tfidf stopwords   (P04 extension: your corpus's own stopwords, and what dropping them changes)
#                     or:  uv run python -m rag.tfidf idf WORD1 WORD2   (P04 Step 3: the df and idf of the words you predicted)
# The one line you edit is the one marked CHAPTER FORMULA: replace it with the idf formula Boot.dev chapter 2 gave you,
# character for character. If the chapter's term frequency is a raw count rather than a count divided by length, change tf too.
import math, sys
from collections import Counter

from rag.corpus import ROOT, load_passages
from rag.preprocess import preprocess

class TfidfIndex:
    # In: a list of passages, plus any keyword arguments for preprocess (stopwords=..., drop_stems=...).
    def __init__(self, units, **pp):
        self.units = units
        self.pp = pp
        self.docs = [Counter(preprocess(u["text"], **pp)) for u in units]   # one Counter of term -> count per passage
        self.lengths = [sum(c.values()) for c in self.docs]                 # terms per passage, after preprocessing
        self.N = len(units)
        self.df = Counter()                                                  # term -> how many passages contain it
        for counts in self.docs:
            self.df.update(counts.keys())

    # In: a term and a passage number.  Out: term frequency: the term's count in that passage, divided by the passage's length.
    def tf(self, term, i):
        return self.docs[i][term] / max(self.lengths[i], 1)

    # In: a term.  Out: inverse document frequency: large for a term in few passages, near zero for one in most of them.
    def idf(self, term):
        return math.log((self.N + 1) / (self.df[term] + 1))   # CHAPTER FORMULA

    def tfidf(self, term, i):
        return self.tf(term, i) * self.idf(term)

    # In: a passage number and how many terms.  Out: that passage's terms, highest tf-idf first (ties broken alphabetically).
    def top_terms(self, i, n=5):
        return sorted(self.docs[i], key=lambda t: (-self.tfidf(t, i), t))[:n]

if __name__ == "__main__":
    passages = load_passages()
    ix = TfidfIndex(passages)
    raw = TfidfIndex(passages, stopwords=frozenset())                         # the same index with no stopwords removed, so "the" is in it

    if len(sys.argv) > 2 and sys.argv[1] == "idf":                            # P04 Step 3: the df and idf of the words you predicted, e.g.  -m rag.tfidf idf ulysses the
        from rag.preprocess import preprocess
        for word in sys.argv[2:]:
            term = preprocess(word, stopwords=frozenset())[0]                 # the index holds stems, so look the word up as its stem, stopwords kept
            print(f"{word}: stem {term!r}  df {raw.df[term]}  idf {raw.idf(term):.4f}")
        sys.exit()
    print("passages:", ix.N, " vocabulary:", len(ix.df))
    once = sum(1 for n in ix.df.values() if n == 1)
    print("terms in exactly one passage:", once, f"({once / len(ix.df):.0%})")
    print("idf('the') with stopwords kept:", round(raw.idf("the"), 4), " df:", raw.df["the"])
    print("largest possible idf:", round(math.log((ix.N + 1) / 2), 4))   # the idf of a term in exactly one passage, by the formula above; change it if yours differs
    for i in (0, ix.N // 2, ix.N - 1):
        print(passages[i]["id"], ix.top_terms(i))

    if len(sys.argv) > 1 and sys.argv[1] == "stopwords":
        THRESHOLD = 0.20                                                       # a term in at least this share of passages is a corpus stopword
        common = sorted(t for t, n in ix.df.items() if n / ix.N >= THRESHOLD)
        for th in (0.10, 0.20, 0.30):
            print(f"df >= {th:.0%}:", sum(1 for n in ix.df.values() if n / ix.N >= th), "terms")
        print("corpus stopwords:", common)
        (ROOT / "rag" / "corpus_stopwords.txt").write_text("\n".join(common) + "\n", encoding="utf-8")

        cut = TfidfIndex(passages, drop_stems=frozenset(common))              # the same index with those stems dropped after stemming
        print("vocabulary before/after:", len(ix.df), len(cut.df))
        before, after = sum(ix.lengths), sum(cut.lengths)
        print(f"term occurrences removed: {1 - after / before:.1%}")
        changed = sum(len(set(ix.top_terms(i)) ^ set(cut.top_terms(i))) // 2 for i in range(ix.N))   # ^ is the symmetric difference: terms in one top-5 but not the other
        print(f"top-5 slots changed: {changed} of {5 * ix.N}")

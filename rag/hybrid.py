# rag/hybrid.py  (P11)
# Chapter 6's reciprocal rank fusion: two ranked lists in, one ranked list out, no scores involved.
# Each list votes 1 / (k + rank) for every unit it ranked, and a unit's fused score is the sum of its votes.
K = 60     # the chapter's k; change it here if the chapter's value differs

# In: a list of rankings (each a list of unit ids, best first), k, and how many to return.  Out: unit ids, best fused score first.
def rrf(rankings, k=K, top=5):
    scores = {}
    for ranking in rankings:
        for rank, uid in enumerate(ranking, start=1):          # rank 1 is the first entry
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda uid: (-scores[uid], uid))[:top]   # highest score first; a tie breaks by id, so the order never changes between runs

# In: a rank (1 = first) and k.  Out: that rank's vote, the chapter's rrf_score.
def rrf_score(rank, k=K):
    return 1.0 / (k + rank)

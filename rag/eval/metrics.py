# rag/eval/metrics.py  (P16)
# Chapter 9's three numbers for one question, plus the reciprocal rank. All four take the ranked chunk ids a
# method returned and the chunk ids you labeled relevant.
#   precision@k = relevant retrieved / k           (how much of what you found is relevant)
#   recall@k    = relevant retrieved / total relevant   (how much of what is relevant you found)
#   f1          = 2 * p * r / (p + r), 0 when both are 0
#   rr          = 1 / rank of the first relevant chunk, 0 if none is in the top k; the mean over questions is MRR

# In: a ranked list of chunk ids and the labeled ids, and k.  Out: a dict of the four numbers for this question.
def score(ranked_ids, relevant, k=5):
    relevant = set(relevant)
    top = ranked_ids[:k]
    hits = len(relevant & set(top))                 # & is set intersection: the labeled chunks that are in the top k
    precision = hits / k
    recall = hits / len(relevant)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    first = None
    for rank, cid in enumerate(top, start=1):
        if cid in relevant:
            first = rank
            break
    return {"precision": precision, "recall": recall, "f1": f1, "rr": 0.0 if first is None else 1.0 / first, "first": first}

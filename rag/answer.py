# rag/answer.py  (P17 rag and summarize, P18 cite)
# Run from the repo root:  uv run python -m rag.answer rag "<a question your corpus answers>"
#                          uv run python -m rag.answer summarize "<a topic>"
#                          uv run python -m rag.answer cite "<a question>"
# Chapter 10 on your corpus: retrieve (hybrid, then the cross-encoder), put the chunks in the prompt, generate.
# Three modes with the chapter's prompts, pointed at your corpus. The cite mode is the real pipeline: every
# chunk goes in with its id, the model must cite ids, every cited id is checked against what was retrieved,
# and a question whose best reranker score is below REFUSE_BELOW is refused before the model is called at all.
# The two lines you edit are CORPUS_NOTE (P17) and REFUSE_BELOW (P18 Step 4). Every answer is logged to data/answers.jsonl.
import json
import sys
import time

from rag import rerank
from rag.corpus import ROOT
from rag.llm import chat

CORPUS_NOTE = "Homer's Iliad and Odyssey in Samuel Butler's prose translation"   # what the corpus is, for the model
REFUSE_BELOW = -100.0        # P18 Step 4 sets this from your own scores. -100 means the gate is off
LOG = ROOT / "data" / "answers.jsonl"

RAG = """You are a RAG agent for a reading service that answers questions about {note}.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

SUMMARIZE = """Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information from each passage.

This should be tailored to readers of {note}.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

CITE = """Answer the query below using ONLY the passages provided.
Each passage starts with its id in square brackets, like [c00412].
The passages come from {note}.

Query: {query}

Passages:
{docs}

Instructions:
- End every sentence of your answer with the id of the passage it came from, like [c00412]
- Never cite an id that is not in the passages above, and never use outside knowledge
- If the passages do not contain the answer, reply with exactly one line: I don't have enough information
- Be direct and informative

Answer:"""

# In: the retrieved chunks and whether to show ids.  Out: the Documents block of the prompt, one chunk per block.
def context(hits, with_ids=True):
    lines = []
    for h in hits:
        text = " ".join(h["text"].split())
        lines.append(f"[{h['id']}] {text}" if with_ids else text)
    return "\n\n".join(lines)

# In: an answer.  Out: the ids it cites, found by scanning for "[c" ... "]" (no regular expressions needed).
def cited_ids(text):
    found = []
    start = text.find("[c")
    while start != -1:
        end = text.find("]", start)
        if end == -1:
            break
        cid = text[start + 1:end]
        if len(cid) == 6 and cid[1:].isdigit() and cid not in found:   # c followed by five digits, like c00412
            found.append(cid)
        start = text.find("[c", end)
    return sorted(found)

# In: a mode, a question, how many chunks.  Out: a record with the answer, what was retrieved, the citations and the gate.
def answer(mode, query, k=5, with_ids=True):
    hits = rerank.rerank(query, k)
    record = {"time": time.time(), "mode": mode, "question": query,
              "retrieved": [(h["id"], round(h["score"], 3)) for h in hits]}
    if mode == "cite" and (not hits or hits[0]["score"] < REFUSE_BELOW):
        record["gate"] = "refused before generation"
        record["answer"] = "I don't have enough information"
    else:
        record["gate"] = "passed" if mode == "cite" else "no gate"
        template = {"rag": RAG, "summarize": SUMMARIZE, "cite": CITE}[mode]
        prompt = template.format(note=CORPUS_NOTE, query=query, docs=context(hits, with_ids))
        record["answer"] = chat(prompt)
    cited = cited_ids(record["answer"])
    retrieved = {h["id"] for h in hits}
    record["cited"] = cited
    record["invalid"] = [c for c in cited if c not in retrieved]       # an id the model cited but was never given
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

if __name__ == "__main__":
    mode = sys.argv[1]
    query = " ".join(sys.argv[2:])
    r = answer(mode, query)
    print("Search Results:")
    for cid, score in r["retrieved"]:
        print(f"- {cid}  score {score}")
    print(f"\n{mode} answer:\n{r['answer']}")
    if mode == "cite":
        print(f"\ncited: {r['cited']}   invalid: {r['invalid']}   gate: {r['gate']}")

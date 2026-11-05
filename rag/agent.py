# rag/agent.py  (P19, Milestone A1)
# Run from the repo root:  uv run python -m rag.agent tool "<a question>"      (the model may call search_corpus)
#                          uv run python -m rag.agent none "<a question>"      (the same model, no tool: the baseline)
# A small agent: the model gets one tool, search_corpus, and decides for itself whether to call it. The loop is
# the one from Unit 0: send the conversation, if the reply is a tool call then run the tool and send the result
# back, otherwise the reply is the answer. In "none" mode the tool is not offered, so the model answers from
# whatever it remembers about your corpus, which is the baseline the milestone measures against.
# The one line you edit is CORPUS_NOTE. Every run is logged to data/agent.jsonl with the calls it made.
import json
import sys
import time

from rag import rerank
from rag.corpus import ROOT
from rag.llm import chat_messages

CORPUS_NOTE = "Homer's Iliad and Odyssey in Samuel Butler's prose translation"   # what the corpus is, for the model
MAX_ROUNDS = 4               # how many tool calls one question may use before the agent has to answer
LOG = ROOT / "data" / "agent.jsonl"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": f"Search {CORPUS_NOTE} and return the five passages that best match a query, each with its id. "
                       "Use it for any question about the people, places, events or wording of that text.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for, in a few words"}},
            "required": ["query"],
        },
    },
}]

SYSTEM_TOOL = f"""You answer questions about {CORPUS_NOTE}.
You have a tool, search_corpus, that returns passages from the text with their ids.
Use it before answering any question about the text, and answer from the passages it returns.
End every sentence of your answer with the id of the passage it came from, like [c00412].
If the passages do not contain the answer, reply with exactly one line: I don't have enough information"""

SYSTEM_NONE = f"""You answer questions about {CORPUS_NOTE} from what you know.
Answer directly in one or two sentences.
If you do not know, reply with exactly one line: I don't have enough information"""

# In: a query.  Out: the tool's result as one string: five passages with ids, the way the model will read them.
def search_corpus(query):
    hits = rerank.rerank(query, 5)
    return "\n\n".join(f"[{h['id']}] {' '.join(h['text'].split())}" for h in hits)

# In: a mode ("tool" or "none") and a question.  Out: a record with the answer, the tool calls made, and timings.
def run(mode, question):
    messages = [{"role": "system", "content": SYSTEM_TOOL if mode == "tool" else SYSTEM_NONE},
                {"role": "user", "content": question}]
    tools = TOOLS if mode == "tool" else None
    record = {"time": time.time(), "mode": mode, "question": question, "calls": [], "rounds": 0, "tool_ms": 0}
    for _ in range(MAX_ROUNDS + 1):
        reply = chat_messages(messages, tools)
        record["rounds"] += 1
        if not reply.get("tool_calls") or record["rounds"] > MAX_ROUNDS:
            record["answer"] = reply.get("content") or ""
            break
        # the model asked for the tool: put its request in the conversation, run the tool, put the result after it
        messages.append({"role": "assistant", "content": reply.get("content"),
                         "tool_calls": [{"id": t["id"], "type": "function",
                                         "function": {"name": t["name"], "arguments": t["arguments"]}} for t in reply["tool_calls"]]})
        for t in reply["tool_calls"]:
            try:
                query = json.loads(t["arguments"]).get("query", "")
            except json.JSONDecodeError:
                query = t["arguments"]
            started = time.time()
            result = search_corpus(query) if t["name"] == "search_corpus" else f"unknown tool {t['name']}"
            record["tool_ms"] += int((time.time() - started) * 1000)
            record["calls"].append({"tool": t["name"], "query": query, "ids": [line[1:7] for line in result.split("\n\n") if line.startswith("[c")]})
            messages.append({"role": "tool", "tool_call_id": t["id"], "content": result})
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

if __name__ == "__main__":
    mode = sys.argv[1]
    question = " ".join(sys.argv[2:])
    r = run(mode, question)
    for c in r["calls"]:
        print(f"tool call: {c['tool']}({c['query']!r}) -> {c['ids']}")
    print(f"rounds {r['rounds']}   tool ms {r['tool_ms']}\n")
    print(r["answer"])

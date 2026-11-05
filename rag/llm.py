# rag/llm.py  (P12)
# Run from the repo root:  uv run python -m rag.llm "Why is the sea wine-dark? One sentence."
# One place that talks to the model, so every later script calls the same function the same way.
# Chapter 7's setup: the OpenAI library pointed at OpenRouter, the key read from .env, the model id "openrouter/free".
# Two things the chapter does not have, both here because the free tier allows about 50 requests a day:
#   a cache (data/llm_cache.json): the same prompt sent twice costs one request, and a rerun of any script is free;
#   a counter: every script prints how many requests it really made, so you know what a day's work costs.
# chat(prompt) is what most scripts use. chat_messages(messages, tools) is the same thing for a whole conversation,
# which the agent in P19 needs.
import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag.corpus import ROOT

MODEL = "openrouter/free"            # the chapter's router: it picks a free model that can handle the request
CACHE = ROOT / "data" / "llm_cache.json"
PENDING = ROOT / "data" / "llm_pending.json"   # only used when LLM_CACHE_ONLY is set (see chat_messages)

_client = None
_cache = None
calls_made = 0                       # how many real requests this run has made

# In: nothing.  Out: the client, built once per run, with the key from .env (never from this file).
def client():
    global _client
    if _client is None:
        load_dotenv(ROOT / ".env")                                # reads OPENROUTER_API_KEY from the repo's .env file
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set: put it in .env at the repo root (P12 Step 2)")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client

# In: nothing.  Out: the cache as a dict, loaded from disk once per run (an empty dict if the file does not exist yet).
def cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    return _cache

def save_cache():
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache(), indent=1, ensure_ascii=False), encoding="utf-8")

# In: the request (model, messages, tools).  Out: a short fingerprint of it, the key the cache files it under.
def fingerprint(model, messages, tools=None):
    raw = json.dumps({"model": model, "messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

# In: a whole conversation (a list of role/content dicts) and optional tool definitions.
# Out: the assistant's message as a plain dict: {"content": str or None, "tool_calls": [...] or None}, plus token counts.
# The reply comes from the cache when this exact request was sent before; otherwise one request goes out and the reply is saved.
def chat_messages(messages, tools=None, model=MODEL, retries=3):
    global calls_made
    key = fingerprint(model, messages, tools)
    if key in cache():
        return cache()[key]["reply"]
    if os.environ.get("LLM_CACHE_ONLY"):                           # no network allowed: remember what was asked, answer with a marker
        pending = json.loads(PENDING.read_text(encoding="utf-8")) if PENDING.exists() else {}
        pending[key] = {"model": model, "messages": messages, "tools": tools}
        PENDING.write_text(json.dumps(pending, indent=1, ensure_ascii=False), encoding="utf-8")
        return {"content": "<<pending>>", "tool_calls": None}
    for attempt in range(retries):
        try:
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            response = client().chat.completions.create(**kwargs)
            calls_made += 1
            msg = response.choices[0].message
            reply = {"content": (msg.content or "").strip() or None,
                     "tool_calls": [{"id": t.id, "name": t.function.name, "arguments": t.function.arguments}
                                    for t in (msg.tool_calls or [])] or None,
                     "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                     "response_tokens": response.usage.completion_tokens if response.usage else None}
            if reply["content"] or reply["tool_calls"]:              # a free model sometimes returns nothing: try again
                cache()[key] = {"model": model, "messages": messages, "tools": tools, "reply": reply}
                save_cache()
                return reply
        except Exception as e:                                      # rate limit, a model that refused, a network blip
            print(f"   (attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
        time.sleep(3 * (attempt + 1))                               # the chapter's advice: wait, then retry
    raise RuntimeError("the model did not answer after several tries; wait a minute and run again")

# In: a prompt, as one user message.  Out: the model's reply as a string.
def chat(prompt, model=MODEL):
    reply = chat_messages([{"role": "user", "content": prompt}], model=model)
    return reply["content"] or ""

# In: a reply that should be a JSON list.  Out: the list, with any ```json fences or stray text around it removed.
def parse_json_list(reply):
    if reply == "<<pending>>":                                     # cache-only mode: nothing to parse yet
        return []
    start = reply.find("[")
    end = reply.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON list in the reply: {reply[:80]!r}")
    return json.loads(reply[start:end + 1])

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    messages = [{"role": "user", "content": prompt}]
    reply = chat_messages(messages)
    print(reply["content"])
    print(f"\nPrompt tokens: {reply.get('prompt_tokens')}")
    print(f"Response tokens: {reply.get('response_tokens')}")
    print(f"requests made this run: {calls_made}   (0 means the cache answered)")

# rag/enhance.py  (P12 spell, P13 rewrite and expand)
# Run from the repo root:  uv run python -m rag.enhance spell "the Ciclops and his sheap"
#                          uv run python -m rag.enhance rewrite "that bit where the guy ties himself to the mast"
#                          uv run python -m rag.enhance expand "the Cyclops"
# Chapter 7's three query enhancements, with the chapter's prompts, pointed at your corpus instead of movies.
# The model never sees the corpus here: it sees the query and one line describing what the corpus is, and the
# improved query goes to the same hybrid search as before. The one line you edit is CORPUS_NOTE.
import sys

from rag import search
from rag.llm import chat

CORPUS_NOTE = "Homer's Iliad and Odyssey in Samuel Butler's prose translation"   # one line: what your corpus is, for the model

SPELL = """Fix any spelling errors in the user-provided search query below.
The query will be searched against {note}.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

REWRITE = """Rewrite the user-provided search query below to be more specific and searchable.
The query will be searched against {note}.

Consider:
- The names, places and events that text is known for
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

EXPAND = """Expand the user-provided search query below with related terms.
The query will be searched against {note}.

Add synonyms and related concepts that might appear in that text.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

User query: "{query}"
"""

PROMPTS = {"spell": SPELL, "rewrite": REWRITE, "expand": EXPAND}

# In: a method name and a query.  Out: the enhanced query (for expand, the original with the new terms appended).
def enhance(method, query):
    prompt = PROMPTS[method].format(note=CORPUS_NOTE, query=query)
    reply = chat(prompt).strip().strip('"')                    # models like to quote the answer; the quotes are not part of the query
    if method == "expand":
        return f"{query} {reply}"
    return reply

if __name__ == "__main__":
    method = sys.argv[1]
    query = " ".join(sys.argv[2:])
    better = enhance(method, query)
    print(f"Enhanced query ({method}): '{query}' -> '{better}'\n")
    for rank, r in enumerate(search.hybrid(better, 5), start=1):
        print(f"{rank}. {r['id']}  {' '.join(r['text'].split())[:90]}...")

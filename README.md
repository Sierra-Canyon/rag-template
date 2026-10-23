# RAG — Honors Software Engineering, applied track

**Section C-JD · Period C · Room U 108 · 2026–27**

One repository per project. This is the third, and it carries the applied track from
P01 to the end of the semester: Boot.dev's *Learn Retrieval Augmented Generation*
chapter by chapter, and after each chapter the same thing built again on your corpus. The
foundations repo stays where it is; nothing from it moves here except the two files that
fetch your corpus.

Git hooks live in `.git/hooks/` and do not travel with a clone, so the secret guard you
installed in the foundations repo protects the foundations repo and nothing else. Every
project template ships its own `setup.sh` for exactly that reason.

---

## First run

```
./setup.sh
```

That installs Python 3.12, NumPy and NLTK, installs the secret guard, and runs the
environment check. The check has two lines the foundations repo did not have, `Go
toolchain` and `Boot.dev CLI`, and both have to say `OK` before P01 is done: every
Boot.dev lesson is submitted through that CLI. It ends by asking you to try committing a
fake key and watch it get refused. Do that.

Then P01 Step 4 brings your corpus over:

```
cp ../hse-2026-2027-foundations-<your-username>/scripts/fetch_corpus.sh scripts/
cp ../hse-2026-2027-foundations-<your-username>/data/SOURCE.md data/
bash scripts/fetch_corpus.sh
```

The sha256 in `SOURCE.md` has to match the file it fetched. Same corpus, byte for byte,
from September to June.

## Boot.dev's own project lives outside this repo

```
~/version_control/bootdev-rag/
```

The course builds its project on its dataset in that folder, and the CLI grades it there.
Nothing from it is committed here. What is committed here is the second version of each
chapter, written for your corpus, under `rag/`.

## Run everything with `uv run`, as a module

```
uv run python -m rag.corpus
```

`-m rag.corpus`, not `rag/corpus.py`. Running a file as a module is what lets
`rag/tfidf.py` say `from rag.corpus import load_passages`. Plain `python` uses whatever is
first on your PATH and will not find NumPy; the symptom is `ModuleNotFoundError` on a
package you just watched install.

## What is where

| Path | What |
|---|---|
| `rag/corpus.py` | **P02.** Your corpus as passages. Set `MIN_CHARS` to the minimum you settled on in A06. |
| `rag/preprocess.py` | **P02.** You type this one. The docstring states your order of operations. |
| `rag/tfidf.py` | **P04.** The index. The line marked `# CHAPTER FORMULA` takes the formula the chapter gives you. |
| `rag/` after that | One module per chapter: BM25, chunking, embeddings, hybrid search, reranking, the grounded answer. |
| `evidence/P<nn>.md` | One file per assignment with a marked slot for every paste, prediction and reflection answer. `scripts/check_evidence.py P01` tells you which slots are still empty. |
| `scratch/` | The chapter's formula copied character for character, your five sentences, your notes. Committed, not graded. |
| `data/` | Your corpus. Git-ignored except `README.md` and `SOURCE.md`. |
| `scripts/check_env.py` | The environment check. Re-run it whenever something breaks. |
| `scripts/install_secret_guard.sh` | Pre-commit hook that refuses staged keys. |

## Rules that apply in every repo you open this year

**Branches.** `dev/<feature>` for work. One pull request per assignment, **jd12** as the
reviewer, and the PR body says where the work is weakest. Your student log is on the
`<your-username>-track` branch from P01 to the end of the semester.

**Type every Boot.dev solution yourself**, and type the corpus version yourself too. The
second time goes quickly only if the first time went through your fingers.

**Never enter a personal credit card.** The seat is bought in class. Where the course
calls a model it uses free models through OpenRouter; a `429` means wait sixty seconds.

**AI is allowed, and you say so.** Every log entry carries an `AI use:` line. Using it
costs you nothing. Not saying so is the only version that is a problem.

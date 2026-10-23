# Your corpus goes here

`.gitignore` excludes everything in this directory except this file and `SOURCE.md`.
P01 copies `scripts/fetch_corpus.sh` and `data/SOURCE.md` from your foundations repo and
re-creates `data/corpus.txt` from them, byte for byte: the sha256 in `SOURCE.md` has to match.

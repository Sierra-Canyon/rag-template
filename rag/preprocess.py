# rag/preprocess.py  (P02)
"""Turn text into search terms.

Order: lowercase -> punctuation and symbols to spaces -> split on whitespace
-> drop one-letter tokens -> drop stopwords -> stem -> drop corpus stems.
"""
import unicodedata

from nltk.stem import PorterStemmer

STOPWORDS = frozenset("""
a about after all also an and any are as at be been but by can could did do does
for from had has have he her him his how i if in into is it its me my no not of on
or our out she so than that the their them then there these they this those to up
was we were what when where which who why will with would you your
""".split())

_stemmer = PorterStemmer()

# In: text.  Out: the same text with every punctuation mark and symbol replaced by a space.
def strip_punctuation(text):
    # unicodedata.category names what a character is: "P..." is punctuation, "S..." is a symbol. Curly quotes and em dashes count; an ASCII list would miss them.
    return "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in text)

# ---- P02 Step 4: type tokenize and preprocess below this line, then the two prints under __main__ ----

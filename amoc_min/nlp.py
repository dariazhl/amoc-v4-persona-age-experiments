"""spaCy layer: lemmas, content words, node typing, edge similarity.

v4 keys every node by its lemma list, not by surface string, and decides
CONCEPT vs PROPERTY from the part of speech. It also keeps every surface form
it has seen for a node and reports the most frequent one. All of that lives
here so the rest of amoc_min never imports spaCy directly.

Edge de-duplication in v4 used OpenAI ada-002 embeddings with cosine > 0.8.
We use spaCy vectors instead (en_core_web_lg carries them), which needs no
second model or API. If only en_core_web_sm is installed there are no vectors,
so we fall back to lemma overlap and say so once.
"""
import logging

import spacy

from . import config as C

_NLP = None
_HAS_VECTORS = False


def nlp():
    global _NLP, _HAS_VECTORS
    if _NLP is None:
        for name in (C.SPACY_MODEL, "en_core_web_md", "en_core_web_sm"):
            try:
                _NLP = spacy.load(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                "No spaCy English model. Install one: "
                "python -m spacy download en_core_web_lg")
        _HAS_VECTORS = _NLP.vocab.vectors_length > 0
        logging.info("spaCy %s loaded, vectors=%s",
                     _NLP.meta["name"], _HAS_VECTORS)
        if not _HAS_VECTORS:
            logging.warning(
                "spaCy model has no word vectors; edge de-duplication falls "
                "back to lemma overlap. Install en_core_web_lg to match v4.")
    return _NLP


def lemmas(text):
    """v4's get_concept_lemmas: the lemma of every token, as the node key."""
    return [t.lemma_.lower() for t in nlp()(str(text)) if not t.is_space]


def is_content_word(token):
    """v4's is_content_word_and_non_stopword: NOUN / PROPN / ADJ, not a stopword."""
    return (token.pos_ in C.CONTENT_POS
            and token.lemma_.lower() not in nlp().Defaults.stop_words)


def content_words(sentence):
    """Returns [(surface, lemma-key, node_type)] for one sentence."""
    out = []
    for t in nlp()(str(sentence)):
        if is_content_word(t):
            kind = "property" if t.pos_ == "ADJ" else "concept"
            out.append((t.text, [t.lemma_.lower()], kind))
    return out


def node_type_of(text):
    """v4's has_noun: a phrase containing a noun is a CONCEPT, else a PROPERTY."""
    return ("concept"
            if any(t.pos_ in ("NOUN", "PROPN") for t in nlp()(str(text)))
            else "property")


def similarity(a, b):
    """v4's Edge.is_similar, by spaCy vectors when available."""
    if a == b:
        return 1.0
    da, db = nlp()(str(a)), nlp()(str(b))
    if _HAS_VECTORS and da.vector_norm and db.vector_norm:
        return float(da.similarity(db))
    la = {t.lemma_.lower() for t in da if not t.is_punct}
    lb = {t.lemma_.lower() for t in db if not t.is_punct}
    return len(la & lb) / len(la | lb) if la | lb else 0.0

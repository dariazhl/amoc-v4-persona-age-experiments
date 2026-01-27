import logging
import re
from typing import List

try:
    from spacy.tokens import Span, Token
except Exception:
    Span = Token = None


def load_spacy():
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_lg")
        except Exception:
            nlp = spacy.load("en_core_web_sm")

        logging.info("Loaded spaCy model.")
        return nlp

    except Exception as e:
        logging.error(f"Could not load spaCy model: {e}")
        return None


def is_content_word_and_non_stopword(nlp, token: Token) -> bool:
    return (
        token.pos_ in {"NOUN", "PROPN", "ADJ"}
        and token.lemma_ not in nlp.Defaults.stop_words
    )


# to revert
def get_content_words_from_sent(nlp, sent: Span) -> List[Token]:
    content_words: list[Token] = []

    EXISTENTIAL_VERBS = {
        "appear",
        "emerge",
        "arrive",
        "materialize",
        "manifest",
        "surface",
        "show",
        "show_up",
        "occur",
    }

    for tok in sent:
        # # --- EXISTENTIAL SUBJECT OVERRIDE ---
        # if (
        #     tok.dep_ == "nsubj"
        #     and tok.head.pos_ == "VERB"
        #     and tok.head.lemma_ in EXISTENTIAL_VERBS
        #     and tok.pos_ in {"NOUN", "PROPN"}
        #     and tok.lemma_ not in nlp.Defaults.stop_words
        # ):
        #     content_words.append(tok)
        #     continue

        # --- ORIGINAL AMoC LOGIC ---
        if (
            tok.pos_ in {"NOUN", "PROPN", "ADJ"}
            and tok.dep_ not in {"det", "aux", "punct", "cc"}
            and tok.lemma_ not in nlp.Defaults.stop_words
        ):
            content_words.append(tok)

    return content_words


def get_verb_with_adverbs(verb: str) -> str:
    adverbs = [
        tkn.lemma_
        for tkn in verb.children
        if tkn.dep_ == "advmod" and tkn.pos_ == "ADV" and tkn.lemma_ not in {"not"}
    ]

    if adverbs:
        return f"{verb.lemma_}({' '.join(adverbs)})"
    return verb.lemma_


def get_concept_lemmas(nlp, concept: str) -> List[str]:
    doc = nlp(concept)
    return [token.lemma_ for token in doc]


def canonicalize_node_text(nlp, text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return text

    # Parse:
    # - underscores/hyphens used as "word separators"
    # - punctuation-wrapped tokens
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    if len(doc) == 0:
        return text

    root = doc[:].root
    alpha_tokens = [t for t in doc if getattr(t, "is_alpha", False)]
    if not alpha_tokens:
        return (root.lemma_ if getattr(root, "lemma_", None) else text).strip()

    noun_tokens = [t for t in alpha_tokens if t.pos_ in {"NOUN", "PROPN"}]
    if noun_tokens:
        chosen = root if root in noun_tokens else noun_tokens[-1]
    else:
        descriptive_tokens = [t for t in alpha_tokens if t.pos_ == "ADJ"]
        if root in descriptive_tokens:
            chosen = root
        else:
            chosen = descriptive_tokens[-1] if descriptive_tokens else alpha_tokens[0]

    # Preserve surface form for proper nouns; otherwise use lemma lowercased.
    if chosen.pos_ == "PROPN":
        return chosen.text.strip()
    return (chosen.lemma_ or chosen.text).strip().lower()


def has_noun(nlp, text: str) -> bool:
    doc = nlp(text)
    return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)

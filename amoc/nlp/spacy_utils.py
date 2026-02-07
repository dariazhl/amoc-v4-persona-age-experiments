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
    """
    Canonicalize node text per AMoC v4 paper specification:
    - Strip determiners (the, a, an) - NEVER part of node labels
    - Return single lowercase lemma as the canonical node key
    - Preserve surface form only for proper nouns

    Per paper Figures 2-4: Nodes are single lemmas like "country", not "the country"
    """
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

    # CRITICAL FIX: Filter out determiners (DET) first - per AMoC paper requirement
    # Node labels are NEVER "the country", always just "country"
    content_tokens = [
        t for t in doc
        if getattr(t, "is_alpha", False) and t.pos_ != "DET"
    ]

    if not content_tokens:
        # Fallback: if no content tokens, try to get any alpha token
        alpha_tokens = [t for t in doc if getattr(t, "is_alpha", False)]
        if not alpha_tokens:
            root = doc[:].root
            return (root.lemma_ if getattr(root, "lemma_", None) else text).strip().lower()
        content_tokens = alpha_tokens

    root = doc[:].root

    # Prefer nouns over adjectives for node label
    noun_tokens = [t for t in content_tokens if t.pos_ in {"NOUN", "PROPN"}]
    if noun_tokens:
        chosen = root if root in noun_tokens else noun_tokens[-1]
    else:
        # Adjectives become PROPERTY nodes
        adj_tokens = [t for t in content_tokens if t.pos_ == "ADJ"]
        if adj_tokens:
            chosen = root if root in adj_tokens else adj_tokens[-1]
        else:
            chosen = content_tokens[0]

    # Preserve surface form for proper nouns; otherwise use lemma lowercased.
    # Per AMoC paper: all node labels are lowercase lemmas
    if chosen.pos_ == "PROPN":
        return chosen.text.strip()
    return (chosen.lemma_ or chosen.text).strip().lower()


def has_noun(nlp, text: str) -> bool:
    doc = nlp(text)
    return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)


# ==========================================================================
# PAPER-FAITHFUL EXTRACTION: Deterministic linguistic extraction
# Per AMoC paper Figures 4-6: Adjectives and prepositional objects are
# extracted deterministically before LLM enrichment.
# ==========================================================================

def extract_adjectival_modifiers(sent: Span) -> List[dict]:
    """
    Extract adjectival modifiers from a sentence per AMoC paper.

    Detects:
    - amod: adjectival modifier (e.g., "young knight")
    - acomp: adjectival complement (e.g., "knight is brave")
    - attr: attribute (e.g., "knight was young")

    Returns list of dicts: {
        'adjective': str (lemma),
        'head_noun': str (lemma),
        'relation': str ('is')
    }

    Per paper: These become PROPERTY nodes attached via 'is' edge.
    """
    modifiers = []

    for token in sent:
        # amod: "young knight" -> young modifies knight
        if token.dep_ == "amod" and token.pos_ == "ADJ":
            head = token.head
            if head.pos_ in {"NOUN", "PROPN"}:
                modifiers.append({
                    'adjective': token.lemma_.lower(),
                    'head_noun': head.lemma_.lower(),
                    'relation': 'is',
                })

        # acomp: "knight is brave" -> brave is complement of is, knight is subject
        elif token.dep_ == "acomp" and token.pos_ == "ADJ":
            verb = token.head
            if verb.pos_ == "AUX" or verb.lemma_ in {"be", "become", "seem", "appear"}:
                # Find the subject of the verb
                for child in verb.children:
                    if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN"}:
                        modifiers.append({
                            'adjective': token.lemma_.lower(),
                            'head_noun': child.lemma_.lower(),
                            'relation': 'is',
                        })
                        break

        # attr: "knight was young" -> young is attribute, knight is subject
        elif token.dep_ == "attr" and token.pos_ == "ADJ":
            verb = token.head
            if verb.pos_ == "AUX" or verb.lemma_ in {"be", "become", "seem", "appear"}:
                for child in verb.children:
                    if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN"}:
                        modifiers.append({
                            'adjective': token.lemma_.lower(),
                            'head_noun': child.lemma_.lower(),
                            'relation': 'is',
                        })
                        break

    return modifiers


def extract_prepositional_objects(sent: Span) -> List[dict]:
    """
    Extract prepositional objects from a sentence per AMoC paper.

    Detects: prep → pobj patterns
    Examples:
    - "rode through the forest" -> (knight, ride_through, forest)
    - "unfamiliar with the country" -> (knight, unfamiliar_with, country)

    Returns list of dicts: {
        'subject': str (lemma of verb's subject),
        'head_word': str (lemma of verb or adjective),
        'preposition': str,
        'object': str (lemma of pobj),
        'label': str (head_preposition)
    }

    Per paper: Objects become CONCEPT nodes, edge label is head_preposition.
    """
    prep_objects = []

    for token in sent:
        # Look for prep dependency
        if token.dep_ == "prep":
            prep_text = token.lemma_.lower()
            head = token.head

            # Find the object of the preposition
            pobj = None
            for child in token.children:
                if child.dep_ == "pobj" and child.pos_ in {"NOUN", "PROPN"}:
                    pobj = child
                    break

            if pobj is None:
                continue

            subject = None
            head_word = None

            # Case 1: Head is a verb (e.g., "rode through the forest")
            if head.pos_ in {"VERB", "AUX"}:
                head_word = head.lemma_.lower()
                # Find the subject of the verb
                for child in head.children:
                    if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN"}:
                        subject = child
                        break
                # Also check if verb is part of a clause with a subject higher up
                if subject is None and head.head and head.head.pos_ in {"NOUN", "PROPN"}:
                    subject = head.head

            # Case 2: Head is an adjective (e.g., "unfamiliar with the country")
            elif head.pos_ == "ADJ":
                head_word = head.lemma_.lower()
                # Find the verb that the adjective is attached to
                adj_head = head.head
                if adj_head.pos_ in {"VERB", "AUX"}:
                    # Find the subject of that verb
                    for child in adj_head.children:
                        if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN"}:
                            subject = child
                            break

            # Case 3: Head is a noun (e.g., "journey to the castle")
            elif head.pos_ in {"NOUN", "PROPN"}:
                head_word = head.lemma_.lower()
                # The noun itself might be the subject, or find its governing verb
                # For now, use the noun as subject if it's a major entity
                subject = head

            if subject is None or head_word is None:
                continue

            # Create edge label: head_preposition
            edge_label = f"{head_word}_{prep_text}"

            prep_objects.append({
                'subject': subject.lemma_.lower(),
                'head_word': head_word,
                'preposition': prep_text,
                'object': pobj.lemma_.lower(),
                'label': edge_label,
            })

    return prep_objects

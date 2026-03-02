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
        if (
            tok.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
            and tok.dep_ not in {"det", "aux", "punct", "cc"}
            and tok.lemma_ not in nlp.Defaults.stop_words
        ):
            content_words.append(tok)

    return content_words


def get_verb_with_adverbs(verb_token: Token) -> str:
    if verb_token is None:
        return ""

    verb_lemma = verb_token.lemma_.lower()

    adverbs = [
        tkn.lemma_.lower()
        for tkn in verb_token.children
        if tkn.dep_ == "advmod"
        and tkn.pos_ == "ADV"
        and tkn.lemma_.lower() not in {"not", "n't"}
    ]

    if adverbs:
        # Format: "verb adverb" (e.g., "ride fast")
        return f"{verb_lemma} {' '.join(adverbs)}"
    return verb_lemma


EXCLUDED_ADVERBS = frozenset({"not", "n't", "never", "no"})

AUXILIARY_VERBS = frozenset(
    {
        "be",
        "is",
        "am",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    }
)

COPULA_VERBS = frozenset({"be", "is", "am", "are", "was", "were", "been", "being"})

MODAL_VERBS = frozenset(
    {
        "want",
        "wants",
        "wanted",
        "wanting",
        "try",
        "tries",
        "tried",
        "trying",
        "plan",
        "plans",
        "planned",
        "planning",
        "intend",
        "intends",
        "intended",
        "intending",
        "hope",
        "hopes",
        "hoped",
        "hoping",
        "wish",
        "wishes",
        "wished",
        "wishing",
        "need",
        "needs",
        "needed",
        "needing",
        "expect",
        "expects",
        "expected",
        "expecting",
        "decide",
        "decides",
        "decided",
        "deciding",
        "attempt",
        "attempts",
        "attempted",
        "attempting",
        "desire",
        "desires",
        "desired",
        "desiring",
        "prefer",
        "prefers",
        "preferred",
        "preferring",
    }
)

SEMANTIC_CLASSES = {
    "MOTION": frozenset(
        {
            "go",
            "goes",
            "ride",
            "rides",
            "walk",
            "walks",
            "run",
            "runs",
            "travel",
            "travels",
            "move",
            "moves",
            "come",
            "comes",
            "leave",
            "leaves",
            "enter",
            "enters",
            "exit",
            "exits",
            "pass",
            "passes",
            "cross",
            "crosses",
            "traverse",
            "traverses",
            "journey",
            "journeys",
            "wander",
            "wanders",
            "gallop",
            "gallops",
            "trot",
            "trots",
            "march",
            "marches",
        }
    ),
    "LOCATION": frozenset(
        {
            "is",
            "be",
            "stay",
            "stays",
            "remain",
            "remains",
            "sit",
            "sits",
            "stand",
            "stands",
            "live",
            "lives",
            "dwell",
            "dwells",
            "reside",
            "resides",
            "locate",
            "locates",
            "exist",
            "exists",
            "inhabit",
            "inhabits",
        }
    ),
    "CAPTURE": frozenset(
        {
            "kidnap",
            "kidnaps",
            "capture",
            "captures",
            "take",
            "takes",
            "seize",
            "seizes",
            "grab",
            "grabs",
            "hold",
            "holds",
            "imprison",
            "imprisons",
            "abduct",
            "abducts",
            "detain",
            "detains",
            "trap",
            "traps",
        }
    ),
    "COMBAT": frozenset(
        {
            "fight",
            "fights",
            "battle",
            "battles",
            "attack",
            "attacks",
            "strike",
            "strikes",
            "hit",
            "hits",
            "defeat",
            "defeats",
            "slay",
            "slays",
            "kill",
            "kills",
            "wound",
            "wounds",
            "combat",
            "combats",
        }
    ),
    "RESCUE": frozenset(
        {
            "save",
            "saves",
            "rescue",
            "rescues",
            "free",
            "frees",
            "liberate",
            "liberates",
            "release",
            "releases",
            "protect",
            "protects",
            "defend",
            "defends",
            "help",
            "helps",
            "aid",
            "aids",
        }
    ),
    "COMMUNICATION": frozenset(
        {
            "say",
            "says",
            "tell",
            "tells",
            "speak",
            "speaks",
            "ask",
            "asks",
            "answer",
            "answers",
            "call",
            "calls",
            "announce",
            "announces",
            "inform",
            "informs",
            "warn",
            "warns",
            "advise",
            "advises",
        }
    ),
    "POSSESSION": frozenset(
        {
            "have",
            "has",
            "own",
            "owns",
            "possess",
            "possesses",
            "keep",
            "keeps",
            "carry",
            "carries",
            "bear",
            "bears",
        }
    ),
    "PERCEPTION": frozenset(
        {
            "see",
            "sees",
            "know",
            "knows",
            "hear",
            "hears",
            "notice",
            "notices",
            "recognize",
            "recognizes",
            "understand",
            "understands",
            "realize",
            "realizes",
            "discover",
            "discovers",
            "find",
            "finds",
            "learn",
            "learns",
        }
    ),
}

_VERB_TO_CLASS = {}
for class_name, verbs in SEMANTIC_CLASSES.items():
    for verb in verbs:
        _VERB_TO_CLASS[verb] = class_name


def get_semantic_class(verb: str) -> str:
    verb_lower = verb.lower().strip()
    return _VERB_TO_CLASS.get(verb_lower, "OTHER")


def are_semantically_equivalent(label1: str, label2: str) -> bool:
    # Normalize both labels
    l1 = label1.lower().strip().replace("_", " ")
    l2 = label2.lower().strip().replace("_", " ")

    # String equality
    if l1 == l2:
        return True

    # Extract main verb
    v1 = l1.split()[0] if l1 else ""
    v2 = l2.split()[0] if l2 else ""

    # Same semantic class
    c1 = get_semantic_class(v1)
    c2 = get_semantic_class(v2)

    if c1 != "OTHER" and c1 == c2:
        return True

    return False


def _is_progressive_verb(token) -> bool:
    if token is None:
        return False

    # Check POS - must be VERB (not AUX)
    if token.pos_ != "VERB":
        return False

    # Check morphology for gerund/progressive
    morph = token.morph
    verb_form = morph.get("VerbForm")
    aspect = morph.get("Aspect")

    # VerbForm=Ger indicates gerund
    if verb_form and "Ger" in verb_form:
        return True

    # Aspect=Prog indicates progressive
    if aspect and "Prog" in aspect:
        return True

    # Fallback: check spaCy tag (VBG = verb, gerund or present participle)
    if token.tag_ == "VBG":
        return True

    # Final check: ends with -ing and is a verb
    if token.text.lower().endswith("ing") and len(token.text) > 3:
        return True

    return False


def _verb_to_present_tense(lemma: str) -> str:
    # Convert a verb lemma to simple present tense (3rd person singular).
    if not lemma:
        return lemma

    lemma = lemma.lower().strip()

    # Irregular verbs
    irregulars = {
        "be": "is",
        "have": "has",
        "do": "does",
        "go": "goes",
    }
    if lemma in irregulars:
        return irregulars[lemma]

    # Verbs ending in -s, -sh, -ch, -x, -z, -o: add -es
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return lemma + "es"

    # Verbs ending in consonant + y: change y to -ies
    if lemma.endswith("y") and len(lemma) > 1:
        prev_char = lemma[-2]
        if prev_char not in "aeiou":
            return lemma[:-1] + "ies"

    # Default: add -s
    return lemma + "s"


def canonicalize_edge_label(nlp, label: str) -> str:
    if not label or not isinstance(label, str):
        return ""

    label_normalized = label.strip().replace("_", " ")
    doc = nlp(label_normalized)
    if not doc:
        return ""

    for tok in doc:
        if tok.lemma_.lower() in MODAL_VERBS:
            return ""

    auxiliaries = []
    main_verb = None
    adjective = None
    preps = []

    for tok in doc:
        if tok.pos_ == "AUX":
            auxiliaries.append(tok)
        elif tok.pos_ == "VERB" and main_verb is None:
            main_verb = tok
        elif tok.pos_ == "ADJ":
            adjective = tok
        elif tok.pos_ == "ADP":
            preps.append(tok.lemma_.lower())

    if auxiliaries and adjective and main_verb is None:
        aux_lemmas = {aux.lemma_.lower() for aux in auxiliaries}
        if aux_lemmas & COPULA_VERBS:
            return ""

    if main_verb:
        base = _verb_to_present_tense(main_verb.lemma_.lower())
        parts = [base] + preps
        result = "_".join(parts)
        result_doc = nlp(result.replace("_", " "))
        if result_doc and result_doc[0].pos_ == "ADJ":
            return ""

        return result

    return ""


def extract_adverbs_from_sentence(sent: Span, verb_token: Token) -> List[str]:
    if verb_token is None:
        return []

    adverbs = []
    for tok in sent:
        # Check if this token is an adverb modifying our verb
        if (
            tok.pos_ == "ADV"
            and tok.dep_ == "advmod"
            and tok.head == verb_token
            and tok.lemma_.lower() not in EXCLUDED_ADVERBS
        ):
            adverbs.append(tok.lemma_.lower())

    return adverbs


def is_adverb_token(token: Token) -> bool:
    if token is None:
        return False
    return token.pos_ == "ADV"


def get_concept_lemmas(nlp, concept: str) -> List[str]:
    if not concept:
        return []
    doc = nlp(concept)
    return [token.lemma_ for token in doc if token.lemma_]


def canonicalize_node_text(nlp, text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return text

    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    if not doc:
        return text.lower()

    content_tokens = [
        t for t in doc if getattr(t, "is_alpha", False) and t.pos_ != "DET"
    ]

    if not content_tokens:
        content_tokens = [t for t in doc if getattr(t, "is_alpha", False)]
        if not content_tokens:
            return text.lower()

    root = doc[:].root

    noun_tokens = [t for t in content_tokens if t.pos_ in {"NOUN", "PROPN"}]
    if noun_tokens:
        chosen = root if root in noun_tokens else noun_tokens[-1]
    else:
        adj_tokens = [t for t in content_tokens if t.pos_ == "ADJ"]
        if adj_tokens:
            chosen = root if root in adj_tokens else adj_tokens[-1]
        else:
            chosen = content_tokens[0]

    if chosen.pos_ == "PROPN":
        return chosen.text.strip()

    return (chosen.lemma_ or chosen.text).strip().lower()


def has_noun(nlp, text: str) -> bool:
    doc = nlp(text)
    return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)


def extract_adjectival_modifiers(sent: Span) -> List[dict]:
    mods = []

    for tok in sent:
        if tok.pos_ == "ADJ":
            head = tok.head
            if head.pos_ in {"NOUN", "PROPN"}:
                mods.append(
                    {
                        "adjective": tok.lemma_.lower(),
                        "head_noun": head.lemma_.lower(),
                        "relation": "is",
                    }
                )

            elif tok.dep_ in {"acomp", "attr"}:
                for child in tok.head.children:
                    if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {
                        "NOUN",
                        "PROPN",
                    }:
                        mods.append(
                            {
                                "adjective": tok.lemma_.lower(),
                                "head_noun": child.lemma_.lower(),
                                "relation": "is",
                            }
                        )
                        break

    return mods


def extract_prepositional_objects(sent: Span) -> List[dict]:
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
                    if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {
                        "NOUN",
                        "PROPN",
                    }:
                        subject = child
                        break
                # Also check if verb is part of a clause with a subject higher up
                if (
                    subject is None
                    and head.head
                    and head.head.pos_ in {"NOUN", "PROPN"}
                ):
                    subject = head.head

            # Case 2: ADJ heads → SKIP ENTIRELY
            elif head.pos_ == "ADJ":
                continue

            # Case 3: Head is a noun (e.g., "journey to the castle")
            elif head.pos_ in {"NOUN", "PROPN"}:
                head_word = head.lemma_.lower()
                subject = head

            if subject is None or head_word is None:
                continue

            # Create edge label: head_preposition
            edge_label = f"{head_word}_{prep_text}"

            prep_objects.append(
                {
                    "subject": subject.lemma_.lower(),
                    "head_word": head_word,
                    "preposition": prep_text,
                    "object": pobj.lemma_.lower(),
                    "label": edge_label,
                }
            )

    return prep_objects

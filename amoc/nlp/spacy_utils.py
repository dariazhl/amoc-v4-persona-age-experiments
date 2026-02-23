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
            tok.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
            and tok.dep_ not in {"det", "aux", "punct", "cc"}
            and tok.lemma_ not in nlp.Defaults.stop_words
        ):
            content_words.append(tok)

    return content_words


def get_verb_with_adverbs(verb_token: Token) -> str:
    """
    Extract verb lemma with any adverb modifiers.

    Per AMoC paper: Adverbs modify actions, not the world.
    Adverbs are absorbed into edge labels, not represented as nodes.

    Args:
        verb_token: A spaCy Token representing a verb

    Returns:
        Canonical verb form with adverbs (e.g., "ride fast", "fight bravely")
    """
    if verb_token is None:
        return ""

    # Get the verb lemma (canonical form)
    verb_lemma = verb_token.lemma_.lower()

    # Collect adverb modifiers (excluding negation)
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


# ==========================================================================
# EDGE LABEL CANONICALIZATION: Per AMoC v4 paper
# Verbs name actions, not time. The graph remembers meaning, not surface form.
# Progressive aspect is linguistic noise and must not enter memory.
# ==========================================================================

# Negation words to exclude from adverb absorption
EXCLUDED_ADVERBS = frozenset({"not", "n't", "never", "no"})

# Auxiliaries to strip from edge labels (these don't carry semantic content)
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

# Copula verbs that can precede adjectives (not progressive constructions)
COPULA_VERBS = frozenset({"be", "is", "am", "are", "was", "were", "been", "being"})

# ==========================================================================
# MODAL VERBS: Must cause edge REJECTION per AMoC v4 paper
# ==========================================================================
# Modal/intentional verbs encode mental states, not world-state changes.
# Edges like "wants to free" or "tries to escape" must be rejected entirely.
# The graph stores WHAT happened, not intentions or attempts.
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

# ==========================================================================
# SEMANTIC CLASSES: For edge equivalence checking
# ==========================================================================
# Edges in the same semantic class between the same node pair are equivalent.
# Only one edge per semantic class is allowed between any ordered node pair.
SEMANTIC_CLASSES = {
    # MOTION: verbs describing movement
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
    # LOCATION: verbs describing position/presence
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
    # CAPTURE: verbs describing taking/holding
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
    # COMBAT: verbs describing fighting
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
    # RESCUE: verbs describing saving/freeing
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
    # COMMUNICATION: verbs describing speaking/telling
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
    # POSSESSION: verbs describing having/owning
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
    # PERCEPTION: verbs describing seeing/knowing
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

# Build reverse lookup: verb -> semantic class
_VERB_TO_CLASS = {}
for class_name, verbs in SEMANTIC_CLASSES.items():
    for verb in verbs:
        _VERB_TO_CLASS[verb] = class_name


def get_semantic_class(verb: str) -> str:
    """
    Get the semantic class for a verb.

    Returns the class name (e.g., "MOTION", "CAPTURE") or "OTHER" if not classified.
    """
    verb_lower = verb.lower().strip()
    return _VERB_TO_CLASS.get(verb_lower, "OTHER")


def are_semantically_equivalent(label1: str, label2: str) -> bool:
    """
    Check if two edge labels are semantically equivalent.

    Two labels are equivalent if:
    1. Their main verbs belong to the same semantic class, OR
    2. They are string-equal after normalization

    This is used to determine if a new edge should replace an existing one.
    """
    # Normalize both labels
    l1 = label1.lower().strip().replace("_", " ")
    l2 = label2.lower().strip().replace("_", " ")

    # String equality
    if l1 == l2:
        return True

    # Extract main verb (first word)
    v1 = l1.split()[0] if l1 else ""
    v2 = l2.split()[0] if l2 else ""

    # Same semantic class
    c1 = get_semantic_class(v1)
    c2 = get_semantic_class(v2)

    # Both in same non-OTHER class
    if c1 != "OTHER" and c1 == c2:
        return True

    return False


def _is_progressive_verb(token) -> bool:
    """
    Check if a token is a progressive/gerund verb form (-ing).

    Detects:
    - VerbForm=Ger (gerund)
    - Tense=Pres with Aspect=Prog (present progressive)
    - Tag VBG (verb, gerund or present participle)

    Returns True if the verb is in progressive form.
    """
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
    """
    Convert a verb lemma to simple present tense (3rd person singular).

    This implements standard English morphology rules:
    - Most verbs: add -s (walk → walks, run → runs)
    - Verbs ending in -s, -sh, -ch, -x, -z, -o: add -es (go → goes, watch → watches)
    - Verbs ending in consonant + y: change y to -ies (fly → flies)

    Per AMoC paper: Graph stores canonical semantic actions, not tense.
    We use present tense as the canonical form.
    """
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
    """
    AMoC v4 rule:
    - Relations must be VERB-headed
    - Copula + adjective constructions NEVER produce relations
    - Adjectives are handled as PROPERTY nodes elsewhere
    - Phrasal verbs are canonicalized by inflecting the verb head and preserving particles
      e.g., "fight for" → "fights_for", "ride through" → "rides_through"
    """
    if not label or not isinstance(label, str):
        return ""

    # -------------------------------------------------------------------------
    # PHRASAL VERB FIX: Normalize underscores to spaces for proper tokenization
    # Labels like "ride_through" must be parsed as "ride through" so spaCy can
    # identify the verb head and preposition/particle separately.
    # -------------------------------------------------------------------------
    label_normalized = label.strip().replace("_", " ")
    doc = nlp(label_normalized)
    if not doc:
        return ""

    # -------------------------------------------------------------------------
    # Reject modal / intentional verbs
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # ### FIX 1: Copula + adjective → REJECT
    # e.g. "is unfamiliar with" must NOT become a relation
    # -------------------------------------------------------------------------
    if auxiliaries and adjective and main_verb is None:
        aux_lemmas = {aux.lemma_.lower() for aux in auxiliaries}
        if aux_lemmas & COPULA_VERBS:
            return ""

    # -------------------------------------------------------------------------
    # Verb cases (allowed)
    # -------------------------------------------------------------------------
    if main_verb:
        base = _verb_to_present_tense(main_verb.lemma_.lower())
        parts = [base] + preps
        result = "_".join(parts)

        # -------------------------------------------------------------------------
        # DEFENSIVE ASSERTION: Reject if first token is an adjective
        # This catches any edge cases where adjective-headed labels slip through.
        # Per AMoC v4: edge labels MUST be verb-headed.
        # -------------------------------------------------------------------------
        result_doc = nlp(result.replace("_", " "))
        if result_doc and result_doc[0].pos_ == "ADJ":
            return ""

        return result

    # -------------------------------------------------------------------------
    # ### FIX 2: No adjective-only fallback
    # -------------------------------------------------------------------------
    return ""


def extract_adverbs_from_sentence(sent: Span, verb_token: Token) -> List[str]:
    """
    Extract adverbs that modify a specific verb in a sentence.

    Per AMoC paper: Adverbs modify actions and should be absorbed
    into edge labels, not represented as separate nodes.

    Args:
        sent: spaCy Span representing the sentence
        verb_token: The verb Token to find adverb modifiers for

    Returns:
        List of adverb lemmas that modify the verb
    """
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
    """
    Check if a token is an adverb.

    Per AMoC paper: Adverbs must never be displayed as nodes.
    This helper is used to filter out adverbs from node creation.
    """
    if token is None:
        return False
    return token.pos_ == "ADV"


def get_concept_lemmas(nlp, concept: str) -> List[str]:
    if not concept:
        return []
    doc = nlp(concept)
    return [token.lemma_ for token in doc if token.lemma_]


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
        t for t in doc if getattr(t, "is_alpha", False) and t.pos_ != "DET"
    ]

    if not content_tokens:
        # Fallback: if no content tokens, try to get any alpha token
        alpha_tokens = [t for t in doc if getattr(t, "is_alpha", False)]
        if not alpha_tokens:
            root = doc[:].root
            return (
                (root.lemma_ if getattr(root, "lemma_", None) else text).strip().lower()
            )
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
    Adjectives → PROPERTY nodes ONLY
    """
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
            # Per AMoC v4: adjectives become PROPERTY nodes, NOT edge labels.
            # "unfamiliar with the country" is handled as PROPERTY(knight, unfamiliar),
            # NOT as an edge label "unfamiliar_with".
            elif head.pos_ == "ADJ":
                continue  # Skip extraction for adjective heads

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

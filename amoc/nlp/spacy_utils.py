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
        if tkn.dep_ == "advmod" and tkn.pos_ == "ADV" and tkn.lemma_.lower() not in {"not", "n't"}
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
AUXILIARY_VERBS = frozenset({
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
})

# Copula verbs that can precede adjectives (not progressive constructions)
COPULA_VERBS = frozenset({"be", "is", "am", "are", "was", "were", "been", "being"})


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
    Canonicalize an edge label per AMoC v4 paper.

    ==========================================================================
    PRESENT-TENSE NORMALIZATION (Paper-Aligned)
    ==========================================================================
    Per AMoC Figures 2–6:
    - Relations are canonical semantic actions
    - Verb tense does not encode narrative time
    - Progressive aspect (-ing) is never represented in the graph
    - Memory stores WHAT happened, not HOW it was phrased

    Progressive constructions are converted to simple present tense:
    - "is walking" → "walks"
    - "was kidnapping" → "kidnaps"
    - "is fighting" → "fights"
    - "running through" → "runs through"

    Copula + adjective constructions are preserved:
    - "is unfamiliar with" → "is unfamiliar with" (this is NOT progressive)

    Also handles:
    - Absorbing adverbs into the label (adverbs modify actions, not the world)
    - Preserving prepositions for verb+prep constructions
    - Preserving phrasal verb particles
    ==========================================================================

    Args:
        nlp: spaCy language model
        label: Raw edge label string

    Returns:
        Canonicalized edge label in simple present tense
    """
    if not label or not isinstance(label, str):
        return label

    label = label.strip()
    if not label:
        return label

    doc = nlp(label)
    if len(doc) == 0:
        return label

    # ==========================================================================
    # PHASE 1: Analyze structure to detect progressive vs copula+adjective
    # ==========================================================================
    tokens = list(doc)
    auxiliaries = []
    main_verb = None
    adjective = None
    other_parts = []  # prepositions, particles, adverbs

    for tok in tokens:
        pos = tok.pos_

        if pos == "AUX":
            # Auxiliary verb (is, was, were, have, etc.)
            auxiliaries.append(tok)
        elif pos == "VERB":
            # Main verb - check if it's progressive
            if main_verb is None:
                main_verb = tok
        elif pos == "ADJ":
            # Adjective - might be copula+adjective construction
            adjective = tok
        elif pos == "ADP":
            # Preposition - keep for verb+prep constructions
            other_parts.append(("prep", tok.text.lower()))
        elif pos == "PART":
            # Particle (phrasal verb particles or infinitive "to")
            other_parts.append(("part", tok.text.lower()))
        elif pos == "ADV":
            # Adverb - absorb if not negation
            if tok.lemma_.lower() not in EXCLUDED_ADVERBS:
                other_parts.append(("adv", tok.text.lower()))

    # ==========================================================================
    # PHASE 2: Determine construction type and normalize
    # ==========================================================================

    # Case 1: Copula + Adjective (e.g., "is unfamiliar with")
    # This is NOT progressive - preserve the structure
    if auxiliaries and adjective and main_verb is None:
        # Check if auxiliary is a copula (be-verb)
        aux_lemmas = {aux.lemma_.lower() for aux in auxiliaries}
        if aux_lemmas & COPULA_VERBS:
            # This is copula + adjective - preserve "is" + adjective
            parts = ["is", adjective.text.lower()]
            # Add prepositions/particles that follow
            for part_type, part_text in other_parts:
                if part_type in ("prep", "part"):
                    parts.append(part_text)
            return " ".join(parts)

    # Case 2: Progressive construction (AUX + VBG like "is walking", "was fighting")
    if main_verb and _is_progressive_verb(main_verb):
        # Convert progressive to simple present tense
        verb_lemma = main_verb.lemma_.lower()
        present_tense = _verb_to_present_tense(verb_lemma)

        # Build result: present-tense verb + prepositions/particles
        parts = [present_tense]
        for part_type, part_text in other_parts:
            if part_type in ("prep", "part"):
                parts.append(part_text)
            elif part_type == "adv":
                # Include adverbs after verb
                parts.append(part_text)
        return " ".join(parts)

    # Case 3: Standalone gerund without auxiliary (e.g., "running through")
    if main_verb and main_verb.text.lower().endswith("ing"):
        # Convert to present tense even without auxiliary
        verb_lemma = main_verb.lemma_.lower()
        present_tense = _verb_to_present_tense(verb_lemma)

        parts = [present_tense]
        for part_type, part_text in other_parts:
            if part_type in ("prep", "part"):
                parts.append(part_text)
            elif part_type == "adv":
                parts.append(part_text)
        return " ".join(parts)

    # Case 4: Regular verb (not progressive) - convert to present tense for consistency
    if main_verb:
        verb_lemma = main_verb.lemma_.lower()
        # Check if already in present tense (simple form)
        if main_verb.tag_ in {"VBZ", "VBP"}:
            # Already present tense - keep as is
            verb_form = main_verb.text.lower()
        else:
            # Convert to present tense
            verb_form = _verb_to_present_tense(verb_lemma)

        parts = [verb_form]
        for part_type, part_text in other_parts:
            if part_type in ("prep", "part"):
                parts.append(part_text)
            elif part_type == "adv":
                parts.append(part_text)
        return " ".join(parts)

    # Case 5: No main verb found - return adjective if present, else original
    if adjective:
        parts = [adjective.text.lower()]
        for part_type, part_text in other_parts:
            if part_type in ("prep", "part"):
                parts.append(part_text)
        return " ".join(parts)

    # Fallback: return original label
    return label


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

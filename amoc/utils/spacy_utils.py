import logging
import re
from dataclasses import dataclass
from typing import List

try:
    from spacy.tokens import Span, Token
except Exception:
    Span = Token = None


COPULA_VERBS = frozenset({"be", "is", "am", "are", "was", "were", "been", "being"})


@dataclass(frozen=True)
class DeterministicRelationCandidate:
    subject_lemma: str
    relation_label: str
    object_lemma: str
    object_is_property: bool


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


def get_content_words_from_sent(nlp, sent: Span) -> List[Token]:
    content_words: list[Token] = []

    for tok in sent:
        if (
            tok.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
            and tok.dep_ not in {"det", "aux", "punct", "cc"}
            and tok.lemma_ not in nlp.Defaults.stop_words
        ):
            content_words.append(tok)

    return content_words


def verb_to_present_tense(lemma: str) -> str:
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
    # add ADV and PREP to verb ie. rode => rode through
    if main_verb:
        # base = verb_to_present_tense(main_verb.lemma_.lower())
        base = main_verb.lemma_.lower()
        parts = [base] + preps
        result = "_".join(parts)
        result_doc = nlp(result.replace("_", " "))
        if result_doc and result_doc[0].pos_ == "ADJ":
            return ""

        return result

    return ""


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


# trouble - "is" case
def extract_deterministic_relation_candidates(
    sent: Span,
) -> List[DeterministicRelationCandidate]:
    candidates: list[DeterministicRelationCandidate] = []
    print(f"DEBUG: Processing sentence: {sent.text}")  # or logging.info

    def _lemma(tok) -> str:
        return (tok.lemma_ or "").lower().strip()

    def _append(subj_lemma: str, rel: str, obj_lemma: str, obj_is_property: bool):
        if not subj_lemma or not rel or not obj_lemma:
            return
        candidates.append(
            DeterministicRelationCandidate(
                subject_lemma=subj_lemma,
                relation_label=rel,
                object_lemma=obj_lemma,
                object_is_property=obj_is_property,
            )
        )
        print(
            f"DEBUG: Appended candidate: ({subj_lemma}, {rel}, {obj_lemma}, property={obj_is_property})"
        )

    for token in sent:
        # Copular adjective / attribute -> subject -is-> property
        if token.dep_ in {"acomp", "attr"} and (
            token.pos_ == "ADJ" or (token.pos_ == "VERB" and token.tag_ == "VBN")
        ):
            if _lemma(token.head) != "be":
                continue
            subj = next(
                (c for c in token.head.children if c.dep_ in {"nsubj", "nsubjpass"}),
                None,
            )
            if subj is None:
                continue
            _append(_lemma(subj), "is", _lemma(token), True)

        # Adjectival modifier -> noun -is-> adjective
        if token.dep_ == "amod" and token.pos_ == "ADJ":
            _append(_lemma(token.head), "is", _lemma(token), True)

        # Verb SVO + prep objects
        if token.pos_ == "VERB" and _lemma(token) != "be":
            subj = next(
                (c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}),
                None,
            )
            if subj is None:
                continue
            subj_lemma = _lemma(subj)
            verb_lemma = _lemma(token)
            if not subj_lemma or not verb_lemma:
                continue

            for obj in (c for c in token.children if c.dep_ in {"dobj", "attr"}):
                _append(subj_lemma, verb_lemma, _lemma(obj), False)
                for conj in (c for c in obj.children if c.dep_ == "conj"):
                    _append(subj_lemma, verb_lemma, _lemma(conj), False)

            for prep in (c for c in token.children if c.dep_ == "prep"):
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                rel = f"{verb_lemma}_{_lemma(prep)}"
                _append(subj_lemma, rel, _lemma(pobj), False)
                for conj in (c for c in pobj.children if c.dep_ == "conj"):
                    _append(subj_lemma, rel, _lemma(conj), False)

        # ROOT copular prepositional phrase -> subject -is_prep-> pobj
        if token.dep_ == "ROOT" and _lemma(token) == "be":
            subj = next(
                (c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}),
                None,
            )
            if subj is None:
                continue
            subj_lemma = _lemma(subj)
            if not subj_lemma:
                continue
            for prep in (c for c in token.children if c.dep_ == "prep"):
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                rel = f"is_{_lemma(prep)}"
                _append(subj_lemma, rel, _lemma(pobj), False)
                for conj in (c for c in pobj.children if c.dep_ == "conj"):
                    _append(subj_lemma, rel, _lemma(conj), False)

    print(f"DEBUG: Total deterministic candidates: {len(candidates)}")
    if candidates:
        print("DEBUG: Candidates list:")
        for c in candidates:
            print(
                f"  {c.subject_lemma} --{c.relation_label}--> {c.object_lemma} (property={c.object_is_property})"
            )
    else:
        print("DEBUG: No deterministic candidates found.")
    return candidates

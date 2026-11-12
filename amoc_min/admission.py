"""Deterministic triplet admission, ported from amoc/admission/.

v5 rejects a triplet before it ever reaches the graph. The nov_11 run shows
why that matters: it admitted `knight has_property young`, `knight comes_from
castle`, `has`, and a spread of underscored labels (`rode_through`,
`was_unfamiliar_with`) that match nothing in the reference vocabulary.

Rules are v5's (amoc/admission/triplet_validator.py), minus the LLM pass:
empty, negation, vague, self-loop, too-short, adjective-subject, and
"relation must contain a verb". Normalisation collapses underscores so
`rode_through` and `rode through` are one label.
"""
import re

from . import nlp as N

COPULAS = frozenset({"is", "am", "are", "was", "were", "be", "been", "being"})

VAGUE = ("related to", "associated with", "connected to", "involved in",
         "concerned with", "pertaining to", "refers to", "regarding",
         "with respect to", "has", "have", "having", "belongs to", "part of")

# Rewritten, not rejected: these are adjectival relations the LLM spells in
# its own vocabulary. `knight has_property young` is a real relation badly
# named -- `is` is what both the reference and the deterministic parser use.
RELATION_REWRITE = {
    "has property": "is", "is property of": "is", "has attribute": "is",
    "has quality": "is", "has characteristic": "is", "has trait": "is",
    "is described as": "is", "is characterized by": "is",
    "has the property of": "is", "property of": "is",
}

NEGATION = ("not related", "no connection", "not available", "not applicable",
            "not connected", "no relation", "no link", "not involved",
            "not associated", "without connection", "without relation",
            "unconnected", "unrelated", "disconnected", "disassociated",
            "nonapplicable", "nonexistent", "unavailable", "uninvolved",
            "not", "no", "never", "neither", "nor", "without")

# v5 force-allows these: they are the reference's three inferred concepts.
FORCE_ALLOW = frozenset({"horse", "sword", "fire"})


def normalize(text):
    """v5's TextNormalizer: underscores to spaces, collapse whitespace, strip
    leading determiners so `a dragon` and `dragon` are the same node."""
    t = re.sub(r"[_\s]+", " ", str(text).strip().lower())
    t = re.sub(r"^(a|an|the)\s+", "", t)
    return t.strip()


def canonical_relation(rel):
    """Normalise, then rewrite the LLM's adjectival spellings onto `is`."""
    r = normalize(rel)
    return RELATION_REWRITE.get(r, r)


def _is_vague(rel):
    r = normalize(rel)
    return any(r == v or r.startswith(v + " ") for v in VAGUE)


def _is_negation(rel):
    r = normalize(rel)
    return any(r == n or r.startswith(n + " ") for n in NEGATION)


def _has_verb(rel):
    r = normalize(rel)
    if r in COPULAS:
        return True
    return any(t.pos_ in ("VERB", "AUX") for t in N.nlp()(r))


def _is_adjective_here(word, sentence):
    """Is this word an adjective IN THIS SENTENCE?

    Judging the bare word misfires: spaCy tags a lone `rescue` as ADJ, which
    rejected four good triplets in the nov_11 run. v5 runs this check with the
    sentence available; without one we decline to judge rather than guess.
    """
    if not sentence:
        return False
    w = normalize(word)
    for t in N.nlp()(str(sentence)):
        if t.lemma_.lower() == w or t.text.lower() == w:
            return t.pos_ == "ADJ"
    return False


def admit(subj, rel, obj, sentence=""):
    """Returns (ok, reason, triplet). Reason is "" when admitted; the triplet
    is normalised, with the relation rewritten onto `is` where applicable."""
    s, r, o = normalize(subj), canonical_relation(rel), normalize(obj)

    tri = [s, r, o]
    if not s or not r or not o:
        return False, "empty subject, relation or object", tri
    if len(r) < 2 or len(s) < 1 or len(o) < 1:
        return False, "too short", tri
    if s == o:
        # determiners are stripped first, so `a dragon` / `dragon` is caught
        return False, f"self-loop on '{s}'", tri
    if s in FORCE_ALLOW or o in FORCE_ALLOW:
        return True, "", tri                 # v5 force-allows horse/sword/fire
    if _is_negation(r):
        return False, f"negation relation '{r}' adds no semantic value", tri
    if _is_vague(r):
        return False, f"vague relation '{r}' -- be more specific", tri
    if not _has_verb(r):
        return False, f"relation '{r}' contains no verb", tri
    if r not in COPULAS and _is_adjective_here(s, sentence):
        return False, f"adjective '{s}' cannot be the subject of '{r}'", tri
    return True, "", tri


def admit_all(triplets, log=None, sentence="", existing=()):
    """Filter a list of [s, r, o], normalising as it goes.

    Connectivity beats admission. A triplet that fails the rules is rescued
    anyway when it is the only thing attaching a concept to the rest of the
    graph -- v5 tracks the same idea as `was_connectivity_critical`, and
    repairs it with FORCED_CONNECTIVITY_EDGE. Every one of the 7 rejections in
    the nov_11 run was connectivity-critical: `knight has armor` was dropped
    as vague and left `armor`/`scorched` permanently severed from the knight.

    `existing` is the edges already in the graph, as (s, r, o).
    Self-loops are never rescued: they connect nothing.
    """
    import networkx as nx

    kept, deferred = [], []
    for t in triplets:
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            continue
        ok, why, tri = admit(t[0], t[1], t[2], sentence)
        if ok:
            if tri not in kept:
                kept.append(tri)
        else:
            deferred.append((tri, why, list(t)))

    g = nx.Graph()
    g.add_edges_from((s, o) for s, _, o in existing)
    g.add_edges_from((s, o) for s, _, o in kept)

    for tri, why, orig in deferred:
        s, _, o = tri
        if s == o:                               # a self-loop joins nothing
            if log is not None:
                log.append({"triplet": orig, "reason": why})
            continue
        isolated = (s not in g or o not in g or not nx.has_path(g, s, o))
        if isolated and tri not in kept:
            kept.append(tri)
            g.add_edge(s, o)
            if log is not None:
                log.append({"triplet": orig, "reason": why,
                            "rescued": "connectivity-critical"})
        elif log is not None:
            log.append({"triplet": orig, "reason": why})
    return kept

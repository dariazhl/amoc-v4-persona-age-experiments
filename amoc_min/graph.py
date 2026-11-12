"""The reader's knowledge graph, and the activation that falls out of it.

Follows v4 (old_code.py) closely:

  * nodes are keyed by their spaCy LEMMA list, not by surface string, and keep
    every surface form seen, reporting the most frequent one;
  * nodes are CONCEPT or PROPERTY, decided by part of speech;
  * edges carry a forget_score that ticks down each sentence, and a new edge
    is dropped if a similar one already joins the same pair;
  * activation is BFS distance from the concepts the sentence states,
    travelling only over still-active edges, scored 5 - 0.5 * distance.

Concept decay is therefore EMERGENT from edge decay plus connectivity, and an
inferred node can raise a text node by shortening its distance.
"""
import heapq

from . import config as C
from . import nlp as N


RELAXED_MODEL_TAGS = ("qwen", "gemma", "phi")   # v5: uses_relaxed_inference


def _triplet_source(edge, model_name):
    src = edge.src.source == "inferred"
    dst = edge.dst.source == "inferred"
    relaxed = any(t in (model_name or "").lower() for t in RELAXED_MODEL_TAGS)
    inferred = (src or dst) if relaxed else (src and dst)
    return "INFERRED" if inferred else "TEXT_BASED"


class Node:
    def __init__(self, key, surface, node_type, source, source_sentence):
        self.key = key                      # tuple of lemmas -- the identity
        self.surfaces = {surface: 1}        # v4: actual_texts
        self.node_type = node_type          # "concept" | "property"
        self.source = source                # "text" | "inferred"
        self.source_sentence = source_sentence   # sentence it entered the graph
        self.last_stated = source_sentence
        self.edges = []

    @property
    def lemma(self):
        return " ".join(self.key)

    def add_surface(self, surface):
        self.surfaces[surface] = self.surfaces.get(surface, 0) + 1

    @property
    def text(self):
        """v4's get_text_representer: the most frequent surface form."""
        return max(self.surfaces, key=self.surfaces.get)

    def __repr__(self):
        return f"<{self.lemma}:{self.node_type}:{self.source}>"


class Edge:
    def __init__(self, src, dst, label, forget, source, source_sentence):
        self.src, self.dst, self.label = src, dst, label
        self.forget = forget                # v5 recorder: edge_visibility
        self.source = source                # "text" | "inferred"
        self.active = True
        self.source_sentence = source_sentence   # v5: created_at_sentence
        self.relevance = 2                  # v5 NARRATIVE_RELEVANCE: 0/1/2
        # per-sentence flags, for the edge_status state machine
        self.forget_at_start = forget
        self.asserted_this_sentence = True
        self.reactivated_this_sentence = False

    def begin_sentence(self):
        self.forget_at_start = self.forget
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = False

    def fade(self):                         # v4: fade_away
        self.forget -= 1
        if self.forget <= 0:
            self.active = False

    def revive(self, forget, reactivated=False):
        self.forget, self.active = forget, True
        if reactivated:
            self.reactivated_this_sentence = True
        else:
            self.asserted_this_sentence = True

    def status(self, sentence, edge_forget):
        """v5 recorder.capture_sentence_edges's edge_status."""
        if self.forget <= 0:
            return "inactive"
        if (self.source_sentence == sentence and self.asserted_this_sentence
                and self.forget == edge_forget):
            return "asserted"
        if self.asserted_this_sentence:
            return "explicit"
        if self.reactivated_this_sentence:
            return "reactivated"
        if self.forget == 1:
            return "decaying"
        return "inactive"

    def other(self, node):
        return self.dst if self.src is node else self.src

    def __repr__(self):
        return f"{self.src.text} -{self.label}-> {self.dst.text}"


class Graph:
    def __init__(self, edge_forget=None):
        self._last_stated = {}
        self.nodes = {}                     # key tuple -> Node
        self.edges = []
        self.edge_forget = C.EDGE_FORGET if edge_forget is None else edge_forget

    # -- construction ----------------------------------------------------
    def node(self, surface, source, sentence, node_type=None):
        """v4's add_or_get_node, keyed on lemmas."""
        key = tuple(N.lemmas(surface))
        if not key:
            return None
        n = self.nodes.get(key)
        if n is None:
            n = self.nodes[key] = Node(
                key, surface, node_type or N.node_type_of(surface),
                source, sentence)
        else:
            n.add_surface(surface)
            if source == "text" and n.source == "inferred":
                n.source = "text"           # the text caught up with the guess
        return n

    def _similar_exists(self, s, o, label):
        """v4's check_if_similar_edge_exists: revive rather than duplicate."""
        for e in self.edges:
            if {e.src, e.dst} != {s, o}:
                continue
            if e.label == label:
                e.revive(self.edge_forget)
                return e
            # a concept-property pair carries one edge, whatever it is called
            if {e.src.node_type, e.dst.node_type} == {"concept", "property"}:
                e.revive(self.edge_forget)
                return e
            if N.similarity(e.label, label) > C.SIMILARITY:
                e.revive(self.edge_forget)
                return e
        return None

    def add(self, subj, rel, obj, source, sentence):
        s = self.node(subj, source, sentence)
        o = self.node(obj, source, sentence)
        if s is None or o is None or s is o:
            return None
        existing = self._similar_exists(s, o, rel)
        if existing is not None:
            return existing
        e = Edge(s, o, rel, self.edge_forget, source, sentence)
        self.edges.append(e)
        s.edges.append(e)
        o.edges.append(e)
        return e

    def begin_sentence(self):
        for e in self.edges:
            e.begin_sentence()

    def fade_all(self, keep=()):
        """Tick decay. Edges in `keep` are exempt (just added this sentence)."""
        for e in self.edges:
            if e.active and e not in keep:
                e.fade()

    # -- activation ------------------------------------------------------
    def distances(self, stated_keys):
        """Weighted shortest path from the concepts this sentence states.

        An edge's cost is set by the relevance the model gave it: a
        CONTINUOUS edge (2) costs one hop, a FADING edge (1) costs two, and a
        DISCONTINUOUS edge (0) is not traversable. Combined with
        score = 5 - 0.5 * distance this yields the reference's own value set
        (5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1) instead of the three values a
        uniform cost can reach.
        """
        seeds = [self.nodes[k] for k in stated_keys if k in self.nodes]
        dist, pq = {}, [(0, id(n), n) for n in seeds]
        heapq.heapify(pq)
        while pq:
            d, _, n = heapq.heappop(pq)
            if n in dist:
                continue
            dist[n] = d
            for e in n.edges:
                if not e.active or e.relevance <= 0:
                    continue
                cost = C.COST_CONTINUOUS if e.relevance >= 2 else C.COST_FADING
                nxt = e.other(n)
                if nxt not in dist:
                    heapq.heappush(pq, (d + cost, id(nxt), nxt))
        return dist

    def scores(self, stated_keys):
        """v4: node.score = BFS distance, 100 when unreachable."""
        dist = self.distances(stated_keys)
        return {n: dist.get(n, 100) for n in self.nodes.values()}

    def activation(self, stated_keys, sentence, history, sentence_lemmas=()):
        """score = 5 - 0.5 * distance, full reset at onset, unreachable -> 0."""
        for k in stated_keys:
            if k in self.nodes:
                self.nodes[k].last_stated = sentence
            self._last_stated[" ".join(k)] = sentence

        # A relation's age is reset by its OWN appearance in the text, not by
        # its endpoints'. Keying off endpoints made every hub-attached
        # relation immortal: `knight` recurs in almost every sentence, so
        # `rode` never aged and sat at 5 for eleven sentences where the
        # reference has it dead by S4.
        lemmas = set(sentence_lemmas)
        for e in self.edges:
            key = N.relation_key(e.label)
            if C.RELATION_AGEING == "relevance":
                if e.relevance >= 2:         # option C -- not the default yet
                    self._last_stated[key] = sentence
            elif key in lemmas:              # option B (default)
                self._last_stated[key] = sentence
        dist = self.distances(stated_keys)
        out = {}

        stated_now = {" ".join(k) for k in stated_keys}
        last = self._last_stated

        def put(token, d, prev, source_sentence=None):
            # A token that has never been stated ages from when it ENTERED the
            # graph. Defaulting to the current sentence gave age 0 forever, so
            # every inferred concept (horse, fire, sword) stayed at full
            # activation for the whole story.
            since = last.get(
                token,
                source_sentence if source_sentence is not None else sentence)
            if d is None or d > C.MATRIX_MAX_DISTANCE:
                return
            age = 0 if token in stated_now else max(0, sentence - since)
            score = max(0.0, 5.0 - C.SCORE_PER_HOP * d
                        - C.SENIORITY_WEIGHT * age)
            if score > 0 and prev == 0.0:
                score = 5.0
            if score > 0:
                out[token] = max(out.get(token, 0.0), round(min(score, 5.0), 2))

        for n in self.nodes.values():
            put(n.lemma, dist.get(n), history.get(n.lemma, [0.0])[-1],
                source_sentence=n.source_sentence)

        # Relation labels are matrix rows too: 7 of the 20 reference tokens
        # (rode, appeared, kidnapping, fought, killed, freed, married) are
        # edge labels. A label sits at the distance of its nearer endpoint.
        for e in self.edges:
            if not e.active:
                continue
            ds = [dist[x] for x in (e.src, e.dst) if x in dist]
            key = N.relation_key(e.label)
            put(key, min(ds) if ds else None, history.get(key, [0.0])[-1],
                source_sentence=e.source_sentence)
        return out

    # -- prompt context (v4 feeds these to the LLM) ----------------------
    def active_nodes(self, threshold=None, only_text=False):
        """v4's get_active_nodes: score <= threshold."""
        t = C.PROMPT_MAX_DISTANCE if threshold is None else threshold
        sc = self._last_scores or {}
        return [n for n, s in sorted(sc.items(), key=lambda kv: kv[1])
                if s <= t and (not only_text or n.source == "text")]

    _last_scores = None
    _last_stated = None

    def refresh_scores(self, stated_keys):
        self._last_scores = self.scores(stated_keys)
        return self._last_scores

    def nodes_str(self, nodes):
        """v4's get_nodes_str: '- knight (type: concept) (score: 0)'."""
        sc = self._last_scores or {}
        return "".join(
            f"- {n.text} (type: {n.node_type}) (score: {sc.get(n, 100)})\n"
            for n in nodes) or "(none)\n"

    def edges_str(self, nodes, only_text=False, only_active=True):
        """v4's get_edges_str: numbered 'a - label (edge) - b'."""
        seen, lines, order = set(), [], []
        for n in nodes:
            for e in n.edges:
                if e in seen or (only_active and not e.active):
                    continue
                if only_text and not (e.src.source == "text"
                                      and e.dst.source == "text"):
                    continue
                seen.add(e)
                order.append(e)
                lines.append(f"{len(lines) + 1}. {e.src.text} - {e.label} "
                             f"(edge) - {e.dst.text}")
        return ("\n".join(lines) or "(none)"), order

    # -- logging / viewing -----------------------------------------------
    def records(self, sentence, sentence_text, meta=None,
                decay_explanations=None, context_shift=False):
        """The triplet log, matching amoc/output/recorder.py's EdgeRecord.

        Every edge is recorded, not only the active ones -- an inactive edge
        is a row with edge_status "inactive", so the decay trail survives.
        Field names and enum spellings match the v5 recorder so the existing
        analysis scripts read these CSVs unchanged.
        """
        meta = meta or {}
        expl = decay_explanations or {}
        SRC = {"text": "TEXT_BASED", "inferred": "INFERENCE_BASED"}
        rows = []
        for e in self.edges:
            if not str(e.label).strip() or e.src is e.dst:
                continue
            triplet = (e.src.text, e.label, e.dst.text)
            status = e.status(sentence, self.edge_forget)
            rows.append({
                "original_index": meta.get("original_index", -1),
                "age_refined": meta.get("age_refined", -1),
                "regime": meta.get("regime", ""),
                "persona_text": meta.get("persona_text", ""),
                "model_name": meta.get("model_name", ""),
                "sentence_idx": sentence,
                "sentence_text": sentence_text,
                "source": e.src.text,
                "verb": e.label,
                "object": e.dst.text,
                "source_provenance": SRC[e.src.source],
                "object_provenance": SRC[e.dst.source],
                # v5 recorder's rule: strict models need BOTH endpoints
                # inferred; relaxed models (qwen/gemma/phi) need either.
                # NOTE this is why triplet_source was 100% TEXT_BASED in the
                # jul_28 data -- under the strict AND rule an edge is almost
                # never INFERRED. Use edge_provenance below to measure the
                # inference channel; triplet_source is kept for compatibility
                # with the existing analysis scripts.
                "triplet_source": _triplet_source(
                    e, meta.get("model_name", "")),
                "edge_provenance": ("INFERRED" if e.source == "inferred"
                                    else "TEXT_BASED"),
                "edge_visibility": e.forget,
                "edge_status": status,
                "decay_explanation": (expl.get(triplet, "")
                                      if status in ("decaying", "inactive",
                                                    "carryover") else ""),
                "context_shift": context_shift,
                # amoc_min additions, beyond the v5 recorder
                "source_lemma": e.src.lemma,
                "object_lemma": e.dst.lemma,
                "source_type": e.src.node_type,
                "object_type": e.dst.node_type,
                "source_sentence": e.source_sentence,
            })
        return rows

    def snapshot(self):
        return {
            "nodes": [{"lemma": n.lemma, "text": n.text, "type": n.node_type,
                       "source": n.source,
                       "source_sentence": n.source_sentence}
                      for n in self.nodes.values()],
            "edges": [{"from": e.src.lemma, "to": e.dst.lemma,
                       "label": e.label, "source": e.source,
                       "active": e.active, "forget": e.forget,
                       "source_sentence": e.source_sentence}
                      for e in self.edges],
        }

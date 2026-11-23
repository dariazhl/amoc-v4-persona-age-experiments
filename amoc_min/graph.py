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
        self.strength = 5                   # --llm-score: the model's 0-5 rating
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

    def rate(self, score, edge_forget, asserted=False):
        """--llm-score: the model's 0-5 rating is this edge's activation.

        `forget` stays on its 0/1/2 scale so status() and the triplet log
        read the same in every mode: 0 is inactive, 1-2 is decaying, and 3+
        counts as held in mind (reported as reactivated).
        """
        self.strength = score
        self.active = score > 0
        self.forget = 0 if score == 0 else (1 if score <= 2 else edge_forget)
        if asserted:
            self.asserted_this_sentence = True
        else:
            self.reactivated_this_sentence = score >= 3

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
    def __init__(self, edge_forget=None, capacity=1.0):
        self._last_stated = {}
        self.nodes = {}                     # key tuple -> Node
        self.edges = []
        self.edge_forget = C.EDGE_FORGET if edge_forget is None else edge_forget
        # Working-memory capacity of THIS reader (C.capacity_for). 1.0 is the
        # default model exactly; below 1 the reader carries fewer concepts
        # into the next sentence and keeps less of their activation.
        self.capacity = float(capacity)
        self.max_carryover = (max(1, round(C.MAX_CARRYOVER_NODES * self.capacity))
                              if C.MAX_CARRYOVER_NODES else 0)
        self.carryover_decay = C.CARRYOVER_DECAY * self.capacity
        self.llm_carryover_cap = (max(1, round(C.LLM_CARRYOVER_CAP * self.capacity))
                                  if C.LLM_CARRYOVER_CAP else 0)

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
        # Decide the edge is possible BEFORE creating either node. Creating
        # them first and then bailing on `s is o` left the node behind with no
        # edge -- a self-loop silently minted an orphan, which is the one way
        # an unconnected node could still enter the graph.
        ks, ko = tuple(N.lemmas(subj)), tuple(N.lemmas(obj))
        if not ks or not ko or ks == ko:
            return None
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

    # -- connectivity ----------------------------------------------------
    def components(self):
        """Connected components over EVERY edge, active or not.

        This is the figure's own definition: plot.draw() renders a decayed
        edge in pale grey, so a node hanging off a grey line is still joined
        to the graph on screen. Activation does NOT use this -- distances()
        walks active edges only, so a grey line keeps a concept visible
        without feeding it score.

        The invariant is one component. A cluster of two or more nodes off
        on its own is always a bug upstream. A SINGLE stated node may briefly
        stand alone, when the sentence states a concept and the model gives it
        no relationship -- "Suddenly, a dragon appeared.": one content word,
        an intransitive verb. Tolerated, not wanted; carry-over plus the
        hub-first prompt exist to make even that case attach.

        It is tolerated rather than deleted because `dragon` is S3's only
        stated word: dropping nodes that have no edge does make every sentence
        connected, but it leaves activation nothing to seed from and takes the
        whole S3 column to zero (rho .421 -> .000, mean .822 -> .790).

        Nothing repairs the graph after the fact. Connectivity is a property
        of what is admitted (Graph.add, admission.admit_all, the attachment
        rule in engine._attaching) and of what the model can see when it is
        asked (Graph.active_nodes' carry-over), never of a repair pass.
        """
        seen, out = set(), []
        for n in self.nodes.values():
            if n in seen:
                continue
            stack, comp = [n], []
            seen.add(n)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for e in cur.edges:
                    nxt = e.other(cur)
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            out.append(comp)
        return sorted(out, key=lambda c: (-len(c), min(x.lemma for x in c)))

    def begin_sentence(self):
        for e in self.edges:
            e.begin_sentence()

    def can_deactivate(self, edge):
        """False when switching this edge off would split the active graph.

        v5's can_remove_edge()/has_alternative_path(). An edge is safe to
        drop only if both endpoints keep an active edge AND they remain
        reachable from one another without it.
        """
        adj = {}
        for e in self.edges:
            if e.active and e is not edge:
                adj.setdefault(e.src, set()).add(e.dst)
                adj.setdefault(e.dst, set()).add(e.src)
        a, b = edge.src, edge.dst
        if a not in adj or b not in adj:
            return False                 # would strand an endpoint
        seen, stack = {a}, [a]
        while stack:
            for y in adj[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return b in seen

    def _protect(self, e):
        """True when this edge must stay active to keep the graph in one piece.

        LLM-RULE ONLY. The text rule is left exactly as it was: its active
        graph already holds together (0 of 533 reader-sentences split), so
        it needs no guard, and adding one would change which edges it drops
        and therefore its matrices. The two ageing rules stay comparable
        only if every brake here is confined to the LLM one.
        """
        return (C.RELATION_AGEING == C.AGEING_LLM
                and C.PROTECT_BRIDGES and e.active and e.forget <= 1
                and not self.can_deactivate(e))

    def fade_all(self, keep=()):
        """Tick decay. Edges in `keep` are exempt (just added this sentence)."""
        for e in self.edges:
            if e.active and e not in keep and not self._protect(e):
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
        llm_rule = C.RELATION_AGEING == C.AGEING_LLM
        def _touch(token):
            """A relevance-2 judgement under the LLM rule.

            LLM_AGE_RETAIN=0 zeroes the age (v5's behaviour, and amoc_min's
            before this): the concept is as fresh as when first read, so it
            can never decay while the model keeps saying yes. Any value above
            0 keeps a fraction of the age, so sustained relevance SLOWS decay
            instead of stopping it and the live set can fall again.
            """
            prev = self._last_stated.get(token, sentence)
            age = max(0, sentence - prev)
            self._last_stated[token] = sentence - int(age * C.LLM_AGE_RETAIN)

        for e in self.edges:
            key = N.relation_key(e.label)
            if llm_rule and e.active and e.relevance >= 2:
                # The model still calls this relation CONTINUOUS: it may slow
                # ageing (LLM_AGE_RETAIN < 1), never stop it.
                _touch(key)
            if key in lemmas:
                # Its own word in the text resets its age under BOTH rules, as
                # amoc-v5 resets seniority on an explicit re-mention. This used
                # to be text-rule only; with LLM_AGE_RETAIN = 1.0 that left a
                # restated relation ageing forever under --llm-decay.
                self._last_stated[key] = sentence

        if llm_rule:
            # ...and so are the concepts it joins. Without this the flag moved
            # only the relation-label rows -- 7 of the reference's 20 -- while
            # every CONCEPT row went on ageing from surface mentions, so the
            # score did not reflect the rule it was named for. A concept is
            # held in mind by the model judging one of its relations still
            # current, not by the text repeating its word.
            for e in self.edges:
                if not (e.active and e.relevance >= 2):
                    continue
                for n in (e.src, e.dst):
                    _touch(n.lemma)
                    n.last_stated = self._last_stated[n.lemma]
        dist = self.distances(stated_keys)
        out = {}

        stated_now = {" ".join(k) for k in stated_keys}
        last = self._last_stated

        def put(token, d, prev, source_sentence=None, inferred=False):
            # A token that has never been stated ages from when it ENTERED the
            # graph. Defaulting to the current sentence gave age 0 forever, so
            # every inferred concept (horse, fire, sword) stayed at full
            # activation for the whole story.
            since = last.get(
                token,
                source_sentence if source_sentence is not None else sentence)
            age = 0 if token in stated_now else max(0, sentence - since)
            # What this sentence activates on its own: 0 when out of reach.
            if d is None or d > C.MATRIX_MAX_DISTANCE:
                fresh = 0.0
            else:
                fresh = max(0.0, 5.0 - C.SCORE_PER_HOP * d
                            - C.SENIORITY_WEIGHT * age)
                if fresh > 0 and prev == 0.0:
                    fresh = 5.0
                # A concept the reader INFERRED never enters as strongly as
                # one the text states. The reference puts horse at 2, sword
                # at 2, fire at 3, against 5 for a stated word. Without this
                # cap an inferred concept entered at 5.0 and took three
                # sentences to decay away, which is most of why the matrix
                # carried 40 live tokens where the reference has 8.
                if inferred and token not in stated_now:
                    fresh = min(fresh, C.INFERRED_ACTIVATION)
            # What the reader still holds from last sentence. This is what
            # keeps `knight` alive through "Suddenly, a dragon appeared."
            carry = prev * self.carryover_decay
            if carry < C.CARRYOVER_FLOOR:
                carry = 0.0
            score = max(fresh, carry)
            if score < C.ACTIVATION_FLOOR:
                score = 0.0
            if score > 0:
                out[token] = max(out.get(token, 0.0), round(min(score, 5.0), 2))

        for n in self.nodes.values():
            put(n.lemma, dist.get(n), history.get(n.lemma, [0.0])[-1],
                source_sentence=n.source_sentence,
                inferred=(n.source == "inferred"))

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

        # v5's MAX_CARRYOVER_NODES. Without a ceiling the relevance rule has
        # no equilibrium -- every sentence can add to the live set and none
        # is obliged to remove from it. Stated concepts are never capped;
        # the rest compete on activation minus seniority, as v5 ranks them.
        if llm_rule and self.llm_carryover_cap:
            carried = [k for k in out if k not in stated_now]
            if len(carried) > self.llm_carryover_cap:
                rank = sorted(
                    carried,
                    key=lambda k: out[k] - C.SENIORITY_WEIGHT *
                    max(0, sentence - last.get(k, sentence)),
                    reverse=True)
                for k in rank[self.llm_carryover_cap:]:
                    del out[k]
        return out

    def rated_activation(self, stated_keys):
        """--llm-score: a matrix column read straight off the ratings.

        A relationship's rating is the activation of its own row and of the
        two concepts it joins; a concept takes its strongest relationship,
        and whatever the sentence states is 5. Nothing else enters, so a
        concept that is not stated is active only through a relationship the
        model still rates above 0 -- it cannot be lit with no live edge.
        """
        out = {}
        for e in self.edges:
            if e.strength <= 0 or not str(e.label).strip():
                continue
            for k in (N.relation_key(e.label), e.src.lemma, e.dst.lemma):
                out[k] = max(out.get(k, 0.0), float(e.strength))
        for k in stated_keys:
            if k in self.nodes:
                out[self.nodes[k].lemma] = 5.0
        return out

    # -- prompt context (v4 feeds these to the LLM) ----------------------
    def active_nodes(self, threshold=None, only_text=False):
        """v4's get_active_nodes (score <= threshold), plus v5's carry-over.

        The score is BFS distance from what THIS sentence states, so a
        concept the sentence does not reach scores 100 and drops out. For an
        unattached node that is a trap: everything else is unreachable from
        it, so the prompt shows it alone with no edges and the model has
        nothing to attach it to -- which is why `dragon` could never be
        joined to `knight` at S3, whatever the prompt said.

        Carry-over tops the list up with the most recently stated nodes
        regardless of reach, so the story's backbone stays on the page.
        """
        t = C.PROMPT_MAX_DISTANCE if threshold is None else threshold
        sc = self._last_scores or {}
        near = [n for n, s in sorted(sc.items(), key=lambda kv: kv[1])
                if s <= t and (not only_text or n.source == "text")]
        if not self.max_carryover:
            return near
        seen = set(near)
        pool = [n for n in self.nodes.values()
                if n not in seen and (not only_text or n.source == "text")]
        # most recently stated first, then best connected -- v5 ranks this
        # with CARRYOVER_SENIORITY_WEIGHT; recency then degree is the same
        # order and needs no extra constant.
        pool.sort(key=lambda n: (-n.last_stated, -len(n.edges), n.lemma))
        return near + pool[:self.max_carryover]

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

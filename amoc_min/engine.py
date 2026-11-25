"""The whole model: story analysis, reader pass, activation, matrix.

Follows v4's analyze() per sentence:

    extract content words (spaCy)  ->  text nodes
    RELATIONSHIPS                  ->  text edges
    INFER_OBJECTS                  ->  inferred concepts / properties
    INFER_RELATIONSHIPS            ->  candidate inferred relationship triplets
    READER_INFERENCES (persona)    ->  which of them this reader makes
    SELECT_RELEVANT_EDGES          ->  revive those, fade the rest
    activation                     ->  one matrix column

Decay is NOT a timer. As in v4's reactivate_relevant_edges, every sentence the
model is asked which edges still relate to the text; those are revived, and
only the unselected ones (bar any added this sentence) fade.

The first five steps are persona-free, so they run once for the story and are
cached: the persona only chooses reader inferences, which keeps faithfulness structural
(it may select, never invent) and keeps concept labels canonical.
"""
import json
import logging
import os
import re
import urllib.request

from . import config as C
from . import nlp as N
from .admission import admit_all, normalize
from .graph import Graph
from .prompts import (INFERENCE_NUDGE, INFER_OBJECTS,
                      INFER_OBJECTS_FIRST,
                      INFER_RELATIONSHIPS, INFER_RELATIONSHIPS_FIRST,
                      READER_INFERENCES, RELATIONSHIPS,
                      RELATIONSHIPS_FIRST, REPLACE_PRONOUNS,
                      NARRATIVE_RELEVANCE, ACTIVATION_RATING,
                      READER_FRAME, READER_GENERATE_FRAME, NEUTRAL_LICENSE,
                      NEUTRAL_LICENSE_LOOSE,
                      READER_INFERENCES_STRICT)


def _parse(text, default):
    """First JSON value in a reply. When the caller's default is a dict (every
    caller here does .get on the result), only ever return a dict: some models
    (Qwen3.8) answer a bare list, or prefix the object with a bracketed
    aside, and the old 'earliest { or [' rule handed that list straight to
    .get(). For a reply that opens with '{' the result is unchanged."""
    text = text or ""
    if not isinstance(default, dict):
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if not m:
            return default
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return default

    def as_dict(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            return v[0]                     # [{...}] -> {...}
        return None

    m = re.search(r"\{.*\}", text, re.S)    # widest {...}: the old behaviour
    if m:
        try:
            v = as_dict(json.loads(m.group(0)))
            if v is not None:
                return v
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            v = as_dict(json.loads(m.group(0)))
            if v is not None:
                return v
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()                # first well-formed {...} anywhere
    for s in re.finditer(r"\{", text):
        try:
            v, _ = dec.raw_decode(text, s.start())
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            return v
    if text.strip():
        logging.warning("unparseable reply, using default: %r", text[:200])
    return default


def _triplets(raw):
    out = []
    for t in raw or []:
        if isinstance(t, dict):
            t = [t.get("subject"), t.get("relation"), t.get("object")]
        if (isinstance(t, (list, tuple)) and len(t) == 3
                and all(isinstance(x, str) and x.strip() for x in t)):
            s, r, o = (x.strip() for x in t)
            if s.lower() != o.lower() and [s, r, o] not in out:
                out.append([s, r, o])
    return out


def _strs(raw):
    return [x.strip() for x in (raw or [])
            if isinstance(x, str) and x.strip()]


def _attaching(items, present):
    """Order `items` so each attaches to the graph, and drop the rest.

    An item attaches when one endpoint is already present. Iterated to a
    fixpoint, so a chain admitted earlier in the same sentence can extend:
    `knight -> pursuit` anchors `pursuit`, which then lets `pursuit -> chase`
    in. Anything still unattached when nothing more can be added is a
    satellite, and is returned to no one.

    `present` is mutated, so the caller chooses the graph this is judged
    against: a hypothetical one (every item admitted) for the persona-free
    candidate set, or the real one, re-read each round, for what a given
    reader actually draws.
    """
    pool, out = list(items), []
    while True:
        moved = False
        for c in list(pool):
            s, o = (tuple(N.lemmas(x))
                    for x in (c["triplet"][0], c["triplet"][2]))
            if s in present or o in present:
                out.append(c)
                pool.remove(c)
                present.update((s, o))
                moved = True
        if not moved:
            return out


class LLM:
    def __init__(self, server_url, model=C.MODEL, temperature=None):
        self.url = server_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = C.TEMPERATURE if temperature is None else temperature

    def ask(self, prompt):
        body = json.dumps({
            "model": self.model, "temperature": self.temperature,
            "max_tokens": 2048,             # never clip a long inferred list
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(self.url, body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=C.TIMEOUT) as r:
            raw = r.read().decode("utf-8", "replace")
        # Fail with the server's actual reply, not a bare NoneType: a new model
        # can answer 200 with no choices, or put everything in its reasoning
        # channel and leave content null (a hybrid left in thinking mode).
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise RuntimeError(f"{self.model}: no choices in server reply: "
                               f"{raw[:500]!r}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content is None:
            raise RuntimeError(
                f"{self.model}: empty content (finish_reason="
                f"{choices[0].get('finish_reason')!r}, reasoning chars="
                f"{len(msg.get('reasoning_content') or msg.get('reasoning') or '')})"
                f" -- if reasoning is non-zero the model is still thinking; "
                f"check VLLM_EXTRA_ARGS / enable_thinking")
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)


def _validate(sentence, triplets, log=None, existing=()):
    """Deterministic admission, ported from amoc/admission/. No LLM call:
    these are structural rules, identical for every reader, so they belong in
    the story analysis and cost nothing."""
    return admit_all(triplets, log, sentence=sentence, existing=existing)


def _nodes_from_text(words):
    """v4 renders these as ' - (surface, node_type)' lines."""
    return "".join(f" - ({w}, {k})\n" for w, _, k in words) or " (none)\n"


def _as_reader(prompt, persona):
    """--persona-decay: ask the what-stays-in-mind question on behalf of one
    reader. Unchanged for the baseline reader and whenever the flag is off."""
    if C.PERSONA_IN_DECAY and persona:
        return READER_FRAME.format(persona=persona) + prompt
    return prompt


def _reactivate(llm, g, nodes, context_text, just_added, sentence_text="",
                persona=None):
    """v5's NARRATIVE_RELEVANCE: the model scores every edge 0/1/2 for whether
    the story's current context still applies to it.

    The score becomes the edge's traversal cost, so activation decays through
    distance rather than snapping to zero: CONTINUOUS edges cost one hop,
    FADING two, DISCONTINUOUS are untraversable. Returns (explanations,
    context_shift) for the triplet log.
    """
    listing, order = g.edges_str(nodes, only_active=False)
    if not order:
        return {}, False
    out = _parse(llm.ask(_as_reader(NARRATIVE_RELEVANCE.format(
        story_context=context_text, current_sentence=sentence_text or context_text,
        active_triplets=listing), persona)), {})
    scored = {}
    for s in (out.get("scores") or []):
        if not isinstance(s, dict):
            continue
        try:
            scored[int(s.get("n")) - 1] = int(s.get("score"))
        except (TypeError, ValueError):
            continue
    reasoning = out.get("reasoning") or ""
    shift = bool(out.get("context_shift"))
    explanations = {}
    for i, e in enumerate(order):
        sc = scored.get(i)
        if sc is None:
            continue                         # unscored -> leave as is
        e.relevance = max(0, min(2, sc))
        # v5 pipeline/decay.py removes on score 1 as well as 0: its rubric
        # makes 1 mean "only the subject is still present", which it treats
        # as superseded. amoc_min left 1 untouched, so a FADING edge could
        # sit active indefinitely. Under the LLM ageing rule that is one of
        # the three things keeping the live set from ever shrinking.
        drop = sc <= 0 or (sc == 1 and C.LLM_DROP_FADING
                           and C.RELATION_AGEING == C.AGEING_LLM)
        if sc >= 2:
            e.revive(g.edge_forget, reactivated=True)
        elif drop and e not in just_added and not g._protect(e):
            e.fade()
            explanations[(e.src.text, e.label, e.dst.text)] = reasoning
    return explanations, shift


def _rate(llm, g, context_text, just_added, sentence_text="", persona=None):
    """--llm-score: the model rates every relationship 0-5 and the rating IS
    its activation. Returns (explanations, context_shift) for the triplet log.

    What the current sentence states is 5 without asking. Everything else is
    listed, inactive relationships included, so one can come back. A
    relationship the reply does not rate -- a clipped or unparseable answer,
    a failed call, no LLM at all -- loses one point rather than keeping its
    old rating, so nothing can stay alive by going unrated.
    """
    stated = [e for e in just_added if e.source == "text"]
    for e in stated:
        e.rate(5, g.edge_forget, asserted=True)
    order = [e for e in g.edges if e not in stated and str(e.label).strip()]
    out = {}
    if llm is not None and order:
        listing = "\n".join(f"{n}. {e.src.text} - {e.label} - {e.dst.text}"
                            for n, e in enumerate(order, 1))
        try:
            out = _parse(llm.ask(_as_reader(ACTIVATION_RATING.format(
                story_context=context_text,
                current_sentence=sentence_text or context_text,
                triplets=listing), persona)), {})
        except Exception:
            logging.exception("activation rating failed; every relationship "
                              "loses a point this sentence")
    raw = out.get("scores") or {}
    if isinstance(raw, list):               # tolerate [{"n":1,"score":3}, ...]
        raw = {str(x.get("n")): x.get("score") for x in raw
               if isinstance(x, dict)}
    reasoning, explanations = out.get("reasoning") or "", {}
    for n, e in enumerate(order, 1):
        try:
            sc = max(0, min(5, int(float(raw[str(n)]))))
        except (KeyError, TypeError, ValueError):
            sc = max(0, e.strength - 1)
        e.rate(sc, g.edge_forget)
        if sc <= 2:
            explanations[(e.src.text, e.label, e.dst.text)] = reasoning
    return explanations, bool(out.get("context_shift"))


def analyze_story(llm, cache=None, sentences=None):
    """Persona-free, so it runs once for the story and is shared by every
    reader: what each sentence states, and what it a reader could infer.

    `sentences` defaults to the knight story; run_texts.py passes the Keefe
    and O'Brien texts through the same analysis."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))

    sentences = list(sentences or C.SENTENCES)
    if C.RESOLVE_PRONOUNS:
        joined = llm.ask(REPLACE_PRONOUNS.format(text=" ".join(sentences)))
        sentences = [s.text.strip() for s in N.nlp()(joined.strip()).sents] \
            or sentences

    g, passes, seen, window = Graph(), [], [], []
    rejected = []
    for i, sent in enumerate(sentences, 1):
        first = i == 1
        words = N.content_words(sent)
        for surface, key, kind in words:
            g.node(surface, "text", i, node_type=kind)
        stated_keys = [tuple(k) for _, k, _ in words]
        g.refresh_scores(stated_keys)
        active = g.active_nodes()
        nft = _nodes_from_text(words)

        tmpl = RELATIONSHIPS_FIRST if first else RELATIONSHIPS
        triplets = _triplets(_parse(llm.ask(tmpl.format(
            nodes_from_text=nft, text=sent,
            nodes_from_graph=g.nodes_str(active),
            edges_from_graph=g.edges_str(active)[0])), {}).get("triplets"))
        triplets = _validate(sent, triplets, rejected,
                             existing=[(x.src.lemma, x.label, x.dst.lemma)
                                       for x in g.edges])
        just_added = [e for e in (g.add(s, r, o, "text", i)
                                  for s, r, o in triplets) if e]

        # v4 shows the inference stage only the TEXT-based subgraph, which is
        # what keeps this whole pass persona-free and therefore cacheable.
        text_active = g.active_nodes(only_text=True)
        tmpl = INFER_OBJECTS_FIRST if first else INFER_OBJECTS
        infer_prompt = tmpl.format(
            nodes_from_text=nft, text=sent,
            inferred_so_far=", ".join(seen) or "(none yet)",
            nodes_from_graph=g.nodes_str(text_active),
            edges_from_graph=g.edges_str(text_active, only_text=True)[0])
        obj = _parse(llm.ask(infer_prompt), {})
        if not (_strs(obj.get("concepts")) or _strs(obj.get("properties"))):
            # v5's INFERENCE_NUDGE: one retry when the stage adds nothing
            obj = _parse(llm.ask(infer_prompt + INFERENCE_NUDGE.format(
                previous=json.dumps(obj)[:200])), {})

        tmpl = INFER_RELATIONSHIPS_FIRST if first else INFER_RELATIONSHIPS
        raw = _parse(llm.ask(tmpl.format(
            nodes_from_text=nft, text=sent,
            concepts=json.dumps(_strs(obj.get("concepts"))),
            properties=json.dumps(_strs(obj.get("properties"))),
            nodes_from_graph=g.nodes_str(text_active),
            edges_from_graph=g.edges_str(text_active, only_text=True)[0])), {})

        inferred_candidates = []
        stated_surfaces = {w.lower() for w, _, _ in words}
        for item in (raw.get("inferences") or []):
            if not isinstance(item, dict):
                continue
            tri = _triplets([item.get("triplet")])
            if not tri:
                continue
            inferred_candidates.append({"triplet": tri[0],
                                        "from": _strs(item.get("from"))})
            for c in (tri[0][0], tri[0][2]):
                if c.lower() not in stated_surfaces and c not in seen:
                    seen.append(c)

        valid = _validate(sent, [c["triplet"] for c in inferred_candidates],
                          rejected,
                          existing=[(x.src.lemma, x.label, x.dst.lemma)
                                    for x in g.edges])
        valid_set = {tuple(t) for t in valid}
        inferred = []
        for c in inferred_candidates:
            key = tuple(normalize(x) for x in c["triplet"])
            if key in valid_set:
                inferred.append({"triplet": list(key),
                                 "from": [normalize(p) for p in c["from"]]})

        window.append(sent)
        if len(window) > C.CONTEXT_LENGTH:
            window.pop(0)
        _reactivate(llm, g, list(g.nodes.values()), " ".join(window),
                    just_added, sentence_text=sent)

        passes.append({"sentence": sent, "words": words,
                       "rejected": [r for r in rejected],
                       "stated": [list(k) for k in stated_keys],
                       "triplets": triplets, "inferred": inferred,
                       "context": " ".join(window)})
        rejected = []

    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp"      # atomic: tasks may race
        with open(tmp, "w") as f:
            json.dump(passes, f, indent=1)
        os.replace(tmp, cache)
    return passes


def _select_reader_inferences(llm, persona, sentence, candidates):
    """The only place the persona enters the model.

    Candidates are candidates as a numbered list and chosen by number, so a
    reader cannot reword a concept even in principle. No cap is applied: the
    count is recorded as a diagnostic, never used to truncate.
    """
    if not candidates:
        return [], None
    lines = [f"{i}. {s} -- {r} -> {o}" for i, (s, r, o) in
             enumerate(c["triplet"] for c in candidates)]
    tmpl = READER_INFERENCES_STRICT if C.STRICT_SELECTION else READER_INFERENCES
    out = _parse(llm.ask(tmpl.format(
        persona=persona, sentence=sentence,
        candidates=chr(10).join(lines))), {})
    picked = []
    for j in (out.get("judgements") or []):
        if not isinstance(j, dict) or not j.get("makes"):
            continue
        try:
            i = int(j.get("n"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(candidates) and candidates[i] not in picked:
            picked.append(candidates[i])
    return picked, len(picked)


def _reader_generates(llm, g, persona, step, first, seen):
    """Step 2: THIS reader proposes its own inferences, through the same two
    prompts the shared analysis uses, with the reader put in front. Then the
    persona-free check of step 3 that does not need a model: the structural
    validation. The neutral judge is _neutral_license. Returns (candidates,
    log): the log has one entry per proposal, with status invalid / valid."""
    words, sent = step["words"], step["sentence"]
    nft = _nodes_from_text(words)
    active = g.active_nodes()               # the reader's OWN graph, inferences included
    frame = READER_GENERATE_FRAME.format(persona=persona)
    tmpl = INFER_OBJECTS_FIRST if first else INFER_OBJECTS
    prompt = frame + tmpl.format(
        nodes_from_text=nft, text=sent,
        inferred_so_far=", ".join(seen) or "(none yet)",
        nodes_from_graph=g.nodes_str(active), edges_from_graph=g.edges_str(active)[0])
    obj = _parse(llm.ask(prompt), {})
    if not (_strs(obj.get("concepts")) or _strs(obj.get("properties"))):
        obj = _parse(llm.ask(prompt + INFERENCE_NUDGE.format(
            previous=json.dumps(obj)[:200])), {})
    tmpl = INFER_RELATIONSHIPS_FIRST if first else INFER_RELATIONSHIPS
    raw = _parse(llm.ask(frame + tmpl.format(
        nodes_from_text=nft, text=sent,
        concepts=json.dumps(_strs(obj.get("concepts"))),
        properties=json.dumps(_strs(obj.get("properties"))),
        nodes_from_graph=g.nodes_str(active), edges_from_graph=g.edges_str(active)[0])), {})

    proposed = []
    for item in (raw.get("inferences") or []):
        if isinstance(item, dict):
            tri = _triplets([item.get("triplet")])
            if tri:
                proposed.append({"triplet": tri[0], "from": _strs(item.get("from"))})
    valid = {tuple(t) for t in _validate(
        sent, [c["triplet"] for c in proposed], None,
        existing=[(x.src.lemma, x.label, x.dst.lemma) for x in g.edges])}
    out, log = [], []
    for c in proposed:
        key = tuple(normalize(x) for x in c["triplet"])
        status = "valid" if key in valid else "invalid"
        log.append({"triplet": list(key), "status": status})
        if status == "valid":
            out.append({"triplet": list(key), "from": [normalize(p) for p in c["from"]]})
    return out, log


def _neutral_license(llm, context, sentence, candidates):
    """Step 3, the neutral judge: keeps only candidates that follow from the
    text. Persona-free, so it is the same test for every reader. If the reply
    cannot be read, every candidate is kept (they already passed the
    structural checks) and the failure is reported, rather than leaving this
    reader with no inferences for the sentence. Returns (kept, failed)."""
    if not candidates:
        return [], False
    lines = [f"{i}. {s} -- {r} -> {o}" for i, (s, r, o) in
             enumerate(c["triplet"] for c in candidates)]
    tmpl = NEUTRAL_LICENSE_LOOSE if C.LOOSE_LICENSE else NEUTRAL_LICENSE
    out = _parse(llm.ask(tmpl.format(
        context=context, sentence=sentence, candidates="\n".join(lines))), {})
    judged = [j for j in (out.get("judgements") or []) if isinstance(j, dict)]
    if not judged:
        logging.warning("neutral license unreadable; keeping %d candidates", len(candidates))
        return list(candidates), True
    keep = []
    for j in judged:
        try:
            i = int(j.get("n"))
        except (TypeError, ValueError):
            continue
        if j.get("inferable") and 0 <= i < len(candidates) and candidates[i] not in keep:
            keep.append(candidates[i])
    return keep, False


def read(passes, llm=None, persona=None,
         snapshots=None, on_sentence=None, meta=None, capacity=1.0):
    """One reader's pass. Returns (columns, reader inferences, triplet records, graph).

    persona=None draws the full candidate inferred relationships: the ideal reader the Landscape
    reference represents. A chained inference is candidates only once its parent
    was drawn by THIS reader, so inference depth is persona-dependent.
    """
    snapshots = [] if snapshots is None else snapshots
    g = Graph(capacity=capacity)
    columns, counts, records, history = [], [], [], {}
    drawn_ever = set()
    generates = bool(C.PERSONA_GENERATES and persona and llm is not None)
    seen = []

    for i, step in enumerate(passes, 1):
        g.begin_sentence()
        for surface, key, kind in step["words"]:
            g.node(surface, "text", i, node_type=kind)
        just_added = [e for e in (g.add(s, r, o, "text", i)
                                  for s, r, o in step["triplets"]) if e]

        # An inferred triplet may enter only if it ATTACHES: one endpoint is
        # already a node, so every inferred node is born connected and the
        # graph cannot grow satellite clusters. Nothing repairs connectivity
        # afterwards -- it is a property of admission.
        #
        # This replaces a `from`-provenance gate that produced exactly the
        # wrong set. It asked whether each parent string was a node, but
        # `from` holds free text: `knight is on pursuit` carried
        # from=['hurried after'] -- a relation label, never a node -- and was
        # rejected, while `pursuit is a chase` carried from=[] and `all([])`
        # is True, so it was admitted. The anchoring inferences were dropped
        # and the floating ones kept; {adventure, chase, pursuit} then sat
        # off the graph from S7 to S13.
        # Judged against the MAIN component, not merely "some node". A node
        # that is itself off the main graph must not be allowed to anchor a
        # cluster onto itself: at S3 that is what grew {cave, dragon, fire,
        # legend} out of the stranded `dragon`. Main-component attachment
        # leaves `dragon` alone until the text relates it to something
        # (S4), and scored better on the reference besides (mean rho .822 vs
        # .813 over the 13 sentences).
        comps = g.components()
        main = {n.key for n in comps[0]} if comps else set()
        gen = {}
        if generates:
            # steps 2-3: the reader proposes, the structural checks and
            # the neutral judge decide what counts as an inference from the text
            g.refresh_scores([tuple(k) for k in step["stated"]])
            own, plog = _reader_generates(llm, g, persona, step, i == 1, seen)
            context = " ".join(p["sentence"] for p in passes[:i])
            licensed, failed = _neutral_license(llm, context, step["sentence"], own)
            gen = {"proposed": len(plog),
                   "valid": len(own),
                   "licensed": len(licensed), "license_failed": failed}
            candidates = _attaching(licensed, main)
        else:
            candidates = _attaching(step["inferred"], main)
        if persona is None:
            drawn, said = candidates, None
        elif generates and not C.STRICT_SELECTION:
            drawn, said = candidates, None    # step 4 optional: the reader already chose
        else:
            drawn, said = _select_reader_inferences(llm, persona, step["sentence"], candidates)

        # Re-checked against the graph as it ACTUALLY is, because a reader who
        # declines an anchor must not receive its children: inference depth is
        # persona-dependent, and no reader can end up with a satellite.
        added = _attaching(drawn, {n.key for n in comps[0]} if comps else set())
        stranded = len(drawn) - len(added)
        for c in added:
            s, r, o = c["triplet"]
            e = g.add(s, r, o, "inferred", i)
            if e:
                just_added.append(e)
            drawn_ever.update((s.lower(), o.lower()))
            for x in (s, o):                  # the reader's own inferred-so-far list
                if generates and x not in seen and x.lower() not in {
                        w.lower() for w, _, _ in step["words"]}:
                    seen.append(x)
        counts.append({"candidates": len(candidates), "drawn": len(drawn),
                       "stranded": stranded, "said": said, **gen})

        if generates:                         # what became of every proposal
            lic = {tuple(c["triplet"]) for c in licensed}
            got = {tuple(c["triplet"]) for c in added}
            for p in plog:
                if p["status"] == "valid":
                    k = tuple(p["triplet"])
                    p["status"] = ("drawn" if k in got else
                                   "licensed_not_drawn" if k in lic else "rejected")
            counts[-1]["proposals"] = plog

        stated_keys = [tuple(k) for k in step["stated"]]
        g.refresh_scores(stated_keys)
        # Decay lives inside reactivation, as in v4. Without an LLM -- offline
        # tests, or a failed call -- fall back to a plain timer fade, so the
        # graph still decays instead of silently keeping every edge alive.
        explanations, shift = {}, False
        score_mode = C.RELATION_AGEING == C.AGEING_SCORE
        if score_mode:
            explanations, shift = _rate(llm, g, step["context"], just_added,
                                        sentence_text=step["sentence"],
                                        persona=persona)
        elif llm is None:
            g.fade_all(keep=just_added)
        else:
            try:
                explanations, shift = _reactivate(
                    llm, g, list(g.nodes.values()), step["context"],
                    just_added, sentence_text=step["sentence"],
                    persona=persona)
            except Exception:
                logging.exception("reactivation failed at sentence %d; "
                                  "falling back to timer fade", i)
                g.fade_all(keep=just_added)

        col = (g.rated_activation(stated_keys) if score_mode else
               g.activation(stated_keys, i, history,
                            sentence_lemmas=N.lemmas(step["sentence"])))
        for t in set(history) | set(col):
            history.setdefault(t, []).append(col.get(t, 0.0))
        columns.append(col)
        records.extend(g.records(i, step["sentence"], meta=meta,
                                 decay_explanations=explanations,
                                 context_shift=shift))
        snapshots.append(g.snapshot())
        if on_sentence is not None:
            on_sentence(i, step, g, col)

    return columns, counts, records, g


def resolve(column, token, token_map=None):
    """v4's get_word_lemma_score: given a word, return that node's score.

    Tries the token as written, then its lemma, then any key in the column
    that TOKEN_MAP canonicalises onto it -- so the reference's `rode` finds
    our `ride`, and `kidnapping` finds `kidnap`. Returns 0.0 when the graph
    has no such node, exactly as an absent row in v4's matrix.
    """
    if token in column:
        return column[token]
    lem = " ".join(N.lemmas(token))
    if lem in column:
        return column[lem]
    if token_map:
        for k, v in column.items():
            if token_map.get(k, k) == token or token_map.get(lem, lem) == k:
                return v
    return 0.0


def to_matrix(columns, vocab=None, token_map=None):
    """Wide matrix in landscape_paper.csv's own format.

    With `vocab`, emit exactly those rows and query the graph for each, as
    v4's matrix step did against the Landscape paper's 20 tokens. Without it,
    emit every token the graph produced.
    """
    if vocab:
        lines = ["word/sentence " + " ".join(str(i) for i in range(1, len(columns) + 1))]
        for t in vocab:
            row = [resolve(c, t, token_map) for c in columns]
            lines.append(t + " " + " ".join(
                str(int(v)) if float(v).is_integer() else str(v) for v in row))
        return "\n".join(lines) + "\n"
    tokens = sorted({t for c in columns for t in c})
    lines = ["word/sentence " + " ".join(str(i) for i in range(1, len(columns) + 1))]
    for t in tokens:
        row = [c.get(t, 0.0) for c in columns]
        lines.append(t + " " + " ".join(
            str(int(v)) if float(v).is_integer() else str(v) for v in row))
    return "\n".join(lines) + "\n"

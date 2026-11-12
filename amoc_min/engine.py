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
from .graph import Graph
from .prompts import (INFERENCE_NUDGE, INFER_OBJECTS,
                      INFER_OBJECTS_FIRST,
                      INFER_RELATIONSHIPS, INFER_RELATIONSHIPS_FIRST,
                      READER_INFERENCES, RELATIONSHIPS,
                      RELATIONSHIPS_FIRST, REPLACE_PRONOUNS,
                      SELECT_RELEVANT_EDGES, VALIDATE_TRIPLETS)


def _parse(text, default):
    m = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
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
            return json.load(r)["choices"][0]["message"]["content"]


def _validate(llm, sentence, triplets):
    """v5's VALIDATE_TRIPLET rules, batched into one call per sentence.

    Persona-FREE on purpose: a malformed triplet is malformed for every
    reader, so this runs inside the story analysis and leaves the shared text
    backbone identical across personas.
    """
    if not triplets:
        return triplets
    listing = chr(10).join(f"{i}. ({s}, {r}, {o})"
                           for i, (s, r, o) in enumerate(triplets))
    out = _parse(llm.ask(VALIDATE_TRIPLETS.format(
        sentence=sentence, triplets=listing)), {})
    judged = {}
    for j in (out.get("judgements") or []):
        if not isinstance(j, dict):
            continue
        try:
            n = int(j.get("n"))
        except (TypeError, ValueError):
            continue
        judged[n] = j
    kept = []
    for i, t in enumerate(triplets):
        j = judged.get(i)
        if j is None:                       # unjudged -> keep, do not silently drop
            kept.append(t)
            continue
        if j.get("valid"):
            kept.append(t)
        else:
            fixed = _triplets([j.get("corrected")])
            if fixed:
                kept.append(fixed[0])
            else:
                logging.info("dropped invalid triplet %s: %s", t,
                             j.get("reason", ""))
    return kept


def _nodes_from_text(words):
    """v4 renders these as ' - (surface, node_type)' lines."""
    return "".join(f" - ({w}, {k})\n" for w, _, k in words) or " (none)\n"


def _reactivate(llm, g, nodes, context_text, just_added):
    """v4's reactivate_relevant_edges: revive what still matters, fade the rest.

    Returns (decay_explanations, context_shift) for the triplet log.
    """
    listing, order = g.edges_str(nodes, only_active=False)
    if not order:
        return {}, False
    out = _parse(llm.ask(SELECT_RELEVANT_EDGES.format(
        edges=listing, text=context_text)), {})
    picked = set()
    for i in (out.get("relevant") or [])[:C.NR_RELEVANT_EDGES]:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(order):
            picked.add(order[i - 1])
    reasoning = out.get("reasoning") or ""
    shift = bool(out.get("context_shift"))
    explanations = {}
    for e in picked:
        e.revive(g.edge_forget, reactivated=True)
    for e in order:
        if e not in picked and e not in just_added and e.active:
            e.fade()
            explanations[(e.src.text, e.label, e.dst.text)] = reasoning
    return explanations, shift


def analyze_story(llm, cache=None):
    """Persona-free, so it runs once for the story and is shared by every
    reader: what each sentence states, and what it a reader could infer."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))

    sentences = list(C.SENTENCES)
    if C.RESOLVE_PRONOUNS:
        joined = llm.ask(REPLACE_PRONOUNS.format(text=" ".join(sentences)))
        sentences = [s.text.strip() for s in N.nlp()(joined.strip()).sents] \
            or sentences

    g, passes, seen, window = Graph(), [], [], []
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
        triplets = _validate(llm, sent, triplets)
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

        valid = _validate(llm, sent, [c["triplet"] for c in inferred_candidates])
        valid_set = {tuple(t) for t in valid}
        inferred = [c for c in inferred_candidates
                    if tuple(c["triplet"]) in valid_set]

        window.append(sent)
        if len(window) > C.CONTEXT_LENGTH:
            window.pop(0)
        _reactivate(llm, g, list(g.nodes.values()), " ".join(window),
                    just_added)

        passes.append({"sentence": sent, "words": words,
                       "stated": [list(k) for k in stated_keys],
                       "triplets": triplets, "inferred": inferred,
                       "context": " ".join(window)})

    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp"      # atomic: tasks may race
        with open(tmp, "w") as f:
            json.dump(passes, f, indent=1)
        os.replace(tmp, cache)
    return passes


def _select_reader_inferences(llm, persona, sentence, candidates, allow_persona_words=False):
    """The only place the persona enters the model.

    Candidates are candidates as a numbered list and chosen by number, so a
    reader cannot reword a concept even in principle. No cap is applied: the
    count is recorded as a diagnostic, never used to truncate.
    """
    if not candidates:
        return [], None
    lines = [f"{i}. {s} -- {r} -> {o}" for i, (s, r, o) in
             enumerate(c["triplet"] for c in candidates)]
    out = _parse(llm.ask(READER_INFERENCES.format(
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


def read(passes, llm=None, persona=None, allow_persona_words=False,
         snapshots=None, on_sentence=None, meta=None):
    """One reader's pass. Returns (columns, reader inferences, triplet records, graph).

    persona=None draws the full candidate inferred relationships: the ideal reader the Landscape
    reference represents. A chained inference is candidates only once its parent
    was drawn by THIS reader, so inference depth is persona-dependent.
    """
    snapshots = [] if snapshots is None else snapshots
    g = Graph()
    columns, counts, records, history = [], [], [], {}
    drawn_ever = set()

    for i, step in enumerate(passes, 1):
        g.begin_sentence()
        for surface, key, kind in step["words"]:
            g.node(surface, "text", i, node_type=kind)
        just_added = [e for e in (g.add(s, r, o, "text", i)
                                  for s, r, o in step["triplets"]) if e]

        candidates = [c for c in step["inferred"]
                   if all(p.lower() in drawn_ever
                          or tuple(N.lemmas(p)) in g.nodes
                          for p in c.get("from", []))]
        if persona is None:
            drawn, said = candidates, None
        else:
            drawn, said = _select_reader_inferences(llm, persona, step["sentence"], candidates,
                                 allow_persona_words)
        for c in drawn:
            s, r, o = c["triplet"]
            e = g.add(s, r, o, "inferred", i)
            if e:
                just_added.append(e)
            drawn_ever.update((s.lower(), o.lower()))
        counts.append({"candidates": len(candidates), "drawn": len(drawn),
                       "said": said})

        stated_keys = [tuple(k) for k in step["stated"]]
        g.refresh_scores(stated_keys)
        # Decay lives inside reactivation, as in v4. Without an LLM -- offline
        # tests, or a failed call -- fall back to a plain timer fade, so the
        # graph still decays instead of silently keeping every edge alive.
        explanations, shift = {}, False
        if llm is None:
            g.fade_all(keep=just_added)
        else:
            try:
                explanations, shift = _reactivate(
                    llm, g, list(g.nodes.values()), step["context"],
                    just_added)
            except Exception:
                logging.exception("reactivation failed at sentence %d; "
                                  "falling back to timer fade", i)
                g.fade_all(keep=just_added)

        col = g.activation(stated_keys, i, history)
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


def to_matrix(columns):
    """Wide matrix in landscape_paper.csv's own format."""
    tokens = sorted({t for c in columns for t in c})
    lines = ["word/sentence " + " ".join(str(i) for i in range(1, len(columns) + 1))]
    for t in tokens:
        row = [c.get(t, 0.0) for c in columns]
        lines.append(t + " " + " ".join(
            str(int(v)) if float(v).is_integer() else str(v) for v in row))
    return "\n".join(lines) + "\n"

from __future__ import annotations

from typing import List

from amoc.nlp.spacy_utils import canonicalize_node_text


# extract blue nodes for new texts
def blue_nodes_from_text(text: str, nlp, min_len: int = 2) -> List[str]:
    if not text:
        return []
    doc = nlp(text)
    nodes = set()
    for tok in doc:
        if tok.is_stop or tok.is_punct or not tok.is_alpha:
            continue
        if tok.pos_ not in {"NOUN", "PROPN"}:
            continue
        canon = canonicalize_node_text(nlp, tok.text)
        if canon and len(canon) >= min_len:
            nodes.add(canon)
    return sorted(nodes)

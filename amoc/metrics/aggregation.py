import logging
import hashlib
import os
from typing import Dict, Any

import pandas as pd

from amoc.metrics.lexical import simple_sentiment_score
from amoc.metrics.lexical import compute_lexical_metrics
from amoc.metrics.graph_metrics import compute_graph_metrics
from amoc.nlp.spacy_utils import load_spacy


_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = load_spacy()
        if _NLP is None:
            raise RuntimeError("spaCy failed to load")
    return _NLP


def make_persona_id(persona_text: str) -> str:
    return hashlib.sha1(persona_text.encode("utf-8")).hexdigest()


TAXONOMIC_RELATIONS = {
    "is_a",
    "type_of",
    "kind_of",
    "class_of",
    "form_of",
}

CAUSAL_RELATIONS = {
    "causes",
    "leads_to",
    "results_in",
    "affects",
    "influences",
}


def classify_relation(r: str) -> str:
    r = r.lower().strip()
    if r in TAXONOMIC_RELATIONS:
        return "taxonomic"
    if r in CAUSAL_RELATIONS:
        return "causal"
    return "event"


def abstract_relation_ratio(relations) -> float:
    if not relations:
        return 0.0

    abstract = 0
    for r in relations:
        if classify_relation(r) in {"taxonomic", "causal"}:
            abstract += 1

    return abstract / len(relations)


def abstract_concept_ratio(concepts) -> float:
    """
    Proxy for conceptual abstraction:
    nouns without named-entity grounding.
    """
    if not concepts:
        return 0.0

    nlp = get_nlp()
    abstract = 0
    total = 0

    for c in concepts:
        doc = nlp(c)
        for tok in doc:
            if tok.is_alpha:
                total += 1
                if tok.pos_ == "NOUN" and tok.ent_type_ == "":
                    abstract += 1

    return abstract / total if total > 0 else 0.0


def process_triplets_file(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, engine="python", on_bad_lines="warn")

    # Drop stray unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if df.empty:
        return pd.DataFrame()

    required_cols = {
        "original_index",
        "age_refined",
        "persona_text",
        "subject",
        "object",
        "regime",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logging.warning(f"Missing required columns in {path}: {missing}")
        return pd.DataFrame()

    if "model_name" not in df.columns:
        df["model_name"] = None

    df["source_file"] = os.path.basename(path)
    df["persona_id"] = df["persona_text"].astype(str).apply(make_persona_id)

    # grouping
    group_cols = [
        "persona_id",
        "persona_text",
        "model_name",
        "regime",
    ]

    if "education_level" in df.columns:
        group_cols.append("education_level")

    records = []

    for keys, g in df.groupby(group_cols, dropna=False):
        ctx = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

        num_triplets = len(g)
        if num_triplets == 0:
            continue

        subjects = g["subject"].astype(str)
        objects = g["object"].astype(str)
        relations = (
            g["relation"].astype(str)
            if "relation" in g.columns
            else pd.Series(["<NO_RELATION>"] * num_triplets, index=g.index)
        )

        # METRICS
        num_unique_subjects = subjects.nunique()
        num_unique_objects = objects.nunique()
        num_unique_relations = relations.nunique()
        num_unique_concepts = len(set(subjects) | set(objects))

        triplets = list(zip(subjects, relations, objects))
        num_unique_triplets = len(set(triplets))
        triplet_repetition_ratio = 1.0 - (num_unique_triplets / num_triplets)

        persona_text = ctx["persona_text"] or ""
        persona_tokens = persona_text.split()
        persona_num_tokens = len(persona_tokens)
        triplets_per_100_tokens = (
            (num_triplets / persona_num_tokens) * 100 if persona_num_tokens > 0 else 0.0
        )
        sentiment_score = simple_sentiment_score(persona_text)
        lex = compute_lexical_metrics(persona_text)
        edges = list(zip(subjects.tolist(), objects.tolist()))
        graph = compute_graph_metrics(edges)

        concepts = list(subjects) + list(objects)
        relation_list = list(relations)
        concept_abstraction = abstract_concept_ratio(concepts)
        relation_abstraction = abstract_relation_ratio(relation_list)

        # age -> retrieved from age_refined
        age_refined = g["age_refined"].iloc[0]
        try:
            age_refined_int = int(age_refined)
        except Exception:
            age_refined_int = None

        record: Dict[str, Any] = {
            "persona_id": ctx["persona_id"],
            "original_index": g["original_index"].iloc[0],  # metadata only
            "source_file": g["source_file"].iloc[0],
            "regime": ctx["regime"],
            "model_name": ctx["model_name"],
            "persona_text": persona_text,
            "age_refined": age_refined_int,
            "num_triplets": num_triplets,
            "num_unique_triplets": num_unique_triplets,
            "num_unique_subjects": num_unique_subjects,
            "num_unique_objects": num_unique_objects,
            "num_unique_concepts": num_unique_concepts,
            "num_unique_relations": num_unique_relations,
            "triplet_repetition_ratio": triplet_repetition_ratio,
            "persona_num_tokens": persona_num_tokens,
            "triplets_per_100_tokens": triplets_per_100_tokens,
            "sentiment_score": sentiment_score,
            "lexical_ttr": lex["lexical_ttr"],
            "lexical_avg_word_len": lex["lexical_avg_word_len"],
            "abstract_concept_ratio": concept_abstraction,
            "abstract_relation_ratio": relation_abstraction,
            **graph,
        }

        if "education_level" in ctx:
            record["education_level"] = ctx["education_level"]

        records.append(record)

    return pd.DataFrame(records)

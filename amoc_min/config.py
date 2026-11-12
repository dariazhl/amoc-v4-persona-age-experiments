"""Every parameter for the persona-modularity branch.

Nothing here is imported from amoc/ and nothing falls back to a hidden
default. The August runs failed because defaults (persona "for focus only,
do not add extra concepts", ALLOW_INFERRED_CHAINS=0, DECAY_STEP=1) silently
suppressed the effect under study. So all knobs live in this one file.
"""
from pathlib import Path

HERE      = Path(__file__).resolve().parent
ROOT      = HERE.parent
# Vendored so amoc_min is self-contained: study_results/ is not synced to the
# cluster, and both the reference matrix and TOKEN_MAP live under it.
REFERENCE = HERE / "reference" / "landscape_paper.csv"
OUT       = HERE / "runs"

SPACY_MODEL  = "en_core_web_lg"   # falls back to _md then _sm
CONTENT_POS  = ("NOUN", "PROPN", "ADJ")   # v4: is_content_word_and_non_stopword
SIMILARITY   = 0.8                # v4: Edge.similarity_threshold
CONTEXT_LENGTH    = 1             # v4: sentences of context for reactivation
RESOLVE_PRONOUNS  = False         # v4 ran the knight story with this off


MODEL   = "meta-llama/Llama-3.3-70B-Instruct"
TIMEOUT = 180

# Deterministic by default. Two reasons: between-persona differences are then
# persona differences rather than sampling noise, and the story analysis is
# stable enough to cache and share. The cost is that within-regime variance
# must come from the 20 distinct persona texts rather than from sampling -- if
# it turns out to be zero, every regime collapses to one row and effect sizes
# become artefactually perfect. Check that on the first run before scaling.
TEMPERATURE = 0.0

# Knight story, 13 sentences == the 13 columns of landscape_paper.csv.
SENTENCES = [
    "A young knight rode through the forest.",
    "The knight was unfamiliar with the country.",
    "Suddenly, a dragon appeared.",
    "The dragon was kidnapping a beautiful princess.",
    "The knight wanted to free the princess.",
    "The knight wanted to marry the princess.",
    "The knight hurried after the dragon.",
    "The knight and the dragon fought for life and death.",
    "Soon, the knight's armor was completely scorched.",
    "At last, the knight killed the dragon.",
    "The knight freed the princess.",
    "The princess was very thankful to the knight.",
    "The princess married the knight.",
]

# --- activation ----------------------------------------------------------
# Derived from the 54 decay transitions in landscape_paper.csv: the rule is
# multiplicative (x0.5 dominant, x0.6/x0.8 observed), never subtractive --
# the smallest drop anywhere in the reference is -1.0, so DECAY_STEP=1 on a
# 0-2 scale could not reproduce it. Anything landing under FLOOR goes to 0.
INFERRED_ACTIVATION = 2.0   # concept inferred, not stated (ref: horse=2, sword=2, fire=3)
SCORE_PER_HOP       = 0.5   # score = 5 - SCORE_PER_HOP * BFS distance.
                            # The AMoC v4 notebook contains no matrix code at
                            # all -- node.score is only ever a hop count. The
                            # 0.5 is recovered from the AMoC v4.0 column of
                            # Fig. 8, whose values are 5 / 4.5 / 4 / 3.5 / 3 /
                            # 2.5 for distances 0..5, unreachable -> 0.
EDGE_FORGET         = 2     # sentences an un-reactivated edge survives (v4: EDGE_VISIBILITY)
# Two different jobs, which v4 conflated because it had no matrix at all.
MAX_CARRYOVER_NODES = 5     # v5's constant of the same name. Nodes kept in
                            # the PROMPT's graph section even when they are
                            # out of reach, ranked by how recently the text
                            # stated them. Without it the context a sentence
                            # sees is whatever is reachable from that
                            # sentence's own words, which is self-
                            # reinforcing: at S3 "a dragon appeared" is
                            # unreachable from everything, so the model was
                            # shown `dragon` alone with no edges and could
                            # not have related it to the knight however it
                            # was asked to. Carry-over is what makes v5's
                            # `dragon -meets-> knight` possible.
PROMPT_MAX_DISTANCE = 2     # v4's MAX_DISTANCE_FROM_ACTIVE_NODES: how many
                            # hops of graph context to show the LLM. Keep at
                            # v4's value -- raising it changes generation and
                            # inflates every prompt.
MATRIX_MAX_DISTANCE = 1     # activation cutoff. score = 5 - 0.5*d, so 2 can
                            # only ever emit {5, 4.5, 4} -- which is exactly
                            # what the nov_11 matrix contained. 8 is the
                            # Kept at 2 so the matrix stays sparse: the value
                            # range comes from SENIORITY, not from distance.
# Why 1 and not 0 or 2, measured on the knight story against the reference:
#   0  rho +0.891, density error 4.8 -- but NO inferred concept ever reaches
#      the matrix (horse, sword, fire all vanish), so the persona, which acts
#      only through which inferences a reader draws, becomes unmeasurable.
#   1  rho +0.913, density error 13.1   <- chosen
#   2  rho +0.913, density error 15.2 -- identical accuracy, denser matrix.
# 1 and 2 score the same, so 2 bought nothing but rows.

COST_CONTINUOUS     = 1     # v5 NARRATIVE_RELEVANCE score 2
COST_FADING         = 2     # score 1; score 0 makes the edge untraversable
# How a relation label's age is reset.
#   "text"      its own word appears in the sentence (the Landscape Model's
#               word-activation idea; deterministic, reproducible)
#   "relevance" NARRATIVE_RELEVANCE scores the edge 2 -- one judgement then
#               drives BOTH distance and age, which compounds. Stronger model,
#               weaker measurement: the matrix becomes largely a function of
#               the LLM's opinion of its own graph.
#
# Selected at run time by `--llm-decay` (run.py), which is driven by LLM_DECAY
# in the SLURM script. Both values are recorded in the run summary, so a
# matrix can always be traced back to the rule that produced it.
RELATION_AGEING     = "text"

# The two names the flag chooses between. Kept here so nothing else in the
# codebase spells the strings out.
AGEING_TEXT, AGEING_LLM = "text", "relevance"

# Carry-over: what the reader still holds from the previous sentence.
#
# Without this, activation is recomputed each sentence purely as distance from
# the words that sentence states, so a sentence naming one peripheral concept
# flattens the graph. "Suddenly, a dragon appeared." states only `dragon`, and
# everything else has to be within MATRIX_MAX_DISTANCE of it -- `knight` fell
# from 5.0 to 0.0 at S3 while the reference holds it at 2.5. That 2.5 is
# decayed carry-over; no distance rule can reach it from `dragon`.
#
# v5 spells this MAX_CARRYOVER / CARRYOVER_SENIORITY_WEIGHT. Here it is the
# Landscape Model's own recurrence: a concept keeps a fraction of what it had,
# and the fresh distance-based score competes with it (whichever is higher).
# Set CARRYOVER_DECAY = 0 to recover the old behaviour exactly.
CARRYOVER_DECAY     = 0.5   # fraction retained per sentence
CARRYOVER_FLOOR     = 1.0   # anything decaying below this is forgotten (-> 0)

# The reference's non-zero values are {1, 1.5, 2, 2.5, 3, 4, 5} -- it never
# emits anything below 1. Seniority leaves fractional dribbles (country 0.5 at
# S4, beautiful 0.5 at S6) that the Landscape Model has no equivalent for, so
# anything under this is forgotten rather than reported.
ACTIVATION_FLOOR    = 1.0

SENIORITY_WEIGHT    = 2.0   # v5's CARRYOVER_SENIORITY_WEIGHT. With distance
                            # capped at 2, this is what produces the value
                            # range: score = 5 - 0.5*d - w*age, where age is
                            # sentences since the concept was last stated
                            # (0 when stated now). Drives scores to 0 over
                            # time, so the matrix gets SPARSER, not denser.

# Decay is persona-INVARIANT by design. Persona acts only on inference
# reader inferences, so all regimes share one activation dynamic and therefore remain
# comparable against a single reference matrix.

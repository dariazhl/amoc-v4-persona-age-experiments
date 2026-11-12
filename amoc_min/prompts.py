"""Prompts. v4's set, plus READER_INFERENCES -- the one place the persona acts.

v4's wording is kept close, with two changes: output is JSON rather than
eval'd Python, and the inference stage is told it may build on an earlier
inference (v4 had no chaining).

Deliberately absent: the August frame's "Persona (for focus only, do not add
extra concepts)", which forbade the persona from touching the inference step
and produced rho = +0.03 between persona level and inference rate.

EXTRACT, INFER_* and SELECT_RELEVANT_EDGES never see the persona.
"""

_GRAPH = """I also have the current knowledge graph.
Nodes (with their type, and a score of how central they are in the story;
0 is most central, 1 less central, and so on):
{nodes_from_graph}
Edges (relationships between the nodes):
{edges_from_graph}
"""

_HEAD = """I want to build a knowledge graph from a story. The graph has two \
kinds of node: CONCEPT nodes, which are the objects and persons of the story \
and are usually nouns, and PROPERTY nodes, which describe concepts and are \
usually adjectives. An edge joins a concept to a concept, or a concept to a \
property, and is labelled with the relationship between them.

I already extracted the nodes of the current sentence, and you should use these:
{nodes_from_text}
"""

# -- 1. relationships stated in the text ---------------------------------
RELATIONSHIPS = _HEAD + _GRAPH + """
Give the relationships between the nodes of this sentence, and between those
nodes and the graph's nodes (prefer the more central ones). The sentence is:
{text}

Reply with JSON and nothing else:
{{"triplets": [["subject", "relation", "object"]]}}"""

RELATIONSHIPS_FIRST = _HEAD + """
Give the relationships between the nodes of this sentence. The sentence is:
{text}

Reply with JSON and nothing else:
{{"triplets": [["subject", "relation", "object"]]}}"""

# -- 2. inference, stage one: what the text supports ----------------------
INFER_OBJECTS = _HEAD + _GRAPH + """
Concepts already inferred earlier in the story: {inferred_so_far}

Name concepts and properties that are NOT written in the text but that the
text supports -- things necessarily or very probably true given what is
stated, which a reader could connect to the story. An inference may build on
an EARLIER INFERENCE rather than on the text directly: if a reader has already
inferred that the knight carries a sword, they may infer what follows from the
sword. Do not name anything that contradicts the text.

Do NOT repeat any of the extracted nodes listed above, and do not use any word
that appears in the sentence.

Example: for "A girl walked to the old mill." good answers are concepts like
"river", "wheel", "flour" and properties like "curious", "quiet"; "girl",
"mill" and "old" are NOT allowed, because they are already in the text.

The sentence is:
{text}

Reply with JSON and nothing else:
{{"concepts": ["..."], "properties": ["..."]}}"""

INFER_OBJECTS_FIRST = _HEAD + """
Name concepts and properties that are NOT written in the text but that the
text supports -- things necessarily or very probably true given what is
stated. Do not name anything that contradicts the text.

Do NOT repeat any of the extracted nodes listed above, and do not use any word
that appears in the sentence.

Example: for "A girl walked to the old mill." good answers are concepts like
"river", "wheel", "flour" and properties like "curious", "quiet"; "girl",
"mill" and "old" are NOT allowed, because they are already in the text.

The sentence is:
{text}

Reply with JSON and nothing else:
{{"concepts": ["..."], "properties": ["..."]}}"""

# -- 3. inference, stage two: connect them --------------------------------
INFER_RELATIONSHIPS = _HEAD + _GRAPH + """
And these additional concepts and properties, which the text supports but
does not state:
concepts: {concepts}
properties: {properties}

Join them to the nodes of the sentence, and to each other where that follows.
For each triplet, say which concepts it follows from, using words from the
sentence or from the inferred list above; use "from": [] when it follows from
the sentence as a whole. The sentence is:
{text}

Reply with JSON and nothing else:
{{"inferences": [{{"triplet": ["subject", "relation", "object"],
                  "from": ["..."]}}]}}"""

INFER_RELATIONSHIPS_FIRST = _HEAD + """
And these additional concepts and properties, which the text supports but
does not state:
concepts: {concepts}
properties: {properties}

Join them to the nodes of the sentence. For each triplet, say which concepts
it follows from; use "from": [] when it follows from the sentence as a whole.
The sentence is:
{text}

Reply with JSON and nothing else:
{{"inferences": [{{"triplet": ["subject", "relation", "object"],
                  "from": ["..."]}}]}}"""

# -- 3b. triplet validation (persona-FREE: a malformed triplet is malformed
# for every reader, so this runs once in the story analysis and never skews
# the shared text backbone). v5's VALIDATE_TRIPLET_PROMPT rules 1-5; rule 6
# (no inferred-to-inferred) is deliberately omitted because chaining is on.
VALIDATE_TRIPLETS = """You are checking candidate relationships for a knowledge
graph, drawn from this sentence:
"{sentence}"

{triplets}

Judge each one by these rules, in order:
1. THE RELATION MUST BE A VERB or verb phrase.
   valid: "wrote", "is", "wears", "rode through"   invalid: "traditional", "none"
2. NO CIRCULAR RELATIONSHIPS: the object must not be the nominalised form of
   the verb.  valid: (knight, unites, tribes)   invalid: (knight, unites, unification)
3. ACTION VERBS REQUIRE NOUN OBJECTS.
   valid: (knight, wears, armor)   invalid: (knight, wears, scorched)
4. COPULAS MAY TAKE ADJECTIVES: "is", "was", "became", "seemed" may link to one.
   valid: (knight, is, young)
5. NO VAGUE RELATIONS: reject "relates to", "associated with", "connected to",
   "involves", "concerns", "pertains to".
   valid: (knight, fought, dragon)   invalid: (knight, relates to, dragon)

Reply with JSON and nothing else, one entry per number:
{{"judgements": [{{"n": <number>, "valid": true/false,
                  "reason": "short", "corrected": ["subject","relation","object"] or null}}]}}"""

INFERENCE_NUDGE = """

Your previous answer added nothing: it was empty, or it only repeated words
already in the sentence ({previous}). Answer again, with concepts and
properties that a reader brings to the sentence from their own knowledge and
that do NOT appear in it."""

# -- 4. which edges stay in mind (v4's reactivate_relevant_edges) ---------
NARRATIVE_RELEVANCE = """You are maintaining a knowledge graph of a story.
Score each relationship by how much the story's CURRENT context still applies
to it -- not by whether its exact words are repeated.

Story context:
{story_context}

Current sentence:
{current_sentence}

Active relationships:
{active_triplets}

SCORE 2 - CONTINUOUS: still part of the scene, topic or characters the story
is discussing. A pronoun, a synonym, or an implied continuation is enough;
literal repetition is NOT required.

SCORE 1 - FADING: the story has moved to different details but is still in the
same broader scene or event. Not the current focus, but not contradicted or
abandoned either.

SCORE 0 - DISCONTINUOUS: the sentence marks a clear shift -- a new scene, a
new time, a new topic or a new set of characters -- so this relationship's
context has been superseded. Also score 0 if you are uncertain.

Judge continuity of TOPIC and SCENE, not word overlap.

Also say whether the CURRENT SENTENCE marks a context shift for the story as a
whole: a new scene, time, topic or set of characters.

Reply with JSON and nothing else, one entry per number:
{{"scores": [{{"n": <number>, "score": 0, 1 or 2}}],
  "context_shift": true/false, "reasoning": "one short sentence"}}"""

# -- 5. the persona's inference reader inferences ----------------------------------
READER_INFERENCES = """You are modelling one specific reader.

Reader: {persona}

Current sentence: "{sentence}"

The text supports these inferred relationships here:
{candidates}

Take each one on its own and say whether THIS reader makes that connection
while reading this sentence. Judge from the reader's age, education,
vocabulary and background knowledge. There is no target number: judge each
one on its own merits.

Reply with JSON and nothing else, one entry per number:
{{"judgements": [{{"n": <number>, "makes": true/false}}]}}"""


# -- 6. pronoun resolution (v4 ran the knight story with this off) --------
REPLACE_PRONOUNS = """Replace the pronouns "he, she, they" with the persons or
nouns from the text that they refer to. Sometimes there is no such reference
and you should leave them as they are. Do not invent names; they must be in
the text. The text is:
{text}

Reply with the rewritten text and nothing else."""

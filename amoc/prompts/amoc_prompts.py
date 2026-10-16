NEW_RELATIONSHIPS_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

I also have the knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

I want you to tell me the relationships (edges) between the nodes from the text themselves. And also between the nodes from the text and the other nodes from the graph (here prioritize the relationships based on the score). The text is:
{text}

IMPORTANT CONNECTIVITY CONSTRAINT: Every relationship you output must include:
- at least one node from the extracted text nodes list ({nodes_from_text}), AND
- at least one node that already exists in the graph nodes list ({nodes_from_graph}).

Do NOT output relationships that only connect new text nodes to other new text nodes if neither side attaches to the existing graph.

HUB-FIRST ATTACHMENT: Among the existing graph nodes, prioritize connections to the node(s) with the LOWEST score (score=0 means most central/hub). Every explicit concept from the current sentence should ideally have at least one relationship connecting it to the most central node (hub) in the graph. If a direct connection to the hub is semantically justified by the text, include it.

List them as a Python list and do not provide additional explanation."""

NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

I want you to tell me the relationships (edges) between the nodes given the text. The text is:
{text}

HUB CONNECTIVITY: Ensure that the resulting graph is well-connected. Choose one central concept (the one most frequently related to other concepts) as the hub, and ensure that all other explicit nodes have at least one relationship path to this hub. Prioritize direct connections to the hub where semantically justified.

List them as a Python list and do not provide additional explanation."""

INFER_OBJECTS_AND_PROPERTIES_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

I also have the current knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Generate a list of concepts and properties that are in line with the overall coherence and sense within the given text and the knowledge graph, but they are not in the text. The text is:
{text}

List them in the following format and explain the role of the text and the knowledge graph in the decesion making process:
{{
    "concepts": ["concept1", "concept2", ...],
    "properties": ["property1", "property2", ...]
}}"""

GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

I also have the current knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

And I already have additional concepts and properties that are in line with the overall coherence and sense within the given text:
concepts: {concepts}
properties: {properties}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Create relationships between the nodes from the text and the new concepts and the new properties (as you see fit). The text is:
{text}

HUB-FIRST ATTACHMENT: When creating inferred relationships, prioritize connections that attach new inferred nodes to the most central explicit node (hub) or to other explicit nodes from the sentence. Inferred nodes should connect to the explicit backbone first, ensuring the graph remains well-connected through the hub.

Provide them in the following format:
{{
    "concept_relationships": [concept_relation1, concept_relation2, ...],
    "property_relationships": [property_relation1, property_relation2, ...]
}}"""

INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Generate a list of concepts and properties that are in line with the overall coherence and sense within the given text, but they are not in the text. The text is:
{text}

List them in the following format:
{{
    "concepts": ["concept1", "concept2", ...],
    "properties": ["property1", "property2", ...]
}}"""

GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

And I already have additional concepts and properties that are in line with the overall coherence and sense within the given text:
concepts: {concepts}
properties: {properties}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Create relationships between the nodes from the text and the new concepts and the new properties (as you see fit). The text is:
{text}

HUB-FIRST ATTACHMENT: When creating inferred relationships, prioritize connections that attach new inferred nodes to the most central explicit node (hub) or to other explicit nodes from the sentence. Inferred nodes should connect to the explicit backbone first, ensuring the graph remains well-connected through the hub.

Provide them in the following format:
{{
    "concept_relationships": [concept_relation1, concept_relation2, ...],
    "property_relationships": [property_relation1, property_relation2, ...]
}}"""

SELECT_RELEVANT_EDGES_PROMPT = """You have the following edges from a knowledge graph in the format: node - edge - node
{edges}

As you can see, they are numbered. Tell me what edges are related to / support / contradict the following text.
{text}

Provide them in the following format (a list of numbers):
[1, 2, 3, ...]
"""

PRONOUN_RESOLUTION_PROMPT = """You are resolving pronouns in a story. Given the story context and a sentence, identify which pronouns refer to which entities.

Story context:
{context}

Sentence:
{sentence}

Return a JSON object mapping each pronoun to the entity it refers to.

Rules:
- Only include pronouns that clearly resolve (he, she, they, him, her, them, his, her, their)
- Use the exact entity name as it appears in the context
- Do not invent entities not in the context
- If a pronoun is ambiguous or unclear, do not include it
- Return ONLY the JSON object, no other text

Example:
Input context: "Charlemagne was a great king. He ruled wisely."
Input sentence: "He conquered many lands."
Output: {{"He": "Charlemagne"}}

Now process this."""

NEW_RELATIONSHIPS_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

I also have the knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

I want you to tell me the relationships (edges) between the nodes from the text themselves. And also between the nodes from the text and the other nodes from the graph (here prioritize the relationships based on the score). The text is:
{text}

IMPORTANT CONNECTIVITY CONSTRAINT: Every relationship you output must include:
- at least one node from the extracted text nodes list ({nodes_from_text}), AND
- at least one node that already exists in the graph nodes list ({nodes_from_graph}).

Do NOT output relationships that only connect new text nodes to other new text nodes if neither side attaches to the existing graph.

HUB-FIRST ATTACHMENT: Among the existing graph nodes, prioritize connections to the node(s) with the LOWEST score (score=0 means most central/hub). Every explicit concept from the current sentence should ideally have at least one relationship connecting it to the most central node (hub) in the graph. If a direct connection to the hub is semantically justified by the text, include it.

List them as a Python list and do not provide additional explanation."""

NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

I want you to tell me the relationships (edges) between the nodes given the text. The text is:
{text}

List them as a Python list and do not provide additional explanation."""

INFER_OBJECTS_AND_PROPERTIES_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

I also have the current knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Generate a list of concepts and properties that are in line with the overall coherence and sense within the given text and the knowledge graph, but they are not in the text. The text is:
{text}

List them in the following format and explain the role of the text and the knowledge graph in the decesion making process:
{{
    "concepts": ["concept1", "concept2", ...],
    "properties": ["property1", "property2", ...]
}}"""

GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

I also have the current knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

And I already have additional concepts and properties that are in line with the overall coherence and sense within the given text:
concepts: {concepts}
properties: {properties}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Create relationships between the nodes from the text and the new concepts and the new properties (as you see fit). The text is:
{text}

HUB-FIRST ATTACHMENT: When creating inferred relationships, prioritize connections that attach new inferred nodes to the most central explicit node (hub) or to other explicit nodes from the sentence. Inferred nodes should connect to the explicit backbone first, ensuring the graph remains well-connected through the hub.

Provide them in the following format:
{{
    "concept_relationships": [concept_relation1, concept_relation2, ...],
    "property_relationships": [property_relation1, property_relation2, ...]
}}"""

INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Generate a list of concepts and properties that are in line with the overall coherence and sense within the given text, but they are not in the text. The text is:
{text}

List them in the following format:
{{
    "concepts": ["concept1", "concept2", ...],
    "properties": ["property1", "property2", ...]
}}"""

GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

And I already have additional concepts and properties that are in line with the overall coherence and sense within the given text:
concepts: {concepts}
properties: {properties}

{level_block}
- which edges you choose to include (more concrete vs more abstract),
- how you phrase the relation (e.g., 'plays_with' vs 'researches', 'learns_about' vs 'teaches').

Create relationships between the nodes from the text and the new concepts and the new properties (as you see fit). The text is:
{text}

HUB-FIRST ATTACHMENT: When creating inferred relationships, prioritize connections that attach new inferred nodes to the most central explicit node (hub) or to other explicit nodes from the sentence. Inferred nodes should connect to the explicit backbone first, ensuring the graph remains well-connected through the hub.

Provide them in the following format:
{{
    "concept_relationships": [concept_relation1, concept_relation2, ...],
    "property_relationships": [property_relation1, property_relation2, ...]
}}"""

SELECT_RELEVANT_EDGES_PROMPT = """You have the following edges from a knowledge graph in the format: node - edge - node
{edges}

As you can see, they are numbered. Tell me what edges are related to / support / contradict the following text.
{text}

Provide them in the following format (a list of numbers):
[1, 2, 3, ...]
"""

HUB_EDGE_LABEL_WITH_EXPLANATION_PROMPT = """I am building a knowledge graph from text. I have extracted the following concepts from the sentence:
{explicit_nodes}

These concepts appear in the sentence but are not directly connected by an edge in my graph.
I need to connect "{node_a}" to "{node_b}" (the hub concept) to maintain graph connectivity.

The sentence is:
"{sentence_text}"

Based on the sentence context, provide:
1. A short relationship label (1-3 words) that describes how "{node_a}" relates to "{node_b}"
2. A brief explanation (1 sentence) of why this connection makes sense in the context

Respond in this exact JSON format:
{{
    "label": "relationship_label",
    "explanation": "Brief explanation of the connection"
}}"""


FORCED_CONNECTIVITY_EDGE_PROMPT = """I am building a knowledge graph from a story. The graph has become disconnected - there are concepts that should be connected but are not.

I need to connect these two concepts to restore graph connectivity:
- Concept A: "{node_a}"
- Concept B: "{node_b}"

Here is the story context so far:
"{story_context}"

Here is the current sentence being processed:
"{current_sentence}"

Based on the story context and current sentence, provide a SHORT relationship label (1-3 words, preferably a verb or verb phrase) that reasonably connects "{node_a}" to "{node_b}".

IMPORTANT GUIDELINES:
1. The relationship should be semantically reasonable given the story context
2. Prefer generic but meaningful relationships like "relates to", "involves", "concerns" if no specific connection is clear
3. Do NOT invent specific actions or events not supported by the text
4. Keep it simple - this is a connectivity bridge, not a primary semantic relationship

Respond in this exact JSON format:
{{
    "label": "relationship_label",
    "explanation": "Brief explanation (1 sentence) of why this connection is reasonable"
}}"""

VALIDATE_TRIPLET_PROMPT = """You are validating a candidate relationship for a knowledge graph.

Given the sentence:
"{sentence}"

Candidate triple: ({subject}, {relation}, {object})

VALIDATION RULES (apply in order):
1. RELATION MUST BE A VERB:
   The relation must be a verb or verb phrase.
   Valid: "wrote", "is", "wears", "takes_place_at"
   Invalid: "traditional", "simple", "na", "none"

2. NO CIRCULAR RELATIONSHIPS:
   Object must not be the nominalized form of the verb.
   Valid: (charlemagne, unites, tribes)
   Invalid: (charlemagne, unites, unification)

3. ACTION VERBS REQUIRE NOUN OBJECTS:
   Verbs like wear, find, prefer need noun objects.
   Valid: (charlemagne, wears, attire), (charlemagne, prefers, simplicity)
   Invalid: (charlemagne, wears, traditional), (charlemagne, prefers, simple)

4. COPULA VERBS CAN TAKE ADJECTIVES:
   "is", "was", "are", "become", "seem", "look" can link to adjectives.
   Valid: (charlemagne, is, handsome), (attire, is, traditional)

5. AVOID VAGUE RELATIONS:
   Reject any relation that is vague or non-specific:
   - "relates to", "related to", "associated with", "connected to"
   - "involves", "concerns", "pertains to", "regarding"
   Valid: (charlemagne, fought, saxons)
   Invalid: (charlemagne, related to, saxons)

{rule_6}
EXAMPLES:
Input: (charlemagne, dresses in, beautiful)
Output: {{"valid": false, "reason": "'dresses in' requires a noun object (what does he wear?)", "corrected_triple": null}}

Input: (charlemagne, prefers, simple)
Output: {{"valid": false, "reason": "'prefers' needs a noun object (what does he prefer?)", "corrected_triple": null}}

Input: (charlemagne, prefers, simplicity)
Output: {{"valid": true, "reason": "Complete: verb + noun object", "corrected_triple": null}}

Input: (charlemagne, is, handsome)
Output: {{"valid": true, "reason": "Copular verb + adjective is valid", "corrected_triple": null}}

Input: (charlemagne, unites, unification)
Output: {{"valid": false, "reason": "Circular - unification IS the act of uniting", "corrected_triple": ["charlemagne", "unites", "tribes"]}}

Input: (clothing, relates to, school)
Output: {{"valid": false, "reason": "Vague relation - specify how clothing relates to school", "corrected_triple": null}}

Return a JSON with:
1. "valid": true/false
2. "reason": brief explanation
3. "corrected_triple": [subject, relation, object] if fixable, else null
"""

NARRATIVE_RELEVANCE_PROMPT = """You are maintaining a knowledge graph of a story. Your task is to score each
relationship by how much the story's CURRENT context still applies to it — not by whether its exact words
are repeated in the current sentence.

Story context (previous sentences):
{story_context}

Current sentence being processed:
{current_sentence}

Active relationships in the reader's memory:
{active_triplets}

SCORING GUIDE (0-2) – TOPICAL/SCENE CONTINUITY:
SCORE 2 - CONTINUOUS (keep active)
- The relationship is still part of the scene, topic, or characters the story is currently discussing.
- Pronouns, synonyms, or an implied continuation of the same idea are enough — literal repetition of the
  relation or object word is NOT required.
- Example: if the story is still following the knight through the forest, (knight, rides, horse) stays
  relevant even if the current sentence doesn't say "rides" or "horse", as long as the scene hasn't changed.

SCORE 1 - FADING (borderline / connectivity-critical exception)
- The story has moved on to different specific details, but is still within the same broader scene or
  event (same characters/location/topic) — this relationship isn't the current focus but hasn't been
  contradicted or abandoned.
- Also use this score if the edge is critical for keeping the graph connected, even while faded.

SCORE 0 - DISCONTINUOUS (no longer applies)
- The current sentence marks a clear shift: a new scene, a new time ("later that day", "meanwhile"), a new
  topic, or a new set of characters — the relationship's context has been superseded and is no longer part
  of what the story is discussing.
- The relation is generic ("is", "has", "involves", "relates to") with no remaining topical anchor.
- You are uncertain whether the topic is still continuous → SCORE 0.

CRITICAL RULES:
1. Judge continuity of TOPIC/SCENE, not literal word overlap. A pronoun, a synonym, or an implied
   continuation of the same idea is enough for score 2 — do not require the exact noun/verb to reappear.
2. Score 0 should correspond to an actual shift in the story (new scene/topic/time/characters), not simply
   "this specific word wasn't used in this sentence."
3. When in doubt about whether the topic changed, score 0.

ALSO report whether the CURRENT SENTENCE ITSELF marks a context shift for the story as a whole — i.e., a
"paragraph break": a new scene, a new time, a new topic, or a new set of characters, such that much of the
established context is no longer what the story is discussing. This is a single yes/no judgment about the
sentence, separate from the individual per-relationship scores above.

Return JSON: {{"scores": {{"(subj, rel, obj)": 0/1/2}}, "context_shift": true/false, "reasoning": "brief"}}
"""

PRUNE_IRRELEVANT_TRIPLETS_BY_NARRATIVE = """You are maintaining a clean knowledge graph.

Story so far:
{story_context}

Current sentence:
{current_sentence}

Active relationships:
{active_triplets}

KEEP a relationship ONLY if:
- The RELATION or OBJECT appears EXPLICITLY in the current sentence.
- OR removing it would isolate a concept that appears in the current sentence.

REMOVE otherwise, including if only the subject appears.

OUTPUT: {{"to_keep": ["(subject, relation, object)"], "reasoning": "..."}}
"""


LEGACY_LEVEL_BLOCK = """IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:"""

READER_LEVEL_BLOCK = """IMPORTANT: The graph represents what THE READER described above understands from the text, not an ideal analysis of it. Take the reader's age, educational level, vocabulary and background knowledge into account so that the graph reflects what this reader is realistically able to understand, notice and infer. A younger or less educated reader may miss abstract or technical relations and connect ideas through concrete, everyday knowledge; a more educated reader may recognise abstract, domain-specific or implied relations.

Every edge must still be grounded in the text or in knowledge this reader would plausibly bring to it. The reader's age and education should influence:"""

VALIDATOR_RULE_6 = """6. NO INFERRED-TO-INFERRED CONNECTIONS:
   If BOTH the subject AND object are inferred concepts (not explicitly mentioned in the sentence), the triple is INVALID.
   A concept is "inferred" if it does NOT appear as a word or clear synonym in the sentence.
   Valid: (charlemagne, is, king) - "charlemagne" and "king" appear in sentence
   Invalid: (strategic, involves, reader) - neither "strategic" nor "reader" appears in sentence
"""

LEGACY_FRAME = """You are a knowledge graph builder. Output ONLY the requested Python list or JSON object. Do not add explanations, thinking process, or extra text.
        Persona (for focus only, do not add extra concepts): {persona}
        {prompt}"""

READER_FRAME = """You are simulating a specific reader building a mental model of a text, expressed as a knowledge graph. Output ONLY the requested Python list or JSON object. Do not add explanations, thinking process, or extra text.
The reader: {persona}
Build the graph as THIS reader would understand the text: include what they would notice, understand and infer, and leave out what they would not.
{prompt}"""

NEUTRAL_FRAME = """You are a careful annotation assistant. Output ONLY the requested Python list or JSON object. Do not add explanations, thinking process, or extra text.
{prompt}"""

NEW_RELATIONSHIPS_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

I also have the knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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

IMPORTANT: Some concept nodes represent persons whose age or educational level is explicitly mentioned or strongly implied in the text (for example: "8-year-old boy", "primary school student", "high school student", "college freshman", "university student"). When you generate relationships, you must take this age or educational level into account so that the relationships reflect what that person is realistically able to understand.

If the age or education level is not mentioned, do NOT invent it. In that case, just create reasonable relationships supported by the text without making assumptions about their level of understanding.

Always base every edge strictly on information that is stated or clearly implied in the text. Age and education should influence:
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
   Prefer specific over "related to", "associated with".
   Valid: (charlemagne, fought, saxons)
   Invalid: (charlemagne, related to, saxons)

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

# Simplified with 1-3 scale
NARRATIVE_RELEVANCE_PROMPT = """You are maintaining a knowledge graph of a story. Your task is to score each relationship by how important it is for understanding the current narrative.

Story context (previous sentences):
{story_context}

Current sentence being processed:
{current_sentence}

Active relationships in the reader's memory:
{active_triplets}

SCORING GUIDE (0-3):

SCORE 3 - HIGH RELEVANCE (Essential to keep)
- Involves main characters (Charlemagne, his family, kingdoms, key figures)
- Directly describes actions or states in the current sentence
- Bridges concepts to explicit story elements - these bridging edges are CRITICAL to preserve

SCORE 2 - MEDIUM RELEVANCE (Useful context)
- Background information about secondary characters
- Generic but meaningful relations ("lives_in", "travels_to", "works_with")
- Provides supporting details not central to current events

SCORE 1 - LOW RELEVANCE (Can be removed)
- Semantically incomplete triples (action verb + adjective without noun):
  (charlemagne, prefers, simple) - what does he prefer?
  (charlemagne, wears, traditional) - wears WHAT?
  (charlemagne, finds, interesting) - finds WHAT interesting?
- Vague relations with no specific meaning:
  (pride, relates_to, ability) - how do they relate?
  (x, associated_with, y) - in what way?
- Semantic duplicates when a better form exists:
  If both (charlemagne, is, skilled) AND (charlemagne, has, skill) exist, the "has" version is score 1
  Prefer "is + adjective" (score 3) over "has + noun" (score 1) for the same attribute
- Minor details from early sentences no longer connected to current narrative

SCORE 0 - COMPLETELY IRRELEVANT (Remove immediately)
- Semantically incoherent or garbage triples
- Complete non-sequiturs that have no connection to the current narrative
- Triplets that are clearly errors or hallucinations
- Must check connectivity protection - only remove if node remains connected

CRITICAL RULES:

1. CONTEXT AWARENESS:
   - A triple's score can CHANGE based on later context
   - Example: (pride, relates_to, ability) in sentence 1 with no context = score 1
   - If sentence 3 discusses how pride affects ability, that same triple becomes score 3

2. CONNECTIVITY PROTECTION:
   - NEVER assign score 0 to an edge if removing it would disconnect a node from the graph
   - Score 0 means immediate removal in the current sentence
   - Score 1 means gradual decay over multiple sentences
   - Check if the node has other connections before marking for removal

SCORING EXAMPLES:

Context: First sentence "Charlemagne preferred simple living."
Active triplets:
- (charlemagne, preferred, simple) → SCORE 1 (incomplete - preferred WHAT?)
- (charlemagne, is, simple) → SCORE 3 (valid property of main character)
- (charlemagne, has, simplicity) → SCORE 1 (duplicate of "is simple" - worse form)

Context: Later sentence "He wore traditional Frankish attire."
Active triplets:
- (charlemagne, wore, traditional) → SCORE 1 (incomplete - wore WHAT?)
- (charlemagne, wore, attire) → SCORE 3 (complete, connects to main character)
- (attire, is, traditional) → SCORE 3 (complete property)

Return a JSON object with:
{{
    "scores": {{
        "(subject1, relation1, object1)": 3,
        "(subject2, relation2, object2)": 1,
        ...
    }},
    "reasoning": "Brief explanation of scoring strategy, noting bridging edges protected and duplicate resolutions"
}}
"""

PRUNE_IRRELEVANT_TRIPLETS_BY_NARRATIVE = """You are maintaining a clean knowledge graph of a story. Your task is to identify which relationships are essential to keep and which are irrelevant and can be removed.

Story so far:
{story_context}

Current sentence:
{current_sentence}

Here are the active relationships in the reader's memory:
{active_triplets}

KEEP ONLY if they meet MULTIPLE of these criteria:

1. DIRECT NARRATIVE RELEVANCE:
   - Directly involves main characters (Charlemagne, his family, kingdoms, key figures)
   - Directly mentioned or clearly implied in the CURRENT sentence
   - Forms a critical bridge that would disconnect the graph if removed

2. SEMANTICALLY COMPLETE:
   - Subject + action verb + specific noun object (e.g., "charlemagne - conquers - saxons")
   - Subject + is + adjective describing a key attribute (e.g., "charlemagne - is - powerful")
   - Must have both subject and object as NOUNS or PROPER NOUNS (not vague concepts)

REMOVE if ANY of these apply:

1. SEMANTICALLY INCOMPLETE:
   - Action verb + adjective without noun (e.g., "has - regal", "prefers - simple", "wears - traditional")
   - Any triplet where the object is an adjective and the verb is not a copula (is/was/are)

2. VAGUE OR GENERIC RELATIONS:
   - "relates_to", "associated_with", "connected_to", "involves", "concerns", "has"
   - Any relation that doesn't specify HOW two concepts are connected

3. SEMANTIC DUPLICATES:
   - When multiple triplets express the same meaning, KEEP ONLY ONE
   - Prefer "is + adjective" over "has + noun" (e.g., keep "is skilled", remove "has skill")
   - Remove any triplet that is redundant with another

4. WEAKLY CONNECTED:
   - Triplets where neither subject nor object appears in the current sentence
   - Minor details about secondary characters not relevant to current plot

5. AMBIGUOUS OR ILLOGICAL:
   - Triplets that don't make logical sense (e.g., "ability - enables - famous")
   - Relations missing required prepositions
   - Generic subjects like "thing", "something", "it" performing actions

DECISION EXAMPLES:

Sentence 1: "Charlemagne conquered the Saxons."
Active triplets:
- (charlemagne, conquered, saxons) - KEEP (directly in current sentence)
- (charlemagne, is, king) - REMOVE (background, not in current sentence)
- (saxons, are, fierce) - REMOVE (inferred, not in current sentence)

Sentence 2: "He was a powerful ruler."
Active triplets:
- (charlemagne, is, powerful) - KEEP (directly in current sentence)
- (charlemagne, conquered, saxons) - REMOVE (not in current sentence)
- (charlemagne, has, army) - REMOVE (not mentioned, inferred)

Sentence 3: "The court scholars wrote manuscripts."
Active triplets:
- (court, employs, scholars) - KEEP (bridges to current)
- (scholars, write, manuscripts) - KEEP (directly in current sentence)
- (charlemagne, has, court) - REMOVE (not mentioned in current sentence)

Return a JSON object with this exact structure:
{{
    "to_keep": [
        "(subject1, relation1, object1)",
        "(subject2, relation2, object2)",
        ...
    ],
    "reasoning": "Explain pruning decisions, noting what was removed and why"
}}
"""

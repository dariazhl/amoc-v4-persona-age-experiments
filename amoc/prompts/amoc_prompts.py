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

**SEMANTIC COMPLETENESS VALIDATION RULES:**

1. **RELATION MUST BE A VERB OR VERB PHRASE:**
   - The relation must be a verb or contain a verb (e.g., "wrote", "is", "wears", "takes_place_at")
   - ✓ VALID: "wrote", "is", "wears", "has", "takes_place_at", "prefers", "finds"
   - ✗ INVALID: "not applicable", "na", "none", adjectives as relations like "traditional", "simple", "interesting"

2. **PREPOSITIONS MUST BE INCLUDED WHEN REQUIRED:**
   - Many verbs require prepositions to complete their meaning
   - ✓ CORRECT: "related to", "lives in", "travels to", "depends on", "wrote about", "believes in"
   - ✗ INCORRECT: "related", "lives", "travels", "depends", "wrote", "believes"
   - If the relation is missing a required preposition, mark as INVALID and provide the corrected version
   - Example: (charlemagne, related, charles) → INVALID, correct to (charlemagne, related to, charles)

3. **NO CIRCULAR OR TAUTOLOGICAL RELATIONSHIPS:**
   - The object must NOT be the nominalized form of the verb (e.g., "unite" -> "unification", "create" -> "creation")
   - The triple should express an action affecting a separate entity, not the action itself
   - ✓ VALID: (charlemagne, unites, tribes) - tribes are separate from the act of uniting
   - ✓ VALID: (charlemagne, seeks, unity) - "unity" is a state, not the action itself
   - ✗ INVALID: (charlemagne, seeks to unite, unification) - "unification" IS the act of uniting
   - ✗ INVALID: (charlemagne, creates, creation) - creation IS the act of creating
   - ✗ INVALID: (charlemagne, governs, governance) - governance IS the act of governing
   - ✗ INVALID: (charlemagne, rules, rule) - rule IS the act of ruling
   - ✗ INVALID: (charlemagne, fights, fight) - fight IS the act of fighting

4. **OBJECT MUST COMPLETE THE VERB'S MEANING:**
   - For transitive verbs (verbs that require an object), the object must be a noun
   - ✓ VALID: (charlemagne, prefers, simplicity) - noun object completes "prefers"
   - ✓ VALID: (charlemagne, wears, attire) - noun object
   - ✓ VALID: (charlemagne, finds, manuscript) - noun object
   - ✗ INVALID: (charlemagne, prefers, simple) - "simple" is an adjective, doesn't complete the meaning
   - ✗ INVALID: (charlemagne, wears, traditional) - "wears" needs a noun (what does he wear?)
   - ✗ INVALID: (charlemagne, finds, interesting) - "finds" needs a noun (what does he find?)

5. **ACTION VERBS REQUIRE NOUN OBJECTS (CRITICAL RULE):**
   - Action verbs (wear, find, use, take, make, create, fight, dress, prefer, etc.) MUST have NOUN objects
   - The object must be a concrete thing that receives the action
   - ✓ VALID: (knight, wears, armor) - armor is a noun receiving the action
   - ✓ VALID: (charlemagne, finds, manuscript) - manuscript is a noun receiving the action
   - ✓ VALID: (charlemagne, prefers, simplicity) - simplicity is a noun (abstract but still noun)
   - ✗ INVALID: (knight, wears, traditional) - "traditional" is an adjective, not a thing
   - ✗ INVALID: (charlemagne, finds, interesting) - "interesting" is an adjective
   - ✗ INVALID: (breech, is part of, famous) - "famous" is an adjective
   - ✗ INVALID: (charlemagne, dresses in, beautiful) - "beautiful" is an adjective
   - ✗ INVALID: (he, finds, interesting) - "interesting" is an adjective
   - The ONLY exception is copular verbs (see Rule 6)

6. **COPULA VERBS ("is", "was", "are") HANDLE ADJECTIVES CORRECTLY:**
   - Copular verbs (is, was, are, become, seem, appear, feel, look, sound, taste, smell) can link to adjectives
   - "is" + adjective is COMPLETE and VALID
   - ✓ VALID: (charlemagne, is, handsome) - complete property description
   - ✓ VALID: (charlemagne, is, skilled) - complete
   - ✓ VALID: (attire, is, traditional) - complete
   - ✓ VALID: (princess, looks, beautiful) - "looks" is copular-like
   - This is correct because copular verbs link a subject to a property/state

7. **NO GENERIC SUBJECTS FOR ACTION VERBS:**
   - Generic words ("thing", "something", "it", "this", "that", "certain", "year") cannot be subjects of action verbs
   - ✓ VALID: (charlemagne, writes, book)
   - ✗ INVALID: (thing, writes, book)
   - ✓ VALID: Generic words can be objects: (charlemagne, writes, something)

8. **REJECT NEGATION RELATIONS:**
   - Relations that express absence of connection add no semantic value to a knowledge graph
   - ✗ REJECT: "not connected", "not related", "no connection", "unrelated", "disconnected", "not connected to", "not related to"
   - ✗ REJECT: (clothing, not connected to, school)
   - ✗ REJECT: (charlemagne, not related to, shirt)
   - Any relation starting with "not" or "no" or expressing absence should be rejected

9. **AVOID VAGUE RELATIONS:**
   - Relations like "related to", "associated with", "connected to", "involves" add little semantic meaning
   - These should be avoided unless they are the only possible connection
   - ✓ BETTER: Use specific relations like "fought", "married", "ruled", "wrote"
   - ✗ AVOID: (charlemagne, related to, pepin) - use (charlemagne, is father of, pepin) if known
   - ✗ AVOID: (pride, relates_to, ability) - use (pride, enhances, ability) or (pride, comes from, ability)
   - If the only connection possible is vague, mark as INVALID with suggestion to use more specific relation


**DECISION EXAMPLES:**

Input: (breech, is part of, famous)
Output: {{"valid": false, "reason": "Action verb phrase 'is part of' requires a noun object - 'famous' is an adjective. What is famous? This needs a noun.", "corrected_triple": null}}

Input: (charlemagne, dresses in, beautiful)
Output: {{"valid": false, "reason": "Action verb 'dresses in' requires a noun object - what does he dress in? 'beautiful' is an adjective describing the missing noun", "corrected_triple": null}}

Input: (he, finds, interesting)
Output: {{"valid": false, "reason": "Action verb 'finds' requires a noun object - what does he find interesting? The adjective doesn't complete the meaning", "corrected_triple": null}}

Input: (knight, wears, traditional)
Output: {{"valid": false, "reason": "'wears' requires a noun object - what does he wear? 'traditional' as an adjective doesn't complete the meaning", "corrected_triple": null}}

Input: (charlemagne, seeks to unite, unification)
Output: {{"valid": false, "reason": "Circular relationship - 'unification' IS the act of uniting. The object should be what is being united (e.g., tribes, kingdoms)", "corrected_triple": ["charlemagne", "seeks to unite", "tribes"]}}

Input: (charlemagne, unites, tribes)
Output: {{"valid": true, "reason": "Complete: subject + verb + concrete noun object", "corrected_triple": null}}

Input: (charlemagne, prefers, simple)
Output: {{"valid": false, "reason": "'prefers' requires a noun object - what does he prefer? 'simple' as an adjective doesn't complete the meaning", "corrected_triple": null}}

Input: (charlemagne, prefers, simplicity)
Output: {{"valid": true, "reason": "Complete: subject + verb + noun object (abstract noun is acceptable)", "corrected_triple": null}}

Input: (charlemagne, is, handsome)
Output: {{"valid": true, "reason": "'is' + adjective is a complete property description (copular verb)", "corrected_triple": null}}

Input: (princess, looks, beautiful)
Output: {{"valid": true, "reason": "'looks' is a copular verb linking subject to adjective", "corrected_triple": null}}

Input: (charlemagne, values, educational)
Output: {{"valid": false, "reason": "'values' requires a noun object - what does he value? 'educational' is an adjective", "corrected_triple": ["charlemagne", "values", "education"]}}

Input: (charlemagne, wears, attire)
Output: {{"valid": true, "reason": "Complete: subject + verb + noun object", "corrected_triple": null}}

Input: (pride, relates_to, ability)
Output: {{"valid": false, "reason": "Vague relation 'relates_to' adds no semantic content - doesn't specify how pride relates to ability", "corrected_triple": null}}

Input: (charlemagne, finds, manuscript)
Output: {{"valid": true, "reason": "Complete with noun object", "corrected_triple": null}}

Input: (festival, takes_place_at, aachen)
Output: {{"valid": true, "reason": "Complete: subject + verb phrase + location object", "corrected_triple": null}}

Input: (festival, take place at, aachen)
Output: {{"valid": false, "reason": "Incorrect verb form - should be 'takes_place_at' for subject 'festival'", "corrected_triple": ["festival", "takes_place_at", "aachen"]}}

Input: (clothing, not connected to, school)
Output: {{"valid": false, "reason": "Negation relation adds no semantic value - absence of connection is not a valid relation", "corrected_triple": null}}

Input: (charlemagne, not related to, pepin)
Output: {{"valid": false, "reason": "Negation relation - absence of relation is not valid for a knowledge graph", "corrected_triple": null}}

Input: (clothing, relates to, school)
Output: {{"valid": false, "reason": "Vague relation 'relates to' doesn't specify how clothing relates to school. Consider 'is uniform for', 'is worn at', or other specific relation", "corrected_triple": null}}


Return a JSON object with:
1. "valid": true or false
2. "reason": brief explanation focusing on semantic completeness
3. "corrected_triple": if you think the triple is almost correct but needs adjustment, provide the corrected (subject, relation, object) as a list. Otherwise, null.
"""

# Simplified with 1-3 scale
NARRATIVE_RELEVANCE_PROMPT = """You are maintaining a knowledge graph of a story. Your task is to score each relationship by how important it is for understanding the current narrative.

Story context (previous sentences):
{story_context}

Current sentence being processed:
{current_sentence}

Active relationships in the reader's memory:
{active_triplets}

**SCORING GUIDE (0-3):**

**SCORE 3 - HIGH RELEVANCE (Essential to keep)**
- Involves main characters (Charlemagne, his family, kingdoms, key figures)
- Directly describes actions or states in the current sentence
- Forms part of a meaningful multi-hop chain - ALL edges in a connected chain should be preserved together, not decayed aggressively
  * Example: (charlemagne, has, court), (court, employs, scholars), (scholars, write, manuscripts) should ALL be scored 3 because they form an explanatory unit
  * Even if individual edges seem less relevant alone, their value is in maintaining the complete chain
- Bridges concepts to explicit story elements - these bridging edges are CRITICAL to preserve

**SCORE 2 - MEDIUM RELEVANCE (Useful context)**
- Background information about secondary characters
- Generic but meaningful relations ("lives_in", "travels_to", "works_with")
- Provides supporting details not central to current events

**SCORE 1 - LOW RELEVANCE (Can be removed if not part of chains)**
- Semantically incomplete triples (action verb + adjective without noun):
  * (charlemagne, prefers, simple) - what does he prefer?
  * (charlemagne, wears, traditional) - wears WHAT?
  * (charlemagne, finds, interesting) - finds WHAT interesting?
- Vague relations with no specific meaning that are NOT part of valuable chains:
  * (pride, relates_to, ability) - how do they relate?
  * (x, associated_with, y) - in what way?
- Semantic duplicates when a better form exists:
  * If both (charlemagne, is, skilled) AND (charlemagne, has, skill) exist, the "has" version is score 1
  * Prefer "is + adjective" (score 3) over "has + noun" (score 1) for the same attribute
- Minor details from early sentences no longer connected to current narrative

**SCORE 0 - COMPLETELY IRRELEVANT (Remove immediately)**
- Semantically incoherent or garbage triples
- Complete non-sequiturs that have no connection to the current narrative
- Triplets that are clearly errors or hallucinations
- **Must check connectivity protection** - only remove if node remains connected

**CRITICAL RULES:**

1. **CONTEXT AWARENESS:**
   - A triple's score can CHANGE based on later context
   - Example: (pride, relates_to, ability) in sentence 1 with no context = score 1
   - If sentence 3 discusses how pride affects ability, that same triple becomes score 3

2. **MULTI-HOP CHAIN PROTECTION:**
   - **ALL edges that are part of a connected multi-hop chain should be preserved together**
   - Do NOT decay individual edges in a chain aggressively - the chain's value is collective
   - When you see a chain like (A) -> (B) -> (C) -> (D), score ALL edges as 3, even if some seem less central
   - Example: (charlemagne, has, court), (court, employs, scholars), (scholars, write, manuscripts) → ALL score 3

3. **CONNECTIVITY PROTECTION:**
   - **NEVER assign score 0 to an edge if removing it would disconnect a node from the graph**
   - Score 0 means immediate removal in the current sentence
   - Score 1 means gradual decay over multiple sentences
   - Check if the node has other connections before marking for removal

**SCORING EXAMPLES:**

Context: First sentence "Charlemagne preferred simple living."
Active triplets:
- (charlemagne, preferred, simple) → SCORE 1 (incomplete - preferred WHAT?)
- (charlemagne, is, simple) → SCORE 3 (valid property of main character)
- (charlemagne, has, simplicity) → SCORE 1 (duplicate of "is simple" - worse form)

Context: Later sentence "He wore traditional Frankish attire."
Active triplets:
- (charlemagne, wore, traditional) → SCORE 1 (incomplete - wore WHAT? - but note this should be replaced by the chain below)
- (charlemagne, wore, attire) → SCORE 3 (complete, part of valuable chain)
- (attire, is, traditional) → SCORE 3 (complete property, part of valuable chain)
- The complete chain (wore, attire) + (attire, is, traditional) is preserved with both edges at score 3

Return a JSON object with:
{{
    "scores": {{
        "(subject1, relation1, object1)": 3,
        "(subject2, relation2, object2)": 1,
        ...
    }},
    "reasoning": "Brief explanation of scoring strategy, noting multi-hop chains preserved as units, bridging edges protected, and duplicate resolutions"
}}
"""

PRUNE_IRRELEVANT_TRIPLETS_BY_NARRATIVE = """You are maintaining a clean knowledge graph of a story. Your task is to identify which relationships are **essential** to keep and which are **irrelevant** and can be removed.

Story so far:
{story_context}

Current sentence:
{current_sentence}

Here are the active relationships in the reader's memory:
{active_triplets}

**KEEP (RELEVANT) if:**

1. **SEMANTICALLY COMPLETE RELATIONSHIPS:**
   - Subject + action verb + noun object (e.g., "charlemagne - wears - attire")
   - Subject + is + adjective (e.g., "charlemagne - is - powerful")
   - Subject + verb + preposition + noun (e.g., "charlemagne - travels_to - aachen")

2. **MAIN CHARACTER CONNECTIONS:**
   - Involves main characters (Charlemagne, his family, kingdoms, key figures)
   - Describes their actions, states, or relationships

3. **MULTI-HOP CHAINS - PRESERVE AS COMPLETE UNITS:**
   - **ALL edges that are part of a connected multi-hop chain should be kept together**
   - Do NOT prune individual edges from a chain even if they seem less relevant alone
   - Example chain to preserve completely:
     * (charlemagne, has, court)
     * (court, employs, scholars)
     * (scholars, write, manuscripts)
     → Keep ALL three, even if "employs" seems less central
   - Chains that connect concepts to explicit text nodes are HIGH VALUE
   - Relationships that explain "how" or "why" things happen should be preserved as complete units

4. **BRIDGING EDGES ARE CRITICAL:**
   - Edges that connect inferred concepts to explicit nodes must be preserved
   - These enable the entire inference structure
   - Even if a bridging edge seems generic, it enables the chain

5. **CONTEXTUALLY GROUNDED:**
   - Relationships mentioned or clearly implied in recent sentences
   - Provides context needed to understand the current sentence
   - Bridges important concepts even if inferred

**REMOVE (IRRELEVANT) if:**

1. **SEMANTICALLY INCOMPLETE:**
   - Action verb + adjective without noun (e.g., "has - regal", "prefers - simple", "wears - traditional", "finds - interesting")
   - These are grammatically incomplete and add confusion, not meaning
   - **Exception: If an incomplete triple is part of a larger pattern that can be corrected, flag it but don't remove the corrected version**

2. **VAGUE OR GENERIC RELATIONS**
   - "relates_to", "associated_with", "connected_to" without specific meaning
   - "involves", "concerns", "has" when used generically
  
3. **SEMANTIC DUPLICATES:**
   - When both (charlemagne, is, skilled) AND (charlemagne, has, skill) exist → KEEP only "is skilled", REMOVE "has skill"
   - When both (strategy, is, strategic) AND (strategy, has, strategy) exist → KEEP only "is strategic"
   - When both (x, is, adjective) AND (x, has, noun_form) describe the same attribute → KEEP "is + adjective", REMOVE "has + noun"

4. **NO LONGER RELEVANT AND NOT PART OF CHAINS:**
   - Minor details from earlier sentences no longer connected to current narrative
   - Background information that has been superseded

5. **STRUCTURALLY DANGEROUS (HANDLE WITH CARE):**
   - **Never remove a relationship if it's the only connection between a node and the rest of the graph**
   - Check connectivity before removing!

6. **SEMANTICALLY AMBIGUOUS OR ILLOGICAL:**
   - Taken together, the triplet's subject, relation and object do not make sense logically e.g. "ability" - "enables" - "famous" OR "charlemagne" - "wears at" - "festival"
   - The relation misses a copular verb where needed
   - These add confusion, not meaning
   - **But if they can be corrected to fit into a chain, note that in reasoning**

**DECISION EXAMPLES:**

Sentence 1: "Charlemagne preferred simple living."
Active triplets after sentence 1:
- (charlemagne, preferred, simple) - REMOVE (incomplete - what did he prefer?)
- (charlemagne, is, simple) - KEEP (valid property)

Sentence: "He wore traditional attire."
Active triplets:
- (charlemagne, wore, traditional) - REMOVE (incomplete - replace with chain)
- (charlemagne, wore, attire) - KEEP (part of valuable chain)
- (attire, is, traditional) - KEEP (part of valuable chain)
- The chain (wore, attire) + (attire, is, traditional) is preserved as a unit

**MULTI-HOP CHAIN EXAMPLE:**
Consider this complete chain:
- (charlemagne, has, court)
- (court, employs, scholars)
- (scholars, write, manuscripts)

Even if "employs" seems less central alone, KEEP ALL THREE. The chain's value is collective - it explains how Charlemagne's court functions and produces manuscripts. Removing any link breaks the explanatory chain.

**BRIDGING EDGE EXAMPLE:**
- (charlemagne, has, court) [explicit]
- (court, employs, scholars) [inferred - BRIDGING EDGE]
- (scholars, write, manuscripts) [inferred]

The edge (court, employs, scholars) is CRITICAL - it connects the explicit "court" to the inferred "scholars". Keep it even if "employs" seems generic.

**DUPLICATE HANDLING EXAMPLE:**
If both exist:
- (charlemagne, is, skilled) - KEEP
- (charlemagne, has, skill) - REMOVE (duplicate)

Return a JSON object with this exact structure:
{{
    "to_keep": [
        "(subject1, relation1, object1)",
        "(subject2, relation2, object2)",
        ...
    ],
    "reasoning": "Explain key pruning decisions, noting any multi-hop chains preserved as complete units, bridging edges protected, and semantic duplicates resolved"
}}
"""

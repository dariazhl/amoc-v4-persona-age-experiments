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

REPLACE_PRONOUNS_PROMPT = 'Replace the pronouns "he, she, they" with the persons / nouns from the text that they are referring to. Sometimes there is no such reference and you should leave them as they are. Do not come up with imaginary names for the pronouns, they must be in the text. The text is:\n'

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

VALIDATE_TRIPLE_PROMPT = """You are validating a candidate relationship for a knowledge graph.

Given the sentence:
"{sentence}"

Determine if the following triple is EXPLICITLY STATED in the sentence:
({subject}, {relation}, {object})

STRICT RULES:
- The triple must be directly and explicitly stated in the sentence, not just implied.
- The exact words or a very close synonym must appear in the sentence.
- The subject must be the actual doer of the action in the sentence.
- The object must be the actual receiver of the action in the sentence.
- If the sentence does NOT explicitly state this relationship, it is INVALID.
- If the triple reverses subject and object, it is INVALID.
- If the triple uses a word not in the sentence (e.g., "knows about" when sentence says "wrote about"), it is INVALID.

**GENERIC WORD RULES:**
- Vague, generic words like "thing", "something", "certain", "some", "any", "other" should NOT appear as subject or object nodes unless they are essential, specific concepts in the context.
- If a generic word appears in the sentence (e.g., "most of the things"), do NOT extract it as a separate node. Instead, extract the specific concepts it refers to (e.g., if "things" refers to "writings" or "stories", use those instead).
- Words like "certain" are adjectives that modify nouns - they should be attached as properties to the noun they modify, not extracted as standalone objects.
- Example: In "wrote most of the things", "things" is too vague - if the context suggests it means "writings" or "stories", use those specific terms instead. If no specific meaning is clear, reject triples containing "thing".
- Abstract concepts like "thing" should NOT be the subject of actions unless the sentence explicitly makes them the subject (e.g., "the thing fell").
- In the sentence "A man wrote about the king", "thing" is the object, not the subject - so "thing writes about king" is INVALID.
- If the object is a generic word like "thing" and the relation is a generic relation like "is", "has", "involves", the triple is INVALID.

Respond with a JSON object containing:
1. "valid": true or false
2. "reason": a brief explanation of your decision (1 sentence)
3. "corrected_triple": if you think the triple is almost correct but needs adjustment (e.g., wrong direction, or replacing a generic word with a more specific one from context), provide the corrected (subject, relation, object) as a list. Otherwise, null.

Example response format:
{{"valid": true, "reason": "The sentence states that the man wrote about the king.", "corrected_triple": null}}

OR if reversed:
{{"valid": false, "reason": "The sentence says the man wrote about the king, not the other way around.", "corrected_triple": ["man", "wrote about", "king"]}}

OR if generic word:
{{"valid": false, "reason": "The object 'thing' is too vague and does not add meaningful context.", "corrected_triple": null}}

OR if completely wrong:
{{"valid": false, "reason": "The sentence does not mention anything about 'thing' knowing about anything.", "corrected_triple": null}}
"""

NARRATIVE_RELEVANCE_PROMPT = """You are evaluating whether a relationship in a knowledge graph is reasonably relevant to the narrative of a story.

Story so far:
{story_context}

Current sentence:
{current_sentence}

Candidate triple:
({subject}, {relation}, {object})

A relationship can be considered narratively relevant if it:
- Captures an action, event, or interaction in the story
- Reveals something meaningful about a character, object, or situation
- Helps explain relationships between entities
- Provides context that helps the reader understand the situation
- Contributes in some way to understanding what is happening in the story

Some relationships may still be acceptable even if they are not central to the plot, as long as they meaningfully describe the story context.

A relationship is likely NOT narratively relevant if it:
- States a very generic or universal fact unrelated to the story
- Uses vague placeholder entities ("thing", "something", "certain") in a way that adds little meaning
- Describes background knowledge that does not relate to the current narrative
- Creates a connection that does not help explain the story or its context

Examples of less relevant triples:
- (year, is_given_in, good) — just states when payment occurred, not why it matters
- (certain, describes, good) — "certain" is a determiner, not a story-relevant property
- (tribe, pays, good) — this one might be relevant IF the payment is important to the plot
- (year, is_unit_of, time)
- (king, is, man)

Examples of narratively relevant triples:
- (conquered tribes, pay tribute to, king)
- (Charlemagne, protects, kingdom)
- (fierce Asiatic tribes, threaten, kingdom)
- (sons, learn to ride from, father)

Respond with a JSON object:
{{
    "relevant": true/false,
    "reason": "Brief explanation of why this relationship is or is not useful for understanding the story",
    "suggested_replacement": "If the triple could be slightly rephrased to better reflect the narrative, suggest an improved (subject, relation, object). Otherwise null."
}}
"""

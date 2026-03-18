OPEN_AI_API_KEY = ""
OPEN_AI_ORGANIZATION = ""
MAX_DISTANCE_FROM_ACTIVE_NODES = 2
MAX_NEW_CONCEPTS = 3
MAX_NEW_PROPERTIES = 3
CONTEXT_LENGTH = 1 # Experimental parameter to include more sentences as a "known" text. Should be equal to 1 to replicate the behaviour specified in the paper.
EDGE_FORGET = 2
NR_RELEVANT_EDGES = 15
DEBUG = False

# install openai library
!pip install openai
!python -m spacy download en_core_web_lg

# imports
from typing import List, Tuple, Dict, Set, Union, Optional
from enum import Enum
from openai.embeddings_utils import get_embedding, cosine_similarity
from collections import deque
import spacy
from spacy.tokens import Token, Doc, Span
import logging
import time
import openai

class NodeType(Enum):
    CONCEPT = 1
    PROPERTY = 2

class NodeSource(Enum):
    TEXT_BASED = 1
    INFERENCE_BASED = 2

class Node:

    def __init__(self, lemmas: List[str], actual_text: str, node_type: NodeType,
                 node_source: NodeSource, score: int) -> None:
        self.lemmas: List[str] = lemmas
        self.actual_texts: Dict[str, int] = {actual_text: 1}
        self.node_type: NodeType = node_type
        self.node_source: NodeSource = node_source
        self.score = score
        self.edges: List[Edge] = []

    def __eq__(self, other: 'Node') -> bool:
        return self.lemmas == other.lemmas

    def __hash__(self) -> int:
        return hash(tuple(self.lemmas))

    def add_actual_text(self, actual_text: str) -> None:
        if actual_text in self.actual_texts:
            self.actual_texts[actual_text] += 1
        else:
            self.actual_texts[actual_text] = 1

    def get_text_representer(self) -> str:
        # return max count from actual_texts
        return max(self.actual_texts, key=self.actual_texts.get) # type: ignore

    def __str__(self) -> str:
        return f"{self.get_text_representer()} ({self.node_type.name}, \
                    {self.node_source.name}, {self.score})"

    def __repr__(self) -> str:
        return str(self)

class Edge:

    def __init__(self, source_node: Node, dest_node: Node, label: str,
                 forget_score: int, active: bool = True) -> None:
        self.source_node: Node = source_node
        self.dest_node: Node = dest_node
        self.active: bool = active
        self.label: str = label
        self.forget_score: int = forget_score
        self.embedding = get_embedding(label, engine='text-embedding-ada-002')
        self.similarity_threshold = 0.8

    def fade_away(self) -> None:
        self.forget_score -= 1
        if self.forget_score <= 0:
            self.active = False

    def is_similar(self, other_edge: 'Edge') -> bool:
        return cosine_similarity(self.embedding, other_edge.embedding) > self.similarity_threshold

    def __eq__(self, other: 'Edge') -> bool:
        return self.source_node == other.source_node and self.dest_node == other.dest_node and self.label == other.label

    def __hash__(self) -> int:
        return hash((self.source_node, self.dest_node, self.label))

    def __str__(self) -> str:
        return f"{self.source_node.get_text_representer()} --{self.label} ({'active' if self.active else 'inactive'})--> {self.dest_node.get_text_representer()} ({self.forget_score})"

    def __repr__(self) -> str:
        return self.__str__()

class Graph:

    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

    def add_or_get_node(self, lemmas: List[str], actual_text: str,
                        node_type: NodeType, node_source: NodeSource) -> Node:
        node = self.get_node(lemmas)
        if node is None:
            node = Node(lemmas, actual_text, node_type, node_source, 0)
            self.nodes.add(node)
        else:
            node.add_actual_text(actual_text)
        return node

    def get_node(self, lemmas: List[str]) -> Optional[Node]:
        for node in self.nodes:
            if node.lemmas == lemmas:
                return node
        return None

    def get_edge_by_nodes_and_label(self, source_node: Node, dest_node: Node, label: str) -> Optional[Edge]:
        for edge in self.edges:
            if edge.source_node == source_node and edge.dest_node == dest_node and edge.label == label:
                return edge
        return None

    def get_edge(self, edge: Edge) -> Optional[Edge]:
        for other_edge in self.edges:
            if edge == other_edge:
                return other_edge
        return None

    def add_edge(self, source_node: Node, dest_node: Node, label: str, edge_forget: int) -> Optional[Edge]:
        edge = Edge(source_node, dest_node, label, edge_forget)
        if self.check_if_similar_edge_exists(edge, edge_forget):
            return None
        self.edges.add(edge)
        source_node.edges.append(edge)
        dest_node.edges.append(edge)
        return edge

    def check_if_similar_edge_exists(self, edge: Edge, edge_forget: int) -> bool:
        if edge in self.edges:
            self.get_edge(edge).forget_score = edge_forget
            self.get_edge(edge).active = True
            return True
        for other_edge in self.edges:
            if edge.source_node == other_edge.source_node and edge.dest_node == other_edge.dest_node:
                if edge.source_node.node_type == NodeType.CONCEPT and edge.dest_node.node_type == NodeType.PROPERTY:
                    other_edge.forget_score = edge_forget
                    other_edge.active = True
                    return True # this edge represents an edge that connects a concept node with a property node
                if edge.dest_node.node_type == NodeType.CONCEPT and edge.source_node.node_type == NodeType.PROPERTY:
                    other_edge.forget_score = edge_forget
                    other_edge.active = True
                    return True # same as above
                if edge.is_similar(other_edge):
                    # update edge type if necessary
                    other_edge.forget_score = edge_forget
                    other_edge.active = True
                    return True
        return False

    def bfs_from_activated_nodes(self, activated_nodes: List[Node]) -> Dict[Node, int]:
        distances = {}
        queue = deque([(node, 0) for node in activated_nodes])
        while queue:
            curr_node, curr_distance = queue.popleft()
            if curr_node not in distances:
                distances[curr_node] = curr_distance
                for edge in curr_node.edges:
                    if edge.active:
                        next_node = edge.dest_node if edge.source_node == curr_node else edge.source_node
                        queue.append((next_node, curr_distance + 1)) # type: ignore
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(self, activated_nodes: List[Node]) -> None:
        distances_to_activated_nodes = self.bfs_from_activated_nodes(activated_nodes)
        for node in self.nodes:
            node.score = distances_to_activated_nodes.get(node, 100)

    def get_word_lemma_score(self, word_lemmas: List[str]) -> Optional[float]:
        for node in self.nodes:
            if node.lemmas == word_lemmas:
                return node.score
        return None

    def get_top_k_nodes(self, nodes: List[Node], k: int) -> List[Node]:
        return sorted(nodes, key=lambda node: node.score)[:k]

    def get_top_concepts_nodes(self, k: int) -> List[Node]:
        nodes = [node for node in self.nodes if node.node_type == NodeType.CONCEPT]
        return self.get_top_k_nodes(nodes, k)

    def get_top_text_based_concepts(self, k: int) -> List[Node]:
        nodes = [node for node in self.nodes if node.node_type == NodeType.CONCEPT and node.node_source == NodeSource.TEXT_BASED]
        return self.get_top_k_nodes(nodes, k)

    def get_active_nodes(self, score_threshold: int, only_text_based: bool = False) -> List[Node]:
        return [node for node in self.nodes if node.score <= score_threshold and (not only_text_based or node.node_source == NodeSource.TEXT_BASED)]

    def get_nodes_str(self, nodes: List[Node]) -> str:
        nodes_str = ""
        for node in sorted(nodes, key=lambda node: node.score):
            nodes_str += "- " + f"{node.get_text_representer()} (type: {node.node_type}) (score: {node.score})" + "\n"
        return nodes_str

    def get_edges_str(self, nodes: List[Node], only_text_based: bool = False, only_active: bool = True) -> Tuple[str, List[Edge]]:
        used_edges = set()
        edges_str = ""
        count = 1
        for node in sorted(nodes, key=lambda node: node.score):
            for edge in node.edges:
                if only_active and edge.active == False:
                    continue
                if edge not in used_edges:
                    if not only_text_based:
                        edges_str += f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}" + "\n"
                        used_edges.add(edge)
                        count += 1
                    else:
                        if edge.source_node.node_source == NodeSource.TEXT_BASED and edge.dest_node.node_source == NodeSource.TEXT_BASED:
                            edges_str += f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}" + "\n"
                            used_edges.add(edge)
                            count += 1
        return edges_str, list(used_edges)

    def get_active_graph_repr(self) -> str:
        edges = [edge for edge in self.edges if edge.active]
        nodes = set()
        for edge in edges:
            nodes.add(edge.source_node)
            nodes.add(edge.dest_node)
        s = "nodes:\n"
        for node in nodes:
            s += str(node) + "\n"
        s += "\nedges:\n"
        for edge in edges:
            s += str(edge) + "\n"
        return s

    def __str__(self) -> str:
        return "nodes: {}\n\nedges: {}".format("\n".join([str(x) for x in self.nodes]), "\n".join([str(x) for x in self.edges]))

    def __repr__(self) -> str:
        return self.__str__()

NEW_RELATIONSHIPS_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

I also have the knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

I want you to tell me the relationships (edges) between the nodes from the text themselves. And also between the nodes from the text and the other nodes from the graph (here prioritize the relationships based on the score). The text is:
{text}

List them as a Python list and do not provide additional explanation."""

NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text and these ones you should use:
{nodes_from_text}

I want you to tell me the relationships (edges) between the nodes given the text. The text is:
{text}

List them as a Pyhon list and do not provide additional explanation."""



INFER_OBJECTS_AND_PROPERTIES_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

I also have the current knowledge graph:
Nodes (with their types, and a score of how central they are in the story (0 is most central, 1 less central, etc.)):
{nodes_from_graph}

Edges (relationships between the nodes):
{edges_from_graph}

Generate a list of concepts and properties that are in line with the overall coherence and sense within the given text and the knowledge graph, but they are not in the text. The text is:
{text}

List them in the following format and explain the role of the text and the knowledge graph in the decesion making process:
{{
    "concepts": [concept1, concept2, ...],
    "properties": [property1, property2, ...]
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

Create relationships between the nodes from the text and the new concepts and the new properties (as you see fit). The text is:
{text}

Provide them in the following format:
{{
    "concept_relationships": [concept_relation1, concept_relation2, ...],
    "property_relationships": [property_relation1, property_relation2, ...]
}}"""


INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

Generate a list of concepts and properties that are in line with the overall coherence and sense within the given text, but they are not in the text. The text is:
{text}

List them in the following format:
{{
    "concepts": [concept1, concept2, ...],
    "properties": [property1, property2, ...]
}}"""

GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT = """I want to build a knowledge graph using the provided text. The graph should consist of two types of nodes: concept nodes and property nodes. Concepts nodes represent objects or persons from the story and are generally represented by nouns in the text. Property nodes describe the concepts nodes and are generally represented by adjectives in the text. An edge connects a concept to another concept or a concept to a property, and it is described by a relationship between the connected nodes.

The format for representing the graph is as follows: ('concept1', 'relation (edge)', 'concept2') or ('concept1', 'relation (edge)', 'property1').

I already extracted the nodes from the text:
{nodes_from_text}

And I already have additional concepts and properties that are in line with the overall coherence and sense within the given text:
concepts: {concepts}
properties: {properties}

Create relationships between the nodes from the text and the new concepts and the new properties (as you see fit). The text is:
{text}

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

class OpenAIChatGPT:

    def __init__(self, debug=False):
        openai.organization = OPEN_AI_ORGANIZATION
        openai.api_key = OPEN_AI_API_KEY
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def read_open_sk_key(self):
        with open('openai.sk') as f:
            return f.read().strip()

    def call_chatgpt(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[
                        {"role": "user", "content": prompt},
                    ],
                temperature=0.8
                )
            if self.debug:
                logging.debug("\n\n" + str(response) + "\n\n")
            return response['choices'][0]['message']['content']
        except:
            time.sleep(10)
            return self.call_chatgpt(prompt)

    def parse_for_dict(self, sentence_word_connections_response):
        try:
            start_index = sentence_word_connections_response.index("{")
            num_braces = 1
            end_index = None
            for i in range(start_index + 1, len(sentence_word_connections_response)):
                if sentence_word_connections_response[i] == "{":
                    num_braces += 1
                elif sentence_word_connections_response[i] == "}":
                    num_braces -= 1
                    if num_braces == 0:
                        end_index = i
                        break
            if end_index is not None:
                json_str = sentence_word_connections_response[start_index:end_index + 1]
                json_obj = eval(json_str)
                return json_obj
            else:
                raise ValueError("No json object found in response")
        except:
            print("No json object found in response")
            return None

    def extract_list_from_string(self, string):
        # Find the start and end indices of the list in the string
        start_index = string.find('[')
        end_index = string.rfind(']')

        # Extract the substring containing the list
        list_string = string[start_index:end_index+1]

        # Use eval to convert the list string to a Python list
        try:
            result = eval(list_string)
            if isinstance(result, list):
                return result
            else:
                logging.error("No list found in the string.")
                return []
        except SyntaxError:
            logging.error("No list found in the string.")
            return []

    def get_new_relationships(self, nodes_from_text, nodes_from_graph, edges_from_graph, text):
        prompt = NEW_RELATIONSHIPS_PROMPT.format(nodes_from_text=nodes_from_text, nodes_from_graph=nodes_from_graph, edges_from_graph=edges_from_graph, text=text)
        response = self.call_chatgpt(prompt)
        return self.extract_list_from_string(response)

    def get_new_relationships_first_sentence(self, nodes_from_text, text):
        prompt = NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT.format(nodes_from_text=nodes_from_text, text=text)
        response = self.call_chatgpt(prompt)
        return self.extract_list_from_string(response)

    def infer_objects_and_properties(self, nodes_from_text, nodes_from_graph, edges_from_graph, text):
        prompt = INFER_OBJECTS_AND_PROPERTIES_PROMPT.format(nodes_from_text=nodes_from_text, nodes_from_graph=nodes_from_graph, edges_from_graph=edges_from_graph, text=text)
        response = self.call_chatgpt(prompt)
        return self.parse_for_dict(response)

    def generate_new_inferred_relationships(self, nodes_from_text, nodes_from_graph, edges_from_graph, concepts, properties, text):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT.format(nodes_from_text=nodes_from_text, nodes_from_graph=nodes_from_graph, edges_from_graph=edges_from_graph, concepts=concepts, properties=properties, text=text)
        response = self.call_chatgpt(prompt)
        return self.parse_for_dict(response)

    def infer_objects_and_properties_first_sentence(self, nodes_from_text, text):
        prompt = INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT.format(nodes_from_text=nodes_from_text, text=text)
        response = self.call_chatgpt(prompt)
        return self.parse_for_dict(response)

    def generate_new_inferred_relationships_first_sentence(self, nodes_from_text, concepts, properties, text):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT.format(nodes_from_text=nodes_from_text, concepts=concepts, properties=properties, text=text)
        response = self.call_chatgpt(prompt)
        return self.parse_for_dict(response)

    def get_relevant_edges(self, edges_from_graph, text):
        prompt = SELECT_RELEVANT_EDGES_PROMPT.format(edges=edges_from_graph, text=text)
        response = self.call_chatgpt(prompt)
        return self.extract_list_from_string(response)

class AMoCv4:
    def __init__(self, max_distance_from_active_nodes: int, max_new_concepts: int, max_new_properties: int,
                 context_length: int, edge_forget: int, nr_relevant_edges: int,
                 debug: bool = False) -> None:
        """
        max_distance_from_active_nodes: maximum distance in terms of hops from activated nodes to consider
        max_new_concepts: maximum number of new concepts that can be inferred
        max_new_properties: maximum number of new properties that can be inferred
        debug: enable debug mode
        """
        self.max_distance_from_active_nodes = max_distance_from_active_nodes
        self.max_new_concepts = max_new_concepts
        self.max_new_properties = max_new_properties
        self.context_length = context_length
        self.edge_forget = edge_forget
        self.nr_relevant_edges = nr_relevant_edges
        self.graph = Graph()
        self.spacy_nlp = spacy.load('en_core_web_lg')
        self.chatgpt = OpenAIChatGPT(debug=debug)
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        logging.info("Hi! I am AMOC Star Plus.")

    def reset_graph(self) -> None:
        self.graph = Graph()

    def resolve_pronouns(self, text: str) -> str:
        prompt = REPLACE_PRONOUNS_PROMPT + text
        return self.chatgpt.call_chatgpt(prompt)

    def analyze(self, text: str, replace_prononuns: bool):
        if replace_prononuns:
          text = self.resolve_pronouns(text)
        doc = self.spacy_nlp(text)
        prev_sentences = []
        current_sentence = ""
        for i, sent in enumerate(doc.sents):
            logging.info("Processing sentence %d: %s" % (i, sent))
            if i == 0:
                current_sentence = sent
                prev_sentences.append(sent)
                self.init_graph(sent)

                inferred_concept_relationships, inferred_property_relationships = \
                        self.infer_new_relationships_step_0(sent)
                print("Inferred concept relationships:\n{inferred_concept_relationships}")
                print("Inferred property relationships:\n{inferred_property_relationships}")
                self.add_inferred_relationships_to_graph_step_0(inferred_concept_relationships, NodeType.CONCEPT, sent)
                self.add_inferred_relationships_to_graph_step_0(inferred_property_relationships, NodeType.PROPERTY, sent)
            else:
                added_edges = []
                current_sentence = sent
                prev_sentences.append(sent)
                if len(prev_sentences) > self.context_length:
                    prev_sentences.pop(0)
                # from current sentence get all the text based nodes
                current_sentence_text_based_nodes, current_sentence_text_based_words = \
                                            self.get_senteces_text_based_nodes([current_sentence], create_unexistent_nodes=True)

                # get new relationships with chatgpt
                current_all_text = sent.text
                graph_active_nodes = self.graph.get_active_nodes(self.max_distance_from_active_nodes)
                active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
                active_nodes_edges_text, _ = self.graph.get_edges_str(graph_active_nodes)

                only_text_based_graph_active_nodes = self.graph.get_active_nodes(self.max_distance_from_active_nodes, only_text_based=True)
                only_text_based_active_nodes_text = self.graph.get_nodes_str(only_text_based_graph_active_nodes)
                only_text_based_active_nodes_edges_text, _ = self.graph.get_edges_str(only_text_based_graph_active_nodes, only_text_based=True)

                new_relationships = self.get_new_relationships(current_all_text,
                                                       current_sentence_text_based_nodes,
                                                       current_sentence_text_based_words,
                                                       active_nodes_text,
                                                       active_nodes_edges_text)
                text_based_activated_nodes = current_sentence_text_based_nodes
                for relationship in new_relationships:
                    if len(relationship) != 3:
                        continue
                    if not relationship[0] or not relationship[2]:
                        continue
                    if relationship[0] == relationship[2]:
                        continue
                    if not isinstance(relationship[0], str) or not isinstance(relationship[2], str):
                        continue
                    source_node = self.get_node_from_new_relationship(relationship[0], graph_active_nodes,
                                                            current_sentence_text_based_nodes,
                                                            current_sentence_text_based_words,
                                                            node_source=NodeSource.TEXT_BASED,
                                                            create_node=True)
                    dest_node = self.get_node_from_new_relationship(relationship[2], graph_active_nodes,
                                                            current_sentence_text_based_nodes,
                                                            current_sentence_text_based_words,
                                                            node_source=NodeSource.TEXT_BASED,
                                                            create_node=True)
                    edge_label = relationship[1].replace("(edge)", "").strip()
                    if source_node is None or dest_node is None:
                        continue

                    print("New text relationship: ", relationship)
                    # update to text based if needed
                    if relationship[0] in current_sentence_text_based_words:
                        source_node.node_source = NodeSource.TEXT_BASED
                    if relationship[2] in current_sentence_text_based_words:
                        dest_node.node_source = NodeSource.TEXT_BASED

                    # create edge
                    potential_new_edge = self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)
                    if potential_new_edge:
                        added_edges.append(potential_new_edge)


                # infer new relationships

                # and then infer
                current_all_text = sent.text
                inferred_concept_relationships, inferred_property_relationships = \
                    self.infer_new_relationships(current_all_text,
                                                        current_sentence_text_based_nodes,
                                                        current_sentence_text_based_words,
                                                        only_text_based_active_nodes_text,
                                                        only_text_based_active_nodes_edges_text
                                                        )

                print("Inferred concept relationships: ", inferred_concept_relationships)
                print("Inferred property relationships: ", inferred_property_relationships)
                # add inferred relationships to graph
                self.add_inferred_relationships_to_graph(inferred_concept_relationships, NodeType.CONCEPT,
                                                            current_sentence_text_based_nodes,
                                                            current_sentence_text_based_words,
                                                            graph_active_nodes,
                                                            added_edges)
                self.add_inferred_relationships_to_graph(inferred_property_relationships, NodeType.PROPERTY,
                                                            current_sentence_text_based_nodes,
                                                            current_sentence_text_based_words,
                                                            graph_active_nodes,
                                                            added_edges)

                # update graph through active nodes
                self.graph.set_nodes_score_based_on_distance_from_active_nodes(text_based_activated_nodes)

                self.reactivate_relevant_edges(self.graph.get_active_nodes(self.max_distance_from_active_nodes), " ".join([s.text for s in prev_sentences]), added_edges)

                self.graph.set_nodes_score_based_on_distance_from_active_nodes(text_based_activated_nodes)


            print("All graph:\n\n")
            print(self.graph)
            print("\n\n")
            print("=====================================================================================================")
            print("\n\n")
            print("Active graph:\n\n")
            print(self.graph.get_active_graph_repr())
            print("\n\n")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("\n\n\n\n")


    def infer_new_relationships_step_0(self, sent: Span) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        # get all the text based nodes
        current_sentence_text_based_nodes, current_sentence_text_based_words = \
                                    self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"

        for _ in range(3):
            try:
                object_properties_dict = self.chatgpt.infer_objects_and_properties_first_sentence(nodes_from_text, sent.text)
                break
            except:
                continue
        else:
            return [], []

        for _ in range(3):
            try:
                new_relationships = self.chatgpt.generate_new_inferred_relationships_first_sentence(nodes_from_text,
                                                            object_properties_dict["concepts"][:self.max_new_concepts],
                                                            object_properties_dict["properties"][:self.max_new_properties],
                                                            sent.text)
                return new_relationships["concept_relationships"], new_relationships["property_relationships"]
            except:
                continue
        return [], []

    def infer_new_relationships(self, text: str,
                                current_sentence_text_based_nodes: List[Node],
                                current_sentence_text_based_words: List[str],
                                graph_nodes_representation: str,
                                graph_edges_representation: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"

        for _ in range(3):
            try:
                object_properties_dict = self.chatgpt.infer_objects_and_properties(nodes_from_text, graph_nodes_representation,
                                                                                   graph_edges_representation, text)
                break
            except:
                continue

        for _ in range(3):
            try:
                new_relationships = self.chatgpt.generate_new_inferred_relationships(nodes_from_text, graph_nodes_representation,
                                                                                     graph_edges_representation,
                                                        object_properties_dict["concepts"][:self.max_new_concepts],
                                                        object_properties_dict["properties"][:self.max_new_properties],
                                                        text)
                return new_relationships["concept_relationships"], new_relationships["property_relationships"]
            except:
                continue
        return [], []

    def reactivate_relevant_edges(self, active_nodes: List[Node], prev_sentences_text: str, newly_added_edges: List[Edge]) -> None:
        edges_text, edges = self.graph.get_edges_str(self.graph.nodes, only_active=False)
        relevant_edges_index = self.chatgpt.get_relevant_edges(edges_text, prev_sentences_text)[:self.nr_relevant_edges]
        for i in relevant_edges_index:
            print("Reactivating edge: ", edges[i-1])
            edges[i-1].forget_score = self.edge_forget
            edges[i-1].active = True
        for j in range(1, len(edges)+1):
            if j not in relevant_edges_index and edges[j-1] not in newly_added_edges:
                print("Fading away: ", edges[j-1])
                edges[j-1].fade_away()

    def init_graph(self, sent: Span) -> None:
        # get all the text based nodes
        current_sentence_text_based_nodes, current_sentence_text_based_words = \
                                    self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=True)

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"

        relationships = self.chatgpt.get_new_relationships_first_sentence(nodes_from_text, sent.text)
        print(f"First sentence edges:\n{relationships}")

        for relationship in relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(relationship[2], str):
                continue
            source_node = self.get_node_from_text(relationship[0],
                                                    current_sentence_text_based_nodes,
                                                    current_sentence_text_based_words,
                                                    node_source=NodeSource.TEXT_BASED,
                                                    create_node=True)
            dest_node = self.get_node_from_text(relationship[2],
                                                    current_sentence_text_based_nodes,
                                                    current_sentence_text_based_words,
                                                    node_source=NodeSource.TEXT_BASED,
                                                    create_node=True)
            edge_label = relationship[1].replace("(edge)", "").strip()
            if source_node is None or dest_node is None:
                continue

            # create edge
            self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)

    def get_new_relationships(self, text,
                                    current_sentence_text_based_nodes: List[Node],
                                    current_sentence_text_based_words: List[str],
                                    graph_nodes_representation: str,
                                    graph_edges_representation: str) -> List[Tuple[str, str, str]]:
        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"

        relationships = self.chatgpt.get_new_relationships(nodes_from_text, graph_nodes_representation, graph_edges_representation, text)
        return relationships

    def add_inferred_relationships_to_graph_step_0(self, inferred_relationships: List[Tuple[str, str, str]], node_type: NodeType,
                                                   sent: Span) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = \
                                    self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        for relationship in inferred_relationships:
            print(relationship)
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(relationship[2], str):
                continue
            source_node = self.get_node_from_text(relationship[0],
                                                    current_sentence_text_based_nodes,
                                                    current_sentence_text_based_words,
                                                    node_source=NodeSource.INFERENCE_BASED,
                                                    create_node=False)
            dest_node = self.get_node_from_text(relationship[2],
                                                    current_sentence_text_based_nodes,
                                                    current_sentence_text_based_words,
                                                    node_source=NodeSource.INFERENCE_BASED,
                                                    create_node=False)
            edge_label = relationship[1].replace("(edge)", "").strip()
            if source_node is None:
                source_node = self.graph.add_or_get_node(self.get_concept_lemmas(relationship[0]), relationship[0], node_type, NodeSource.INFERENCE_BASED)

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(self.get_concept_lemmas(relationship[2]), relationship[2], node_type, NodeSource.INFERENCE_BASED)

            self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)

    def add_inferred_relationships_to_graph(self, inferred_relationships: List[Tuple[str, str, str]], node_type: NodeType,
                                                curr_sentences_nodes: List[Node],
                                                curr_sentences_words: List[str],
                                                active_graph_nodes: List[Node],
                                                added_edges: List[Edge]) -> None:
        for relationship in inferred_relationships:
            print(relationship)
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(relationship[2], str):
                continue
            source_node = self.get_node_from_new_relationship(relationship[0], active_graph_nodes,
                                                    curr_sentences_nodes,
                                                    curr_sentences_words,
                                                    node_source=NodeSource.INFERENCE_BASED,
                                                    create_node=False)
            dest_node = self.get_node_from_new_relationship(relationship[2], active_graph_nodes,
                                                    curr_sentences_nodes,
                                                    curr_sentences_words,
                                                    node_source=NodeSource.INFERENCE_BASED,
                                                    create_node=False)
            edge_label = relationship[1].replace("(edge)", "").strip()
            if source_node is None:
                source_node = self.graph.add_or_get_node(self.get_concept_lemmas(relationship[0]), relationship[0], node_type, NodeSource.INFERENCE_BASED)

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(self.get_concept_lemmas(relationship[2]), relationship[2], node_type, NodeSource.INFERENCE_BASED)

            # create edge
            potential_edge = self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)
            if potential_edge:
                added_edges.append(potential_edge)

    def get_node_from_text(self, text: str,
                                curr_sentences_nodes: List[Node],
                                curr_sentences_words: List[str],
                                node_source: NodeSource,
                                create_node: bool) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        if create_node:
            lemmas = self.get_concept_lemmas(text)
            if self.has_noun(text):
                new_node = self.graph.add_or_get_node(lemmas, text, NodeType.CONCEPT, node_source)
            else:
                new_node = self.graph.add_or_get_node(lemmas, text, NodeType.PROPERTY, node_source)
            return new_node
        return None

    def get_node_from_new_relationship(self, text: str, graph_active_nodes: List[Node],
                                curr_sentences_nodes: List[Node],
                                curr_sentences_words: List[str],
                                node_source: NodeSource,
                                create_node: bool) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        else:
            lemmas = self.get_concept_lemmas(text)
            for node in graph_active_nodes:
                if lemmas == node.lemmas:
                    return node
        if create_node:
            lemmas = self.get_concept_lemmas(text)
            if self.has_noun(text):
                new_node = self.graph.add_or_get_node(lemmas, text, NodeType.CONCEPT, node_source)
            else:
                new_node = self.graph.add_or_get_node(lemmas, text, NodeType.PROPERTY, node_source)
            return new_node
        return None

    def is_content_word_and_non_stopword(self, token: Token, pos_list: List[str] = ["NOUN", "PROPN", "ADJ",]) -> bool:
        #  "VERB", "ADV" not included by default
        return (token.pos_ in pos_list) and (token.lemma_ not in self.spacy_nlp.Defaults.stop_words)

    def get_content_words_from_sent(self, sent: Span) -> List[Token]:
        return [token for token in sent if self.is_content_word_and_non_stopword(token)]

    def get_concept_lemmas(self, concept: str):
        doc = self.spacy_nlp(concept)
        return [token.lemma_ for token in doc]

    def has_noun(self, text: str) -> bool:
        span = self.spacy_nlp(text)
        for token in span:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                return True
        return False

    def get_sentences_concept_nodes(self, previous_sentences: List[Span]) -> List[Node]:
        concept_nodes = []
        for sent in previous_sentences:
            content_words = self.get_content_words_from_sent(sent)
            for word in content_words:
                node = self.graph.get_node(word.lemma_)
                if node is not None and node.node_type == NodeType.CONCEPT:
                    concept_nodes.append(node)
        return concept_nodes

    def get_senteces_text_based_nodes(self, previous_sentences: List[Span],
                                      create_unexistent_nodes: bool = True) -> Tuple[List[Token], List[str]]:
        text_based_nodes = []
        text_based_words = []
        for sent in previous_sentences:
            content_words = self.get_content_words_from_sent(sent)
            for word in content_words:
                node = self.graph.get_node([word.lemma_])
                if node is not None: # node exists in the graph, don't care about the node type
                    node.add_actual_text(word.text)
                    text_based_nodes.append(node)
                    text_based_words.append(word.text)
                else: # node doesn't exist in the graph
                    if create_unexistent_nodes:
                        if word.pos_ == "ADJ":
                            new_node = self.graph.add_or_get_node([word.lemma_], word.text, NodeType.PROPERTY, NodeSource.TEXT_BASED)
                        else:
                            new_node = self.graph.add_or_get_node([word.lemma_], word.text, NodeType.CONCEPT, NodeSource.TEXT_BASED)
                        text_based_nodes.append(new_node)
                        text_based_words.append(word.text)
        return text_based_nodes, text_based_words

amoc = AMoCv4(max_distance_from_active_nodes=MAX_DISTANCE_FROM_ACTIVE_NODES,
              max_new_concepts=MAX_NEW_CONCEPTS,
              max_new_properties=MAX_NEW_PROPERTIES,
              context_length=CONTEXT_LENGTH,
              edge_forget=EDGE_FORGET,
              nr_relevant_edges=NR_RELEVANT_EDGES,
              debug=DEBUG)
text = "A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. The knight and the dragon fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. The princess married the knight."
amoc.analyze(text, replace_prononuns=False)

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"
import logging
import sys
import time
from typing import List, Tuple, Dict, Set, Optional, Union
from enum import Enum
from collections import deque
import spacy
from spacy.tokens import Token, Span
import glob
import csv
import pandas as pd
from enum import Enum
from collections import deque
import spacy
from vllm import LLM, SamplingParams
import argparse
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import glob
from scipy import stats
import seaborn as sns
import json
from datetime import datetime
import pandas as pd
import re
from typing import Dict, Any, List, Tuple


# Constants

MAX_DISTANCE_FROM_ACTIVE_NODES = 2
MAX_NEW_CONCEPTS = 3
MAX_NEW_PROPERTIES = 3
CONTEXT_LENGTH = 1
EDGE_FORGET = 2
NR_RELEVANT_EDGES = 15
DEBUG = False

# Configure Logging
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Global spaCy load – shared across all personas / AMoC instances
# ---------------------------------------------------------------------
try:
    import spacy

    try:
        GLOBAL_NLP = spacy.load("en_core_web_lg")
    except Exception:
        GLOBAL_NLP = spacy.load("en_core_web_sm")
    logging.info("Loaded spaCy model globally.")
except Exception as e:
    GLOBAL_NLP = None
    logging.error(f"Could not load spaCy model: {e}")


# --- CONFIGURATION FOR OUTPUT & INPUT (kept but unused by new runner) ---
# Root folders for data and outputs
INPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/balanced_dfs/normalized"
OUTPUT_DIR = (
    "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets"
)
OUTPUT_ANALYSIS_DIR = os.path.join(
    "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output", "amoc_analysis"
)
os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

VLLM_MODELS = {
    "qwen3:30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",
    "gpt-oss:120b": "openai/gpt-oss-120b",
}

# Story Text Constant
STORY_TEXT = "A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. The knight and the dragon fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. The princess married the knight."


# --- PROMPTS (Kept mostly original, Persona is injected via System Prompt) ---

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


# ---------------------------------------------------------------------
# AMoC LOGIC
# ---------------------------------------------------------------------


class NodeType(Enum):
    CONCEPT = 1
    PROPERTY = 2


class NodeSource(Enum):
    TEXT_BASED = 1
    INFERENCE_BASED = 2


class Node:
    def __init__(
        self,
        lemmas: List[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
        score: int,
    ) -> None:
        self.lemmas: List[str] = lemmas
        self.actual_texts: Dict[str, int] = {actual_text: 1}
        self.node_type: NodeType = node_type
        self.node_source: NodeSource = node_source
        self.score = score
        self.edges: List["Edge"] = []

    def __eq__(self, other: "Node") -> bool:
        return self.lemmas == other.lemmas

    def __hash__(self) -> int:
        return hash(tuple(self.lemmas))

    def add_actual_text(self, actual_text: str) -> None:
        if actual_text in self.actual_texts:
            self.actual_texts[actual_text] += 1
        else:
            self.actual_texts[actual_text] = 1

    def get_text_representer(self) -> str:
        return max(self.actual_texts, key=self.actual_texts.get)

    def __str__(self) -> str:
        return f"{self.get_text_representer()} ({self.node_type.name}, {self.node_source.name}, {self.score})"

    def __repr__(self) -> str:
        return str(self)


class Edge:
    def __init__(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        forget_score: int,
        active: bool = True,
    ) -> None:
        self.source_node: Node = source_node
        self.dest_node: Node = dest_node
        self.active: bool = active
        self.label: str = label
        self.forget_score: int = forget_score
        self.similarity_threshold = 0.8

    def fade_away(self) -> None:
        self.forget_score -= 1
        if self.forget_score <= 0:
            self.active = False

    def is_similar(self, other_edge: "Edge") -> bool:
        # Placeholder: strict equality for now to avoid loading a second model for embeddings
        return self.label == other_edge.label
        # return cosine_similarity(self.embedding, other_edge.embedding) > self.similarity_threshold

    def __eq__(self, other: "Edge") -> bool:
        return (
            self.source_node == other.source_node
            and self.dest_node == other.dest_node
            and self.label == other.label
        )

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

    def add_or_get_node(
        self,
        lemmas: List[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
    ) -> Node:
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

    def get_edge_by_nodes_and_label(
        self, source_node: Node, dest_node: Node, label: str
    ) -> Optional[Edge]:
        for edge in self.edges:
            if (
                edge.source_node == source_node
                and edge.dest_node == dest_node
                and edge.label == label
            ):
                return edge
        return None

    def get_edge(self, edge: Edge) -> Optional[Edge]:
        for other_edge in self.edges:
            if edge == other_edge:
                return other_edge
        return None

    def add_edge(
        self, source_node: Node, dest_node: Node, label: str, edge_forget: int
    ) -> Optional[Edge]:
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
            if (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
            ):
                if (
                    edge.source_node.node_type == NodeType.CONCEPT
                    and edge.dest_node.node_type == NodeType.PROPERTY
                ):
                    other_edge.forget_score = edge_forget
                    other_edge.active = True
                    return True
                if (
                    edge.dest_node.node_type == NodeType.CONCEPT
                    and edge.source_node.node_type == NodeType.PROPERTY
                ):
                    other_edge.forget_score = edge_forget
                    other_edge.active = True
                    return True
                if edge.is_similar(other_edge):
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
                        next_node = (
                            edge.dest_node
                            if edge.source_node == curr_node
                            else edge.source_node
                        )
                        queue.append((next_node, curr_distance + 1))
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List[Node]
    ) -> None:
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
        nodes = [
            node
            for node in self.nodes
            if node.node_type == NodeType.CONCEPT
            and node.node_source == NodeSource.TEXT_BASED
        ]
        return self.get_top_k_nodes(nodes, k)

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List[Node]:
        return [
            node
            for node in self.nodes
            if node.score <= score_threshold
            and (not only_text_based or node.node_source == NodeSource.TEXT_BASED)
        ]

    def get_nodes_str(self, nodes: List[Node]) -> str:
        nodes_str = ""
        for node in sorted(nodes, key=lambda node: node.score):
            nodes_str += (
                "- "
                + f"{node.get_text_representer()} (type: {node.node_type}) (score: {node.score})"
                + "\n"
            )
        return nodes_str

    def get_edges_str(
        self, nodes: List[Node], only_text_based: bool = False, only_active: bool = True
    ) -> Tuple[str, List[Edge]]:
        used_edges = set()
        edges_str = ""
        count = 1
        for node in sorted(nodes, key=lambda node: node.score):
            for edge in node.edges:
                if only_active and edge.active == False:
                    continue
                if edge not in used_edges:
                    if not only_text_based:
                        edges_str += (
                            f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}"
                            + "\n"
                        )
                        used_edges.add(edge)
                        count += 1
                    else:
                        if (
                            edge.source_node.node_source == NodeSource.TEXT_BASED
                            and edge.dest_node.node_source == NodeSource.TEXT_BASED
                        ):
                            edges_str += (
                                f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}"
                                + "\n"
                            )
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
        return "nodes: {}\n\nedges: {}".format(
            "\n".join([str(x) for x in self.nodes]),
            "\n".join([str(x) for x in self.edges]),
        )

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------
# VLLM CLIENT (from vllm_adapted_minimal.py)
# ---------------------------------------------------------------------


class VLLMClient:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        tp_size: int = 1,
        debug: bool = False,
    ):
        self.debug = debug
        self.model_name = model_name
        self.tp_size = tp_size
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        logging.info(f"Initializing vLLM with model: {model_name}, tp_size={tp_size}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.80,
            max_model_len=5000,
        )
        self.sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=1024
        )

        import torch

        num_gpus = torch.cuda.device_count()
        if tp_size > num_gpus:
            raise ValueError(
                f"Requested tensor parallel size {tp_size}, but only {num_gpus} GPUs are available."
            )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.llm is None:
            logging.error("VLLM not initialized.")
            return "[]"

        # Manual prompt template construction
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            prompt += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        # Force Start to 'final' channel (The "Anti-Loop" Fix)
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\nfinal\n"

        try:
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text

            # Split by 'final'
            if "final" in raw_text:
                result = raw_text.split("final")[-1]
            else:
                result = raw_text

            result = result.replace("assistant", "").replace("!", "").strip()

            if not result.startswith("[") and result.endswith("]"):
                result = "[" + result
            elif not result.startswith("{") and result.endswith("}"):
                result = "{" + result

            return result

        except Exception as e:
            logging.exception(f"VLLM runtime error: {e}")
            return "[]"

    def call_vllm(self, prompt: str, persona: str) -> str:
        # Inject persona into system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph builder and reasoning agent.\n"
                    "You must reason about the following persona and age, and extract factual relationships about them (and related concepts) only when they are supported by the text.\n\n"
                    f"Persona description:\n{persona}\n\n"
                    "Do not invent new attributes or relationships that are not supported by the persona description itself."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.generate(messages)

    # --- Helpers adapted from original class ---

    def parse_for_dict(self, sentence_word_connections_response):
        try:
            start_index = sentence_word_connections_response.find("{")
            end_index = sentence_word_connections_response.rfind("}")
            if start_index != -1 and end_index != -1:
                json_str = sentence_word_connections_response[
                    start_index : end_index + 1
                ]
                return eval(json_str)
            return None
        except:
            return None

    def extract_list_from_string(self, string):
        start_index = string.find("[")
        end_index = string.rfind("]")
        if start_index != -1 and end_index != -1:
            list_string = string[start_index : end_index + 1]
            try:
                result = eval(list_string)
                print(result)
                return result if isinstance(result, list) else []
            except:
                return []
        return []

    # --- Wrappers for AMoC Logic ---
    # Now accepting 'persona' arg passed down from AMoC class

    def get_new_relationships(
        self, nodes_from_text, nodes_from_graph, edges_from_graph, text, persona
    ):
        prompt = NEW_RELATIONSHIPS_PROMPT.format(
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return self.extract_list_from_string(response)

    def get_new_relationships_first_sentence(self, nodes_from_text, text, persona):
        prompt = NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return self.extract_list_from_string(response)

    def infer_objects_and_properties(
        self, nodes_from_text, nodes_from_graph, edges_from_graph, text, persona
    ):
        prompt = INFER_OBJECTS_AND_PROPERTIES_PROMPT.format(
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return self.parse_for_dict(response)

    def generate_new_inferred_relationships(
        self,
        nodes_from_text,
        nodes_from_graph,
        edges_from_graph,
        concepts,
        properties,
        text,
        persona,
    ):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT.format(
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            concepts=concepts,
            properties=properties,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return self.parse_for_dict(response)

    def infer_objects_and_properties_first_sentence(
        self, nodes_from_text, text, persona
    ):
        prompt = INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return self.parse_for_dict(response)

    def generate_new_inferred_relationships_first_sentence(
        self, nodes_from_text, concepts, properties, text, persona
    ):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text,
            concepts=concepts,
            properties=properties,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return self.parse_for_dict(response)

    def get_relevant_edges(self, edges_from_graph, text, persona):
        prompt = SELECT_RELEVANT_EDGES_PROMPT.format(edges=edges_from_graph, text=text)
        response = self.call_vllm(prompt, persona)
        return self.extract_list_from_string(response)

    def resolve_pronouns(self, text, persona):
        prompt = REPLACE_PRONOUNS_PROMPT + text
        return self.call_vllm(prompt, persona)


# ---------------------------------------------------------------------
# AMoCv4 – vLLM-adapted version from vllm_adapted_minimal.py
# ---------------------------------------------------------------------
# Global cache of VLLMClient instances per model
VLLM_CLIENT_CACHE: Dict[str, VLLMClient] = {}


class AMoCv4:
    def __init__(
        self,
        persona_description: str,
        vllm_client: VLLMClient,
        max_distance_from_active_nodes: int,
        max_new_concepts: int,
        max_new_properties: int,
        context_length: int,
        edge_forget: int,
        nr_relevant_edges: int,
        debug: bool = False,
        spacy_nlp=None,
    ) -> None:
        self.persona = persona_description
        self.client = vllm_client
        self.model_name = vllm_client.model_name
        self.max_distance_from_active_nodes = max_distance_from_active_nodes
        self.max_new_concepts = max_new_concepts
        self.max_new_properties = max_new_properties
        self.context_length = context_length
        self.edge_forget = edge_forget
        self.nr_relevant_edges = nr_relevant_edges
        self.graph = Graph()

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        if spacy_nlp is not None:
            self.spacy_nlp = spacy_nlp
        else:
            if GLOBAL_NLP is None:
                raise RuntimeError(
                    "AMoCv4 requires spaCy, but GLOBAL_NLP is None. "
                    "Make sure the global spaCy load at the top succeeded."
                )
            self.spacy_nlp = GLOBAL_NLP

    def reset_graph(self) -> None:
        self.graph = Graph()

    def resolve_pronouns(self, text: str) -> str:
        return self.client.resolve_pronouns(text, self.persona)

        # plus all the other initialization logic from your original AMoCv4:
        # - init_graph
        # - methods to infer relationships, properties, etc.

    def analyze(self, text: str, replace_prononuns: bool) -> List[Tuple[str, str, str]]:
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

                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships_step_0(sent)
                )

                self.add_inferred_relationships_to_graph_step_0(
                    inferred_concept_relationships, NodeType.CONCEPT, sent
                )
                self.add_inferred_relationships_to_graph_step_0(
                    inferred_property_relationships, NodeType.PROPERTY, sent
                )
            else:
                added_edges = []
                current_sentence = sent
                prev_sentences.append(sent)
                if len(prev_sentences) > self.context_length:
                    prev_sentences.pop(0)

                current_sentence_text_based_nodes, current_sentence_text_based_words = (
                    self.get_senteces_text_based_nodes(
                        [current_sentence], create_unexistent_nodes=True
                    )
                )

                current_all_text = sent.text
                graph_active_nodes = self.graph.get_active_nodes(
                    self.max_distance_from_active_nodes
                )
                active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
                active_nodes_edges_text, _ = self.graph.get_edges_str(
                    graph_active_nodes
                )

                nodes_from_text = ""
                for idx, node in enumerate(current_sentence_text_based_nodes):
                    nodes_from_text += f" - ({current_sentence_text_based_words[idx]}, {node.node_type})\n"

                # Fetch new relationships (CORRECTED ARGUMENT ORDER)
                # Signature: (nodes_from_text, nodes_from_graph, edges_from_graph, text, persona)
                new_relationships = self.client.get_new_relationships(
                    nodes_from_text,  # 1. Nodes from Text
                    active_nodes_text,  # 2. Nodes from Graph
                    active_nodes_edges_text,  # 3. Edges from Graph
                    current_all_text,  # 4. Text
                    self.persona,  # 5. Persona
                )

                text_based_activated_nodes = current_sentence_text_based_nodes
                for idx, relationship in enumerate(new_relationships):
                    # Skip None or scalar junk (int, float, bool, etc.)
                    if relationship is None or isinstance(
                        relationship, (int, float, bool)
                    ):
                        logging.error(
                            f"[AMoC] Skipping non-iterable relationship at {idx}: {relationship!r}"
                        )
                        continue

                    # If relationship is a dict, try to convert it
                    if isinstance(relationship, dict):
                        subj = relationship.get("subject") or relationship.get("head")
                        rel = relationship.get("relation") or relationship.get(
                            "predicate"
                        )
                        obj = relationship.get("object") or relationship.get("tail")
                        if not (subj and rel and obj):
                            logging.error(
                                f"[AMoC] Skipping malformed dict relationship at {idx}: {relationship!r}"
                            )
                            continue
                        relationship = (str(subj), str(rel), str(obj))

                    # Must be list/tuple from this point on
                    if not isinstance(relationship, (list, tuple)):
                        logging.error(
                            f"[AMoC] Skipping unexpected relationship type at {idx}: {type(relationship)} → {relationship!r}"
                        )
                        continue

                    # Must have exactly 3 elements
                    if len(relationship) != 3:
                        logging.error(
                            f"[AMoC] Skipping relationship with wrong length at {idx}: {relationship!r}"
                        )
                        continue

                    # Unpack
                    subj, rel, obj = relationship

                    # Validate subject/object strings
                    if not subj or not obj:
                        continue
                    if subj == obj:
                        continue
                    if not isinstance(subj, str) or not isinstance(obj, str):
                        continue

                    # Continue with your original code
                    source_node = self.get_node_from_new_relationship(
                        subj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )

                    dest_node = self.get_node_from_new_relationship(
                        relationship[2],
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )
                    edge_label = relationship[1].replace("(edge)", "").strip()
                    if source_node is None or dest_node is None:
                        continue

                    if relationship[0] in current_sentence_text_based_words:
                        source_node.node_source = NodeSource.TEXT_BASED
                    if relationship[2] in current_sentence_text_based_words:
                        dest_node.node_source = NodeSource.TEXT_BASED

                    potential_new_edge = self.graph.add_edge(
                        source_node, dest_node, edge_label, self.edge_forget
                    )
                    if potential_new_edge:
                        added_edges.append(potential_new_edge)

                # infer new relationships logic...
                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships(
                        current_all_text,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        self.graph.get_nodes_str(
                            self.graph.get_active_nodes(
                                self.max_distance_from_active_nodes,
                                only_text_based=True,
                            )
                        ),
                        self.graph.get_edges_str(
                            self.graph.get_active_nodes(
                                self.max_distance_from_active_nodes,
                                only_text_based=True,
                            ),
                            only_text_based=True,
                        )[0],
                    )
                )

                self.add_inferred_relationships_to_graph(
                    inferred_concept_relationships,
                    NodeType.CONCEPT,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    graph_active_nodes,
                    added_edges,
                )
                self.add_inferred_relationships_to_graph(
                    inferred_property_relationships,
                    NodeType.PROPERTY,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    graph_active_nodes,
                    added_edges,
                )

                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    text_based_activated_nodes
                )
                self.reactivate_relevant_edges(
                    self.graph.get_active_nodes(self.max_distance_from_active_nodes),
                    " ".join([s.text for s in prev_sentences]),
                    added_edges,
                )
                self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                    text_based_activated_nodes
                )

        # Return triplets for external saving
        return [
            (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            for edge in self.graph.edges
        ]

    def infer_new_relationships_step_0(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        for _ in range(3):
            try:
                object_properties_dict = (
                    self.client.infer_objects_and_properties_first_sentence(
                        nodes_from_text, sent.text, self.persona
                    )
                )
                break
            except:
                continue
        else:
            return [], []

        for _ in range(3):
            try:
                new_relationships = (
                    self.client.generate_new_inferred_relationships_first_sentence(
                        nodes_from_text,
                        object_properties_dict["concepts"][: self.max_new_concepts],
                        object_properties_dict["properties"][: self.max_new_properties],
                        sent.text,
                        self.persona,
                    )
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def infer_new_relationships(
        self,
        text: str,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        graph_nodes_representation: str,
        graph_edges_representation: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        for _ in range(3):
            try:
                object_properties_dict = self.client.infer_objects_and_properties(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    text,
                    self.persona,
                )
                break
            except:
                continue

        for _ in range(3):
            try:
                new_relationships = self.client.generate_new_inferred_relationships(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    object_properties_dict["concepts"][: self.max_new_concepts],
                    object_properties_dict["properties"][: self.max_new_properties],
                    text,
                    self.persona,
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def reactivate_relevant_edges(
        self,
        active_nodes: List[Node],
        prev_sentences_text: str,
        newly_added_edges: List[Edge],
    ) -> None:
        edges_text, edges = self.graph.get_edges_str(
            self.graph.nodes, only_active=False
        )
        relevant_edges_index = self.client.get_relevant_edges(
            edges_text, prev_sentences_text, self.persona
        )[: self.nr_relevant_edges]
        for i in relevant_edges_index:
            # print("Reactivating edge: ", edges[i-1]) # Reduced verbosity
            edges[i - 1].forget_score = self.edge_forget
            edges[i - 1].active = True
        for j in range(1, len(edges) + 1):
            if j not in relevant_edges_index and edges[j - 1] not in newly_added_edges:
                # print("Fading away: ", edges[j-1])
                edges[j - 1].fade_away()

    def init_graph(self, sent: Span) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=True)
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        relationships = self.client.get_new_relationships_first_sentence(
            nodes_from_text, sent.text, self.persona
        )
        # print(f"First sentence edges:\n{relationships}")

        for relationship in relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue
            source_node = self.get_node_from_text(
                relationship[0],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            dest_node = self.get_node_from_text(
                relationship[2],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if source_node is None or dest_node is None:
                continue
            self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )
        for relationship in inferred_relationships:
            # print(relationship)
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue
            source_node = self.get_node_from_text(
                relationship[0],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_text(
                relationship[2],
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if source_node is None:
                source_node = self.graph.add_or_get_node(
                    self.get_concept_lemmas(relationship[0]),
                    relationship[0],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(
                    self.get_concept_lemmas(relationship[2]),
                    relationship[2],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            self.graph.add_edge(source_node, dest_node, edge_label, self.edge_forget)

    def add_inferred_relationships_to_graph(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        active_graph_nodes: List[Node],
        added_edges: List[Edge],
    ) -> None:
        for relationship in inferred_relationships:
            # print(relationship)
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue
            source_node = self.get_node_from_new_relationship(
                relationship[0],
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_new_relationship(
                relationship[2],
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            if source_node is None:
                source_node = self.graph.add_or_get_node(
                    self.get_concept_lemmas(relationship[0]),
                    relationship[0],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            if dest_node is None:
                dest_node = self.graph.add_or_get_node(
                    self.get_concept_lemmas(relationship[2]),
                    relationship[2],
                    node_type,
                    NodeSource.INFERENCE_BASED,
                )

            potential_edge = self.graph.add_edge(
                source_node, dest_node, edge_label, self.edge_forget
            )
            if potential_edge:
                added_edges.append(potential_edge)

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        if create_node:
            lemmas = self.get_concept_lemmas(text)
            if self.has_noun(text):
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.CONCEPT, node_source
                )
            else:
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.PROPERTY, node_source
                )
            return new_node
        return None

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
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
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.CONCEPT, node_source
                )
            else:
                new_node = self.graph.add_or_get_node(
                    lemmas, text, NodeType.PROPERTY, node_source
                )
            return new_node
        return None

    def is_content_word_and_non_stopword(
        self,
        token: Token,
        pos_list: List[str] = [
            "NOUN",
            "PROPN",
            "ADJ",
        ],
    ) -> bool:
        return (token.pos_ in pos_list) and (
            token.lemma_ not in self.spacy_nlp.Defaults.stop_words
        )

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

    def get_senteces_text_based_nodes(
        self, previous_sentences: List[Span], create_unexistent_nodes: bool = True
    ) -> Tuple[List[Node], List[str]]:
        text_based_nodes = []
        text_based_words = []
        for sent in previous_sentences:
            content_words = self.get_content_words_from_sent(sent)
            for word in content_words:
                node = self.graph.get_node([word.lemma_])
                if node is not None:
                    node.add_actual_text(word.text)
                    text_based_nodes.append(node)
                    text_based_words.append(word.text)
                else:
                    if create_unexistent_nodes:
                        if word.pos_ == "ADJ":
                            new_node = self.graph.add_or_get_node(
                                [word.lemma_],
                                word.text,
                                NodeType.PROPERTY,
                                NodeSource.TEXT_BASED,
                            )
                        else:
                            new_node = self.graph.add_or_get_node(
                                [word.lemma_],
                                word.text,
                                NodeType.CONCEPT,
                                NodeSource.TEXT_BASED,
                            )
                        text_based_nodes.append(new_node)
                        text_based_words.append(word.text)
        return text_based_nodes, text_based_words


# ---------------------------------------------------------------------
# Batch helpers from original file (still present but not used by new CLI)
# ---------------------------------------------------------------------


class AgeAwareAMoCEngine:
    def __init__(self, vllm_client: VLLMClient):
        self.vllm_client = vllm_client

    def _build_analysis_text(self, persona_text: str, age_int: int) -> str:
        # This is the text that both AMoC and the LLM see as "persona"
        return f"Age: {age_int} years old.\n{persona_text}"

    def run(
        self, persona_text: str, age, replace_pronouns: bool
    ) -> List[Tuple[str, str, str]]:
        # 1) Normalize age to int
        try:
            age_int = int(age)
        except Exception:
            age_int = int(float(age)) if pd.notna(age) else -1

        # 2) Build age-aware persona description
        persona_description = self._build_analysis_text(persona_text, age_int)

        # 3) Instantiate AMoCv4 with:
        #    - this persona_description
        #    - the shared vLLM client
        #    - global AMoC hyperparameters
        #    - GLOBAL_NLP (no spaCy reload per persona!)
        amoc = AMoCv4(
            persona_description=persona_description,
            vllm_client=self.vllm_client,
            max_distance_from_active_nodes=MAX_DISTANCE_FROM_ACTIVE_NODES,
            max_new_concepts=MAX_NEW_CONCEPTS,
            max_new_properties=MAX_NEW_PROPERTIES,
            context_length=CONTEXT_LENGTH,
            edge_forget=EDGE_FORGET,
            nr_relevant_edges=NR_RELEVANT_EDGES,
            debug=DEBUG,
            spacy_nlp=GLOBAL_NLP,  # <--- this assumes you applied the earlier change
        )

        # 4) Analyze the fixed STORY_TEXT (or pass any story text you want here)
        triplets = amoc.analyze(STORY_TEXT, replace_prononuns=replace_pronouns)

        # 5) Ensure we return a list of (s, r, o) tuples
        #    (If analyze() already returns that, we just pass it through.)
        return list(triplets)


def process_persona_file(filename: str, vllm_client: VLLMClient):
    # Original batch logic – now superseded by the CSV runner below
    pass


def run_all_files(vllm_client: VLLMClient, files_to_process: List[str]):
    print(f"Starting Batch Analysis for {len(files_to_process)} files...")
    print(f"Output Directory: {OUTPUT_DIR}")
    total_start = time.time()
    for filename in files_to_process:
        process_persona_file(filename, vllm_client)
    print(f"\nAll processing complete in {time.time() - total_start:.2f} seconds.")


# ---------------------------------------------------------------------
# 4. APPLICATION LAYER – run over all CSVs in INPUT_DIR
# ---------------------------------------------------------------------


def robust_read_persona_csv(filename: str) -> pd.DataFrame:
    short_filename = os.path.basename(filename)

    try:
        df = pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"   [Info] UTF-8 failed for {short_filename}, trying 'cp1252'...")
        try:
            df = pd.read_csv(filename, encoding="cp1252")
        except Exception as e:
            print(f"   [Error] Could not read file: {e}")
            raise
    except Exception as e:
        # Fallback for parquet or other formats if passed
        try:
            if filename.endswith(".parquet"):
                df = pd.read_parquet(filename)
            elif filename.endswith(".pkl"):
                df = pd.read_pickle(filename)
            else:
                raise e
        except Exception as e2:
            print(f"   [Error] Could not read file structure: {e2}")
            raise

    # Normalize columns if needed
    if "persona" in df.columns and "persona_text" not in df.columns:
        df["persona_text"] = df["persona"]

    return df


# Build a model+file specific checkpoint path.


def get_checkpoint_path(
    output_dir: str,
    short_filename: str,
    model_name: str,
) -> str:
    safe_model_name = model_name.replace(":", "-").replace("/", "-")
    base_name = os.path.splitext(short_filename)[0]
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"ckpt_{safe_model_name}_{base_name}.json")


# Load checkpoint JSON if it exists, otherwise return empty structure


def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    if not os.path.isfile(ckpt_path):
        return {
            "personas_processed": 0,
            "processed_indices": [],
            "failures": [],
            "last_update": None,
        }
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # If something is wrong with the file, start fresh but don't crash.
        return {
            "personas_processed": 0,
            "processed_indices": [],
            "failures": [],
            "last_update": None,
        }


# Save checkpoint JSON. Overwrites previous.


def save_checkpoint(ckpt_path: str, ckpt: Dict[str, Any]) -> None:
    ckpt["last_update"] = datetime.utcnow().isoformat()
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp_path, ckpt_path)


def process_persona_csv(
    filename: str,
    model_names: List[str],
    max_rows: Optional[int] = None,
    replace_pronouns: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.8,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    output_dir: str = OUTPUT_DIR,
    resume_only: bool = False,
) -> None:
    short_filename = os.path.basename(filename)
    print(f"\n=== Processing File: {short_filename} ===")

    # 1. Load Data
    df = robust_read_persona_csv(filename)

    # Ensure required columns
    if "persona_text" not in df.columns or "age" not in df.columns:
        print(
            f"   [Skip] File {short_filename} missing 'persona_text' or 'age' columns."
        )
        return

    # Ensure valid age
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"])  # drop rows with invalid age

    # Apply max_rows limit if provided
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if df.empty:
        print(f"   [Skip] File {short_filename} has no valid rows after age filtering.")
        return

    # 2. Instantiate AgeAwareAMoCEngine engines (one per model)
    engines: Dict[str, AgeAwareAMoCEngine] = {}
    for model_name in model_names:
        if model_name in VLLM_CLIENT_CACHE:
            print(f"   [Info] Reusing existing VLLMClient for model: {model_name}")
            vllm_client = VLLM_CLIENT_CACHE[model_name]
        else:
            print(f"   [Info] Initializing NEW VLLMClient for model: {model_name}")
            tp_size = tensor_parallel_size
            vllm_client = VLLMClient(
                model_name=model_name,
                tp_size=tp_size,
                debug=DEBUG,
            )
            VLLM_CLIENT_CACHE[model_name] = vllm_client

        engines[model_name] = AgeAwareAMoCEngine(vllm_client)

    # 3. For each model, collect triplets into a table (incremental write)
    os.makedirs(output_dir, exist_ok=True)

    CONTROL_TOKENS = [
        "|eot_id|",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<s>",
        "</s>",
        "assistant",
        "user",
        "system",
    ]

    def is_bad(x: str) -> bool:
        """Check if any control tokens appear inside the string."""
        if not isinstance(x, str):
            return False
        return any(tok in x for tok in CONTROL_TOKENS)

    def repair_triplet(e1: str, e2: str, e3: str):
        """Repair corrupted triplets by replacing bad subject/object."""
        e1, e2, e3 = str(e1).strip(), str(e2).strip(), str(e3).strip()

        # Fix subject
        if is_bad(e1):
            e1 = e3 if not is_bad(e3) else "UNKNOWN"

        # Fix object
        if is_bad(e3):
            e3 = e1 if not is_bad(e1) else "UNKNOWN"

        # Fix relation (optional rule)
        if is_bad(e2) or e2 == "":
            e2 = "related_to"

        return e1, e2, e3

    # 3. For each model, collect triplets into a table (incremental write)
    os.makedirs(output_dir, exist_ok=True)

    CONTROL_TOKENS = [
        "|eot_id|",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<s>",
        "</s>",
        "assistant",
        "user",
        "system",
    ]

    # Common CSV header
    CSV_HEADERS = [
        "original_index",
        "age",
        "persona_text",
        "model_name",
        "subject",
        "relation",
        "object",
    ]

    for model_name, engine in engines.items():
        safe_model_name = model_name.replace(":", "-").replace("/", "-")
        output_filename = f"model_{safe_model_name}_triplets_{short_filename}"
        if not output_filename.lower().endswith(".csv"):
            output_filename += ".csv"
        output_path = os.path.join(output_dir, output_filename)

        # checkpoint path for this model+file
        ckpt_path = get_checkpoint_path(output_dir, short_filename, model_name)
        ckpt = load_checkpoint(ckpt_path)
        processed_indices = set(ckpt.get("processed_indices", []))
        failures = ckpt.get("failures", [])
        personas_processed = ckpt.get("personas_processed", 0)

        print(f"   [Model] {model_name}: writing to {output_path}")
        print(
            f"   [Model] {model_name}: checkpoint at {ckpt_path} "
            f"(already processed {personas_processed} personas; "
            f"{len(processed_indices)} indices)"
        )

        # Ensure CSV exists with header (once)
        if not os.path.isfile(output_path):
            pd.DataFrame([], columns=CSV_HEADERS).to_csv(
                output_path, index=False, encoding="utf-8"
            )
            print(f"   [Model] {model_name}: initialized empty CSV at {output_path}")

        all_extracted_data: List[Dict[str, Any]] = []
        start_model_time = time.time()

        try:
            for idx, (row_idx, row) in enumerate(df.iterrows(), start=1):
                # Skip if already processed in a previous run
                if resume_only:
                    # Only process rows that are NOT in the checkpoint
                    if row_idx in processed_indices:
                        # Already completed in a previous run → skip entirely
                        continue
                else:
                    # Normal mode → process everything
                    # But still skip already-processed rows to avoid double-writing
                    if row_idx in processed_indices:
                        continue

                persona_text = str(row["persona_text"])
                age = row["age"]

                try:
                    age_int = int(age)
                except Exception:
                    age_int = int(float(age)) if pd.notna(age) else -1

                print(
                    f"      [{idx}/{len(df)}] "
                    f"Age: {age_int} | Persona: {persona_text[:50]}..."
                )

                start_time = time.time()
                try:
                    triplets = engine.run(
                        persona_text=persona_text,
                        age=age_int,
                        replace_pronouns=replace_pronouns,
                    )

                    row_records: List[Dict[str, Any]] = []
                    for s, r, o in triplets:
                        s, r, o = repair_triplet(s, r, o)
                        rec = {
                            "original_index": row_idx,
                            "age": age_int,
                            "persona_text": persona_text,
                            "model_name": model_name,
                            "subject": s,
                            "relation": r,
                            "object": o,
                        }
                        all_extracted_data.append(rec)
                        row_records.append(rec)

                    # Incremental flush per persona
                    if row_records:
                        df_chunk = pd.DataFrame(row_records)
                        # header=False because we ensured file exists above
                        df_chunk.to_csv(
                            output_path,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8",
                        )

                    time_taken = time.time() - start_time
                    print(
                        f"         -> extracted {len(triplets)} triplets in {time_taken:.2f}s",
                        flush=True,
                    )

                    # 🔹 Update checkpoint
                    personas_processed += 1
                    processed_indices.add(row_idx)
                    ckpt["personas_processed"] = personas_processed
                    ckpt["processed_indices"] = sorted(processed_indices)
                    save_checkpoint(ckpt_path, ckpt)

                except Exception as e:
                    time_taken = time.time() - start_time
                    print(
                        f"{persona_text[:10]:<10} | {model_name:<15} | "
                        f"{time_taken:<10.2f} | !! FAILED !!",
                        flush=True,
                    )
                    err_info = {
                        "row_index": int(row_idx),
                        "age": age_int,
                        "persona_snippet": persona_text[:80],
                        "error": str(e),
                        "time": datetime.utcnow().isoformat(),
                    }
                    failures.append(err_info)
                    ckpt["failures"] = failures
                    save_checkpoint(ckpt_path, ckpt)
                    logging.error(
                        f"Failed run: idx={row_idx}, model={model_name}. Error: {e}",
                        exc_info=True,
                    )
                    # continue to next persona

        finally:
            elapsed_model = time.time() - start_model_time
            ckpt["elapsed_seconds"] = elapsed_model
            ckpt["failures"] = failures
            ckpt["personas_processed"] = personas_processed
            save_checkpoint(ckpt_path, ckpt)

            print(
                f"   [Model] {model_name}: processed {personas_processed} personas "
                f"(skipped {len(processed_indices) - personas_processed} already done) "
                f"in {elapsed_model:.2f}s. "
                f"Checkpoint: {ckpt_path}"
            )

        # 4. Optional in-memory summary
        if all_extracted_data:
            print(
                f"   [Model] {model_name}: Total triplets in memory this run: "
                f"{len(all_extracted_data)}. CSV was written incrementally to: {output_path}"
            )
        else:
            print(
                f"   [Model] {model_name}: No triplets extracted for {short_filename}."
            )


# --- Sentiment + lexical helpers --------------------------------------------

_POSITIVE_WORDS: List[str] = [
    "good",
    "great",
    "happy",
    "love",
    "enjoy",
    "excited",
    "optimistic",
    "positive",
    "amazing",
    "fantastic",
    "wonderful",
    "proud",
    "satisfied",
    "confident",
    "hopeful",
]

_NEGATIVE_WORDS: List[str] = [
    "bad",
    "sad",
    "hate",
    "angry",
    "upset",
    "worried",
    "anxious",
    "negative",
    "terrible",
    "awful",
    "depressed",
    "lonely",
    "frustrated",
    "guilty",
    "ashamed",
]


def simple_sentiment_score(text: str) -> float:
    if not text:
        return 0.0

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0

    pos = sum(t in _POSITIVE_WORDS for t in tokens)
    neg = sum(t in _NEGATIVE_WORDS for t in tokens)
    return (pos - neg) / len(tokens)


def compute_lexical_metrics(text: str) -> Dict[str, float]:
    if not text:
        return {"lexical_ttr": 0.0, "lexical_avg_word_len": 0.0}

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return {"lexical_ttr": 0.0, "lexical_avg_word_len": 0.0}

    unique_tokens = len(set(tokens))
    lexical_ttr = unique_tokens / len(tokens)
    lexical_avg_word_len = sum(len(t) for t in tokens) / len(tokens)

    return {
        "lexical_ttr": lexical_ttr,
        "lexical_avg_word_len": lexical_avg_word_len,
    }


# --- Graph helpers -----------------------------------------------------------


def compute_graph_metrics(edges: List[Tuple[str, str]]) -> Dict[str, float]:
    # Collect nodes and adjacency (undirected)
    nodes = set()
    adj = {}

    for u, v in edges:
        if u is None or v is None:
            continue
        u = str(u)
        v = str(v)
        nodes.add(u)
        nodes.add(v)

        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    n = len(nodes)
    m = len(edges)

    if n == 0:
        return {
            "graph_num_nodes": 0,
            "graph_num_edges": 0,
            "graph_avg_degree": 0.0,
            "graph_density": 0.0,
            "graph_num_components": 0,
            "graph_largest_component_size": 0,
            "graph_largest_component_ratio": 0.0,
        }

    # Average degree
    degrees = [len(adj.get(node, [])) for node in nodes]
    graph_avg_degree = sum(degrees) / n if n > 0 else 0.0

    # Density (undirected, simple graph)
    if n > 1:
        graph_density = (2.0 * m) / (n * (n - 1))
    else:
        graph_density = 0.0

    # Connected components (undirected)
    visited = set()
    graph_num_components = 0
    largest_component_size = 0

    for node in nodes:
        if node in visited:
            continue
        graph_num_components += 1
        stack = [node]
        comp_size = 0
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp_size += 1
            for nei in adj.get(x, []):
                if nei not in visited:
                    stack.append(nei)
        if comp_size > largest_component_size:
            largest_component_size = comp_size

    graph_largest_component_ratio = largest_component_size / n if n > 0 else 0.0

    return {
        "graph_num_nodes": n,
        "graph_num_edges": m,
        "graph_avg_degree": graph_avg_degree,
        "graph_density": graph_density,
        "graph_num_components": graph_num_components,
        "graph_largest_component_size": largest_component_size,
        "graph_largest_component_ratio": graph_largest_component_ratio,
    }


def process_triplets_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame()

    required_cols = {"original_index", "age", "persona_text", "subject", "object"}
    missing = required_cols - set(df.columns)
    if missing:
        print("Missing required columns:", missing)
        return pd.DataFrame()

    # Add model_name column if missing
    if "model_name" not in df.columns:
        df["model_name"] = None

    group_cols = ["original_index", "age", "persona_text", "model_name"]
    if "education_level" in df.columns:
        group_cols.append("education_level")

    groups = df.groupby(group_cols, dropna=False)

    records = []

    for keys, g in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        ctx = dict(zip(group_cols, keys))

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

        # -------- Unique Counts --------
        num_unique_subjects = subjects.nunique()
        num_unique_objects = objects.nunique()
        num_unique_relations = relations.nunique()
        concept_set = set(subjects.tolist()) | set(objects.tolist())
        num_unique_concepts = len(concept_set)

        # -------- Unique Triplets --------
        triplets = list(zip(subjects.tolist(), relations.tolist(), objects.tolist()))
        num_unique_triplets = len(set(triplets))
        triplet_repetition_ratio = 1.0 - (num_unique_triplets / num_triplets)

        # -------- Persona Text --------
        persona_text = ctx["persona_text"]
        persona_tokens = persona_text.split() if persona_text else []
        persona_num_tokens = len(persona_tokens)
        triplets_per_100_tokens = (
            (num_triplets / persona_num_tokens) * 100 if persona_num_tokens > 0 else 0
        )

        # -------- Sentiment + Lexical Complexity --------
        sentiment_score = simple_sentiment_score(persona_text)
        lex = compute_lexical_metrics(persona_text)

        # -------- Graph Metrics --------
        edges = list(zip(subjects.tolist(), objects.tolist()))
        graph = compute_graph_metrics(edges)

        # -------- Build Output Record --------
        record = {
            # Raw identifying columns
            "original_index": ctx["original_index"],
            "age": ctx["age"],
            "persona_text": ctx["persona_text"],
            # Raw triplet columns — duplicated per persona
            "subject": "; ".join(subjects.tolist()),
            "relation": "; ".join(relations.tolist()),
            "object": "; ".join(objects.tolist()),
            # Metrics
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
            # Graph complexity
            "graph_num_nodes": graph["graph_num_nodes"],
            "graph_num_edges": graph["graph_num_edges"],
            "graph_avg_degree": graph["graph_avg_degree"],
            "graph_density": graph["graph_density"],
            "graph_num_components": graph["graph_num_components"],
            "graph_largest_component_size": graph["graph_largest_component_size"],
            "graph_largest_component_ratio": graph["graph_largest_component_ratio"],
        }

        if "education_level" in ctx:
            record["education_level"] = ctx["education_level"]

        records.append(record)

    return pd.DataFrame(records)


# ==========================================
# MAIN STATISTICAL ANALYSIS
# ==========================================


def annotate_stats(ax, pearson_r, pearson_p, spearman_r, spearman_p):
    """Places Pearson and Spearman stats as a text box inside the given axis."""
    text = (
        f"Pearson r = {pearson_r:.3f} (p={pearson_p:.3g})\n"
        f"Spearman r = {spearman_r:.3f} (p={spearman_p:.3g})"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
    )


def canonicalize_model_name(name: str) -> str:
    name = name.lower().strip()

    # DO NOT strip prefixes like google/ or meta/
    # Only normalize common pattern variations:
    if "gemma3" in name:
        name = name.replace("gemma3", "gemma-3")
    elif "phi4" in name:
        name = name.replace("phi4", "phi-4")
    elif "llama3.3" in name:
        name = name.replace("llama3.3", "Llama-3.3")
    elif "qwen3" in name:
        name = name.replace("qwen3", "Qwen3")

    return name


def run_statistical_analysis(model_name: str):
    print("\n" + "=" * 60)
    print(f"AMoC AGE STATISTICAL ANALYSIS — MODEL: {model_name}")
    print("=" * 60)

    # canonicalize the model name first
    print("DEBUG BEFORE:", model_name)
    model_name = canonicalize_model_name(model_name)
    print("DEBUG AFTER:", model_name)
    safe_tag = model_name.replace("/", "-").replace(":", "-").replace(" ", "_")
    # Apply Qwen-specific capitalization rules
    safe_tag = safe_tag.lower()  # ensure deterministic base

    if "qwen" in safe_tag:
        parts = safe_tag.split("-")
        new_parts = []
        for p in parts:
            if p.startswith("qwen"):
                # qwen -> Qwen, qwen3 -> Qwen3
                new_parts.append("Qwen" + p[4:])
            elif p.endswith("b") and any(c.isdigit() for c in p):
                # 30b -> 30B, a3b -> A3B
                new_parts.append(p.upper())
            elif p.isalpha():
                # instruct -> Instruct
                new_parts.append(p.capitalize())
            else:
                # numbers etc. unchanged (2507)
                new_parts.append(p)

        safe_tag = "-".join(new_parts)
    print("DEBUG SAFE_TAG:", safe_tag)

    pattern = f"model_{safe_tag}_triplets_*.csv"
    search_path = os.path.join(OUTPUT_DIR, pattern)

    print(f"Looking for CSV files with pattern: {search_path}")

    files_to_analyze = glob.glob(search_path)

    if not files_to_analyze:
        print(f"No CSV files found for model '{model_name}' (pattern: {pattern})")
        return

    print(f"Found {len(files_to_analyze)} CSV file(s):")
    for f in files_to_analyze:
        print("  -", os.path.basename(f))

    all_metrics = []
    for full_path in files_to_analyze:
        df_metrics = process_triplets_file(full_path)
        if df_metrics is not None and not df_metrics.empty:
            df_metrics["model"] = model_name
            all_metrics.append(df_metrics)

    if not all_metrics:
        print("No usable metric data found.")
        return

    df_master = pd.concat(all_metrics, ignore_index=True)
    print(f"\nTotal personas analyzed: {len(df_master)}")

    # Clean age
    df_master["age"] = pd.to_numeric(df_master["age"], errors="coerce")
    df_master = df_master.dropna(subset=["age"])
    if df_master.empty:
        print("No valid age data after filtering.")
        return

    # 2. Correlations — updated
    dependent_vars = ["num_triplets", "num_unique_concepts", "graph_density"]
    correlation_stats = {}

    print("\n" + "-" * 30)
    print("   CORRELATION RESULTS")
    print("-" * 30)

    for dep_var in dependent_vars:
        if dep_var not in df_master.columns:
            print(f"Skipping {dep_var}: missing column.")
            continue

        print(f"\n>>> Age vs {dep_var}")

        r_pearson, p_pearson = stats.pearsonr(df_master["age"], df_master[dep_var])
        r_spearman, p_spearman = stats.spearmanr(df_master["age"], df_master[dep_var])

        print(f"    Pearson r:  {r_pearson:.4f} (p={p_pearson:.4g})")
        print(f"    Spearman r: {r_spearman:.4f} (p={p_spearman:.4g})")

        if p_spearman < 0.05:
            direction = "INCREASES" if r_spearman > 0 else "DECREASES"
            print(f"    SIGNIFICANT: As age increases, {dep_var} {direction}")
        else:
            print("    No significant correlation found.")

        correlation_stats[dep_var] = {
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_r": r_spearman,
            "spearman_p": p_spearman,
        }

    # 3. Plotting
    print("\nGenerating plots...")

    plt.figure(figsize=(20, 6))

    # Plot 1 — Age vs num_triplets
    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_master, x="age", y="num_triplets", alpha=0.6, ax=ax1)
    sns.regplot(data=df_master, x="age", y="num_triplets", scatter=False, ax=ax1)
    ax1.set_title(f"Age vs Nr. Triplets — {model_name}")
    if "num_triplets" in correlation_stats:
        s = correlation_stats["num_triplets"]
        annotate_stats(
            ax1, s["pearson_r"], s["pearson_p"], s["spearman_r"], s["spearman_p"]
        )

    # Plot 2 — Age vs num_unique_concepts
    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_master, x="age", y="num_unique_concepts", alpha=0.6, ax=ax2)
    sns.regplot(data=df_master, x="age", y="num_unique_concepts", scatter=False, ax=ax2)
    ax2.set_title(f"Age vs Unique Concepts — {model_name}")
    if "num_unique_concepts" in correlation_stats:
        s = correlation_stats["num_unique_concepts"]
        annotate_stats(
            ax2, s["pearson_r"], s["pearson_p"], s["spearman_r"], s["spearman_p"]
        )

    # Plot 3 — Age vs graph_density
    ax3 = plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_master, x="age", y="graph_density", alpha=0.6, ax=ax3)
    sns.regplot(data=df_master, x="age", y="graph_density", scatter=False, ax=ax3)
    ax3.set_title(f"Age vs Graph Density — {model_name}")
    if "graph_density" in correlation_stats:
        s = correlation_stats["graph_density"]
        annotate_stats(
            ax3, s["pearson_r"], s["pearson_p"], s["spearman_r"], s["spearman_p"]
        )

    plt.tight_layout()

    # 4. Save results
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

    plot_path = os.path.join(OUTPUT_ANALYSIS_DIR, f"stats_plots_{safe_tag}.png")
    table_path = os.path.join(
        OUTPUT_ANALYSIS_DIR, f"final_statistical_table_{safe_tag}.csv"
    )

    plt.savefig(plot_path)
    plt.close()
    df_master.to_csv(table_path, index=False)

    print(f"\nPlots saved to: {plot_path}")
    print(f"Statistical table saved to: {table_path}")


# ==========================================
# MAIN FUNCTION
# ==========================================


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run AMoCv4 over all persona CSVs in INPUT_DIR using one or more vLLM models "
            "(age-aware, persona-aware prompts)."
        )
    )
    p.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated list of vLLM model names "
            "(e.g. 'Qwen/Qwen3-30B-A3B-Instruct-2507,openai/gpt-oss-120b')"
        ),
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of rows per file (for smoke testing).",
    )
    p.add_argument(
        "--replace-pronouns",
        action="store_true",
        help="Use AMoC's pronoun resolution.",
    )
    p.add_argument(
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (TP). Must match number of GPUs per node.",
    )
    p.add_argument(
        "--resume-only",
        action="store_true",
        help="Only process persona rows that were NOT previously completed (based on checkpoint).",
    )

    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("--models must contain at least one model name")

    tp_size = args.tp
    # Discover all persona CSVs under INPUT_DIR
    import glob

    files_to_process = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files_to_process:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    print(f"Discovered {len(files_to_process)} persona files under {INPUT_DIR}")
    print(f"Models: {model_names}")
    print(f"Output root: {OUTPUT_DIR}")

    total_start = time.time()

    try:
        for filename in files_to_process:
            try:
                process_persona_csv(
                    filename=filename,
                    model_names=model_names,
                    max_rows=args.max_rows,
                    replace_pronouns=args.replace_pronouns,
                    # The following are now fixed in-script; change here if needed.
                    max_tokens=512,
                    temperature=0.8,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=0.8,
                    output_dir=OUTPUT_DIR,
                )
            except Exception as e:
                logging.error(
                    f"Fatal error while processing {filename}: {e}", exc_info=True
                )
                print(
                    f"[ERROR] Aborting file {filename} due to unexpected error, "
                    f"but will continue with remaining files."
                )
    finally:
        total_time = time.time() - total_start
        print(
            f"Extraction phase finished (possibly with errors) in {total_time:.2f} seconds."
        )

        for model in model_names:
            print(f"\n[STATS] Running statistical analysis for model: {model}")
            try:
                run_statistical_analysis(model)
            except Exception as e:
                logging.error(
                    f"Error in statistical analysis for {model}: {e}", exc_info=True
                )
                print(f"[ERROR] Statistical analysis failed for {model}: {e}")


if __name__ == "__main__":
    # Optional: set multiprocessing start method if you later add parallelism
    try:
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass

    main(sys.argv[1:])

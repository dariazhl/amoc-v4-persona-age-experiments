from amoc.llm.vllm_client import VLLMClient


class LLMExtractor:
    def __init__(self, client: VLLMClient):
        self._client = client

    def get_new_relationships(
        self, nodes_from_graph: str, edges_from_graph: str, text: str, persona: str
    ):
        return self._client.get_new_relationships(
            nodes_from_graph, edges_from_graph, text, persona
        )

    def get_new_relationships_first_sentence(
        self, nodes_from_text: str, text: str, persona: str
    ):
        return self._client.get_new_relationships_first_sentence(
            nodes_from_text, text, persona
        )

    def infer_objects_and_properties(
        self,
        nodes_from_text: str,
        graph_nodes_representation: str,
        graph_edges_representation: str,
        text: str,
        persona: str,
    ):
        return self._client.infer_objects_and_properties(
            nodes_from_text,
            graph_nodes_representation,
            graph_edges_representation,
            text,
            persona,
        )

    def infer_objects_and_properties_first_sentence(
        self, nodes_from_text: str, text: str, persona: str
    ):
        return self._client.infer_objects_and_properties_first_sentence(
            nodes_from_text, text, persona
        )

    def generate_new_inferred_relationships(
        self,
        nodes_from_text: str,
        graph_nodes_representation: str,
        graph_edges_representation: str,
        concepts,
        properties,
        text: str,
        persona: str,
    ):
        return self._client.generate_new_inferred_relationships(
            nodes_from_text,
            graph_nodes_representation,
            graph_edges_representation,
            concepts,
            properties,
            text,
            persona,
        )

    def generate_new_inferred_relationships_first_sentence(
        self, nodes_from_text: str, concepts, properties, text: str, persona: str
    ):
        return self._client.generate_new_inferred_relationships_first_sentence(
            nodes_from_text, concepts, properties, text, persona
        )

    def get_relevant_edges(self, edges_from_graph: str, text: str, persona: str):
        return self._client.get_relevant_edges(edges_from_graph, text, persona)

    def resolve_pronouns(self, text: str, persona: str):
        return self._client.resolve_pronouns(text, persona)

    def get_forced_connectivity_edge_label(
        self,
        node_a: str,
        node_b: str,
        story_context: str,
        current_sentence: str,
        persona: str = "",
    ):
        return self._client.get_forced_connectivity_edge_label(
            node_a, node_b, story_context, current_sentence, persona
        )

    def generate_raw(self, prompt_text: str, temperature: float = 0.0):
        return self._client.generate_raw(prompt_text, temperature=temperature)

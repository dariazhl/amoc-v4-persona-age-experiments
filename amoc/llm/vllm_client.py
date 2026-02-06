import logging
from typing import List, Dict

from vllm import LLM, SamplingParams

from amoc.llm.parsing import (
    parse_for_dict,
    extract_list_from_string,
)

from amoc.prompts.amoc_prompts import (
    NEW_RELATIONSHIPS_PROMPT,
    NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT,
    INFER_OBJECTS_AND_PROPERTIES_PROMPT,
    GENERATE_NEW_INFERRED_RELATIONSHIPS_PROMPT,
    INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT,
    GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT,
    SELECT_RELEVANT_EDGES_PROMPT,
    REPLACE_PRONOUNS_PROMPT,
    HUB_EDGE_LABEL_WITH_EXPLANATION_PROMPT,
    FORCED_CONNECTIVITY_EDGE_PROMPT,
)


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
            max_model_len=8200,
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

            # Keep output as-is; downstream parsing helpers extract the first
            # [...] or {...} block. Mutating here can easily corrupt otherwise
            # parseable responses (e.g., prefixing '[' before prose containing a list).
            if "final" in raw_text:
                return raw_text.split("final")[-1].strip()
            return raw_text.strip()

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
        return extract_list_from_string(response)

    def get_new_relationships_first_sentence(self, nodes_from_text, text, persona):
        prompt = NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

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
        return parse_for_dict(response)

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
        return parse_for_dict(response)

    def infer_objects_and_properties_first_sentence(
        self, nodes_from_text, text, persona
    ):
        prompt = INFER_OBJECTS_AND_PROPERTIES_FIRST_SENTENCE_PROMPT.format(
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

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
        return parse_for_dict(response)

    def get_relevant_edges(self, edges_from_graph, text, persona):
        prompt = SELECT_RELEVANT_EDGES_PROMPT.format(edges=edges_from_graph, text=text)
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

    def resolve_pronouns(self, text, persona):
        prompt = REPLACE_PRONOUNS_PROMPT + text
        return self.call_vllm(prompt, persona)

    def get_edge_label(
        self, node_a: str, node_b: str, sentence_text: str, persona: str
    ) -> str:
        result = self.get_edge_label_with_explanation(
            node_a, node_b, sentence_text, [], persona
        )
        return result.get("label", "")

    def get_edge_label_with_explanation(
        self,
        node_a: str,
        node_b: str,
        sentence_text: str,
        explicit_nodes: List[str],
        persona: str,
    ) -> Dict[str, str]:
        explicit_nodes_str = (
            ", ".join(explicit_nodes) if explicit_nodes else f"{node_a}, {node_b}"
        )
        prompt = HUB_EDGE_LABEL_WITH_EXPLANATION_PROMPT.format(
            explicit_nodes=explicit_nodes_str,
            node_a=node_a,
            node_b=node_b,
            sentence_text=sentence_text,
        )
        response = self.call_vllm(prompt, persona)
        result = parse_for_dict(response)
        if not isinstance(result, dict):
            return {"label": "", "explanation": ""}
        return {
            "label": result.get("label", ""),
            "explanation": result.get("explanation", ""),
        }

    # ==========================================================================
    # TASK 2: FORCED CONNECTIVITY EDGE GENERATION
    # ==========================================================================
    def get_forced_connectivity_edge_label(
        self,
        node_a: str,
        node_b: str,
        story_context: str,
        current_sentence: str,
        persona: str,
    ) -> Dict[str, str]:
        """
        TASK 2: Generate an edge label to restore graph connectivity.

        This method is called ONLY when:
        1. The active graph has become disconnected
        2. No existing edges in cumulative memory can restore connectivity
        3. A minimal connecting edge must be created

        Args:
            node_a: First node text (from isolated component)
            node_b: Second node text (from focus component)
            story_context: Previous sentences for context
            current_sentence: The current sentence being processed
            persona: The persona description for LLM context

        Returns:
            Dict with 'label' and 'explanation' keys.
            Returns {"label": "relates to", "explanation": "..."} as fallback.
        """
        prompt = FORCED_CONNECTIVITY_EDGE_PROMPT.format(
            node_a=node_a,
            node_b=node_b,
            story_context=story_context,
            current_sentence=current_sentence,
        )
        response = self.call_vllm(prompt, persona)
        result = parse_for_dict(response)

        if not isinstance(result, dict) or not result.get("label"):
            # Fallback to generic relationship if LLM fails
            logging.warning(
                "[Connectivity] LLM failed to generate edge label for %s -> %s, using fallback",
                node_a,
                node_b,
            )
            return {
                "label": "relates to",
                "explanation": "Fallback connectivity edge (LLM response invalid)",
            }

        return {
            "label": result.get("label", "relates to"),
            "explanation": result.get("explanation", ""),
        }

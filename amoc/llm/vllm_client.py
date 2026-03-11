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
    PRONOUN_RESOLUTION_PROMPT,
    HUB_EDGE_LABEL_WITH_EXPLANATION_PROMPT,
    FORCED_CONNECTIVITY_EDGE_PROMPT,
    VALIDATE_TRIPLET_PROMPT,
    NARRATIVE_RELEVANCE_PROMPT,
    PRUNE_IRRELEVANT_TRIPLETS_BY_NARRATIVE,
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

    def generate(
        self, messages: List[Dict[str, str]], temperature: float = None
    ) -> str:
        if self.llm is None:
            logging.error("VLLM not initialized.")
            return "[]"

        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            prompt += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\nfinal\n"

        try:
            # Clone sampling params so you don’t mutate global config
            sampling_params = self.sampling_params

            if temperature is not None:
                from copy import deepcopy

                sampling_params = deepcopy(self.sampling_params)
                sampling_params.temperature = temperature
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text

            if "final" in raw_text:
                return raw_text.split("final")[-1].strip()
            return raw_text.strip()

        except Exception as e:
            logging.exception(f"VLLM runtime error: {e}")
            return "[]"

    # wrapper to generate response without persona injection for repair - design
    def generate_raw(
        self,
        prompt_text: str,
        temperature: float = 0.3,
    ) -> str:
        messages = [{"role": "user", "content": prompt_text}]
        return self.generate(messages, temperature=temperature)

    def call_vllm(self, prompt: str, persona: str) -> str:
        # Inject persona into system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph builder and reasoning agent.\n\n"
                    "IMPORTANT: You are processing story text to build a knowledge graph.\n"
                    "- ALL concepts, properties, and relationships must come from THE STORY TEXT provided.\n"
                    "- The persona below is provided ONLY to help you understand what aspects of the story are most relevant or salient.\n"
                    "- Do NOT create nodes or edges from the persona description itself.\n"
                    "- The persona influences what you focus on, not what you extract.\n\n"
                    f"Persona (for salience weighting only):\n{persona}\n\n"
                    "Use this persona to guide which story elements deserve emphasis, but ensure every node and edge you output is grounded in the story text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.generate(messages)

    # query the LLM to generate relationships that may not be captured by deterministic rules
    # it uses the full context of the current sentence and the existing graph
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

    # Old design: Ask LLM to re-write entire sentence
    # Issue: risk of contamination with LLM garbage text
    # New design: Identify pronouns and store them in a dict: {"He": "Charlemagne", "his": "Charlemagne"}
    def resolve_pronouns(self, sentence, context, persona):
        prompt = PRONOUN_RESOLUTION_PROMPT.format(context=context, sentence=sentence)
        response = self.call_vllm(prompt, persona)
        result = parse_for_dict(response)

        if not isinstance(result, dict):
            return {}

        return result

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

    def get_forced_connectivity_edge_label(
        self,
        node_a: str,
        node_b: str,
        story_context: str,
        current_sentence: str,
        persona: str,
    ) -> Dict[str, str]:
        # call method when the activate graph is disconnected
        prompt = FORCED_CONNECTIVITY_EDGE_PROMPT.format(
            node_a=node_a,
            node_b=node_b,
            story_context=story_context,
            current_sentence=current_sentence,
        )
        response = self.call_vllm(prompt, persona)
        result = parse_for_dict(response)

        if not isinstance(result, dict) or not result.get("label"):
            # Fallback
            logging.warning(
                " LLM failed to generate edge label for %s -> %s, using fallback",
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

    # Ask LLM to validate if a triple makes sense given the sentence
    def validate_triplet(
        self,
        sentence: str,
        subject: str,
        relation: str,
        object: str,
        persona: str,
    ) -> Dict[str, any]:
        prompt = VALIDATE_TRIPLET_PROMPT.format(
            sentence=sentence,
            subject=subject,
            relation=relation,
            object=object,
        )
        response = self.call_vllm(prompt, persona)
        result = parse_for_dict(response)

        if not isinstance(result, dict):
            logging.warning(f"LLM validation returned invalid format: {response}")
            return {
                "valid": True,
                "reason": "Validation failed, accepting by default",
                "corrected_triple": None,
            }

        return {
            "valid": result.get("valid", True),
            "reason": result.get("reason", ""),
            "corrected_triple": result.get("corrected_triple", None),
        }

    # call in sentrene builder before adding the edges
    def prune_irrelevant_triplets_by_narrative(
        self, story_context, current_sentence, active_triplets, persona
    ):
        prompt = PRUNE_IRRELEVANT_TRIPLETS_BY_NARRATIVE.format(
            story_context=story_context,
            current_sentence=current_sentence,
            active_triplets=active_triplets,
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

    # Check if a triple is narratively relevant to the story using LLM only
    def check_narrative_relevance(
        self, story_context, current_sentence, active_triplets, persona
    ):
        prompt = NARRATIVE_RELEVANCE_PROMPT.format(
            story_context=story_context,
            current_sentence=current_sentence,
            active_triplets=active_triplets,
        )
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

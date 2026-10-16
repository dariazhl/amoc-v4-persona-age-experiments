import logging
import re
import time
from copy import deepcopy
from typing import List, Dict, Optional

import requests
from transformers import AutoTokenizer

from amoc.llm.parsing import (
    parse_for_dict,
    extract_list_from_string,
)

from amoc.config.constants import allow_inferred_chains, persona_mode
from amoc.prompts.amoc_prompts import (
    LEGACY_FRAME,
    LEGACY_LEVEL_BLOCK,
    NEUTRAL_FRAME,
    READER_FRAME,
    READER_LEVEL_BLOCK,
    VALIDATOR_RULE_6,
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


def level_block() -> str:
    return READER_LEVEL_BLOCK if persona_mode() == "reader" else LEGACY_LEVEL_BLOCK


call_log = logging.getLogger("amoc.llm_calls")
if not call_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s pid=%(process)d %(message)s"))
    call_log.addHandler(_handler)
    call_log.setLevel(logging.INFO)
    call_log.propagate = False


class ServerUnavailableError(RuntimeError):
    pass


class VLLMClient:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        tp_size: int = 1,
        debug: bool = False,
        server_url: Optional[str] = None,
    ):
        self.debug = debug
        self.model_name = model_name
        self.tp_size = tp_size
        self.server_url = (server_url or "").rstrip("/") or None
        self._session = None
        self.server_failures = 0

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.top_p = 0.95
        self.max_tokens = 1024

        if self.server_url:
            logging.info(f"using vllm server at {self.server_url} for model: {model_name}")
            self.llm = None
            self._session = requests.Session()
            return

        import torch
        from vllm import LLM

        logging.info(f"starting up vllm with model: {model_name}, tp_size={tp_size}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,      
            max_model_len=8200,               
        )

        num_gpus = torch.cuda.device_count()
        if tp_size > num_gpus:
            raise ValueError(
                f"Requested tensor parallel size {tp_size}, but only {num_gpus} GPUs are available."
            )

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            logging.warning(f"apply_chat_template failed: {e}, using fallback Qwen format")
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    def _clean_response(self, raw_text: str) -> str:
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        raw_text = re.sub(r'</think>', '', raw_text)
        raw_text = re.sub(r'```(?:json|python)?\n?', '', raw_text)
        raw_text = re.sub(r'```\n?', '', raw_text)
        raw_text = re.sub(r'`', '', raw_text)
        match = re.search(r'[\[\{]', raw_text)
        if not match:
            return ""
        start = match.start()
        stack = []
        end = start
        for i, ch in enumerate(raw_text[start:], start=start):
            if ch in '[{':
                stack.append(ch)
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
                if not stack:
                    end = i + 1
                    break
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    end = i + 1
                    break
        if end == start:
            end = raw_text.rfind(']') if raw_text[start] == '[' else raw_text.rfind('}')
            if end == -1:
                end = len(raw_text)
            else:
                end += 1
        return raw_text[start:end]

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        if self.llm is None and self.server_url is None:
            logging.error("VLLM not initialized.")
            return "[]"

        prompt = self._apply_chat_template(messages)

        try:
            start = time.time()
            if self.server_url:
                raw_text, prompt_tokens, output_tokens, finish = self._generate_http(
                    prompt, temperature
                )
            else:
                raw_text, prompt_tokens, output_tokens, finish = self._generate_local(
                    prompt, temperature
                )
            call_log.info(
                f"llm_call prompt_tokens={prompt_tokens} output_tokens={output_tokens} "
                f"finish={finish} latency={time.time() - start:.1f}s"
            )
            raw_text = raw_text.strip()
            cleaned = self._clean_response(raw_text)
            if self.debug:
                logging.debug(f"Cleaned response: {cleaned[:200]}")
            return cleaned
        except ServerUnavailableError:
            raise
        except Exception as e:
            logging.exception(f"VLLM runtime error: {e}")
            return "[]"

    def _generate_local(self, prompt: str, temperature: float):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        out = outputs[0]
        completion = out.outputs[0]
        return (
            completion.text,
            len(out.prompt_token_ids or []),
            len(completion.token_ids),
            completion.finish_reason,
        )

    def _generate_http(self, prompt: str, temperature: float, retries: int = 8):
        if self.server_failures:
            raise ServerUnavailableError(f"vllm server at {self.server_url} already failed")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(retries):
            try:
                resp = self._session.post(
                    f"{self.server_url}/v1/completions", json=payload, timeout=1800
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return (
                    choice["text"],
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    choice.get("finish_reason"),
                )
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code < 500:
                    raise
                error = e
            except requests.ConnectionError as e:
                error = e
            except requests.Timeout as e:
                self.server_failures += 1
                raise ServerUnavailableError(f"vllm server request timed out: {e}") from e
            if attempt == retries - 1:
                self.server_failures += 1
                raise ServerUnavailableError(
                    f"vllm server at {self.server_url} unreachable after {retries} attempts: {error}"
                ) from error
            wait = 2 ** attempt
            logging.warning(f"vllm server request failed ({error}); retrying in {wait}s")
            time.sleep(wait)

    def generate_raw(self, prompt_text: str, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt_text}]
        return self.generate(messages, temperature=temperature)

    def call_vllm(self, prompt: str, persona: str, role: str = "reader") -> str:
        if persona_mode() == "legacy":
            full_prompt = LEGACY_FRAME.format(persona=persona, prompt=prompt)
        elif role == "neutral":
            full_prompt = NEUTRAL_FRAME.format(prompt=prompt)
        else:
            full_prompt = READER_FRAME.format(persona=persona, prompt=prompt)
        messages = [{"role": "user", "content": full_prompt}]
        return self.generate(messages, temperature=0.0)

    def get_new_relationships(
        self, nodes_from_text, nodes_from_graph, edges_from_graph, text, persona
    ):
        prompt = NEW_RELATIONSHIPS_PROMPT.format(
            level_block=level_block(),
            nodes_from_text=nodes_from_text,
            nodes_from_graph=nodes_from_graph,
            edges_from_graph=edges_from_graph,
            text=text,
        )
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

    def get_new_relationships_first_sentence(self, nodes_from_text, text, persona):
        prompt = NEW_RELATIONSHIPS_FOR_FIRST_SENTENCE_PROMPT.format(
            level_block=level_block(),
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        return extract_list_from_string(response)

    def infer_objects_and_properties(
        self, nodes_from_text, nodes_from_graph, edges_from_graph, text, persona
    ):
        prompt = INFER_OBJECTS_AND_PROPERTIES_PROMPT.format(
            level_block=level_block(),
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
            level_block=level_block(),
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
            level_block=level_block(),
            nodes_from_text=nodes_from_text, text=text
        )
        response = self.call_vllm(prompt, persona)
        logging.error(f"RAW INFERENCE RESPONSE (first 500 chars): {response[:500]}")
        return parse_for_dict(response)

    def generate_new_inferred_relationships_first_sentence(
        self, nodes_from_text, concepts, properties, text, persona
    ):
        prompt = GENERATE_NEW_INFERRED_RELATIONSHIPS_FIRST_SENTENCE_PROMPT.format(
            level_block=level_block(),
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

    def resolve_pronouns(self, sentence, context, persona):
        prompt = PRONOUN_RESOLUTION_PROMPT.format(context=context, sentence=sentence)
        response = self.call_vllm(prompt, persona, role="neutral")
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
        prompt = FORCED_CONNECTIVITY_EDGE_PROMPT.format(
            node_a=node_a,
            node_b=node_b,
            story_context=story_context,
            current_sentence=current_sentence,
        )
        response = self.call_vllm(prompt, persona)
        result = parse_for_dict(response)
        if not isinstance(result, dict) or not result.get("label"):
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

    def validate_triplet(
        self,
        sentence: str,
        subject: str,
        relation: str,
        object: str,
        persona: str,
    ) -> Dict[str, any]:
        prompt = VALIDATE_TRIPLET_PROMPT.format(
            rule_6="" if allow_inferred_chains() else VALIDATOR_RULE_6,
            sentence=sentence,
            subject=subject,
            relation=relation,
            object=object,
        )
        response = self.call_vllm(prompt, persona, role="neutral")
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

    def prune_irrelevant_triplets_by_narrative(
        self,
        story_context,
        current_sentence,
        active_triplets,
        persona,
        aggressive=False,
    ):
        prompt = PRUNE_IRRELEVANT_TRIPLETS_BY_NARRATIVE.format(
            story_context=story_context,
            current_sentence=current_sentence,
            active_triplets=active_triplets,
        )
        if aggressive:
            prompt += "\n\nThis is a SECOND PASS. Be EVEN MORE AGGRESSIVE. Remove anything that is not absolutely essential."
        response = self.call_vllm(prompt, persona)
        return parse_for_dict(response)

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
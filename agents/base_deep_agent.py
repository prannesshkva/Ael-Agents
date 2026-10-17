import time
from typing import Any, Callable, Dict, List, Optional

import torch
from deepagents import create_deep_agent

from core.api_queue import APIKeyQueue
from core.embeddings import SemanticEncoder, get_encoder
from core.llm_utils import LLMCallError, llm_call
from core.parameter_registry import DomainParameterRegistry

ScientificSemanticEncoder = SemanticEncoder

class _LazyEncoder:

    def encode(self, text: str, device: str = "cpu") -> torch.Tensor:
        return get_encoder().encode(text, device=device)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        return get_encoder().encode_batch(texts)

GLOBAL_ENCODER = _LazyEncoder()

class ScientificDeepAgent:
    def __init__(
        self,
        name: str,
        domain_title: str,
        param_index: int,
        system_prompt: str,
        domain_tools: Optional[List[Callable]] = None,
        embedding_dim: int = 0,
    ):
        self.name = name
        self.domain_title = domain_title
        self.param_index = param_index
        self.system_prompt = system_prompt
        self.domain_tools = domain_tools or []
        self.embedding_dim = embedding_dim

    def execute_parallel(
        self, task: str, key_queue: APIKeyQueue, param_registry: DomainParameterRegistry
    ) -> Dict[str, Any]:
        t0 = time.time()
        last_error: Optional[str] = None
        hypothesis_text = ""
        used_key_id = "UNKNOWN"

        model_candidates = [key_queue.model_name] + [
            m for m in getattr(key_queue, "FALLBACK_MODELS", ()) if m != key_queue.model_name
        ]
        model_idx = 0
        for attempt in range(4):
            active_model = model_candidates[min(model_idx, len(model_candidates) - 1)]
            token_budget = 1536 + (512 * attempt)
            try:
                with key_queue.lease_key(self.name) as (key_id, api_key):
                    used_key_id = f"KEY_{key_id:02d}"
                    llm = key_queue.create_chat_model(
                        api_key=api_key,
                        temperature=0.45,
                        max_tokens=token_budget,
                        model_override=active_model,
                    )
                    graph = create_deep_agent(
                        model=llm,
                        tools=self.domain_tools,
                        system_prompt=self.system_prompt,
                        name=f"deep_agent_{self.name.lower()}",
                    )
                    user_prompt = (
                        f"RESEARCH QUESTION (answer from your domain: {self.domain_title}):\n{task}\n\n"
                        "Respond concisely with exactly these sections:\n"
                        "1. [DOMAIN AXIOMS & MECHANISM]: your field's relevant laws, equations or mechanisms.\n"
                        "2. [STRUCTURAL INVARIANT]: the abstract mathematical/structural pattern behind them.\n"
                        "3. [DMN IMAGINATION BRIDGE]: an abstract analogy another field could recognise.\n"
                        "Be explicit about uncertainty; do not invent citations."
                    )
                    result = graph.invoke({"messages": [{"role": "user", "content": user_prompt}]})
                    last_msg = result["messages"][-1]
                    hypothesis_text = (last_msg.content or "").strip()
                    if not hypothesis_text and hasattr(last_msg, "additional_kwargs"):
                        hypothesis_text = str(last_msg.additional_kwargs.get("reasoning_content") or "").strip()
                    if hypothesis_text and len(hypothesis_text) > 10:
                        break
                    hypothesis_text = ""
                    last_error = "empty response (reasoning exhausted token budget)"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                err_str = str(e).lower()
                if "tokens per day" in err_str or "model_not_found" in err_str or "404" in err_str:
                    model_idx += 1
                time.sleep(1.5 * (attempt + 1))

        failed = not hypothesis_text
        if failed:
            hypothesis_text = (
                f"[{self.domain_title}] NO RESPONSE (agent call failed: {last_error}). "
                f"This entry carries no domain information."
            )

        latency_s = round(time.time() - t0, 2)
        device = str(param_registry.weights.device)
        e_i = get_encoder().encode(hypothesis_text, device=device)
        w_i = param_registry.get_agent_parameter(self.name)
        h_i = w_i * e_i

        return {
            "agent_name": self.name,
            "domain_title": self.domain_title,
            "param_id": f"w_{self.param_index}",
            "param_value": float(w_i.detach().item()),
            "hypothesis": hypothesis_text,
            "raw_embedding": e_i,
            "weighted_tensor": h_i,
            "latency_s": latency_s,
            "key_id": used_key_id,
            "failed": failed,
            "error": last_error if failed else None,
        }

    def short_answer(self, stimulus: str, key_queue: APIKeyQueue, max_tokens: int = 90) -> str:
        user = (
            f"Concept: '{stimulus}'.\n"
            "In at most two sentences, state what this concept is and what it is most closely "
            "related to, from your domain's viewpoint."
        )
        try:
            return llm_call(
                key_queue, self.system_prompt, user,
                name=f"{self.name}_short", temperature=0.0, max_tokens=max_tokens,
            )
        except LLMCallError:
            return ""

BaseTrainableDeepAgent = ScientificDeepAgent

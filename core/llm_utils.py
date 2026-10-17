import json
import re
import time
from typing import Any, Dict, Optional

from core.api_queue import APIKeyQueue

class LLMCallError(RuntimeError):
    pass

def llm_call(
    key_queue: APIKeyQueue,
    system: str,
    user: str,
    *,
    name: str = "llm_call",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    attempts: int = 6,
) -> str:
    effective_max_tokens = int(max_tokens)
    last: Optional[Exception] = None
    model_candidates = [key_queue.model_name] + [
        m for m in getattr(key_queue, "FALLBACK_MODELS", ()) if m != key_queue.model_name
    ]
    model_idx = 0
    for attempt in range(attempts):
        active_model = model_candidates[model_idx % len(model_candidates)]
        call_tokens = min(effective_max_tokens, 950) if "qwen" in active_model else effective_max_tokens
        try:
            with key_queue.lease_key(name) as (_, api_key):
                llm = key_queue.create_chat_model(
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=call_tokens,
                    model_override=active_model,
                )
                resp = llm.invoke(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}]
                )
                text = (resp.content or "").strip()
                if not text and hasattr(resp, "additional_kwargs"):
                    reasoning = str(resp.additional_kwargs.get("reasoning_content") or "").strip()
                    if reasoning:
                        text = reasoning
                if text:
                    return text
                last = LLMCallError("empty response (reasoning exhausted token budget)")
                effective_max_tokens = min(1536, int(effective_max_tokens * 1.2))
        except Exception as e:
            last = e
            err_str = str(e).lower()
            if "otpm" in err_str or "limit 1000" in err_str:
                effective_max_tokens = 900
            if "tokens per day" in err_str or "model_not_found" in err_str or "404" in err_str or "rate_limit" in err_str:
                model_idx += 1
        time.sleep(min(10.0, 1.0 * (1.8 ** attempt)))
    raise LLMCallError(f"{name} failed after {attempts} attempts: {last}")

def extract_json(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        candidate = text[start : end + 1]
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        sanitized = re.sub(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", candidate)
        return json.loads(sanitized, strict=False)

import os
import queue
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq

class ChatOpenAI(ChatGroq):

    base_url: str = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.4,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        api_base = base_url.replace("/openai/v1", "").rstrip("/") if base_url else "https://api.groq.com"
        super().__init__(
            model=model,
            api_key=api_key,
            groq_api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.base_url = base_url

def configure_langsmith_env() -> bool:
    load_dotenv()
    ls_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if ls_key and "YOUR_" not in ls_key:
        os.environ["LANGSMITH_API_KEY"] = ls_key
        os.environ["LANGCHAIN_API_KEY"] = ls_key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        proj = (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "Ael-Agents-Cortex").strip()
        os.environ["LANGSMITH_PROJECT"] = proj
        os.environ["LANGCHAIN_PROJECT"] = proj
        return True
    return False

def load_keys_from_env() -> List[str]:
    configure_langsmith_env()
    keys: List[str] = []
    bulk = os.getenv("GROQ_API_KEYS", "")
    keys.extend(k.strip() for k in bulk.split(",") if k.strip())
    i = 1
    while True:
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if k is None:
            if i > 64:
                break
            i += 1
            continue
        if k.strip() and "YOUR_" not in k.strip():
            keys.append(k.strip())
        i += 1
    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique

class APIKeyQueue:

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model_name: str = "openai/gpt-oss-20b",
    ):
        keys = list(api_keys) if api_keys else load_keys_from_env()
        if not keys:
            raise RuntimeError(
                "No API keys found. Set GROQ_API_KEY_1..N (or GROQ_API_KEYS) in the environment "
                "or in a local .env file. Keys are never stored in source code."
            )
        self.api_keys = keys
        self.base_url = os.getenv("GROQ_BASE_URL", base_url)
        self.model_name = os.getenv("GROQ_MODEL", model_name)

        self._queue: "queue.Queue[Tuple[int, str]]" = queue.Queue()
        for idx, k in enumerate(self.api_keys):
            self._queue.put((idx + 1, k))

        self._lock = threading.Lock()
        self.lease_telemetry: Dict[str, Dict[str, Any]] = {}

    @property
    def size(self) -> int:
        return len(self.api_keys)

    @contextmanager
    def lease_key(self, agent_name: str, timeout: float = 120.0) -> Generator[Tuple[int, str], None, None]:
        t_start = time.time()
        key_id, api_key = self._queue.get(timeout=timeout)
        wait_ms = (time.time() - t_start) * 1000.0
        try:
            with self._lock:
                self.lease_telemetry[agent_name] = {
                    "key_id": f"KEY_{key_id:02d}",
                    "queue_wait_ms": round(wait_ms, 2),
                    "thread_id": threading.get_ident(),
                    "timestamp": time.time(),
                }
            yield key_id, api_key
        finally:
            self._queue.put((key_id, api_key))

    FALLBACK_MODELS: Tuple[str, ...] = (
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
    )

    def create_chat_model(
        self,
        api_key: str,
        temperature: float = 0.4,
        max_tokens: int = 900,
        model_override: Optional[str] = None,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_override or self.model_name,
            api_key=api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

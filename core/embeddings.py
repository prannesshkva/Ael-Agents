import threading
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

class EncoderInfo(dict):

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    @property
    def __dict__(self) -> Dict[str, Any]:
        return dict(self)

class SemanticEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        allow_fallback: bool = True,
        fallback_dim: int = 256,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.device = device
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._backend_type = "none"
        self.is_fallback = False
        self.dim: int = 0
        self._load_error: Optional[str] = None

        loaded = False
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(model_name, device=device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(model_name, device=device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self._backend_type = "sentence-transformers"
            loaded = True
        except Exception as st_err:
            try:
                from transformers import AutoModel, AutoTokenizer

                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    self._model = AutoModel.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._model = AutoModel.from_pretrained(model_name)
                self._model.to(device)
                self._model.eval()
                self.dim = int(self._model.config.hidden_size)
                self._backend_type = "transformers-bge"
                loaded = True
            except Exception as hf_err:
                self._load_error = f"ST({type(st_err).__name__}: {st_err}); HF({type(hf_err).__name__}: {hf_err})"

        if not loaded:
            if not allow_fallback:
                raise RuntimeError(f"Could not load embedding model '{model_name}': {self._load_error}")
            warnings.warn(
                f"Could not load neural encoder '{model_name}' ({self._load_error}). "
                "Falling back to LEXICAL hashed embeddings; semantic claims are NOT valid.",
                RuntimeWarning,
            )
            self._init_fallback(fallback_dim, seed)

    def _init_fallback(self, dim: int, seed: int) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.is_fallback = True
        self._backend_type = "hashed-lexical-FALLBACK"
        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=2048, ngram_range=(1, 2), stop_words="english", norm="l2", alternate_sign=True
        )
        rng = np.random.RandomState(seed)
        proj = (rng.randn(2048, dim) / np.sqrt(dim)).astype(np.float32)
        self._proj = torch.from_numpy(proj)

    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo({
            "backend": self._backend_type,
            "model": None if self.is_fallback else self.model_name,
            "model_name": "hashed-lexical-FALLBACK" if self.is_fallback else self.model_name,
            "dim": self.dim,
            "semantic_claims_valid": not self.is_fallback,
            "load_error": self._load_error,
        })

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        if self.is_fallback:
            rows = []
            for t in texts:
                v = torch.from_numpy(self._vectorizer.transform([t]).toarray()[0].astype(np.float32))
                rows.append(torch.mv(self._proj.t(), v))
            return F.normalize(torch.stack(rows), p=2, dim=-1, eps=1e-8)

        with self._lock:
            if self._backend_type == "sentence-transformers":
                emb = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return F.normalize(emb.detach().to(torch.float32).cpu(), p=2, dim=-1, eps=1e-8)

            all_vecs = []
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    out = self._model(**encoded)
                    cls_emb = out.last_hidden_state[:, 0]
                    cls_emb = F.normalize(cls_emb.to(torch.float32), p=2, dim=-1, eps=1e-8)
                    all_vecs.append(cls_emb.cpu())
            return torch.cat(all_vecs, dim=0)

    def encode(self, text: str, device: str = "cpu") -> torch.Tensor:
        return self.encode_batch([text])[0].to(device)

_ENCODER: Optional[SemanticEncoder] = None
_ENCODER_LOCK = threading.Lock()

def get_encoder(model_name: str = DEFAULT_MODEL) -> SemanticEncoder:
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None or (_ENCODER.model_name != model_name and not _ENCODER.is_fallback):
            _ENCODER = SemanticEncoder(model_name=model_name)
        return _ENCODER

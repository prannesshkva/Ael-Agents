import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import yaml

from agents.base_deep_agent import ScientificDeepAgent
from agents.domain_specialists import DOMAIN_SPECS_11
from core.embeddings import SemanticEncoder
from core.llm_utils import extract_json, llm_call

ROSTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rosters")

@dataclass
class DomainSpec:
    name: str
    domain_title: str
    focus: str = ""
    system_prompt: Optional[str] = None

    @property
    def description(self) -> str:
        return self.focus or self.domain_title

    def prompt_for(self, index: int) -> str:
        if self.system_prompt:
            return self.system_prompt
        return (
            f"You are the {self.domain_title} Deep-Agent (Parameter w_{index}). "
            f"Analyze the problem EXCLUSIVELY through {self.focus}. "
            "State the governing principles, mechanisms and assumptions of your field that bear on "
            "the problem, flag where your field's knowledge is uncertain, and do not borrow "
            "formalisms from other fields. Keep your response sharp, dense, and under 180 words."
        )

def default_roster() -> List[DomainSpec]:
    return [
        DomainSpec(
            name=s["name"],
            domain_title=s["domain_title"],
            focus=s["domain_title"],
            system_prompt=s["system_prompt"],
        )
        for s in DOMAIN_SPECS_11
    ]

def load_roster(name_or_path: str = "default") -> List[DomainSpec]:
    if name_or_path in ("default", "default_11_sciences"):
        return default_roster()
    path = name_or_path
    if not os.path.isfile(path):
        path = os.path.join(ROSTER_DIR, f"{name_or_path}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Roster '{name_or_path}' not found (looked for {path}).")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _specs_from_dicts(data["domains"])

def _specs_from_dicts(items: List[Dict[str, Any]]) -> List[DomainSpec]:
    specs, seen = [], set()
    for it in items:
        name = "".join(ch for ch in str(it["name"]) if ch.isalnum() or ch == "_")
        if not name or name in seen:
            raise ValueError(f"Roster has an empty or duplicate domain name: {it.get('name')!r}")
        seen.add(name)
        specs.append(
            DomainSpec(
                name=name,
                domain_title=str(it.get("title", name)),
                focus=str(it.get("focus", it.get("title", name))),
                system_prompt=it.get("system_prompt"),
            )
        )
    if len(specs) < 2:
        raise ValueError("A roster needs at least 2 domains.")
    return specs

def create_agents(roster: List[DomainSpec]) -> List[ScientificDeepAgent]:
    return [
        ScientificDeepAgent(
            name=s.name,
            domain_title=s.domain_title,
            param_index=i + 1,
            system_prompt=s.prompt_for(i + 1),
        )
        for i, s in enumerate(roster)
    ]

def propose_roster(question: str, key_queue, n: int = 4) -> List[DomainSpec]:
    system = (
        "You are a Computational Drug Discovery & Therapeutics Architect. Study the problem and segregate "
        "it into mutually exclusive sub-expert pieces. Output ONLY a JSON object."
    )
    user = (
        f"Therapeutic research question:\n{question}\n\n"
        f"Study this problem and segregate it into exactly {n} mutually exclusive, non-overlapping "
        "sub-expert pieces whose combined analysis solves the complete therapeutic challenge. "
        "Return JSON: {\"domains\": [{\"name\": \"CamelCaseNoSpaces\", "
        "\"title\": \"Sub-discipline title\", \"focus\": \"the exact segregated piece, equations, and "
        "mechanisms this expert owns\"}]}"
    )
    last: Optional[Exception] = None
    for _ in range(3):
        try:
            raw = llm_call(key_queue, system, user, name="roster_planner", temperature=0.3, max_tokens=1024)
            specs = _specs_from_dicts(extract_json(raw)["domains"])
            if len(specs) == n:
                return specs
            last = ValueError(f"planner returned {len(specs)} domains, expected {n}")
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not obtain a valid roster from the planner: {last}")

def domain_prior_distance(roster: List[DomainSpec], encoder: SemanticEncoder) -> torch.Tensor:
    texts = [f"{s.domain_title}: {s.focus}" for s in roster]
    E = encoder.encode_batch(texts)
    sim = torch.clamp(E @ E.t(), -1.0, 1.0)
    dist = 1.0 - sim
    n = len(roster)
    off = dist[~torch.eye(n, dtype=torch.bool)]
    lo, hi = float(off.min()), float(off.max())
    scaled = 0.3 + 0.7 * (dist - lo) / max(hi - lo, 1e-8)
    return scaled * (1.0 - torch.eye(n))

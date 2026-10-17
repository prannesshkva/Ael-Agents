import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

SCIENTIFIC_DOMAINS_11 = [
    "StructuralBiophysics",
    "SystemsPharmacology",
    "MedicinalChemistryADMET",
    "ClinicalPKPDControl",
]

class DomainParameterRegistry(nn.Module):

    def __init__(
        self,
        domain_names: Optional[List[str]] = None,
        init_value: float = 1.0,
        temperature: float = 0.5
    ):
        super().__init__()
        self.domain_names = list(domain_names or SCIENTIFIC_DOMAINS_11)
        self.num_domains = len(self.domain_names)
        self.temperature = temperature

        self.domain_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.domain_names)
        }

        self.weights = nn.Parameter(
            torch.full((self.num_domains,), float(init_value), dtype=torch.float32)
        )

    def get_agent_parameter(self, domain_name: str) -> torch.Tensor:
        idx = self.domain_to_idx[domain_name]
        return self.weights[idx]

    def attach_parameter_to_embedding(
        self,
        domain_name: str,
        embedding: torch.Tensor
    ) -> torch.Tensor:
        w_i = self.get_agent_parameter(domain_name)
        return w_i * embedding

    def get_gating_distribution(self, weights_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = weights_override if weights_override is not None else self.weights
        return F.softmax(w / self.temperature, dim=0)

    def get_parameter_table(self) -> List[Dict[str, float]]:
        with torch.no_grad():
            alphas = self.get_gating_distribution()
            rows = []
            for idx, name in enumerate(self.domain_names):
                grad_val = (
                    float(self.weights.grad[idx].item())
                    if self.weights.grad is not None
                    else 0.0
                )
                rows.append({
                    "param_id": f"w_{idx + 1}",
                    "domain": name,
                    "value": float(self.weights[idx].item()),
                    "alpha_gate": float(alphas[idx].item()),
                    "grad": grad_val
                })
            return rows

    def total_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

AgentParameterRegistry = DomainParameterRegistry

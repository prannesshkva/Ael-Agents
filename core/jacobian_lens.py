import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

class SalienceJacobianNetwork:

    def __init__(
        self,
        temperature: float = 0.5,
        interaction_gamma: float = 0.35,
        working_memory_capacity: int = 5,
        top_matches_k: int = 3,
        use_similarity: bool = True,
        use_jacobian: bool = True,
    ):
        self.temperature = temperature
        self.interaction_gamma = interaction_gamma
        self.working_memory_capacity = working_memory_capacity
        self.top_matches_k = top_matches_k
        self.use_similarity = use_similarity
        self.use_jacobian = use_jacobian

    def combine_with_11_parameters(
        self,
        weights_11: torch.Tensor,
        embeddings_11: torch.Tensor,
        sim_matrix: torch.Tensor
    ) -> torch.Tensor:
        alphas = F.softmax(weights_11 / self.temperature, dim=0)
        effective_w = 0.5 * weights_11 + 0.5 * (alphas * len(weights_11))

        primary_field = torch.mv(embeddings_11.t(), effective_w)

        N = weights_11.shape[0]
        coupling_field = torch.zeros_like(primary_field)
        if self.use_similarity:
            for i in range(N):
                for j in range(i + 1, N):
                    pair_weight = alphas[i] * alphas[j] * sim_matrix[i, j]
                    bridge_vec = 0.5 * (embeddings_11[i] + embeddings_11[j])
                    coupling_field = coupling_field + pair_weight * bridge_vec

        y = primary_field + self.interaction_gamma * coupling_field
        return F.normalize(y, p=2, dim=-1, eps=1e-8)

    def compute_jacobian_and_salience(
        self,
        weights_11: torch.Tensor,
        domain_embeddings: Dict[str, torch.Tensor],
        domain_names: List[str],
        sim_matrix: torch.Tensor,
        dmn_bridges: List[Dict[str, Any]],
        create_graph: bool = False,
        working_memory_capacity: Optional[int] = None,
        enforce_match_domains: bool = True,
    ) -> Dict[str, Any]:
        E = torch.stack([domain_embeddings[n] for n in domain_names], dim=0).detach()
        S = sim_matrix.detach()

        def forward_w(w: torch.Tensor) -> torch.Tensor:
            return self.combine_with_11_parameters(w, E, S)

        y_synth = forward_w(weights_11)

        J = torch.autograd.functional.jacobian(
            forward_w,
            weights_11,
            create_graph=create_graph,
            strict=False
        )

        G = torch.mm(J.t(), J)

        sensitivities_tensor = torch.norm(J, p=2, dim=0)
        alphas_tensor = F.softmax(weights_11 / self.temperature, dim=0)

        sensitivities_dict = {
            domain_names[i]: round(float(sensitivities_tensor[i].detach().item()), 4)
            for i in range(len(domain_names))
        }
        gating_dict = {
            domain_names[i]: round(float(alphas_tensor[i].detach().item()), 4)
            for i in range(len(domain_names))
        }

        scored_matches = []
        for bridge in dmn_bridges:
            i, j = bridge["idx_a"], bridge["idx_b"]
            s_i = float(sensitivities_tensor[i].detach().item())
            s_j = float(sensitivities_tensor[j].detach().item())
            g_ij = abs(float(G[i, j].detach().item()))
            w_i = float(alphas_tensor[i].detach().item())
            w_j = float(alphas_tensor[j].detach().item())

            param_priority = (w_i * w_j) ** 0.5
            if self.use_jacobian:
                jacobian_leverage = ((s_i * s_j) ** 0.5) * (1.0 + g_ij)
                leverage_factor = 0.5 + jacobian_leverage
            else:
                leverage_factor = 1.0
            salience_score = bridge["dmn_imagination_score"] * leverage_factor * (0.5 + param_priority)

            scored_matches.append({
                **bridge,
                "jacobian_co_sensitivity": round(g_ij, 5),
                "salience_beauty_score": round(float(salience_score), 5)
            })

        scored_matches.sort(key=lambda x: x["salience_beauty_score"], reverse=True)
        top_beautiful_matches = scored_matches[: self.top_matches_k]

        if self.use_jacobian:
            domain_salience = {
                n: gating_dict[n] * 0.65 + sensitivities_dict[n] * 0.35 for n in domain_names
            }
        else:
            domain_salience = {n: gating_dict[n] for n in domain_names}
        domain_salience_rank = sorted(domain_names, key=lambda n: domain_salience[n], reverse=True)

        capacity = int(working_memory_capacity or self.working_memory_capacity)
        capacity = max(1, min(capacity, len(domain_names)))
        working_memory_domains: List[str] = []
        if enforce_match_domains and top_beautiful_matches and capacity >= 2:
            for d_key in ("domain_a", "domain_b"):
                d_name = top_beautiful_matches[0][d_key]
                if d_name not in working_memory_domains:
                    working_memory_domains.append(d_name)
        for n in domain_salience_rank:
            if len(working_memory_domains) >= capacity:
                break
            if n not in working_memory_domains:
                working_memory_domains.append(n)

        return {
            "synthesized_tensor": y_synth,
            "jacobian_matrix": J,
            "jacobian_gram_matrix": G,
            "sensitivities_tensor": sensitivities_tensor,
            "jacobian_sensitivities": sensitivities_dict,
            "gating_weights": gating_dict,
            "domain_salience": domain_salience,
            "domain_salience_rank": domain_salience_rank,
            "salient_beautiful_matches": top_beautiful_matches,
            "working_memory_domains": working_memory_domains
        }

ExternalJacobianLens = SalienceJacobianNetwork

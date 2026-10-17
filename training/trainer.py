import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from core.parameter_registry import DomainParameterRegistry
from core.rsa_engine import DefaultModeRSAEngine
from core.jacobian_lens import SalienceJacobianNetwork
from training.loss_functions import UnifiedCortexLoss

class CortexTrainer:

    def __init__(
        self,
        param_registry: DomainParameterRegistry,
        dmn_engine: DefaultModeRSAEngine,
        salience_network: SalienceJacobianNetwork,
        loss_fn: UnifiedCortexLoss,
        learning_rate: float = 0.08
    ):
        self.param_registry = param_registry
        self.dmn_engine = dmn_engine
        self.salience_network = salience_network
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam([self.param_registry.weights], lr=learning_rate)

    def build_salience_guided_target(
        self,
        state: Dict[str, Any],
        domain_names: List[str]
    ) -> torch.Tensor:
        cen_emb = state["metadata"]["cen_target_embedding"].detach()
        E = torch.stack([state["domain_embeddings"][n].detach() for n in domain_names], dim=0)

        match_boost = torch.zeros_like(cen_emb)
        for rank, m in enumerate(state["salient_beautiful_matches"]):
            weight = 1.0 / (rank + 1.0)
            e_a = E[m["idx_a"]]
            e_b = E[m["idx_b"]]
            match_boost = match_boost + weight * 0.5 * (e_a + e_b)

        target_y = F.normalize(0.55 * cen_emb + 0.45 * F.normalize(match_boost, p=2, dim=-1), p=2, dim=-1)
        return target_y

    def train_step_on_state(
        self,
        state: Dict[str, Any],
        domain_names: List[str],
        target_y: torch.Tensor
    ) -> Dict[str, float]:
        self.optimizer.zero_grad()

        sn_out = self.salience_network.compute_jacobian_and_salience(
            weights_11=self.param_registry.weights,
            domain_embeddings=state["domain_embeddings"],
            domain_names=domain_names,
            sim_matrix=state["rsa_similarity_matrix"],
            dmn_bridges=state["dmn_spontaneous_bridges"],
            create_graph=True
        )

        total_loss, loss_metrics = self.loss_fn(
            weights_11=self.param_registry.weights,
            synthesized_y=sn_out["synthesized_tensor"],
            target_y=target_y,
            salient_matches=sn_out["salient_beautiful_matches"],
            jacobian_sensitivities_tensor=sn_out["sensitivities_tensor"]
        )

        total_loss.backward()
        grad_norm = float(torch.norm(self.param_registry.weights.grad, p=2).item())
        self.optimizer.step()

        return {
            **loss_metrics,
            "train/grad_norm_11_params": round(grad_norm, 5)
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple

class UnifiedCortexLoss(nn.Module):

    def __init__(
        self,
        task_loss_weight: float = 1.0,
        rsa_salience_weight: float = 0.45,
        jacobian_loss_weight: float = 0.10,
        temperature: float = 0.5
    ):
        super().__init__()
        self.task_loss_weight = task_loss_weight
        self.rsa_salience_weight = rsa_salience_weight
        self.jacobian_loss_weight = jacobian_loss_weight
        self.temperature = temperature

    def forward(
        self,
        weights_11: torch.Tensor,
        synthesized_y: torch.Tensor,
        target_y: torch.Tensor,
        salient_matches: List[Dict[str, Any]],
        jacobian_sensitivities_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        norm_y = F.normalize(synthesized_y, p=2, dim=-1)
        norm_t = F.normalize(target_y, p=2, dim=-1)
        cosine_dist = 1.0 - torch.sum(norm_y * norm_t)
        mse_dist = F.mse_loss(norm_y, norm_t)
        task_loss = cosine_dist + 0.5 * mse_dist

        alphas = F.softmax(weights_11 / self.temperature, dim=0)
        salience_reward = torch.tensor(0.0, device=weights_11.device)
        for m in salient_matches:
            i, j = m["idx_a"], m["idx_b"]
            beauty = float(m["salience_beauty_score"])
            salience_reward = salience_reward + (alphas[i] + alphas[j]) * beauty

        uniform_kl = torch.sum(alphas * torch.log(alphas * len(weights_11) + 1e-8))
        rsa_salience_loss = F.relu(0.5 - salience_reward) + 0.15 * uniform_kl

        jacobian_loss = torch.mean(jacobian_sensitivities_tensor ** 2)

        total_loss = (
            self.task_loss_weight * task_loss
            + self.rsa_salience_weight * rsa_salience_loss
            + self.jacobian_loss_weight * jacobian_loss
        )

        metrics = {
            "loss/total": float(total_loss.detach().item()),
            "loss/task": float(task_loss.detach().item()),
            "loss/rsa_salience": float(rsa_salience_loss.detach().item()),
            "loss/jacobian": float(jacobian_loss.detach().item())
        }
        return total_loss, metrics

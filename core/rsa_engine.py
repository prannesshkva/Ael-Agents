import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from .parameter_registry import SCIENTIFIC_DOMAINS_11

DOMAIN_EPISTEMIC_COORDS = {
    "ClassicalPhysics": [1.0, 0.5, -1.0],
    "QuantumPhysics":   [1.0, -1.0, -1.0],
    "Chemistry":        [0.6, -0.7, -0.3],
    "Biology":          [-0.2, -0.2, 1.0],
    "Mathematics":      [1.0, 0.0, 0.0],
    "ComputerScience":  [0.8, 0.3, 0.0],
    "Neuroscience":     [-0.4, 0.2, 0.9],
    "MaterialsScience": [0.8, -0.4, -0.8],
    "EarthScience":     [0.5, 1.0, -0.2],
    "Biomedicine":      [-0.6, -0.1, 1.0],
    "Cybernetics":      [0.4, 0.6, 0.4],
}

class DefaultModeRSAEngine(nn.Module):

    def __init__(self, diversity_margin: float = 0.25, eps: float = 1e-8):
        super().__init__()
        self.diversity_margin = diversity_margin
        self.eps = eps

    def compute_similarity_and_rdm(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = embeddings.shape[0]
        norm_e = F.normalize(embeddings, p=2, dim=-1, eps=self.eps)
        sim_matrix = torch.clamp(torch.mm(norm_e, norm_e.t()), -1.0, 1.0)
        rdm = (1.0 - sim_matrix) * (1.0 - torch.eye(N, device=embeddings.device))
        return sim_matrix, rdm

    def compute_epistemic_prior_distance(self, domain_names: List[str], device: torch.device) -> torch.Tensor:
        coords = [DOMAIN_EPISTEMIC_COORDS.get(name, [0.0, 0.0, 0.0]) for name in domain_names]
        c_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        dist = torch.cdist(c_tensor, c_tensor, p=2)
        max_d = dist.max().clamp(min=self.eps)
        norm_dist = 0.3 + 0.7 * (dist / max_d)
        norm_dist = norm_dist * (1.0 - torch.eye(len(domain_names), device=device))
        return norm_dist

    def compute_participation_ratio(self, embeddings: torch.Tensor) -> float:
        with torch.no_grad():
            norm_e = F.normalize(embeddings, p=2, dim=-1, eps=self.eps)
            gram = torch.mm(norm_e, norm_e.t())
            eigvals = torch.clamp(torch.linalg.eigvalsh(gram), min=0.0)
            sum_eig = torch.sum(eigvals)
            sum_sq = torch.sum(eigvals ** 2)
            return float(((sum_eig ** 2) / (sum_sq + self.eps)).item())

    def forward(
        self,
        domain_embeddings: Dict[str, torch.Tensor],
        domain_names: Optional[List[str]] = None,
        prior_dist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], Dict[str, float]]:
        names = list(domain_names or sorted(domain_embeddings.keys()))
        E = torch.stack([domain_embeddings[n] for n in names], dim=0)

        sim_matrix, rdm = self.compute_similarity_and_rdm(E)
        if prior_dist is None:
            prior_dist = self.compute_epistemic_prior_distance(names, E.device)
        else:
            prior_dist = prior_dist.to(E.device)

        N = len(names)
        dmn_bridges = []
        for i in range(N):
            for j in range(i + 1, N):
                raw_sim = float(sim_matrix[i, j].detach().item())
                resonance = max(0.0, (raw_sim + 1.0) / 2.0)
                epistemic_surprise = float(prior_dist[i, j].detach().item())
                dmn_bridges.append({
                    "domain_a": names[i],
                    "domain_b": names[j],
                    "idx_a": i,
                    "idx_b": j,
                    "cosine_similarity": round(raw_sim, 4),
                    "rdm_dissimilarity": round(float(rdm[i, j].detach().item()), 4),
                    "structural_resonance": round(resonance, 4),
                    "epistemic_surprise": round(epistemic_surprise, 4),
                    "dmn_imagination_score": round(resonance * epistemic_surprise, 4)
                })

        dmn_bridges.sort(key=lambda x: x["dmn_imagination_score"], reverse=True)

        pr = self.compute_participation_ratio(E)
        triu = torch.triu_indices(N, N, offset=1)
        mean_rdm = float(rdm[triu[0], triu[1]].mean().detach().item())

        metrics = {
            "dmn_rsa/participation_ratio": round(pr, 4),
            "dmn_rsa/mean_dissimilarity": round(mean_rdm, 4),
            "dmn_rsa/candidate_bridges_count": len(dmn_bridges)
        }

        return sim_matrix, rdm, dmn_bridges, metrics

ExternalRSALens = DefaultModeRSAEngine

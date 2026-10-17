from typing import Dict, List, Any, Optional, TypedDict, Annotated
import torch

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    res = dict(a or {})
    if b:
        res.update(b)
    return res

def append_list(a: List[Any], b: List[Any]) -> List[Any]:
    return list(a or []) + list(b or [])

class AgenticCortexState(TypedDict):
    task: str
    messages: Annotated[List[Dict[str, Any]], append_list]

    domain_solutions: Annotated[Dict[str, str], merge_dicts]
    domain_embeddings: Annotated[Dict[str, torch.Tensor], merge_dicts]
    domain_weighted_tensors: Annotated[Dict[str, torch.Tensor], merge_dicts]
    domain_telemetry: Annotated[Dict[str, Dict[str, Any]], merge_dicts]

    rsa_similarity_matrix: Optional[torch.Tensor]
    rdm_matrix: Optional[torch.Tensor]
    dmn_spontaneous_bridges: List[Dict[str, Any]]

    jacobian_matrix: Optional[torch.Tensor]
    jacobian_gram_matrix: Optional[torch.Tensor]
    jacobian_sensitivities: Dict[str, float]
    gating_weights: Dict[str, float]
    salient_beautiful_matches: List[Dict[str, Any]]
    working_memory_domains: List[str]
    domain_salience: Dict[str, float]

    synthesized_tensor: Optional[torch.Tensor]
    final_solution: str

    evaluation_scores: Dict[str, float]
    metadata: Annotated[Dict[str, Any], merge_dicts]

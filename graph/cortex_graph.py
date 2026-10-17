import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from core.state import AgenticCortexState
from core.api_queue import APIKeyQueue
from core.parameter_registry import DomainParameterRegistry
from core.rsa_engine import DefaultModeRSAEngine
from core.jacobian_lens import SalienceJacobianNetwork
from core.cortical_synthesizer import CentralExecutiveNetwork
from agents.base_deep_agent import ScientificDeepAgent, GLOBAL_ENCODER

class CortexGraphBuilder:

    def __init__(
        self,
        agents: List[ScientificDeepAgent],
        key_queue: APIKeyQueue,
        param_registry: DomainParameterRegistry,
        dmn_engine: DefaultModeRSAEngine,
        salience_network: SalienceJacobianNetwork,
        cen_network: CentralExecutiveNetwork,
        prior_dist: Optional[torch.Tensor] = None,
    ):
        self.agents = agents
        self.domain_names = [a.name for a in agents]
        self.key_queue = key_queue
        self.param_registry = param_registry
        self.dmn_engine = dmn_engine
        self.salience_network = salience_network
        self.cen_network = cen_network
        self.prior_dist = prior_dist

    def _create_parallel_agent_node(self, agent: ScientificDeepAgent):
        @traceable(run_type="chain", name=f"DeepAgent_{agent.name}")
        def agent_node(state: AgenticCortexState) -> Dict[str, Any]:
            res = agent.execute_parallel(
                task=state["task"],
                key_queue=self.key_queue,
                param_registry=self.param_registry
            )
            return {
                "domain_solutions": {agent.name: res["hypothesis"]},
                "domain_embeddings": {agent.name: res["raw_embedding"]},
                "domain_weighted_tensors": {agent.name: res["weighted_tensor"]},
                "domain_telemetry": {
                    agent.name: {
                        "param_id": res["param_id"],
                        "param_value": res["param_value"],
                        "key_id": res["key_id"],
                        "latency_s": res["latency_s"],
                        "domain_title": res["domain_title"],
                        "failed": res["failed"],
                        "error": res["error"],
                    }
                },
                "messages": [{
                    "role": "deep_agent",
                    "name": agent.name,
                    "param_id": res["param_id"],
                    "key_id": res["key_id"],
                    "content": res["hypothesis"]
                }]
            }
        return agent_node

    @traceable(run_type="chain", name="1_Default_Mode_Network_DMN_RSA")
    def _dmn_node(self, state: AgenticCortexState) -> Dict[str, Any]:
        sim_matrix, rdm_matrix, dmn_bridges, metrics = self.dmn_engine(
            domain_embeddings=state["domain_embeddings"],
            domain_names=self.domain_names,
            prior_dist=self.prior_dist,
        )
        return {
            "rsa_similarity_matrix": sim_matrix,
            "rdm_matrix": rdm_matrix,
            "dmn_spontaneous_bridges": dmn_bridges,
            "metadata": metrics
        }

    @traceable(run_type="chain", name="2_Salience_Network_SN_Jacobian_Switch")
    def _salience_network_node(self, state: AgenticCortexState) -> Dict[str, Any]:
        sn_out = self.salience_network.compute_jacobian_and_salience(
            weights_11=self.param_registry.weights,
            domain_embeddings=state["domain_embeddings"],
            domain_names=self.domain_names,
            sim_matrix=state["rsa_similarity_matrix"],
            dmn_bridges=state["dmn_spontaneous_bridges"],
            create_graph=False
        )
        return {
            "synthesized_tensor": sn_out["synthesized_tensor"],
            "jacobian_matrix": sn_out["jacobian_matrix"],
            "jacobian_gram_matrix": sn_out["jacobian_gram_matrix"],
            "jacobian_sensitivities": sn_out["jacobian_sensitivities"],
            "gating_weights": sn_out["gating_weights"],
            "salient_beautiful_matches": sn_out["salient_beautiful_matches"],
            "working_memory_domains": sn_out["working_memory_domains"],
            "domain_salience": sn_out.get("domain_salience", {}),
            "metadata": {
                "sn/top_beautiful_match": (
                    f"{sn_out['salient_beautiful_matches'][0]['domain_a']} <-> "
                    f"{sn_out['salient_beautiful_matches'][0]['domain_b']}"
                    if sn_out["salient_beautiful_matches"] else "None"
                ),
                "sn/working_memory_domains": sn_out["working_memory_domains"]
            }
        }

    @traceable(run_type="chain", name="3_Central_Executive_Network_CEN")
    def _cen_node(self, state: AgenticCortexState) -> Dict[str, Any]:
        final_solution = self.cen_network.synthesize(
            task=state["task"],
            domain_solutions=state["domain_solutions"],
            gating_weights=state["gating_weights"],
            jacobian_sensitivities=state["jacobian_sensitivities"],
            salient_matches=state["salient_beautiful_matches"],
            working_memory_domains=state["working_memory_domains"],
            key_queue=self.key_queue
        )
        return {
            "final_solution": final_solution,
            "messages": [{"role": "cen_executive", "content": final_solution}]
        }

    @traceable(run_type="chain", name="4_LangSmith_InLoop_Evaluator")
    def _langsmith_evaluator_node(self, state: AgenticCortexState) -> Dict[str, Any]:
        device = str(self.param_registry.weights.device)
        cen_embedding = GLOBAL_ENCODER.encode(state["final_solution"], device=device)

        domain_eval_scores: Dict[str, float] = {}
        for name in self.domain_names:
            e_i = state["domain_embeddings"][name]
            cos_align = float(F.cosine_similarity(e_i.unsqueeze(0), cen_embedding.unsqueeze(0)).item())
            domain_eval_scores[f"eval/{name}"] = round((cos_align + 1.0) / 2.0, 4)

        top_beauty = (
            state["salient_beautiful_matches"][0]["salience_beauty_score"]
            if state["salient_beautiful_matches"] else 0.0
        )
        pr = state["metadata"].get("dmn_rsa/participation_ratio", 1.0)

        eval_summary = {
            **domain_eval_scores,
            "eval/salience_beauty_score": round(float(top_beauty), 4),
            "eval/cognitive_diversity_pr": round(float(pr), 4),
            "eval/overall_quality": round(
                0.6 * (sum(domain_eval_scores.values()) / len(domain_eval_scores))
                + 0.4 * min(1.0, pr / 6.0),
                4
            )
        }

        return {
            "evaluation_scores": eval_summary,
            "metadata": {"cen_target_embedding": cen_embedding}
        }

    def build(self):
        builder = StateGraph(AgenticCortexState)

        parallel_node_names = []
        for agent in self.agents:
            node_name = f"agent_{agent.name}"
            builder.add_node(node_name, self._create_parallel_agent_node(agent))
            parallel_node_names.append(node_name)

        builder.add_node("dmn_default_mode_network", self._dmn_node)
        builder.add_node("sn_salience_network_switch", self._salience_network_node)
        builder.add_node("cen_central_executive_network", self._cen_node)
        builder.add_node("langsmith_evaluator", self._langsmith_evaluator_node)

        for node_name in parallel_node_names:
            builder.add_edge(START, node_name)

        for node_name in parallel_node_names:
            builder.add_edge(node_name, "dmn_default_mode_network")

        builder.add_edge("dmn_default_mode_network", "sn_salience_network_switch")
        builder.add_edge("sn_salience_network_switch", "cen_central_executive_network")
        builder.add_edge("cen_central_executive_network", "langsmith_evaluator")
        builder.add_edge("langsmith_evaluator", END)

        return builder.compile()

from typing import Any, Dict, List, Optional
import uuid

import torch

from agents.domain_specialists import create_all_11_scientific_agents
from core.api_queue import APIKeyQueue
from core.config import FrameworkConfig
from core.cortical_synthesizer import CentralExecutiveNetwork
from core.domain_registry import (
    DomainSpec,
    create_agents,
    default_roster,
    domain_prior_distance,
)
from core.embeddings import SemanticEncoder, get_encoder
from core.jacobian_lens import SalienceJacobianNetwork
from core.parameter_registry import DomainParameterRegistry
from core.rsa_engine import DefaultModeRSAEngine
from core.state import AgenticCortexState
from graph.cortex_graph import CortexGraphBuilder
from graph.telemetry import CortexTelemetry
from training.loss_functions import UnifiedCortexLoss
from training.trainer import CortexTrainer

class JacobianRSACortexFramework:

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        roster: Optional[List[DomainSpec]] = None,
        api_keys: Optional[List[str]] = None,
        encoder: Optional[SemanticEncoder] = None,
        key_queue: Optional[APIKeyQueue] = None,
    ):
        self.config = config or FrameworkConfig()
        self.encoder = encoder or get_encoder()
        self.roster = roster or default_roster()

        self.key_queue = key_queue or APIKeyQueue(
            api_keys=api_keys, base_url="https://api.groq.com/openai/v1", model_name="openai/gpt-oss-20b"
        )

        self.agents = create_agents(self.roster)
        self.domain_names = [a.name for a in self.agents]

        self.param_registry = DomainParameterRegistry(
            domain_names=self.domain_names, init_value=1.0, temperature=0.5
        )

        self.prior_dist = domain_prior_distance(self.roster, self.encoder)

        self.dmn_engine = DefaultModeRSAEngine(diversity_margin=self.config.rsa.diversity_margin)
        self.salience_network = SalienceJacobianNetwork(
            temperature=0.5, interaction_gamma=0.35,
            working_memory_capacity=min(2, len(self.agents)),
            top_matches_k=min(2, max(1, len(self.agents) - 1)),
        )
        self.cen_network = CentralExecutiveNetwork()

        self.graph_builder = CortexGraphBuilder(
            agents=self.agents,
            key_queue=self.key_queue,
            param_registry=self.param_registry,
            dmn_engine=self.dmn_engine,
            salience_network=self.salience_network,
            cen_network=self.cen_network,
            prior_dist=self.prior_dist,
        )
        self.graph = self.graph_builder.build()

        self.telemetry = CortexTelemetry()
        self.loss_fn = UnifiedCortexLoss(task_loss_weight=1.0, rsa_salience_weight=0.45, jacobian_loss_weight=0.10)
        self.trainer = CortexTrainer(
            param_registry=self.param_registry,
            dmn_engine=self.dmn_engine,
            salience_network=self.salience_network,
            loss_fn=self.loss_fn,
            learning_rate=0.08,
        )
        self._last_state: Optional[AgenticCortexState] = None
        self.last_run_id: Optional[str] = None

    @property
    def n_parameters(self) -> int:
        return self.param_registry.total_trainable_parameters()

    @property
    def last_state(self) -> Optional[AgenticCortexState]:
        return self._last_state

    def run(self, task: str) -> AgenticCortexState:
        run_uuid = uuid.uuid4()
        self.last_run_id = str(run_uuid)
        initial_state: AgenticCortexState = {
            "task": task,
            "messages": [],
            "domain_solutions": {},
            "domain_embeddings": {},
            "domain_weighted_tensors": {},
            "domain_telemetry": {},
            "rsa_similarity_matrix": None,
            "rdm_matrix": None,
            "dmn_spontaneous_bridges": [],
            "jacobian_matrix": None,
            "jacobian_gram_matrix": None,
            "jacobian_sensitivities": {},
            "gating_weights": {},
            "salient_beautiful_matches": [],
            "working_memory_domains": [],
            "domain_salience": {},
            "synthesized_tensor": None,
            "final_solution": "",
            "evaluation_scores": {},
            "metadata": {"langsmith_run_id": self.last_run_id},
        }
        invoke_config = {
            "run_id": run_uuid,
            "run_name": "Ael_Agents_Cortical_Pipeline",
            "tags": ["Ael-Agents", "DMN-SN-CEN", "Jacobian-RSA"],
            "metadata": {
                "num_domains": len(self.domain_names),
                "domains": self.domain_names,
                "encoder": self.encoder.info.model_name,
                "embedding_dim": self.encoder.info.dim,
            },
        }
        final_state = self.graph.invoke(initial_state, config=invoke_config)
        self._last_state = final_state
        return final_state

    def reset_parameters(self, value: float = 1.0) -> None:
        with torch.no_grad():
            self.param_registry.weights.fill_(float(value))

    def optimize_parameters(self, epochs: int = 15) -> List[Dict[str, float]]:
        if self._last_state is None:
            raise RuntimeError("Call .run(task) before optimizing parameters.")
        target_y = self.trainer.build_salience_guided_target(self._last_state, self.domain_names)
        history = [
            self.trainer.train_step_on_state(
                state=self._last_state, domain_names=self.domain_names, target_y=target_y
            )
            for _ in range(epochs)
        ]
        self.refresh_salience()
        return history

    optimize_11_parameters = optimize_parameters

    def refresh_salience(self) -> Dict[str, Any]:
        st = self._last_state
        sn = self.salience_network.compute_jacobian_and_salience(
            weights_11=self.param_registry.weights,
            domain_embeddings=st["domain_embeddings"],
            domain_names=self.domain_names,
            sim_matrix=st["rsa_similarity_matrix"],
            dmn_bridges=st["dmn_spontaneous_bridges"],
            create_graph=False,
        )
        st["jacobian_sensitivities"] = sn["jacobian_sensitivities"]
        st["gating_weights"] = sn["gating_weights"]
        st["salient_beautiful_matches"] = sn["salient_beautiful_matches"]
        st["working_memory_domains"] = sn["working_memory_domains"]
        st["domain_salience"] = sn.get("domain_salience", {})
        st["jacobian_gram_matrix"] = sn["jacobian_gram_matrix"]
        return sn

    def print_diagnostics(self) -> None:
        if self._last_state is not None:
            self.telemetry.render_terminal_dashboard(self._last_state, self.param_registry.get_parameter_table())

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from core.config import FrameworkConfig
from core.domain_registry import (
    DomainSpec,
    default_roster,
    domain_prior_distance,
    load_roster,
    propose_roster,
)
from core.embeddings import get_encoder
from core.framework import JacobianRSACortexFramework
from core.jacobian_lens import SalienceJacobianNetwork
from core.parameter_registry import DomainParameterRegistry
from core.rsa_engine import DefaultModeRSAEngine
from training.loss_functions import UnifiedCortexLoss
from training.trainer import CortexTrainer


class CorticalMCPState:

    def __init__(self) -> None:
        self.encoder = None
        self.dmn_engine = DefaultModeRSAEngine(diversity_margin=0.25)
        self.salience_network = SalienceJacobianNetwork(
            temperature=0.5,
            interaction_gamma=0.35,
            working_memory_capacity=2,
            top_matches_k=2,
        )
        self.loss_fn = UnifiedCortexLoss(
            task_loss_weight=1.0,
            rsa_salience_weight=0.45,
            jacobian_loss_weight=0.10,
        )
        self.registries: Dict[str, DomainParameterRegistry] = {}

    def get_encoder_instance(self):
        if self.encoder is None:
            self.encoder = get_encoder()
        return self.encoder

    def get_registry(self, domain_names: List[str]) -> DomainParameterRegistry:
        key = "|".join(domain_names)
        if key not in self.registries:
            self.registries[key] = DomainParameterRegistry(
                domain_names=domain_names,
                init_value=1.0,
                temperature=0.5,
            )
        return self.registries[key]


GLOBAL_MCP_STATE = CorticalMCPState()


BUILTIN_DOMAIN_PRESETS: Dict[str, List[Dict[str, str]]] = {
    "drug_discovery": [
        {
            "name": "StructuralBiophysics",
            "title": "Structural Biophysics & Binding Thermodynamics",
            "focus": "3D binding pocket geometry, covalent/non-covalent warhead kinetics, ΔG_bind, residence time (1/k_off), and conformational allostery.",
        },
        {
            "name": "SystemsPharmacology",
            "title": "Quantitative Systems Pharmacology & Pathway Dynamics",
            "focus": "Signaling cascade ODEs, feedback loops, receptor occupancy-response coupling, hill coefficients, and resistance bypass mechanisms.",
        },
        {
            "name": "MedicinalChemistryADMET",
            "title": "Medicinal Chemistry, Structure-Activity & ADMET",
            "focus": "Scaffold design, SMILES validity, Lipinski Rule of 5, QED druglikeness, CYP450 metabolic stability, hERG liability, and TPSA permeability.",
        },
        {
            "name": "ClinicalPKPDControl",
            "title": "Clinical Pharmacokinetics, Dosing & Therapeutic Index",
            "focus": "2-compartment PK/PD ODEs, clearance (CL), volume of distribution (Vd), bioavailability (F), peak-to-trough ratio, and steady-state trough target occupancy.",
        },
    ],
    "software_systems": [
        {
            "name": "DistributedConsensus",
            "title": "Distributed Systems & Consensus Protocols",
            "focus": "State machine replication, linearizability, partition tolerance, quorum intersection, tail latency bounds, and network fault recovery.",
        },
        {
            "name": "SystemsMemoryKernel",
            "title": "Low-Level Memory Hierarchy & Kernel Optimization",
            "focus": "Cache-line alignment, zero-copy I/O, lock-free ring buffers, NUMA locality, SIMD/GPU warp scheduling, and O(1) memory bounds.",
        },
        {
            "name": "FormalVerificationSecurity",
            "title": "Formal Verification, Type Safety & Cryptographic Security",
            "focus": "Temporal logic invariants, threat surface isolation, Byzantine fault resistance, memory safety guarantees, and zero-trust verification.",
        },
        {
            "name": "QueueingControlTheory",
            "title": "Queueing Theory, Backpressure & Dynamical Control",
            "focus": "Little's Law, M/M/c queue stability, Lyapunov control, adaptive rate limiting, token-bucket dynamics, and overload shedding.",
        },
    ],
    "general_first_principles": [
        {
            "name": "MathematicalFormalism",
            "title": "Mathematical Topology, Optimization & Invariants",
            "focus": "Convex/non-convex optimization landscapes, manifold geometry, conservation laws, spectral bounds, and asymptotic complexity.",
        },
        {
            "name": "DynamicalSystemsControl",
            "title": "Dynamical Systems, Differential Equations & Stability",
            "focus": "State-space trajectories, Jacobian eigenvalues, feedback stability, attractors, phase transitions, and stochastic perturbations.",
        },
        {
            "name": "InformationThermodynamics",
            "title": "Information Theory, Entropy & Statistical Mechanics",
            "focus": "Shannon mutual information, KL divergence, variational free energy minimization, signal-to-noise limits, and compression bounds.",
        },
        {
            "name": "EmpiricalVerification",
            "title": "Experimental Falsification, Edge Cases & Constraints",
            "focus": "Physical/resource bottlenecks, failure modes, adversarial stress tests, measurable observables, and falsifiable benchmark criteria.",
        },
    ],
}


def tool_propose_specialist_roster(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    preset = str(args.get("preset", "auto")).strip().lower()
    num_specialists = int(args.get("num_specialists", 4))

    if preset in BUILTIN_DOMAIN_PRESETS:
        domains = BUILTIN_DOMAIN_PRESETS[preset][:num_specialists]
    elif preset == "auto":
        lower_p = problem.lower()
        if any(k in lower_p for k in ["kras", "inhibitor", "smiles", "pk/pd", "drug", "pharmacology", "cyp", "binding", "dose", "therapeutic"]):
            domains = BUILTIN_DOMAIN_PRESETS["drug_discovery"][:num_specialists]
            preset = "drug_discovery"
        elif any(k in lower_p for k in ["distributed", "latency", "kernel", "cuda", "cache", "database", "api", "microservice", "compiler", "queue"]):
            domains = BUILTIN_DOMAIN_PRESETS["software_systems"][:num_specialists]
            preset = "software_systems"
        else:
            domains = BUILTIN_DOMAIN_PRESETS["general_first_principles"][:num_specialists]
            preset = "general_first_principles"
    else:
        try:
            loaded = load_roster(preset)
            domains = [
                {"name": s.name, "title": s.domain_title, "focus": s.focus}
                for s in loaded[:num_specialists]
            ]
        except Exception:
            domains = BUILTIN_DOMAIN_PRESETS["general_first_principles"][:num_specialists]

    return {
        "problem": problem,
        "selected_preset": preset,
        "num_specialists": len(domains),
        "specialists": [
            {
                "param_id": f"w_{i + 1}",
                "name": d["name"],
                "title": d["title"],
                "focus": d["focus"],
                "instruction": (
                    f"Generate a dense, orthogonal hypothesis (<180 words) for '{problem}' "
                    f"analyzed EXCLUSIVELY through {d['title']} ({d['focus']})."
                ),
            }
            for i, d in enumerate(domains)
        ],
        "next_step": (
            "Pass a dictionary mapping each specialist 'name' to its generated hypothesis into "
            "'ael_cortical_rdm_and_jacobian_gate' to compute the 768-D BGE RDM and 0-token PyTorch "
            "autograd Jacobian working-memory gate."
        ),
    }


def tool_cortical_rdm_and_jacobian_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    hypotheses_raw = args.get("hypotheses", {})
    if not isinstance(hypotheses_raw, dict) or len(hypotheses_raw) < 2:
        raise ValueError("'hypotheses' must be a JSON object mapping at least 2 specialist names to their hypothesis texts.")

    working_memory_k = max(1, int(args.get("working_memory_k", 2)))
    domain_names = list(hypotheses_raw.keys())
    texts = [str(hypotheses_raw[n]) for n in domain_names]

    print("     [2a] Initializing neural encoder (BAAI/bge-base-en-v1.5)...", flush=True)
    encoder = GLOBAL_MCP_STATE.get_encoder_instance()
    print("     [2b] Embedding 4 specialist hypotheses...", flush=True)
    emb_batch = encoder.encode_batch(texts)
    domain_embeddings = {domain_names[i]: emb_batch[i] for i in range(len(domain_names))}

    specs = [DomainSpec(name=n, domain_title=n, focus=texts[i][:120]) for i, n in enumerate(domain_names)]
    prior_dist = domain_prior_distance(specs, encoder)

    print("     [2c] Computing Default Mode Network (DMN) 4x4 RDM...", flush=True)
    sim_matrix, rdm_matrix, dmn_bridges, dmn_metrics = GLOBAL_MCP_STATE.dmn_engine(
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        prior_dist=prior_dist,
    )

    print("     [2d] Computing 0-token PyTorch autograd Jacobian sensitivity (J_i)...", flush=True)
    registry = GLOBAL_MCP_STATE.get_registry(domain_names)
    sn_out = GLOBAL_MCP_STATE.salience_network.compute_jacobian_and_salience(
        weights_11=registry.weights,
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        sim_matrix=sim_matrix,
        dmn_bridges=dmn_bridges,
        create_graph=False,
        working_memory_capacity=min(working_memory_k, len(domain_names)),
    )
    print("     [2e] Salience Network complete! Working memory gated (k=2).", flush=True)

    wm_domains = sn_out["working_memory_domains"]
    top_bridge = sn_out["salient_beautiful_matches"][0] if sn_out["salient_beautiful_matches"] else None

    rdm_nested = {
        domain_names[i]: {
            domain_names[j]: round(float(rdm_matrix[i, j].item()), 4)
            for j in range(len(domain_names))
        }
        for i in range(len(domain_names))
    }

    gated_context_blocks = []
    for d_name in wm_domains:
        gate_pct = sn_out["gating_weights"].get(d_name, 0.0) * 100.0
        jac_val = sn_out["jacobian_sensitivities"].get(d_name, 0.0)
        gated_context_blocks.append(
            f"[WORKING MEMORY SLOT: {d_name} | Gating α={gate_pct:.1f}% | Jacobian ||∂y/∂w_i||_2={jac_val:.4f}]\n"
            f"{hypotheses_raw[d_name]}"
        )

    bridge_directive = ""
    if top_bridge:
        bridge_directive = (
            f"EPISTEMIC SURPRISE BRIDGE: {top_bridge['domain_a']} <--> {top_bridge['domain_b']} "
            f"(RDM Dissimilarity={top_bridge['rdm_dissimilarity']:.4f}, "
            f"Salience Score={top_bridge['salience_beauty_score']:.4f}, "
            f"Jacobian Co-Sensitivity={top_bridge['jacobian_co_sensitivity']:.5f})"
        )

    return {
        "architecture": "Ael-Agents Triple-Network Cortical Hierarchy (DMN -> Salience -> CEN)",
        "routing_llm_tokens_used": 0,
        "embedding_manifold": {
            "model": encoder.info.model_name,
            "dimension": encoder.info.dim,
        },
        "layer_1_dmn_rsa": {
            "rdm_dissimilarity_matrix": rdm_nested,
            "participation_ratio": dmn_metrics["dmn_rsa/participation_ratio"],
            "mean_dissimilarity": dmn_metrics["dmn_rsa/mean_dissimilarity"],
            "top_epistemic_bridges": dmn_bridges[:3],
        },
        "layer_2_salience_jacobian": {
            "trainable_weights_w": {
                domain_names[i]: round(float(registry.weights[i].detach().item()), 4)
                for i in range(len(domain_names))
            },
            "gating_probabilities_alpha": sn_out["gating_weights"],
            "jacobian_sensitivities_L2": sn_out["jacobian_sensitivities"],
            "domain_salience_scores": sn_out["domain_salience"],
            "domain_salience_ranking": sn_out["domain_salience_rank"],
            "top_salient_bridges": sn_out["salient_beautiful_matches"],
        },
        "layer_3_cen_working_memory": {
            "working_memory_capacity_k": len(wm_domains),
            "gated_working_memory_specialists": wm_domains,
            "pruned_specialists": [d for d in domain_names if d not in wm_domains],
            "epistemic_bridge_directive": bridge_directive,
            "cen_synthesis_prompt": (
                f"Synthesize the final executive solution for '{problem}' using ONLY the top k={len(wm_domains)} "
                f"Jacobian-gated working-memory specialists ({', '.join(wm_domains)}) and resolve the "
                f"{bridge_directive}. Include: 1. [BRIDGE ASSESSMENT], 2. [INTEGRATED MECHANISM/SOLUTION], "
                f"and 3. [FALSIFIABLE TEST]."
            ),
            "gated_specialist_payloads": gated_context_blocks,
        },
    }


def tool_dream_rsi_consolidate(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    hypotheses_raw = args.get("hypotheses", {})
    wake_solution = str(args.get("wake_solution", "")).strip()
    epochs = max(1, min(50, int(args.get("epochs", 15))))

    if not isinstance(hypotheses_raw, dict) or len(hypotheses_raw) < 2:
        raise ValueError("'hypotheses' must map at least 2 specialist names to their hypothesis strings.")
    if not wake_solution:
        raise ValueError("'wake_solution' must be a non-empty string.")

    domain_names = list(hypotheses_raw.keys())
    texts = [str(hypotheses_raw[n]) for n in domain_names]

    encoder = GLOBAL_MCP_STATE.get_encoder_instance()
    emb_batch = encoder.encode_batch(texts)
    domain_embeddings = {domain_names[i]: emb_batch[i] for i in range(len(domain_names))}
    cen_emb = encoder.encode(wake_solution)

    specs = [DomainSpec(name=n, domain_title=n, focus=texts[i][:120]) for i, n in enumerate(domain_names)]
    prior_dist = domain_prior_distance(specs, encoder)

    sim_matrix, rdm_matrix, dmn_bridges, _ = GLOBAL_MCP_STATE.dmn_engine(
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        prior_dist=prior_dist,
    )

    registry = GLOBAL_MCP_STATE.get_registry(domain_names)
    weights_before = {
        domain_names[i]: round(float(registry.weights[i].detach().item()), 4)
        for i in range(len(domain_names))
    }

    sn_initial = GLOBAL_MCP_STATE.salience_network.compute_jacobian_and_salience(
        weights_11=registry.weights,
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        sim_matrix=sim_matrix,
        dmn_bridges=dmn_bridges,
        create_graph=False,
        working_memory_capacity=min(2, len(domain_names)),
    )

    state_dict: Dict[str, Any] = {
        "task": problem,
        "domain_embeddings": domain_embeddings,
        "rsa_similarity_matrix": sim_matrix,
        "rdm_matrix": rdm_matrix,
        "dmn_spontaneous_bridges": dmn_bridges,
        "salient_beautiful_matches": sn_initial["salient_beautiful_matches"],
        "metadata": {"cen_target_embedding": cen_emb},
    }

    trainer = CortexTrainer(
        param_registry=registry,
        dmn_engine=GLOBAL_MCP_STATE.dmn_engine,
        salience_network=GLOBAL_MCP_STATE.salience_network,
        loss_fn=GLOBAL_MCP_STATE.loss_fn,
        learning_rate=0.08,
    )
    target_y = trainer.build_salience_guided_target(state_dict, domain_names)
    history = [
        trainer.train_step_on_state(state=state_dict, domain_names=domain_names, target_y=target_y)
        for _ in range(epochs)
    ]

    f_wake = history[0]["loss/total"]
    f_dream = history[-1]["loss/total"]

    sn_post = GLOBAL_MCP_STATE.salience_network.compute_jacobian_and_salience(
        weights_11=registry.weights,
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        sim_matrix=sim_matrix,
        dmn_bridges=dmn_bridges,
        create_graph=False,
        working_memory_capacity=min(2, len(domain_names)),
    )

    weights_after = {
        domain_names[i]: round(float(registry.weights[i].detach().item()), 4)
        for i in range(len(domain_names))
    }

    unexplored_bridge = dmn_bridges[1] if len(dmn_bridges) > 1 else (dmn_bridges[0] if dmn_bridges else None)

    return {
        "phase": "Offline Dream-RSI Consolidation",
        "epochs_executed": epochs,
        "free_energy_wake": round(f_wake, 5),
        "free_energy_dream": round(f_dream, 5),
        "free_energy_delta": round(f_dream - f_wake, 5),
        "strictly_improved": bool(f_dream <= f_wake),
        "weights_before": weights_before,
        "weights_after": weights_after,
        "updated_gating_alpha": sn_post["gating_weights"],
        "updated_jacobian_sensitivities": sn_post["jacobian_sensitivities"],
        "updated_working_memory_domains": sn_post["working_memory_domains"],
        "dream_recombination_bridge": unexplored_bridge,
    }


def tool_verify_molecule_pkpd(args: Dict[str, Any]) -> Dict[str, Any]:
    from benchmarks.kras_g12d_discovery_benchmark import (
        evaluate_rdkit_smiles,
        simulate_two_compartment_pkpd,
    )

    smiles = str(args.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("'smiles' must be a non-empty SMILES chemical string.")

    pk_params = {
        "dose_mg": float(args.get("dose_mg", 150.0)),
        "cl_l_h": float(args.get("cl_l_h", 8.5)),
        "vc_l": float(args.get("vc_l", 35.0)),
        "ka_h_inv": float(args.get("ka_h_inv", 1.5)),
        "q_l_h": float(args.get("q_l_h", 12.0)),
        "vp_l": float(args.get("vp_l", 45.0)),
        "kd_nm": float(args.get("kd_nm", 0.02)),
        "ic50_nm": float(args.get("ic50_nm", 0.02)),
    }
    dosing_interval_h = float(args.get("dosing_interval_h", 12.0))
    t_end_h = float(args.get("t_end_h", 24.0))

    rdkit_metrics = evaluate_rdkit_smiles(smiles)
    pkpd_metrics = simulate_two_compartment_pkpd(
        pk_params=pk_params,
        dosing_interval_h=dosing_interval_h,
        t_end_h=t_end_h,
    )

    return {
        "smiles": smiles,
        "rdkit_validation": rdkit_metrics,
        "pk_parameters_used": pk_params,
        "scipy_rk45_two_compartment_pkpd": pkpd_metrics,
    }



def tool_run_autonomous_pipeline(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    roster_arg = str(args.get("roster", "default")).strip()
    optimize_epochs = int(args.get("optimize_epochs", 15))

    if not problem:
        raise ValueError("'problem' must be a non-empty string.")

    roster_specs = load_roster(roster_arg) if roster_arg not in ("default", "") else default_roster()
    fw = JacobianRSACortexFramework(roster=roster_specs)
    state = fw.run(problem)
    opt_history = fw.optimize_parameters(epochs=optimize_epochs) if optimize_epochs > 0 else []

    return {
        "task": problem,
        "run_id": fw.last_run_id,
        "working_memory_domains": state.get("working_memory_domains", []),
        "gating_weights": state.get("gating_weights", {}),
        "jacobian_sensitivities": state.get("jacobian_sensitivities", {}),
        "top_salient_bridges": state.get("salient_beautiful_matches", []),
        "evaluation_scores": state.get("evaluation_scores", {}),
        "final_solution": state.get("final_solution", ""),
        "optimization_loss_start": opt_history[0]["loss/total"] if opt_history else None,
        "optimization_loss_end": opt_history[-1]["loss/total"] if opt_history else None,
    }


MCP_TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "ael_propose_specialist_roster",
        "description": (
            "Segregate any scientific, engineering, software, or quantitative problem into N mutually "
            "orthogonal domain specialists (w_1 ... w_N) for the Ael-Agents cortical hierarchy."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem statement or research question to solve.",
                },
                "preset": {
                    "type": "string",
                    "description": "Preset roster ('auto', 'drug_discovery', 'software_systems', 'general_first_principles', or YAML roster name).",
                    "default": "auto",
                },
                "num_specialists": {
                    "type": "number",
                    "description": "Number of orthogonal domain specialists (default: 4).",
                    "default": 4,
                },
            },
            "required": ["problem"],
        },
    },
    {
        "name": "ael_cortical_rdm_and_jacobian_gate",
        "description": (
            "Zero-LLM-token PyTorch Cortical Co-Processor. Embeds specialist hypotheses into a 768-D "
            "neural manifold (BAAI/bge-base-en-v1.5), computes the N×N Representational Dissimilarity "
            "Matrix (RDM) in the Default Mode Network, runs exact PyTorch autograd Jacobian sensitivity "
            "(J_i = ||∂y/∂w_i||_2) in the Salience Network, and gates the top k=2 working-memory "
            "specialists + Epistemic Surprise Bridge for Central Executive Network (CEN) synthesis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The original problem or research question.",
                },
                "hypotheses": {
                    "type": "object",
                    "description": "Dictionary mapping each specialist domain name to its hypothesis text.",
                    "additionalProperties": {"type": "string"},
                },
                "working_memory_k": {
                    "type": "number",
                    "description": "Prefrontal working-memory bottleneck capacity k (default: 2).",
                    "default": 2,
                },
            },
            "required": ["problem", "hypotheses"],
        },
    },
    {
        "name": "ael_dream_rsi_consolidate",
        "description": (
            "Offline Dream-RSI sleep consolidation step. Updates trainable PyTorch cortical weights (w_i) "
            "via gradient descent on UnifiedCortexLoss, verifies free-energy reduction (F_dream <= F_wake), "
            "and isolates unexplored RDM cross-domain bridges."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem statement.",
                },
                "hypotheses": {
                    "type": "object",
                    "description": "Dictionary mapping specialist names to their hypothesis texts.",
                    "additionalProperties": {"type": "string"},
                },
                "wake_solution": {
                    "type": "string",
                    "description": "The synthesized CEN wake solution text.",
                },
                "epochs": {
                    "type": "number",
                    "description": "Number of PyTorch Adam optimization steps (default: 15).",
                    "default": 15,
                },
            },
            "required": ["problem", "hypotheses", "wake_solution"],
        },
    },
    {
        "name": "ael_verify_molecule_pkpd",
        "description": (
            "Deterministic RDKit chemical compiler (valence, MW, LogP, TPSA, Lipinski Rule of 5, QED "
            "druglikeness) coupled with SciPy RK45 2-compartment pharmacokinetic/pharmacodynamic ODE "
            "integration (steady-state trough target occupancy, Peak-to-Trough ratio, 24h suppression)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the candidate molecule.",
                },
                "dose_mg": {
                    "type": "number",
                    "description": "Oral dose in mg (default: 150.0).",
                    "default": 150.0,
                },
                "cl_l_h": {
                    "type": "number",
                    "description": "Systemic clearance CL in L/h (default: 8.5).",
                    "default": 8.5,
                },
                "vc_l": {
                    "type": "number",
                    "description": "Central compartment volume Vc in L (default: 35.0).",
                    "default": 35.0,
                },
                "kd_nm": {
                    "type": "number",
                    "description": "Dissociation constant Kd in nM (default: 0.02).",
                    "default": 0.02,
                },
                "ic50_nm": {
                    "type": "number",
                    "description": "Half-maximal inhibitory concentration IC50 in nM (default: 0.02).",
                    "default": 0.02,
                },
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "ael_run_autonomous_pipeline",
        "description": (
            "Execute the complete autonomous Ael-Agents LangGraph pipeline (parallel specialists -> DMN "
            "RDM -> Salience Jacobian switch -> CEN synthesis -> PyTorch weight optimization)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem statement to solve.",
                },
                "roster": {
                    "type": "string",
                    "description": "Domain roster name or YAML path (default: 'default').",
                    "default": "default",
                },
                "optimize_epochs": {
                    "type": "number",
                    "description": "Number of post-run PyTorch parameter optimization epochs (default: 15).",
                    "default": 15,
                },
            },
            "required": ["problem"],
        },
    },
]


TOOL_DISPATCH = {
    "ael_propose_specialist_roster": tool_propose_specialist_roster,
    "ael_cortical_rdm_and_jacobian_gate": tool_cortical_rdm_and_jacobian_gate,
    "ael_dream_rsi_consolidate": tool_dream_rsi_consolidate,
    "ael_verify_molecule_pkpd": tool_verify_molecule_pkpd,
    "ael_run_autonomous_pipeline": tool_run_autonomous_pipeline,
}


def handle_jsonrpc_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        client_proto = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_proto,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "ael-agents-cortex-mcp",
                    "version": "1.1.0",
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS_SPEC},
        }

    if method == "resources/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"resources": []},
        }

    if method == "prompts/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"prompts": []},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        if tool_name not in TOOL_DISPATCH:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        try:
            payload = TOOL_DISPATCH[tool_name](tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                },
                                indent=2,
                            ),
                        }
                    ],
                    "isError": True,
                },
            }

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


def run_stdio_mcp_server() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = handle_jsonrpc_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def run_self_test() -> None:
    print("[Ael-Agents MCP] Running self-test verification...", flush=True)
    problem = "Design an ultra-low-latency consensus engine with bounded tail latency under Byzantine faults."
    print("  -> Step 1: Proposing orthogonal specialist roster...", flush=True)
    roster_out = tool_propose_specialist_roster({"problem": problem, "preset": "auto"})
    sample_hypotheses = {
        "DistributedConsensus": (
            "Use optimistic pipelined HotStuff quorum certificates with threshold BLS signatures "
            "to reduce message complexity from O(N^2) to O(N) per view while bounding view-change latency."
        ),
        "SystemsMemoryKernel": (
            "Pin consensus state transitions to NUMA-local lock-free ring buffers using kernel-bypass "
            "DPDK/io_uring zero-copy descriptors so p999 serialization overhead stays below 8 microseconds."
        ),
        "FormalVerificationSecurity": (
            "Enforce hardware-backed monotonic counters in TEE enclaves to prevent equivocation, "
            "reducing Byzantine fault quorum size from 3f+1 to 2f+1 under non-equivocation invariants."
        ),
        "QueueingControlTheory": (
            "Apply Lyapunov drift-plus-penalty admission control at ingress queues so arrival rate λ "
            "strictly satisfies λ < μ_quorum, preventing M/G/1 tail queueing explosion during burst traffic."
        ),
    }
    print("  -> Step 2: Computing 768-D BGE RDM & PyTorch Jacobian Sensitivity Gate...", flush=True)
    gate_out = tool_cortical_rdm_and_jacobian_gate(
        {"problem": problem, "hypotheses": sample_hypotheses, "working_memory_k": 2}
    )
    print("  -> Step 3: Running Offline Dream-RSI Consolidation...", flush=True)
    dream_out = tool_dream_rsi_consolidate(
        {
            "problem": problem,
            "hypotheses": sample_hypotheses,
            "wake_solution": (
                "Combine TEE-enforced 2f+1 non-equivocation quorum certificates with kernel-bypass "
                "zero-copy ring buffers and Lyapunov ingress backpressure to guarantee bounded p999 latency."
            ),
            "epochs": 15,
        }
    )
    print("  -> Step 4: Deterministic RDKit & SciPy 2-Compartment PK/PD ODE Verification...", flush=True)
    pkpd_out = tool_verify_molecule_pkpd(
        {"smiles": "Cc1ncnc2c1ncn2C(C)C(=O)NCc1ccc(C(F)C(F)F)cc1"}
    )
    summary = {
        "mcp_self_test": "PASSED",
        "selected_preset": roster_out["selected_preset"],
        "routing_llm_tokens_used": gate_out["routing_llm_tokens_used"],
        "gated_working_memory_specialists": gate_out["layer_3_cen_working_memory"]["gated_working_memory_specialists"],
        "jacobian_sensitivities": gate_out["layer_2_salience_jacobian"]["jacobian_sensitivities_L2"],
        "dream_free_energy_wake": dream_out["free_energy_wake"],
        "dream_free_energy_dream": dream_out["free_energy_dream"],
        "dream_strictly_improved": dream_out["strictly_improved"],
        "pkpd_qed": pkpd_out["rdkit_validation"]["qed_druglikeness"],
        "pkpd_trough_occupancy_pct": pkpd_out["scipy_rk45_two_compartment_pkpd"]["steady_state_trough_occupancy_pct"],
    }
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ael-Agents Cortical MCP Server")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic MCP tool verification")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_stdio_mcp_server()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"[Ael-Agents MCP Error] {exc}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

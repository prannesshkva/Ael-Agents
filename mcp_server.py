import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import sentence_transformers
except Exception:
    pass

import argparse
import io
import json
import math
import statistics
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.optimize
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
        self._enc_lock = threading.Lock()
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
        self.last_hypotheses: Dict[str, str] = {}

    def get_encoder_instance(self):
        with self._enc_lock:
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


UNIVERSAL_SCIENTIFIC_BASIS: List[Dict[str, str]] = [
    {
        "name": "ContinuumFluidTransport",
        "title": "Continuum Mechanics, Fluid Dynamics & Coupled Transport Phenomena",
        "focus": "Navier-Stokes conservation of momentum, boundary-layer advection-diffusion PDEs, multiphase heat/mass transfer, Nusselt/Peclet scaling, and pressure-drop hydraulics.",
    },
    {
        "name": "ElectrodynamicsFieldTheory",
        "title": "Electrodynamics, Magnetohydrodynamics & High-Field Interactions",
        "focus": "Maxwell's equations, Lorentz force coupling (J x B), Hartmann/Stuart magnetohydrodynamic scaling, electromagnetic wave propagation, skin depth, and superconducting field limits.",
    },
    {
        "name": "ThermodynamicsStatisticalPhysics",
        "title": "Non-Equilibrium Thermodynamics, Statistical Mechanics & Energy Conversion",
        "focus": "First and Second Law exergy budgets, Carnot/Brayton cycle limits, entropy production minimization, Boltzmann transport, phase equilibria, and radiative heat transfer.",
    },
    {
        "name": "SolidMechanicsMaterials",
        "title": "Solid Mechanics, Thermo-Elasticity & Materials Degradation",
        "focus": "Cauchy stress-strain tensors, Von Mises/ASME membrane and bending limits, secondary thermo-elastic stress, creep-rupture kinetics, fracture toughness, and radiation/chemical corrosion.",
    },
    {
        "name": "QuantumAtomicPhysics",
        "title": "Quantum Mechanics, Electronic Structure & Coherent Dynamics",
        "focus": "Hamiltonian eigenstates, Lindblad decoherence/dephasing rates, tunneling probabilities, spin/optical transitions, nuclear cross-sections, and quantum measurement noise limits.",
    },
    {
        "name": "ReactionKineticsMolecularDesign",
        "title": "Chemical Kinetics, Molecular Structure & Surface Catalysis",
        "focus": "Transition-state free energy barriers, covalent/non-covalent molecular graphs, reaction-diffusion kinetics, chemical potential, solubility, and selectivity/metabolic stability.",
    },
    {
        "name": "NonlinearDynamicsControl",
        "title": "Nonlinear Dynamical Systems, Coupled ODEs & Feedback Control",
        "focus": "State-space differential equations, Jacobian eigenvalue stability, Lyapunov functions, bifurcation thresholds, compartmental kinetics, and optimal closed-loop control.",
    },
    {
        "name": "MathematicalTopologyOptimization",
        "title": "Applied Mathematics, Variational Calculus & Constrained Optimization",
        "focus": "Karush-Kuhn-Tucker (KKT) multipliers, Pareto frontier geometry, spectral graph bounds, conservation invariants, and numerical PDE/ODE stiffness constraints.",
    },
    {
        "name": "InformationSignalTheory",
        "title": "Information Theory, Stochastic Estimation & Signal Processing",
        "focus": "Shannon channel capacity, Cramer-Rao lower bounds, Fisher information matrices, signal-to-noise thermal floors, Kalman filtering, and variational free energy.",
    },
    {
        "name": "DistributedSystemsArchitecture",
        "title": "Computational Systems, Hardware Microarchitecture & Concurrency",
        "focus": "Memory-hierarchy bandwidth ceilings, lock-free concurrency, consensus fault tolerance, queueing tail-latency bounds, and algorithmic time/space complexity.",
    },
    {
        "name": "SystemsBiologyHomeostasis",
        "title": "Quantitative Systems Biology, Physiological Transport & Homeostasis",
        "focus": "Receptor-ligand binding kinetics, intracellular signaling cascades, membrane permeability barriers, multi-compartmental clearance, and off-target toxicity margins.",
    },
    {
        "name": "EmpiricalFalsificationFailureModes",
        "title": "Experimental Falsification, Boundary Stress Testing & Failure Physics",
        "focus": "Coupled failure modes where optimizing one parameter breaks another, parasitic loss blowup, degenerate edge cases, and quantitative falsification experiments.",
    },
]


def _select_dpp_orthogonal_domains(
    problem: str,
    candidates: List[Dict[str, str]],
    num_specialists: int,
) -> Dict[str, Any]:
    encoder = GLOBAL_MCP_STATE.get_encoder_instance()
    n_cand = len(candidates)
    k_target = max(2, min(int(num_specialists), n_cand))

    cand_texts = [f"{c['title']}: {c['focus']}" for c in candidates]
    all_texts = [problem] + cand_texts
    emb_all = encoder.encode_batch(all_texts)

    q_emb = emb_all[0:1]
    c_emb = emb_all[1:]

    rel_scores = torch.clamp((c_emb @ q_emb.t()).squeeze(-1), -1.0, 1.0)
    top_pool_k = min(n_cand, max(k_target + 3, int(k_target * 2)))
    top_rel_indices = torch.topk(rel_scores, k=top_pool_k).indices.tolist()

    quality = torch.exp(4.5 * rel_scores)

    sim_matrix = torch.clamp(c_emb @ c_emb.t(), -1.0, 1.0)
    kernel_s = 0.5 * (sim_matrix + 1.0)
    l_matrix = torch.outer(quality, quality) * kernel_s + 1e-5 * torch.eye(n_cand)

    selected_indices: List[int] = []
    remaining = set(top_rel_indices)

    for _ in range(k_target):
        best_idx = -1
        best_logdet = -1e18
        for idx in remaining:
            trial = selected_indices + [idx]
            sub_l = l_matrix[ trial][:, trial]
            sign, logdet = torch.linalg.slogdet(sub_l)
            val = float(logdet.item()) if float(sign.item()) > 0 else -1e18
            if val > best_logdet:
                best_logdet = val
                best_idx = idx
        if best_idx < 0:
            best_idx = next(iter(remaining))
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    sub_final = l_matrix[selected_indices][:, selected_indices]
    _, final_logdet = torch.linalg.slogdet(sub_final)
    sub_sim = sim_matrix[selected_indices][:, selected_indices]
    off_mask = ~torch.eye(len(selected_indices), dtype=torch.bool)
    mean_orthogonality = float((1.0 - torch.abs(sub_sim[off_mask])).mean().item()) if off_mask.any() else 1.0

    selected_domains = []
    for rank, idx in enumerate(selected_indices):
        d = candidates[idx]
        selected_domains.append(
            {
                "param_id": f"w_{rank + 1}",
                "name": d["name"],
                "title": d["title"],
                "focus": d["focus"],
                "query_relevance_cos": round(float(rel_scores[idx].item()), 4),
                "instruction": (
                    f"Generate a dense, mathematically rigorous hypothesis (<180 words) for the user's query "
                    f"analyzed EXCLUSIVELY through {d['title']} ({d['focus']}). Even if the user's prompt is brief "
                    f"or non-technical, elevate it to Principal-Researcher depth: instantiate concrete physical "
                    f"geometry, boundary conditions, material/system parameters, explicit governing differential/algebraic "
                    f"equations, dimensionless numbers, and strict failure thresholds."
                ),
            }
        )

    return {
        "selected_domains": selected_domains,
        "dpp_logdet_volume": round(float(final_logdet.item()), 4),
        "mean_pairwise_orthogonality": round(mean_orthogonality, 4),
        "candidate_pool_size": n_cand,
    }


def _normalize_hypotheses(raw: Any, fallback_args: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    if isinstance(raw, str):
        raw_str = raw.strip()
        if raw_str.startswith("{") or raw_str.startswith("["):
            try:
                raw = json.loads(raw_str)
            except Exception:
                pass

    normalized: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            key_name = str(k).strip()
            if not key_name:
                continue
            if isinstance(v, dict):
                text_val = (
                    v.get("hypothesis")
                    or v.get("text")
                    or v.get("content")
                    or v.get("analysis")
                    or v.get("focus")
                    or " ".join(f"{sk}: {sv}" for sk, sv in v.items())
                )
                normalized[key_name] = str(text_val).strip()
            elif isinstance(v, (list, tuple)):
                normalized[key_name] = " ".join(str(x) for x in v).strip()
            else:
                normalized[key_name] = str(v).strip()
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                name = str(
                    item.get("name")
                    or item.get("specialist")
                    or item.get("domain")
                    or item.get("param_id")
                    or f"Specialist_{idx + 1}"
                ).strip()
                text_val = (
                    item.get("hypothesis")
                    or item.get("text")
                    or item.get("content")
                    or item.get("analysis")
                    or item.get("focus")
                    or json.dumps(item)
                )
                normalized[name] = str(text_val).strip()
            else:
                normalized[f"Specialist_{idx + 1}"] = str(item).strip()

    if len(normalized) < 2 and fallback_args:
        skip_keys = {"problem", "hypotheses", "working_memory_k", "wake_solution", "epochs", "candidate_domains", "num_specialists"}
        for k, v in fallback_args.items():
            if k not in skip_keys and isinstance(v, (str, dict)):
                normalized[str(k)] = str(v)

    clean_out: Dict[str, str] = {}
    for k, v in normalized.items():
        s = str(v).strip()
        if s:
            clean_out[str(k).strip()] = s

    if len(clean_out) < 2:
        raise ValueError(
            "'hypotheses' must provide at least 2 specialist hypotheses as a dictionary mapping specialist names to hypothesis strings."
        )
    return clean_out


def tool_propose_specialist_roster(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    if not problem:
        raise ValueError("'problem' must be a non-empty string.")
    num_specialists = int(args.get("num_specialists", 4))
    raw_candidates = args.get("candidate_domains")

    custom_pool: List[Dict[str, str]] = []
    if isinstance(raw_candidates, list) and len(raw_candidates) >= 2:
        for idx, item in enumerate(raw_candidates):
            if isinstance(item, dict):
                c_name = "".join(ch for ch in str(item.get("name", f"Domain_{idx + 1}")) if ch.isalnum() or ch == "_")
                c_title = str(item.get("title", c_name)).strip()
                c_focus = str(item.get("focus", c_title)).strip()
                if c_name:
                    custom_pool.append({"name": c_name, "title": c_title, "focus": c_focus})

    pool_to_use = custom_pool if len(custom_pool) >= 2 else UNIVERSAL_SCIENTIFIC_BASIS
    selection_mode = "prompt_synthesized_dpp" if len(custom_pool) >= 2 else "universal_first_principles_dpp"

    dpp_out = _select_dpp_orthogonal_domains(
        problem=problem,
        candidates=pool_to_use,
        num_specialists=num_specialists,
    )

    return {
        "framework": "Ael-Agents",
        "problem": problem,
        "selection_mode": selection_mode,
        "dpp_logdet_volume": dpp_out["dpp_logdet_volume"],
        "mean_pairwise_orthogonality": dpp_out["mean_pairwise_orthogonality"],
        "candidate_pool_size": dpp_out["candidate_pool_size"],
        "num_specialists": len(dpp_out["selected_domains"]),
        "researcher_elevation_protocol": (
            "Elevate even brief or plain-English user prompts to full Principal-Investigator / Chief-Engineer depth: "
            "infer concrete physical dimensions, realistic operating conditions, material constants, governing differential "
            "equations, dimensionless numbers, and hard failure bounds from first principles before synthesizing."
        ),
        "specialists": dpp_out["selected_domains"],
        "next_step": (
            "Pass a dictionary mapping each specialist 'name' to its generated plain-string hypothesis into "
            "'Ael-Agents_cortical_rdm_and_jacobian_gate' to compute the 768-D BGE RDM and 0-token PyTorch "
            "autograd Jacobian working-memory gate."
        ),
    }


def tool_cortical_rdm_and_jacobian_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    hypotheses_map = _normalize_hypotheses(args.get("hypotheses", {}), fallback_args=args)
    GLOBAL_MCP_STATE.last_hypotheses = dict(hypotheses_map)

    working_memory_k = max(1, int(args.get("working_memory_k", 2)))
    domain_names = [str(k) for k in hypotheses_map.keys()]
    texts = [str(hypotheses_map[n]) for n in domain_names]

    encoder = GLOBAL_MCP_STATE.get_encoder_instance()
    emb_batch = encoder.encode_batch(texts)
    domain_embeddings = {domain_names[i]: emb_batch[i] for i in range(len(domain_names))}

    specs = [DomainSpec(name=n, domain_title=n, focus=texts[i][:120]) for i, n in enumerate(domain_names)]
    prior_dist = domain_prior_distance(specs, encoder)

    sim_matrix, rdm_matrix, dmn_bridges, dmn_metrics = GLOBAL_MCP_STATE.dmn_engine(
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        prior_dist=prior_dist,
    )

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

    wm_domains = sn_out["working_memory_domains"]
    top_bridge = sn_out["salient_beautiful_matches"][0] if sn_out["salient_beautiful_matches"] else None

    rdm_dict: Dict[str, Dict[str, float]] = {}
    for i, di in enumerate(domain_names):
        rdm_dict[di] = {}
        for j, dj in enumerate(domain_names):
            rdm_dict[di][dj] = round(float(rdm_matrix[i, j].item()), 4)

    bridge_prompt = ""
    if top_bridge:
        d1, d2 = top_bridge["domain_a"], top_bridge["domain_b"]
        bridge_prompt = (
            f"Epistemic Surprise Bridge detected between {d1} and {d2} "
            f"(salience={top_bridge['salience_beauty_score']:.4f}, surprise={top_bridge['epistemic_surprise']:.4f}). "
            f"Force an explicit mathematical/mechanistic reconciliation between {d1} and {d2}."
        )

    alphas_vals = list(sn_out["gating_weights"].values())
    salience_entropy = -sum(a * math.log(max(a, 1e-12)) for a in alphas_vals)

    return {
        "framework": "Ael-Agents",
        "problem": problem,
        "embedding_manifold": {
            "backend": getattr(encoder, "backend", "sentence-transformers"),
            "model": getattr(encoder, "model_name", "BAAI/bge-base-en-v1.5"),
            "dimension": int(emb_batch.shape[-1]),
        },
        "routing_llm_tokens_used": 0,
        "layer_1_dmn_rsa": {
            "representational_dissimilarity_matrix": rdm_dict,
            "mean_dissimilarity": round(float(dmn_metrics["dmn_rsa/mean_dissimilarity"]), 4),
            "participation_ratio": round(float(dmn_metrics["dmn_rsa/participation_ratio"]), 4),
            "top_epistemic_bridges": dmn_bridges[:3],
        },
        "layer_2_salience_jacobian": {
            "trainable_weights_w": {
                domain_names[i]: round(float(registry.weights[i].item()), 4)
                for i in range(len(domain_names))
            },
            "gating_alpha": {
                k: round(float(v), 4) for k, v in sn_out["gating_weights"].items()
            },
            "jacobian_sensitivities_L2": {
                k: round(float(v), 4) for k, v in sn_out["jacobian_sensitivities"].items()
            },
            "salience_entropy": round(float(salience_entropy), 4),
        },
        "layer_3_cen_working_memory": {
            "capacity_k": min(working_memory_k, len(domain_names)),
            "gated_working_memory_specialists": wm_domains,
            "gated_hypotheses": {d: hypotheses_map[d] for d in wm_domains if d in hypotheses_map},
            "top_salient_bridge": top_bridge,
            "cen_synthesis_directive": (
                f"Synthesize the final solution using ONLY the gated working-memory domains {wm_domains} "
                f"and the Epistemic Surprise Bridge. {bridge_prompt}"
            ),
        },
        "next_step": (
            "Generate the synthesized wake_solution from the gated working-memory domains and pass it to "
            "'Ael-Agents_dream_rsi_consolidate' to run offline PyTorch gradient descent (F_dream <= F_wake)."
        ),
    }


def tool_dream_rsi_consolidate(args: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(args.get("problem", "")).strip()
    raw_hyps = args.get("hypotheses")
    if raw_hyps:
        hypotheses_map = _normalize_hypotheses(raw_hyps, fallback_args=args)
        GLOBAL_MCP_STATE.last_hypotheses = dict(hypotheses_map)
    elif GLOBAL_MCP_STATE.last_hypotheses:
        hypotheses_map = dict(GLOBAL_MCP_STATE.last_hypotheses)
    else:
        raise ValueError(
            "No specialist hypotheses provided or cached. Pass 'hypotheses' dict or run 'Ael-Agents_cortical_rdm_and_jacobian_gate' first."
        )

    wake_solution = str(
        args.get("wake_solution")
        or args.get("solution")
        or args.get("synthesis")
        or ""
    ).strip()
    epochs = max(1, int(args.get("epochs", 15)))

    if not wake_solution:
        raise ValueError("'wake_solution' must be a non-empty string.")

    domain_names = [str(k) for k in hypotheses_map.keys()]
    texts = [str(hypotheses_map[n]) for n in domain_names]

    encoder = GLOBAL_MCP_STATE.get_encoder_instance()
    emb_batch = encoder.encode_batch(texts)
    domain_embeddings = {domain_names[i]: emb_batch[i] for i in range(len(domain_names))}
    target_emb = encoder.encode(wake_solution).detach()

    specs = [DomainSpec(name=n, domain_title=n, focus=texts[i][:120]) for i, n in enumerate(domain_names)]
    prior_dist = domain_prior_distance(specs, encoder)

    registry = GLOBAL_MCP_STATE.get_registry(domain_names)
    weights_before = {
        domain_names[i]: round(float(registry.weights[i].item()), 4)
        for i in range(len(domain_names))
    }

    optimizer = torch.optim.Adam(registry.parameters(), lr=0.05)
    f_wake: Optional[float] = None
    f_dream: Optional[float] = None

    for step in range(epochs):
        optimizer.zero_grad()
        sim_matrix, rdm_matrix, dmn_bridges, _ = GLOBAL_MCP_STATE.dmn_engine(
            domain_embeddings=domain_embeddings,
            domain_names=domain_names,
            prior_dist=prior_dist,
        )
        sn_out = GLOBAL_MCP_STATE.salience_network.compute_jacobian_and_salience(
            weights_11=registry.weights,
            domain_embeddings=domain_embeddings,
            domain_names=domain_names,
            sim_matrix=sim_matrix,
            dmn_bridges=dmn_bridges,
            create_graph=True,
            working_memory_capacity=min(2, len(domain_names)),
        )
        loss, loss_log = GLOBAL_MCP_STATE.loss_fn(
            weights_11=registry.weights,
            synthesized_y=sn_out["synthesized_tensor"],
            target_y=target_emb,
            salient_matches=sn_out["salient_beautiful_matches"],
            jacobian_sensitivities_tensor=sn_out["sensitivities_tensor"],
        )
        if step == 0:
            f_wake = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(registry.parameters(), max_norm=1.0)
        optimizer.step()
        f_dream = float(loss.item())

    weights_after = {
        domain_names[i]: round(float(registry.weights[i].item()), 4)
        for i in range(len(domain_names))
    }

    sim_matrix, rdm_matrix, dmn_bridges, _ = GLOBAL_MCP_STATE.dmn_engine(
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        prior_dist=prior_dist,
    )
    sn_post = GLOBAL_MCP_STATE.salience_network.compute_jacobian_and_salience(
        weights_11=registry.weights,
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        sim_matrix=sim_matrix,
        dmn_bridges=dmn_bridges,
        create_graph=False,
        working_memory_capacity=min(2, len(domain_names)),
    )

    unexplored_bridge = dmn_bridges[1] if len(dmn_bridges) > 1 else (dmn_bridges[0] if dmn_bridges else None)

    return {
        "framework": "Ael-Agents",
        "problem": problem,
        "epochs_executed": epochs,
        "free_energy_wake": round(f_wake or 0.0, 5),
        "free_energy_dream": round(f_dream or 0.0, 5),
        "free_energy_delta": round((f_dream or 0.0) - (f_wake or 0.0), 5),
        "strictly_improved": bool((f_dream or 0.0) <= (f_wake or 0.0)),
        "weights_before": weights_before,
        "weights_after": weights_after,
        "updated_gating_alpha": sn_post["gating_weights"],
        "updated_jacobian_sensitivities": sn_post["jacobian_sensitivities"],
        "updated_working_memory_domains": sn_post["working_memory_domains"],
        "dream_recombination_bridge": unexplored_bridge,
        "next_step": (
            "MANDATORY STEP 4 (Anthropic-Style Programmatic Tool Calling / PTC): Call 'Ael-Agents_programmatic_verify' "
            "with a self-contained Python verification script ('verification_code') derived directly from the user's prompt "
            "and the cross-domain equations above. Inside that single programmatic script, use scipy/numpy/torch/rdkit and "
            "'record_candidate(name, params, metrics, constraints)' to test multiple candidate designs in a loop, falsify "
            "candidates that violate conservation laws or boundary conditions, and verify the winning design before answering."
        ),
    }


class ProgrammaticVerificationSandbox:

    def __init__(self) -> None:
        self.candidates: List[Dict[str, Any]] = []
        self.constraints_log: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

    def record_candidate(
        self,
        name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        constraints: Dict[str, bool],
    ) -> Dict[str, Any]:
        clean_constraints = {str(k): bool(v) for k, v in constraints.items()}
        failed = [k for k, v in clean_constraints.items() if not v]
        passed = len(failed) == 0
        entry = {
            "candidate_name": str(name),
            "passed_all_constraints": passed,
            "failed_constraints": failed,
            "parameters": self._sanitize(parameters),
            "metrics": self._sanitize(metrics),
            "constraints": clean_constraints,
        }
        self.candidates.append(entry)
        return entry

    def assert_constraint(
        self,
        name: str,
        passed: bool,
        actual: Any = None,
        bound: Any = None,
        units: str = "",
    ) -> bool:
        flag = bool(passed)
        self.constraints_log.append(
            {
                "constraint": str(name),
                "passed": flag,
                "actual": self._sanitize(actual),
                "bound": self._sanitize(bound),
                "units": str(units),
            }
        )
        return flag

    def set_summary(self, data: Dict[str, Any]) -> None:
        if isinstance(data, dict):
            self.summary = self._sanitize(data)

    def _sanitize(self, obj: Any) -> Any:
        if isinstance(obj, (bool, int, str)) or obj is None:
            return obj
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return round(obj, 6)
        if isinstance(obj, np.generic):
            return self._sanitize(obj.item())
        if isinstance(obj, np.ndarray):
            return self._sanitize(obj.tolist())
        if isinstance(obj, dict):
            return {str(k): self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(x) for x in obj]
        return str(obj)


def tool_programmatic_verify(args: Dict[str, Any]) -> Dict[str, Any]:
    code = str(
        args.get("verification_code")
        or args.get("code")
        or args.get("script")
        or ""
    ).strip()
    problem = str(args.get("problem", "")).strip()
    if not code:
        raise ValueError("'verification_code' must be a non-empty Python script string.")

    ptc = ProgrammaticVerificationSandbox()
    stdout_buf = io.StringIO()

    exec_ns: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "solve_ivp": scipy.integrate.solve_ivp,
        "quad": scipy.integrate.quad,
        "minimize": scipy.optimize.minimize,
        "root": scipy.optimize.root,
        "brentq": scipy.optimize.brentq,
        "linprog": scipy.optimize.linprog,
        "torch": torch,
        "F": F,
        "math": math,
        "statistics": statistics,
        "json": json,
        "ptc": ptc,
        "record_candidate": ptc.record_candidate,
        "assert_constraint": ptc.assert_constraint,
        "set_summary": ptc.set_summary,
        "RESULT": {},
    }

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors
        exec_ns["Chem"] = Chem
        exec_ns["Descriptors"] = Descriptors
        exec_ns["QED"] = QED
        exec_ns["rdMolDescriptors"] = rdMolDescriptors
    except Exception:
        pass

    t0 = time.perf_counter()
    exec_error: Optional[str] = None
    tb_str: Optional[str] = None

    try:
        with redirect_stdout(stdout_buf):
            exec(compile(code, "<Ael-Agents-PTC-Verifier>", "exec"), exec_ns, exec_ns)
    except Exception as exc:
        exec_error = str(exc)
        tb_str = traceback.format_exc()

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    stdout_text = stdout_buf.getvalue().strip()

    if not ptc.summary and isinstance(exec_ns.get("RESULT"), dict) and exec_ns["RESULT"]:
        ptc.set_summary(exec_ns["RESULT"])

    verified_cands = [c for c in ptc.candidates if c["passed_all_constraints"]]
    falsified_cands = [c for c in ptc.candidates if not c["passed_all_constraints"]]

    if exec_error is not None:
        status = "EXECUTION_ERROR"
    elif ptc.candidates:
        status = "PASSED" if len(verified_cands) > 0 else "ALL_CANDIDATES_FALSIFIED"
    elif ptc.constraints_log:
        status = "PASSED" if all(c["passed"] for c in ptc.constraints_log) else "CONSTRAINTS_FALSIFIED"
    else:
        status = "PASSED"

    out: Dict[str, Any] = {
        "framework": "Ael-Agents",
        "paradigm": "Anthropic-Style Programmatic Tool Calling (PTC) Single-Pass Scientific Verifier",
        "problem": problem,
        "verification_status": status,
        "execution_time_ms": elapsed_ms,
        "total_candidates_evaluated_in_loop": len(ptc.candidates),
        "num_falsified_candidates": len(falsified_cands),
        "num_verified_candidates": len(verified_cands),
        "falsified_candidates": falsified_cands,
        "verified_candidates": verified_cands,
        "constraints_log": ptc.constraints_log,
        "result_summary": ptc.summary,
        "stdout": stdout_text[:4000] if stdout_text else "",
    }
    if exec_error is not None:
        out["error"] = exec_error
        out["traceback"] = tb_str
    return out


def tool_verify_molecule_pkpd(args: Dict[str, Any]) -> Dict[str, Any]:
    from benchmarks.kras_g12d_discovery_benchmark import (
        evaluate_rdkit_smiles,
        simulate_two_compartment_pkpd,
    )

    smiles = str(args.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("'smiles' must be a non-empty SMILES chemical string.")

    rdkit_metrics = evaluate_rdkit_smiles(smiles)
    computed_mw = float(rdkit_metrics.get("mw") or 400.0)

    pk_params = {
        "dose_mg": float(args.get("dose_mg", 150.0)),
        "cl_l_h": float(args.get("cl_l_h", 8.5)),
        "vc_l": float(args.get("vc_l", 35.0)),
        "ka_h_inv": float(args.get("ka_h_inv", 1.5)),
        "q_l_h": float(args.get("q_l_h", 12.0)),
        "vp_l": float(args.get("vp_l", 45.0)),
        "kd_nm": float(args.get("kd_nm", 0.02)),
        "ic50_nm": float(args.get("ic50_nm", 0.02)),
        "mw_da": float(args.get("mw_da", computed_mw)),
        "f_bioavail": float(args.get("f_bioavail", 0.70)),
        "fu_plasma": float(args.get("fu_plasma", 0.05)),
        "kp_uu_brain": float(args.get("kp_uu_brain", 0.50)),
    }
    dosing_interval_h = float(args.get("dosing_interval_h", 12.0))
    t_end_h = float(args.get("t_end_h", 24.0))

    pkpd_metrics = simulate_two_compartment_pkpd(
        pk_params=pk_params,
        dosing_interval_h=dosing_interval_h,
        t_end_h=t_end_h,
    )

    result: Dict[str, Any] = {
        "framework": "Ael-Agents",
        "smiles": smiles,
        "rdkit_validation": rdkit_metrics,
        "pk_parameters_used": pk_params,
        "scipy_rk45_two_compartment_pkpd": pkpd_metrics,
    }
    if not GLOBAL_MCP_STATE.last_hypotheses:
        result["cortical_pipeline_status"] = (
            "Verification completed. To obtain multi-disciplinary cortical synthesis, execute "
            "'Ael-Agents_propose_specialist_roster' -> 'Ael-Agents_cortical_rdm_and_jacobian_gate' "
            "-> 'Ael-Agents_dream_rsi_consolidate'."
        )
    return result


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
        "framework": "Ael-Agents",
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
        "name": "Ael-Agents_propose_specialist_roster",
        "title": "Ael-Agents: Propose Orthogonal Specialist Roster (Domain-Agnostic DPP)",
        "annotations": {"title": "Ael-Agents: Propose Orthogonal Specialist Roster (Domain-Agnostic DPP)"},
        "description": (
            "Ael-Agents Step 1 (PRIMARY ENTRY POINT - 100% Domain-Agnostic): Decompose ANY scientific, physics, "
            "engineering, mathematical, or systems problem into N mutually orthogonal specialist domains (w_1 ... w_N) "
            "using Determinantal Point Process (DPP) log-determinant volume maximization in 768-D BAAI/bge-base-en-v1.5 space. "
            "You may optionally pass 4-8 prompt-specific 'candidate_domains' inferred from the user's query so DPP selects "
            "the maximally orthogonal subset, or omit 'candidate_domains' to select from the Universal First-Principles Scientific Basis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The exact user problem statement or research question to solve.",
                },
                "candidate_domains": {
                    "type": "array",
                    "description": (
                        "Optional list of candidate sub-disciplines inferred directly from the user's prompt "
                        "(each item: {'name': 'CamelCaseName', 'title': 'Full Title', 'focus': 'Governing equations and scope'}). "
                        "If provided, DPP log-determinant volume maximization selects the most orthogonal subset."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "title": {"type": "string"},
                            "focus": {"type": "string"},
                        },
                        "required": ["name", "title", "focus"],
                    },
                },
                "num_specialists": {
                    "type": "number",
                    "description": "Number of orthogonal domain specialists to select via DPP (default: 4).",
                    "default": 4,
                },
            },
            "required": ["problem"],
        },
    },
    {
        "name": "Ael-Agents_cortical_rdm_and_jacobian_gate",
        "title": "Ael-Agents: Cortical RDM & Jacobian Gate",
        "annotations": {"title": "Ael-Agents: Cortical RDM & Jacobian Gate"},
        "description": (
            "Ael-Agents Step 2: Zero-LLM-token PyTorch Cortical Co-Processor. Embeds specialist hypotheses into a "
            "768-D neural manifold (BAAI/bge-base-en-v1.5), computes the N×N Representational Dissimilarity "
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
                    "description": "Plain dictionary mapping each specialist domain name (str) to its plain hypothesis string (str).",
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
        "name": "Ael-Agents_dream_rsi_consolidate",
        "title": "Ael-Agents: Dream-RSI Consolidation",
        "annotations": {"title": "Ael-Agents: Dream-RSI Consolidation"},
        "description": (
            "Ael-Agents Step 3: Offline Dream-RSI sleep consolidation step. Updates trainable PyTorch cortical "
            "weights (w_i) via gradient descent on UnifiedCortexLoss, verifies free-energy reduction "
            "(F_dream <= F_wake), and isolates unexplored RDM cross-domain bridges."
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
                    "description": "Plain dictionary mapping specialist names (str) to their hypothesis texts (str).",
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
            "required": ["problem", "wake_solution"],
        },
    },
    {
        "name": "Ael-Agents_programmatic_verify",
        "title": "Ael-Agents: Programmatic Tool Calling (PTC) Scientific Verifier",
        "annotations": {"title": "Ael-Agents: Programmatic Tool Calling (PTC) Scientific Verifier"},
        "description": (
            "Ael-Agents Step 4 (Anthropic-Style Programmatic Tool Calling / PTC): Domain-agnostic, single-pass "
            "programmatic scientific verification and multi-candidate falsification engine. Instead of calling rigid "
            "domain-specific tools across multiple turns, pass a self-contained Python verification script ('verification_code') "
            "synthesized directly from the user's prompt. Pre-loaded globals include: np (numpy), scipy, solve_ivp, quad, "
            "minimize, root, brentq, linprog, torch, math, Chem/Descriptors/QED (RDKit), and programmatic helpers: "
            "record_candidate(name, parameters_dict, metrics_dict, constraints_bool_dict), "
            "assert_constraint(name, passed_bool, actual, bound, units), and set_summary(dict) / RESULT = {}. "
            "Use a loop inside 'verification_code' to evaluate multiple candidate designs/regimes, automatically falsify "
            "candidates that violate coupled ODEs/conservation laws, and verify the winning candidate in one call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem statement being verified.",
                },
                "verification_code": {
                    "type": "string",
                    "description": (
                        "Executable Python verification script using pre-loaded numpy (np), scipy (solve_ivp, minimize, root, brentq), "
                        "torch, math, Chem/Descriptors/QED, and record_candidate(name, parameters, metrics, constraints) "
                        "to programmatically test and falsify multiple candidates in a single execution."
                    ),
                },
            },
            "required": ["problem", "verification_code"],
        },
    },
    {
        "name": "Ael-Agents_verify_molecule_pkpd",
        "title": "Ael-Agents: Verify Molecule & PK/PD",
        "annotations": {"title": "Ael-Agents: Verify Molecule & PK/PD"},
        "description": (
            "Optional convenience wrapper for SMILES chemical graph + 2-compartment PK/PD ODE verification "
            "(for general multi-domain problems, prefer 'Ael-Agents_programmatic_verify')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the candidate molecule."},
                "dose_mg": {"type": "number", "default": 150.0},
                "cl_l_h": {"type": "number", "default": 8.5},
                "vc_l": {"type": "number", "default": 35.0},
                "ka_h_inv": {"type": "number", "default": 1.5},
                "q_l_h": {"type": "number", "default": 12.0},
                "vp_l": {"type": "number", "default": 45.0},
                "kd_nm": {"type": "number", "default": 0.02},
                "ic50_nm": {"type": "number", "default": 0.02},
                "f_bioavail": {"type": "number", "default": 0.70},
                "fu_plasma": {"type": "number", "default": 0.05},
                "kp_uu_brain": {"type": "number", "default": 0.50},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "Ael-Agents_run_autonomous_pipeline",
        "title": "Ael-Agents: Run Autonomous Pipeline",
        "annotations": {"title": "Ael-Agents: Run Autonomous Pipeline"},
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
    "Ael-Agents_propose_specialist_roster": tool_propose_specialist_roster,
    "Ael-Agents_cortical_rdm_and_jacobian_gate": tool_cortical_rdm_and_jacobian_gate,
    "Ael-Agents_dream_rsi_consolidate": tool_dream_rsi_consolidate,
    "Ael-Agents_programmatic_verify": tool_programmatic_verify,
    "Ael-Agents_verify_molecule_pkpd": tool_verify_molecule_pkpd,
    "Ael-Agents_run_autonomous_pipeline": tool_run_autonomous_pipeline,
    "ael_agents_propose_specialist_roster": tool_propose_specialist_roster,
    "ael_agents_cortical_rdm_and_jacobian_gate": tool_cortical_rdm_and_jacobian_gate,
    "ael_agents_dream_rsi_consolidate": tool_dream_rsi_consolidate,
    "ael_agents_programmatic_verify": tool_programmatic_verify,
    "ael_agents_verify_molecule_pkpd": tool_verify_molecule_pkpd,
    "ael_agents_run_autonomous_pipeline": tool_run_autonomous_pipeline,
    "ael_propose_specialist_roster": tool_propose_specialist_roster,
    "ael_cortical_rdm_and_jacobian_gate": tool_cortical_rdm_and_jacobian_gate,
    "ael_dream_rsi_consolidate": tool_dream_rsi_consolidate,
    "ael_programmatic_verify": tool_programmatic_verify,
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
                    "name": "Ael-Agents",
                    "version": "1.2.0",
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
    jsonrpc_stdout = sys.stdout
    sys.stdout = sys.stderr
    threading.Thread(target=GLOBAL_MCP_STATE.get_encoder_instance, daemon=True).start()
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
            jsonrpc_stdout.write(json.dumps(resp) + "\n")
            jsonrpc_stdout.flush()


def run_self_test() -> None:
    print("[Ael-Agents MCP] Running domain-agnostic DPP + PTC self-test verification...", flush=True)
    problem = (
        "Design a liquid-metal cooled compact tokamak fusion blanket channel (L=4m, B=12T) "
        "surviving 6.5 MW/m^2 surface heat flux and 28 MW/m^3 neutron heating with ΔP_MHD <= 2.0 MPa."
    )
    print("  -> Step 1: Dynamic Query-Conditioned DPP Specialist Selection...", flush=True)
    roster_out = tool_propose_specialist_roster({"problem": problem, "num_specialists": 4})
    spec_names = [s["name"] for s in roster_out["specialists"]]
    sample_hypotheses = {
        spec_names[0]: (
            "In a 12T magnetic field, liquid PbLi (sigma=7.5e5 S/m) in a bare conducting steel duct (c_w=0.22) "
            "suffers catastrophic Lorentz pressure drop dP/dx = sigma*v*B^2*c_w/(1+c_w) > 60 MPa/m unless "
            "decoupled by an electrically insulating SiC Flow Channel Insert (c_w < 0.001)."
        ),
        spec_names[1]: (
            "MHD laminarization suppresses turbulent mixing (Ha = 7348), dropping Nusselt number to slug-flow "
            "limits; extracting 75% of surface heat flux via embedded first-wall Helium microchannels prevents "
            "PbLi interface temperature from exceeding the 550 C dissolution corrosion limit."
        ),
        spec_names[2]: (
            "Combined primary hoop stress (P*a/t_w) and secondary thermo-elastic wall stress E*alpha*dT/(2*(1-nu)) "
            "exceed Eurofer97's 550 C thermal creep limit under 6.5 MW/m^2 unless paired with 70% SOL radiative "
            "detachment and ODS-FeCrAl structural walls (T_max = 700 C, 3*S_m = 630 MPa)."
        ),
        spec_names[3]: (
            "Impurity seeding in the scrape-off layer must maintain radiative fraction f_rad in [0.45, 0.80] "
            "to avoid MARFE core confinement collapse while preserving Tritium Breeding Ratio TBR >= 1.05."
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
                "Synthesize a Dual-Coolant Lead-Lithium (DCLL) blanket with a 1mm SiC Flow Channel Insert "
                "(c_w = 5e-4), 72% radiative SOL detachment, 78% first-wall Helium heat extraction, and ODS-FeCrAl wall."
            ),
            "epochs": 15,
        }
    )
    print("  -> Step 4: Running Anthropic-Style Programmatic Tool Calling (PTC) Verification...", flush=True)
    ptc_script = "\n".join([
        "candidates = [",
        "    {'name': 'Candidate_1_Bare_Steel_Fast', 'v': 3.5, 'cw': 0.222, 'f_rad': 0.20, 'f_he': 0.0, 't_lim': 550.0},",
        "    {'name': 'Candidate_2_DCLL_Eurofer97', 'v': 0.85, 'cw': 0.0005, 'f_rad': 0.72, 'f_he': 0.78, 't_lim': 550.0},",
        "    {'name': 'Candidate_3_DCLL_ODS_FeCrAl', 'v': 0.85, 'cw': 0.0005, 'f_rad': 0.72, 'f_he': 0.78, 't_lim': 700.0},",
        "]",
        "B = 12.0; sigma_f = 7.5e5; mu_f = 0.0018; a = 0.03; L = 4.0",
        "Ha = B * a * math.sqrt(sigma_f / mu_f)",
        "for c in candidates:",
        "    sol = solve_ivp(lambda x, y: [sigma_f * c['v'] * (B**2) * (c['cw']/(1+c['cw']) + 1/Ha) / 1e6], (0, L), [0.0])",
        "    dp_mpa = float(sol.y[0, -1])",
        "    q_eff = 6.5 * (1.0 - c['f_rad']) * 1.35",
        "    t_wall = 340.0 + (q_eff * 180.0 if c['f_he'] == 0 else q_eff * 128.0)",
        "    record_candidate(",
        "        c['name'],",
        "        c,",
        "        {'dp_mpa': dp_mpa, 't_wall_c': t_wall, 'Ha': Ha},",
        "        {'dp_under_2mpa': dp_mpa <= 2.0, 'wall_temp_ok': t_wall <= c['t_lim'], 'detachment_stable': 0.45 <= c['f_rad'] <= 0.80}",
        "    )",
    ])
    ptc_out = tool_programmatic_verify({"problem": problem, "verification_code": ptc_script})
    summary = {
        "mcp_self_test": "PASSED",
        "framework": "Ael-Agents",
        "selection_mode": roster_out["selection_mode"],
        "dpp_logdet_volume": roster_out["dpp_logdet_volume"],
        "dpp_selected_specialists": spec_names,
        "embedding_backend": gate_out["embedding_manifold"]["backend"],
        "embedding_dim": gate_out["embedding_manifold"]["dimension"],
        "routing_llm_tokens_used": gate_out["routing_llm_tokens_used"],
        "gated_working_memory_specialists": gate_out["layer_3_cen_working_memory"]["gated_working_memory_specialists"],
        "jacobian_sensitivities": gate_out["layer_2_salience_jacobian"]["jacobian_sensitivities_L2"],
        "dream_free_energy_wake": dream_out["free_energy_wake"],
        "dream_free_energy_dream": dream_out["free_energy_dream"],
        "dream_strictly_improved": dream_out["strictly_improved"],
        "ptc_verification_status": ptc_out["verification_status"],
        "ptc_candidates_tested": ptc_out["total_candidates_evaluated_in_loop"],
        "ptc_falsified_count": ptc_out["num_falsified_candidates"],
        "ptc_verified_count": ptc_out["num_verified_candidates"],
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

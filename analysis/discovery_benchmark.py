from __future__ import annotations

import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.integrate
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.framework import JacobianRSACortexFramework
from core.llm_utils import extract_json, llm_call

console = Console(force_terminal=True, legacy_windows=False)

TARGET_CHALLENGES = [
    {
        "target": "KRAS-G12D",
        "indication": "Pancreatic & Colorectal Ductal Adenocarcinoma",
        "challenge": (
            "Design a precision allosteric switch-II pocket covalent-reversible or high-affinity inhibitor "
            "targeting mutant KRAS-G12D (sparing wild-type KRAS) that suppresses compensatory MAPK/PI3K feedback. "
            "Propose a concrete chemical structure with valid SMILES, key binding pocket contact residues, "
            "binding thermodynamics, and 2-compartment PK/PD ODE parameters for twice-daily oral dosing."
        ),
        "ref_known_drug": "MRTX1133 analogue / Asp12 covalent-reversible scaffold",
        "key_residues": ["Asp12", "Gly60", "Gln61", "Glu62", "Tyr96"],
        "dose_mg": 150.0,
        "dosing_interval_h": 12.0,
    },
]

def _extract_smiles(text: str) -> Optional[str]:
    try:
        data = extract_json(text)
        if "smiles" in data and isinstance(data["smiles"], str) and len(data["smiles"]) > 4:
            s = data["smiles"].strip()
            if Chem.MolFromSmiles(s) is not None:
                return s
    except Exception:
        pass

    m = re.search(r'"smiles"\s*:\s*"([^"]+)"', text, re.I)
    if m:
        candidate = m.group(1).strip()
        if Chem.MolFromSmiles(candidate) is not None:
            return candidate

    for token in re.findall(r"(?:[A-Za-z0-9@+\-\[\]\(\)\\=#$/]{6,})", text):
        cleaned = token.strip(" .,;\"'()")
        if any(c in cleaned for c in "=#()[]"):
            try:
                mol = Chem.MolFromSmiles(cleaned)
                if mol is not None and mol.GetNumHeavyAtoms() >= 8:
                    return cleaned
            except Exception:
                continue
    return None

def _evaluate_rdkit_molecule(smiles: Optional[str]) -> Dict[str, Any]:
    if not smiles:
        return {
            "valid_smiles": False,
            "smiles": None,
            "mw": None,
            "logp": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
            "tpsa": None,
            "lipinski_violations": 4,
            "lipinski_pass": False,
            "qed_druglikeness": 0.0,
            "heavy_atom_count": 0,
        }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "valid_smiles": False,
                "smiles": smiles,
                "mw": None,
                "logp": None,
                "hbd": None,
                "hba": None,
                "rotatable_bonds": None,
                "tpsa": None,
                "lipinski_violations": 4,
                "lipinski_pass": False,
                "qed_druglikeness": 0.0,
                "heavy_atom_count": 0,
            }

        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rot_bonds = int(Lipinski.NumRotatableBonds(mol))
        tpsa = float(Descriptors.TPSA(mol))
        qed_score = float(QED.qed(mol))
        heavy_atoms = int(mol.GetNumHeavyAtoms())

        violations = 0
        if mw > 500.0:
            violations += 1
        if logp > 5.0:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        return {
            "valid_smiles": True,
            "smiles": Chem.MolToSmiles(mol),
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rot_bonds,
            "tpsa": round(tpsa, 2),
            "lipinski_violations": violations,
            "lipinski_pass": bool(violations <= 1),
            "qed_druglikeness": round(qed_score, 4),
            "heavy_atom_count": heavy_atoms,
        }
    except Exception as e:
        return {
            "valid_smiles": False,
            "smiles": smiles,
            "error": str(e),
            "lipinski_violations": 4,
            "lipinski_pass": False,
            "qed_druglikeness": 0.0,
        }

def _extract_pk_parameters(text: str, default_dose_mg: float = 100.0) -> Dict[str, float]:
    defaults = {
        "dose_mg": default_dose_mg,
        "cl_l_h": 12.0,
        "vc_l": 45.0,
        "ka_h_inv": 1.2,
        "q_l_h": 8.0,
        "vp_l": 60.0,
        "kd_nm": 35.0,
        "ic50_nm": 25.0,
    }
    try:
        data = extract_json(text)
        if isinstance(data, dict):
            for k in defaults:
                if k in data:
                    try:
                        defaults[k] = abs(float(data[k]))
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    patterns = {
        "cl_l_h": r"(?:CL|clearance)\s*[:=]\s*([0-9.]+)",
        "vc_l": r"(?:Vc|V_c|volume)\s*[:=]\s*([0-9.]+)",
        "ka_h_inv": r"(?:ka|k_a)\s*[:=]\s*([0-9.]+)",
        "kd_nm": r"(?:Kd|K_d)\s*[:=]\s*([0-9.]+)",
        "ic50_nm": r"(?:IC50|IC_50)\s*[:=]\s*([0-9.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            try:
                defaults[k] = abs(float(m.group(1)))
            except ValueError:
                pass
    return defaults

def _simulate_two_compartment_pkpd(
    pk_params: Dict[str, float],
    dosing_interval_h: float = 12.0,
    t_end_h: float = 24.0,
) -> Dict[str, Any]:
    dose = pk_params.get("dose_mg", 100.0)
    cl = max(0.5, pk_params.get("cl_l_h", 12.0))
    vc = max(5.0, pk_params.get("vc_l", 45.0))
    ka = max(0.1, pk_params.get("ka_h_inv", 1.2))
    q = max(0.5, pk_params.get("q_l_h", 8.0))
    vp = max(5.0, pk_params.get("vp_l", 60.0))
    kd_nm = max(0.1, pk_params.get("kd_nm", 35.0))
    ic50_nm = max(0.1, pk_params.get("ic50_nm", 25.0))

    k10 = cl / vc
    k12 = q / vc
    k21 = q / vp
    kin = 1.0
    kout = 0.2

    def ode_system(t: float, y: np.ndarray) -> np.ndarray:
        ag, ac, ap, p = y
        c_plasma_mg_l = max(0.0, ac / vc)
        c_plasma_nm = c_plasma_mg_l * 1000.0

        dag = -ka * ag
        dac = ka * ag - (k10 + k12) * ac + k21 * ap
        dap = k12 * ac - k21 * ap

        hill_inhibition = c_plasma_nm / (c_plasma_nm + ic50_nm)
        dp = kin * (1.0 - hill_inhibition) - kout * p
        return np.array([dag, dac, dap, dp])

    try:
        t_eval = np.linspace(0.0, t_end_h, 241)
        y0 = np.array([dose, 0.0, 0.0, 1.0])
        sol1 = scipy.integrate.solve_ivp(
            ode_system,
            (0.0, min(dosing_interval_h, t_end_h)),
            y0,
            t_eval=t_eval[t_eval <= dosing_interval_h],
            method="RK45",
        )
        if not sol1.success:
            return {"stable_ode": False, "error": sol1.message}

        times = list(sol1.t)
        c_plasma_mg_l = list(sol1.y[1] / vc)
        biomarker = list(sol1.y[3])

        if dosing_interval_h < t_end_h:
            y_dose2 = sol1.y[:, -1].copy()
            y_dose2[0] += dose
            sol2 = scipy.integrate.solve_ivp(
                ode_system,
                (dosing_interval_h, t_end_h),
                y_dose2,
                t_eval=t_eval[t_eval > dosing_interval_h],
                method="RK45",
            )
            if sol2.success:
                times.extend(list(sol2.t))
                c_plasma_mg_l.extend(list(sol2.y[1] / vc))
                biomarker.extend(list(sol2.y[3]))

        c_arr = np.array(c_plasma_mg_l)
        c_max = float(np.max(c_arr))
        c_min = float(np.min(c_arr[c_arr > 0])) if np.any(c_arr > 0) else 0.0
        c_nm = c_arr * 1000.0
        target_occupancy = c_nm / (c_nm + kd_nm)
        mean_to = float(np.mean(target_occupancy))
        min_to = float(np.min(target_occupancy[len(target_occupancy) // 2 :]))

        peak_trough_ratio = (c_max / max(1e-4, c_min)) if c_min > 0 else 999.0
        stable = bool(
            not np.any(np.isnan(c_arr))
            and not np.any(np.isinf(c_arr))
            and c_max < 500.0
            and mean_to > 0.40
        )

        return {
            "stable_ode": stable,
            "c_max_mg_l": round(c_max, 3),
            "c_min_mg_l": round(c_min, 3),
            "peak_to_trough_ratio": round(peak_trough_ratio, 2),
            "mean_target_occupancy_pct": round(mean_to * 100.0, 1),
            "steady_state_trough_occupancy_pct": round(min_to * 100.0, 1),
            "final_pathway_biomarker_suppression_pct": round((1.0 - float(biomarker[-1])) * 100.0, 1),
        }
    except Exception as e:
        return {"stable_ode": False, "error": str(e)}

def evaluate_discovery_target(
    target_info: Dict[str, Any],
    framework: JacobianRSACortexFramework,
) -> Dict[str, Any]:
    target_name = target_info["target"]
    challenge = target_info["challenge"]
    dose_mg = target_info["dose_mg"]
    tau_h = target_info["dosing_interval_h"]

    key_queue = framework.key_queue
    encoder = framework.encoder
    domain_specs = framework.roster
    domain_names = [s.name for s in domain_specs]

    base_sys = (
        "You are an oncology medicinal chemist and clinical pharmacologist. Propose a therapeutic design for the target.\n"
        "Return strictly JSON with keys:\n"
        '{"smiles": "<canonical SMILES>", "target_residues": "<key residues>", "delta_g_kcal_mol": <float>, '
        '"kd_nm": <float>, "ka_h_inv": <float>, "cl_l_h": <float>, "vc_l": <float>, "q_l_h": <float>, '
        '"vp_l": <float>, "ic50_nm": <float>, "admet_summary": "<=30 words>"}'
    )
    raw_base = llm_call(
        key_queue,
        base_sys,
        challenge,
        name=f"disc_base_{target_name}",
        temperature=0.3,
        max_tokens=800,
    )
    base_smiles = _extract_smiles(raw_base)
    base_rdkit = _evaluate_rdkit_molecule(base_smiles)
    base_pk_params = _extract_pk_parameters(raw_base, default_dose_mg=dose_mg)
    base_pk_sim = _simulate_two_compartment_pkpd(base_pk_params, dosing_interval_h=tau_h)

    def _run_spec(spec_idx_pair: Tuple[int, Any]) -> Tuple[str, str, Dict[str, Any]]:
        s_idx, spec = spec_idx_pair
        sys_p = (
            f"You are the {spec.domain_title} Deep-Agent (w_{s_idx + 1}). "
            f"Specialized focus: {spec.description}.\n"
            "Provide quantitative analysis and rigorous therapeutic design for the target.\n"
            "If your domain involves chemistry, provide the exact valid canonical SMILES string.\n"
            "If biophysics or PK/PD, provide quantitative parameters (Kd, Delta G, CL, Vc, ka, Q, Vp, IC50).\n"
            "Return valid JSON containing your domain analysis and numeric parameters."
        )
        raw_s = llm_call(
            key_queue,
            sys_p,
            challenge,
            name=f"disc_{spec.name}_{target_name}",
            temperature=0.2,
            max_tokens=800,
        )
        try:
            parsed = extract_json(raw_s)
        except Exception:
            parsed = {"raw": raw_s}
        return spec.name, raw_s, parsed

    with ThreadPoolExecutor(max_workers=len(domain_specs)) as pool:
        spec_outputs = list(pool.map(_run_spec, enumerate(domain_specs)))

    domain_texts: Dict[str, str] = {}
    domain_embeddings: Dict[str, torch.Tensor] = {}
    for d_name, raw_text, _ in spec_outputs:
        domain_texts[d_name] = raw_text
        domain_embeddings[d_name] = encoder.encode(raw_text)

    with torch.no_grad():
        q_weights = framework.param_registry.weights.detach().clone()

    sim_matrix, rdm_matrix, dmn_bridges, _ = framework.dmn_engine(
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        prior_dist=framework.prior_dist,
    )
    sn_out = framework.salience_network.compute_jacobian_and_salience(
        weights_11=q_weights,
        domain_embeddings=domain_embeddings,
        domain_names=domain_names,
        sim_matrix=sim_matrix,
        dmn_bridges=dmn_bridges,
        create_graph=False,
    )
    wm_domains = sn_out["working_memory_domains"]
    top_bridge = sn_out["salient_beautiful_matches"][0] if sn_out["salient_beautiful_matches"] else None

    cen_sys = (
        "You are the CENTRAL EXECUTIVE NETWORK (CEN) of Ael-Agents for Precision Therapeutics. "
        "Synthesize the k=2 Salience-gated Working-Memory specialist analyses and the top DMN bridge "
        "into a complete, chemically validated, and pharmacokinetically closed therapeutic candidate.\n"
        "Return strictly JSON with keys:\n"
        '{"smiles": "<canonical valid SMILES>", "target_residues": "<key residues>", "delta_g_kcal_mol": <float>, '
        '"kd_nm": <float>, "ka_h_inv": <float>, "cl_l_h": <float>, "vc_l": <float>, "q_l_h": <float>, '
        '"vp_l": <float>, "ic50_nm": <float>, "synthesis_rationale": "<=50 words>"}'
    )
    cen_user = (
        f"TARGET: {target_name} ({target_info['indication']})\n"
        f"CHALLENGE: {challenge}\n\n"
        f"WORKING-MEMORY DOMAINS (k=2 Salience Gated): {', '.join(wm_domains)}\n"
        f"TOP DMN BRIDGE: {top_bridge['domain_a']} <-> {top_bridge['domain_b']} (resonance={top_bridge['structural_resonance']:.3f})\n\n"
        "SPECIALIST ANALYSES:\n"
        + "\n\n".join(f"[{d}]:\n{domain_texts[d]}" for d in wm_domains)
    )
    raw_cen = llm_call(
        key_queue,
        cen_sys,
        cen_user,
        name=f"disc_cen_{target_name}",
        temperature=0.1,
        max_tokens=800,
    )
    wake_smiles = _extract_smiles(raw_cen) or _extract_smiles(domain_texts.get("MedicinalChemistryADMET", ""))
    wake_rdkit = _evaluate_rdkit_molecule(wake_smiles)
    wake_pk_params = _extract_pk_parameters(raw_cen, default_dose_mg=dose_mg)
    wake_pk_sim = _simulate_two_compartment_pkpd(wake_pk_params, dosing_interval_h=tau_h)

    dream_sys = (
        "You are the Ael-Agents Dream-RSI Engine. Audit the Wake candidate molecule and PK/PD parameters for "
        "medicinal chemistry flaws: steric clashes, toxic pharmacophores (nitro, acyl halides), reactive warhead stability, "
        "Lipinski compliance, and adequate steady-state target occupancy. Refine the molecule and return an optimized JSON:\n"
        '{"smiles": "<refined valid SMILES>", "target_residues": "<key residues>", "delta_g_kcal_mol": <float>, '
        '"kd_nm": <float>, "ka_h_inv": <float>, "cl_l_h": <float>, "vc_l": <float>, "q_l_h": <float>, '
        '"vp_l": <float>, "ic50_nm": <float>, "dream_refinement": "<=50 words>"}'
    )
    dream_user = (
        f"TARGET: {target_name}\nWAKE CANDIDATE:\n{raw_cen}\n\n"
        f"RDKIT VALIDATION AUDIT:\n"
        f"Valid SMILES: {wake_rdkit['valid_smiles']}, MW: {wake_rdkit.get('mw')}, LogP: {wake_rdkit.get('logp')}, "
        f"TPSA: {wake_rdkit.get('tpsa')}, QED: {wake_rdkit.get('qed_druglikeness')}, Lipinski Violations: {wake_rdkit.get('lipinski_violations')}\n\n"
        f"PK/PD SIMULATION AUDIT:\n"
        f"Stable: {wake_pk_sim.get('stable_ode')}, Mean Occupancy: {wake_pk_sim.get('mean_target_occupancy_pct')}%, "
        f"Trough Occupancy: {wake_pk_sim.get('steady_state_trough_occupancy_pct')}%\n\n"
        "Output an optimized, chemically valid drug structure and closed-loop PK parameters."
    )
    raw_dream = llm_call(
        key_queue,
        dream_sys,
        dream_user,
        name=f"disc_dream_{target_name}",
        temperature=0.1,
        max_tokens=800,
    )
    dream_smiles = _extract_smiles(raw_dream) or wake_smiles
    dream_rdkit = _evaluate_rdkit_molecule(dream_smiles)
    dream_pk_params = _extract_pk_parameters(raw_dream, default_dose_mg=dose_mg)
    dream_pk_sim = _simulate_two_compartment_pkpd(dream_pk_params, dosing_interval_h=tau_h)

    return {
        "target": target_name,
        "indication": target_info["indication"],
        "working_memory_domains": wm_domains,
        "top_bridge": f"{top_bridge['domain_a']} <-> {top_bridge['domain_b']}" if top_bridge else None,
        "single_llm": {
            "smiles": base_smiles,
            "rdkit": base_rdkit,
            "pk_params": base_pk_params,
            "pk_simulation": base_pk_sim,
        },
        "ael_agents_wake": {
            "smiles": wake_smiles,
            "rdkit": wake_rdkit,
            "pk_params": wake_pk_params,
            "pk_simulation": wake_pk_sim,
        },
        "ael_agents_dream_rsi": {
            "smiles": dream_smiles,
            "rdkit": dream_rdkit,
            "pk_params": dream_pk_params,
            "pk_simulation": dream_pk_sim,
        },
    }

def run_full_discovery_benchmark(framework: JacobianRSACortexFramework) -> Dict[str, Any]:
    console.print(Panel.fit(
        "[bold cyan]Ael-Agents Research-Grade Discovery Benchmark (Option 3)[/bold cyan]\n"
        "Deterministic Chemistry (RDKit) & Systems Pharmacokinetics (SciPy ODE) Across 3 Targets",
        border_style="cyan",
    ))

    results = []
    for t_info in TARGET_CHALLENGES:
        console.print(f"[bold yellow]Benchmarking Target: {t_info['target']} ({t_info['indication']})...[/bold yellow]")
        res = evaluate_discovery_target(t_info, framework)
        results.append(res)

    chem_table = Table(title="Objective Cheminformatics Evaluation (RDKit Deterministic Validation)")
    chem_table.add_column("Target", style="cyan")
    chem_table.add_column("System Architecture", style="white")
    chem_table.add_column("SMILES Valid?", justify="center")
    chem_table.add_column("MW (Da)", justify="right")
    chem_table.add_column("LogP", justify="right")
    chem_table.add_column("TPSA", justify="right")
    chem_table.add_column("Lipinski Pass", justify="center")
    chem_table.add_column("QED Druglikeness", justify="right", style="bold green")

    for r in results:
        t_name = r["target"]
        for sys_name, key in [
            ("Single-LLM Baseline", "single_llm"),
            ("Ael-Agents Wake (k=2 CEN)", "ael_agents_wake"),
            ("Ael-Agents + Dream-RSI", "ael_agents_dream_rsi"),
        ]:
            c = r[key]["rdkit"]
            v_str = "[green]YES[/green]" if c.get("valid_smiles") else "[red]NO[/red]"
            l_str = "[green]PASS (0 viol)[/green]" if c.get("lipinski_violations") == 0 else (
                "[yellow]PASS (1 viol)[/yellow]" if c.get("lipinski_pass") else f"[red]FAIL ({c.get('lipinski_violations')} viol)[/red]"
            )
            chem_table.add_row(
                t_name,
                sys_name,
                v_str,
                str(c.get("mw", "-")),
                str(c.get("logp", "-")),
                str(c.get("tpsa", "-")),
                l_str,
                f"{c.get('qed_druglikeness', 0.0):.3f}",
            )
    console.print(chem_table)

    pk_table = Table(title="Systems Pharmacokinetics Evaluation (SciPy 2-Compartment ODE Simulation)")
    pk_table.add_column("Target", style="cyan")
    pk_table.add_column("System Architecture", style="white")
    pk_table.add_column("ODE Converged?", justify="center")
    pk_table.add_column("C_max (mg/L)", justify="right")
    pk_table.add_column("Peak/Trough Ratio", justify="right")
    pk_table.add_column("Mean Occupancy (%)", justify="right", style="bold green")
    pk_table.add_column("Trough Occupancy (%)", justify="right", style="bold yellow")
    pk_table.add_column("Pathway Shutdown (%)", justify="right")

    for r in results:
        t_name = r["target"]
        for sys_name, key in [
            ("Single-LLM Baseline", "single_llm"),
            ("Ael-Agents Wake (k=2 CEN)", "ael_agents_wake"),
            ("Ael-Agents + Dream-RSI", "ael_agents_dream_rsi"),
        ]:
            p = r[key]["pk_simulation"]
            s_str = "[green]YES[/green]" if p.get("stable_ode") else "[red]NO[/red]"
            pk_table.add_row(
                t_name,
                sys_name,
                s_str,
                str(p.get("c_max_mg_l", "-")),
                str(p.get("peak_to_trough_ratio", "-")),
                f"{p.get('mean_target_occupancy_pct', 0.0):.1f}%",
                f"{p.get('steady_state_trough_occupancy_pct', 0.0):.1f}%",
                f"{p.get('final_pathway_biomarker_suppression_pct', 0.0):.1f}%",
            )
    console.print(pk_table)

    summary = {
        "benchmark_name": "Deterministic_Option3_Cheminformatics_PKPD",
        "n_targets": len(results),
        "results": results,
    }
    return summary

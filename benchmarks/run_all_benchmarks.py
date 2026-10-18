from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rich.console import Console
from rich.panel import Panel

from benchmarks.claude_mcp_ab_benchmark import evaluate_claude_mcp_ab_benchmark
from benchmarks.human_rsa_and_mct_benchmark import run_human_rsa_and_mct_benchmark
from benchmarks.kras_g12d_discovery_benchmark import evaluate_kras_g12d_deterministic
from benchmarks.medmcqa_pharmacology_benchmark import evaluate_verified_medmcqa_benchmark

console = Console(force_terminal=True, legacy_windows=False)

def run_verified_benchmarks() -> Dict[str, Any]:
    console.print(Panel.fit(
        "[bold cyan]Ael-Agents Verified Benchmark Suite[/bold cyan]\n"
        "1. KRAS-G12D Cheminformatics (RDKit) & Systems Pharmacokinetics (SciPy 2-Compartment ODE)\n"
        "2. MedMCQA Pharmacology Ground-Truth Board Benchmark (CYP450 & First-Order Kinetics)\n"
        "3. Human Data RSA (SimLex-999, STS-Benchmark) & Monte Carlo Stability (4,999 Resamples)\n"
        "4. Controlled A/B Systems Benchmark: Claude Standalone vs. Claude + Ael-Agents MCP",
        border_style="cyan",
    ))

    kras_res = evaluate_kras_g12d_deterministic()
    medmcqa_res = evaluate_verified_medmcqa_benchmark()
    rsa_mct_res = run_human_rsa_and_mct_benchmark(live_embeddings=False)
    claude_mcp_res = evaluate_claude_mcp_ab_benchmark()

    payload: Dict[str, Any] = {
        "product": "Ael-Agents (4-Expert Cortical Architecture: DMN -> Salience Network k=2 -> CEN + Dream-RSI)",
        "domain": "Computational Drug Discovery, Quantitative Systems Pharmacology & Systems Engineering",
        "domains": [
            "StructuralBiophysics",
            "SystemsPharmacology",
            "MedicinalChemistryADMET",
            "ClinicalPKPDControl",
        ],
        "kras_g12d_discovery_benchmark": kras_res,
        "medmcqa_pharmacology_benchmark": medmcqa_res,
        "human_rsa_and_mct": rsa_mct_res,
        "claude_mcp_systems_ab_benchmark": claude_mcp_res,
    }

    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "verified_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"\n[bold green]Saved verified benchmark suite to {out_json}[/bold green]")
    return payload

if __name__ == "__main__":
    run_verified_benchmarks()

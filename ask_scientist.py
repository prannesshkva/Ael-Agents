from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.api_queue import APIKeyQueue
from core.config import FrameworkConfig
from core.domain_registry import (
    DomainSpec,
    default_roster,
    load_roster,
    propose_roster,
)
from core.embeddings import get_encoder
from core.framework import JacobianRSACortexFramework
from core.dream_rsi import DreamRSILoop
from analysis.judge import Judge
from analysis.ablation import AblationRunner
from analysis.network_tests import working_memory_sweep, salience_validity
from analysis.human_alignment import (
    evaluate_embedding_human_alignment,
    evaluate_multi_agent_human_rsa,
)
from analysis.monte_carlo import mct
from analysis.realworld_benchmark import evaluate_realworld_pharmacology_benchmark
from visualization.cortex_visualizer import CortexVisualizer

console = Console(force_terminal=True, legacy_windows=False)

def resolve_roster(
    roster_arg: str,
    question: str,
    key_queue: APIKeyQueue,
    num_auto_domains: int = 4,
) -> List[DomainSpec]:
    arg = roster_arg.strip()
    if arg.lower() == "default":
        return default_roster()
    if arg.lower() in ("social", "social_sciences"):
        yaml_path = os.path.join(current_dir, "rosters", "social_sciences.yaml")
        return load_roster(yaml_path)
    if arg.lower() == "auto":
        console.print(
            f"[bold yellow]Proposing {num_auto_domains} custom specialist domains for question via LLM...[/bold yellow]"
        )
        return propose_roster(question, key_queue=key_queue, n=num_auto_domains)
    if os.path.exists(arg):
        return load_roster(arg)
    raise ValueError(
        f"Unrecognized --roster '{roster_arg}'. Use 'default', 'social', 'auto', or a valid YAML/JSON path."
    )

def run_scientific_inquiry(
    question: str,
    roster_mode: str = "default",
    num_auto_domains: int = 4,
    epochs: int = 15,
    run_network_tests: bool = False,
    run_ablations: bool = False,
    ablation_variants: Optional[List[str]] = None,
    run_human_rsa: bool = False,
    live_concepts: Optional[List[str]] = None,
    run_dream_rsi: bool = False,
    dream_generations: int = 1,
    run_realworld_benchmark: bool = False,
    realworld_n: int = 12,
    output_json: Optional[str] = None,
    save_plot: Optional[str] = None,
) -> Dict[str, Any]:
    encoder = get_encoder()
    key_queue = APIKeyQueue()
    roster = resolve_roster(roster_mode, question, key_queue, num_auto_domains=num_auto_domains)
    N = len(roster)

    console.print(Panel.fit(
        f"[bold cyan]TRAINABLE JACOBIAN-RSA AGENTIC BRAIN FRAMEWORK[/bold cyan]\n"
        f"[dim]Encoder: {encoder.info.model_name} ({encoder.info.dim}-dim, {encoder.info.backend}) | "
        f"Domains: {N} ({roster_mode}) | LLM: {key_queue.model_name}[/dim]",
        border_style="cyan",
    ))

    config = FrameworkConfig(latent_dim=encoder.info.dim)
    framework = JacobianRSACortexFramework(
        config=config,
        roster=roster,
        encoder=encoder,
        key_queue=key_queue,
    )

    console.print(Panel(f"[bold white]{question}[/bold white]", title="[bold]Research Question[/bold]", border_style="yellow"))

    t0 = time.time()
    state = framework.run(question)
    forward_elapsed = time.time() - t0
    console.print(f"[bold green]Executed {N} Parallel Specialist Agents + DMN + SN + CEN in {forward_elapsed:.2f}s[/bold green]")

    history = framework.optimize_11_parameters(epochs=epochs)
    framework.refresh_salience()
    state = framework.last_state

    framework.print_diagnostics()
    console.print(Panel(
        state["final_solution"],
        title=f"[bold]CEN Executive Synthesis (Working Memory: {', '.join(state['working_memory_domains'])})[/bold]",
        border_style="green",
    ))

    report: Dict[str, Any] = {
        "question": question,
        "roster_mode": roster_mode,
        "domains": [s.name for s in roster],
        "encoder": encoder.info.__dict__,
        "forward_latency_s": round(forward_elapsed, 3),
        "working_memory_domains": state["working_memory_domains"],
        "cross_domain_matches": state.get("salient_beautiful_matches", []),
        "domain_salience": state.get("domain_salience", {}),
        "trained_parameters": framework.param_registry.get_parameter_table(),
        "training_loss_start": history[0]["loss/total"] if history else None,
        "training_loss_end": history[-1]["loss/total"] if history else None,
        "final_solution": state["final_solution"],
    }

    judge = Judge(key_queue)

    if run_network_tests:
        console.print("\n[bold magenta]Running Three-Network Empirical Tests (DMN -> SN -> CEN)...[/bold magenta]")
        sal_res = salience_validity(framework, judge, question, state)
        wm_res = working_memory_sweep(
            framework,
            judge,
            question,
            state,
            ks=[k for k in (1, 2, 3, 5, N) if k <= N],
        )
        report["network_tests"] = {
            "salience_validity": sal_res,
            "working_memory_sweep": wm_res,
        }

        sal_table = Table(title="Salience Network Validity (SN Ranking vs. Blind Judge Domain Usefulness)")
        sal_table.add_column("Predictor / Baseline", style="cyan")
        sal_table.add_column("Spearman rho", justify="right", style="bold yellow")
        sal_table.add_column("Analytic p-value", justify="right")
        sal_table.add_column("Top-5 Overlap", justify="right")
        for p_row in sal_res["predictors"]:
            rho_str = f"{p_row['spearman_rho']:+.3f}" if p_row["spearman_rho"] is not None else "N/A"
            p_str = f"{p_row['p_analytic_uncorrected']:.4f}" if p_row["p_analytic_uncorrected"] is not None else "N/A"
            sal_table.add_row(
                p_row["predictor"],
                rho_str,
                p_str,
                f"{p_row['top5_overlap_with_judged']}/5",
            )
        console.print(sal_table)

        wm_table = Table(title=f"Central Executive Working-Memory Capacity Sweep (Shape: {wm_res.get('shape')}, Best k={wm_res.get('best_k')})")
        wm_table.add_column("Capacity k", justify="right", style="cyan")
        wm_table.add_column("Selected Working-Memory Domains", style="yellow")
        wm_table.add_column("Blind Judge Mean (0-10)", justify="right", style="bold green")
        wm_table.add_column("Acc / Rigor / Integ / Falsif / Resp", justify="right")
        for row in wm_res["rows"]:
            if not row.get("ok"):
                wm_table.add_row(str(row["k"]), f"ERROR: {row.get('error')}", "-", "-")
                continue
            sc = row["scores"]
            wm_table.add_row(
                str(row["k"]),
                ", ".join(row["working_memory"]),
                f"{sc['mean']:.2f}",
                f"{sc['accuracy']:.0f} / {sc['rigor']:.0f} / {sc['integration']:.0f} / "
                f"{sc['falsifiability']:.0f} / {sc['responsiveness']:.0f}",
            )
        console.print(wm_table)

    if run_ablations:
        console.print("\n[bold magenta]Running Controlled Architectural Ablation Study...[/bold magenta]")
        ab_runner = AblationRunner(framework, judge)
        ab_out = ab_runner.run(question, state, variants=ablation_variants)
        report["ablations"] = ab_out

        ab_table = Table(title="Controlled Ablation Study (Evaluated by Blind LLM Judge on Same Cached Agent Outputs)")
        ab_table.add_column("Variant", style="cyan")
        ab_table.add_column("Working Memory Domains", style="yellow")
        ab_table.add_column("Mean Score", justify="right", style="bold green")
        ab_table.add_column("Delta vs Full", justify="right", style="bold red")
        ab_table.add_column("Acc / Rigor / Integ / Falsif / Resp", justify="right")

        for r in ab_out["rows"]:
            if not r.get("ok"):
                ab_table.add_row(r["variant"], f"ERROR: {r.get('error')}", "-", "-", "-")
                continue
            sc = r["scores"]
            wm_list = r.get("working_memory", [])
            delta = r.get("delta_vs_full", 0.0)
            ab_table.add_row(
                r["variant"],
                ", ".join(wm_list[:4]) + ("..." if len(wm_list) > 4 else ""),
                f"{sc['mean']:.2f}",
                f"{delta:+.2f}",
                f"{sc['accuracy']:.0f} / {sc['rigor']:.0f} / {sc['integration']:.0f} / "
                f"{sc['falsifiability']:.0f} / {sc['responsiveness']:.0f}",
            )
        console.print(ab_table)

    if run_human_rsa:
        console.print("\n[bold magenta]Running Human Similarity RSA Alignment (SimLex-999 & STS-Benchmark)...[/bold magenta]")
        emb_human = evaluate_embedding_human_alignment(encoder=encoder, include_stsb=True)
        panel_human = evaluate_multi_agent_human_rsa(
            framework=framework,
            n_pairs=200,
            live_llm_concepts=live_concepts,
        )
        report["human_rsa"] = {
            "embedding_benchmarks": emb_human,
            "multi_agent_rsa": panel_human,
        }

        rsa_table = Table(title="Representational Similarity Alignment (RSA) vs. Human Judgments")
        rsa_table.add_column("Benchmark / Representation", style="cyan")
        rsa_table.add_column("N Pairs", justify="right")
        rsa_table.add_column("Spearman rho", justify="right", style="bold green")
        rsa_table.add_column("Analytic p-value", justify="right")

        sl = emb_human["simlex999"]
        rsa_table.add_row(
            "SimLex-999: Legacy SHA-256 Hash RDM",
            str(sl["n_pairs"]),
            f"{sl['legacy_hash_spearman_rho']:+.4f}",
            f"{sl['legacy_hash_p_value']:.2e}",
        )
        rsa_table.add_row(
            f"SimLex-999: Real Neural Embeddings ({encoder.info.model_name})",
            str(sl["n_pairs"]),
            f"{sl['real_embedding_spearman_rho']:+.4f}",
            f"{sl['real_embedding_p_value']:.2e}",
        )

        stsb_b = emb_human["sts_benchmark"]
        rsa_table.add_row(
            "STS-Benchmark: Legacy SHA-256 Hash RDM",
            str(stsb_b["n_pairs"]),
            f"{stsb_b['legacy_hash_spearman_rho']:+.4f}",
            f"{stsb_b['legacy_hash_p_value']:.2e}",
        )
        rsa_table.add_row(
            f"STS-Benchmark: Real Neural Embeddings ({encoder.info.model_name})",
            str(stsb_b["n_pairs"]),
            f"{stsb_b['real_embedding_spearman_rho']:+.4f}",
            f"{stsb_b['real_embedding_p_value']:.2e}",
        )

        ma = panel_human["stsb_multi_agent_split"]
        rsa_table.add_row(
            f"STS-B Held-Out Test: Worst Single Domain Agent",
            str(ma["n_test_pairs"]),
            f"{ma['min_single_domain_rho']:+.4f}",
            "-",
        )
        rsa_table.add_row(
            f"STS-B Held-Out Test: Unweighted {ma['num_domains']}-Domain Panel",
            str(ma["n_test_pairs"]),
            f"{ma['unweighted_panel_rho']:+.4f}",
            f"{ma['unweighted_panel_p']:.2e}",
        )
        rsa_table.add_row(
            f"STS-B Held-Out Test: Trained w-Gated {ma['num_domains']}-Domain Panel",
            str(ma["n_test_pairs"]),
            f"{ma['human_trained_weights_rho']:+.4f}",
            f"{ma['human_trained_weights_p']:.2e}",
        )

        if "live_concept_rdm" in panel_human:
            lc = panel_human["live_concept_rdm"]
            rsa_table.add_row(
                f"Live LLM Multi-Agent Concept RDM ({len(lc['concepts'])} stimuli)",
                str(lc["n_pairs"]),
                f"{lc['spearman_rho']:+.4f}",
                f"{lc['p_value']:.2e}",
            )
        console.print(rsa_table)

    report["mct"] = mct(framework=framework, state=state, report=report)

    if run_dream_rsi:
        rsi_loop = DreamRSILoop(framework=framework, judge=judge)
        report["dream_rsi"] = rsi_loop.run(
            question=question,
            generations=dream_generations,
            initial_state=state,
        )
        state = framework.last_state
        report["final_solution"] = state["final_solution"]

    if run_realworld_benchmark:
        console.print(
            f"\n[bold magenta]Running Real-World MedMCQA Pharmacology Ground-Truth Benchmark (N={realworld_n})...[/bold magenta]"
        )
        report["realworld_benchmark"] = evaluate_realworld_pharmacology_benchmark(
            framework=framework,
            num_questions=realworld_n,
        )

    report["langsmith"] = framework.telemetry.sync_to_langsmith(
        run_id=framework.last_run_id,
        state=state,
        param_rows=framework.param_registry.get_parameter_table(),
        report=report,
        share_public=True,
    )

    if save_plot:
        CortexVisualizer.plot_cortex_diagnostics(
            state=state,
            param_rows=framework.param_registry.get_parameter_table(),
            title=f"{N}-Domain Jacobian-RSA Brain Architecture (DMN -> SN -> CEN)",
            save_path=save_plot,
        )
        console.print(f"[bold cyan]Saved diagnostic plot to:[/bold cyan] {save_plot}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"[bold cyan]Saved structured JSON report to:[/bold cyan] {output_json}")

    return report

def main():
    parser = argparse.ArgumentParser(
        description="Ask any research question to the Trainable Jacobian-RSA Brain Architecture."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=(
            "Design a precision allosteric covalent-reversible inhibitor and closed-loop PK/PD dosing regimen "
            "that selectively targets mutant KRAS-G12D signaling, prevents compensatory MAPK/PI3K feedback "
            "reactivation, and maintains hepatic/cardiac ADMET safety margins."
        ),
        help="The therapeutic research question to investigate.",
    )
    parser.add_argument(
        "--roster",
        "-r",
        type=str,
        default="default",
        help="Domain roster: 'default' (4 therapeutic sub-experts), 'social' (social sciences YAML), 'auto' (LLM-proposed 4 pieces), or path to YAML/JSON.",
    )
    parser.add_argument(
        "--num-auto-domains",
        type=int,
        default=4,
        help="Number of segregated sub-expert pieces when --roster auto is used.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Optimization epochs for domain weights.")
    parser.add_argument("--test-networks", action="store_true", help="Run DMN->SN->CEN network validity & capacity tests.")
    parser.add_argument("--ablate", action="store_true", help="Run controlled ablation study.")
    parser.add_argument("--test-human-rsa", action="store_true", help="Run SimLex-999 & STS-B human RSA alignment tests.")
    parser.add_argument("--dream-rsi", action="store_true", help="Run the Dream-RSI Wake-Dream-Improve loop.")
    parser.add_argument("--dream-generations", type=int, default=1, help="Number of Dream-RSI generations.")
    parser.add_argument("--realworld-benchmark", action="store_true", help="Run real-world MedMCQA Pharmacology ground-truth benchmark.")
    parser.add_argument("--realworld-n", type=int, default=12, help="Number of MedMCQA Pharmacology questions to evaluate.")
    parser.add_argument("--output-json", type=str, default=None, help="Path to write JSON evaluation report.")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save diagnostic figure.")

    args = parser.parse_args()
    run_scientific_inquiry(
        question=args.question,
        roster_mode=args.roster,
        num_auto_domains=args.num_auto_domains,
        epochs=args.epochs,
        run_network_tests=args.test_networks,
        run_ablations=args.ablate,
        run_human_rsa=args.test_human_rsa,
        run_dream_rsi=args.dream_rsi,
        dream_generations=args.dream_generations,
        run_realworld_benchmark=args.realworld_benchmark,
        realworld_n=args.realworld_n,
        output_json=args.output_json,
        save_plot=args.save_plot,
    )

if __name__ == "__main__":
    main()

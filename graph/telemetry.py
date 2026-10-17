import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

class CortexTelemetry:

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console(force_terminal=True, legacy_windows=False)

    def render_terminal_dashboard(
        self,
        state: Dict[str, Any],
        param_rows: List[Dict[str, float]],
    ) -> None:
        telemetry_map = state.get("domain_telemetry", {})
        sensitivities = state.get("jacobian_sensitivities", {})
        wm_domains = set(state.get("working_memory_domains", []))
        eval_scores = state.get("evaluation_scores", {})

        param_table = Table(
            title=f"[bold cyan]{len(param_rows)} Specialist Deep-Agents & Attached Trainable Parameters (w_1 ... w_{len(param_rows)})[/bold cyan]",
            border_style="cyan",
        )
        param_table.add_column("Param", justify="center", style="bold white")
        param_table.add_column("Domain Agent", style="bold yellow")
        param_table.add_column("Groq Key", justify="center", style="dim cyan")
        param_table.add_column("Time(s)", justify="right", style="dim")
        param_table.add_column("Value (w_i)", justify="right", style="green")
        param_table.add_column("Gate (alpha_i)", justify="right", style="magenta")
        param_table.add_column("Jacobian ||dy/dw_i||", justify="right", style="red")
        param_table.add_column("CEN Working Mem", justify="center", style="bold green")
        param_table.add_column("LangSmith Eval", justify="right", style="blue")

        for row in param_rows:
            d_name = row["domain"]
            t_info = telemetry_map.get(d_name, {})
            in_wm = "ACTIVE [CEN]" if d_name in wm_domains else "background"
            eval_val = eval_scores.get(f"eval/{d_name}", 0.0)
            param_table.add_row(
                row["param_id"],
                d_name,
                str(t_info.get("key_id", "KEY")),
                f"{t_info.get('latency_s', 0.0):.2f}s",
                f"{row['value']:.4f}",
                f"{row['alpha_gate']*100:.1f}%",
                f"{sensitivities.get(d_name, 0.0):.4f}",
                in_wm,
                f"{eval_val:.3f}",
            )
        self.console.print(param_table)

        matches = state.get("salient_beautiful_matches", [])
        if matches:
            sn_table = Table(
                title="[bold magenta]Salience Network (SN) Toggling Switch: Top 'Useful / Beautiful' Cross-Domain Matches[/bold magenta]",
                border_style="magenta",
            )
            sn_table.add_column("Rank", justify="center", style="bold white")
            sn_table.add_column("Domain Pair (Isomorphic Bridge)", style="bold yellow")
            sn_table.add_column("DMN Cosine Sim", justify="right", style="cyan")
            sn_table.add_column("Epistemic Surprise", justify="right", style="green")
            sn_table.add_column("Jacobian G_ij", justify="right", style="red")
            sn_table.add_column("Salience Beauty Score", justify="right", style="bold magenta")

            for idx, m in enumerate(matches, 1):
                sn_table.add_row(
                    f"#{idx}",
                    f"{m['domain_a']} <--> {m['domain_b']}",
                    f"{m['cosine_similarity']:.4f}",
                    f"{m['epistemic_surprise']:.4f}",
                    f"{m['jacobian_co_sensitivity']:.5f}",
                    f"{m['salience_beauty_score']:.5f}",
                )
            self.console.print(sn_table)

        rdm = state.get("rdm_matrix")
        if rdm is not None:
            rdm_np = rdm.detach().cpu().numpy() if isinstance(rdm, torch.Tensor) else np.array(rdm)
            names = [r["domain"] for r in param_rows]
            rdm_table = Table(
                title=f"[bold blue]Default Mode Network (DMN): {len(names)}x{len(names)} External RSA Dissimilarity Matrix (RDM)[/bold blue]",
                border_style="blue",
            )
            rdm_table.add_column("Domain", style="bold yellow")
            for n in names:
                rdm_table.add_column(n[:5], justify="center")
            for i, n in enumerate(names):
                vals = [f"{rdm_np[i, j]:.2f}" for j in range(len(names))]
                rdm_table.add_row(n[:14], *vals)
            self.console.print(rdm_table)

    def sync_to_langsmith(
        self,
        run_id: Optional[str],
        state: Dict[str, Any],
        param_rows: List[Dict[str, float]],
        report: Optional[Dict[str, Any]] = None,
        share_public: bool = True,
    ) -> Dict[str, Any]:
        ls_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
        project = (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "Ael-Agents-Cortex").strip()
        if not ls_key or "YOUR_" in ls_key or not run_id:
            return {"enabled": False, "project": project, "run_id": run_id}

        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers
            from langsmith import Client

            wait_for_all_tracers()
            client = Client(api_key=ls_key)

            session_id = None
            for _ in range(8):
                try:
                    run_obj = client.read_run(run_id)
                    session_id = getattr(run_obj, "session_id", None)
                    break
                except Exception:
                    time.sleep(1.0)

            feedback_scores: Dict[str, float] = {}
            eval_scores = state.get("evaluation_scores", {})
            for k in ("eval/overall_quality", "eval/salience_beauty_score", "eval/cognitive_diversity_pr"):
                if k in eval_scores:
                    feedback_scores[k] = float(eval_scores[k])

            wm_domains = set(state.get("working_memory_domains", []))
            for row in param_rows:
                if row["domain"] in wm_domains:
                    feedback_scores[f"gate/{row['domain']}"] = round(float(row["alpha_gate"]), 4)

            if report:
                if "mct" in report:
                    mc = report["mct"].get("rdm_structure_mc_test", {})
                    if "mc_p_value" in mc:
                        feedback_scores["mct/rdm_p_value"] = float(mc["mc_p_value"])
                    if "observed_rdm_dispersion" in mc:
                        feedback_scores["mct/rdm_dispersion"] = float(mc["observed_rdm_dispersion"])

                if "human_rsa" in report:
                    emb_b = report["human_rsa"].get("embedding_benchmarks", {})
                    if "simlex999" in emb_b:
                        feedback_scores["rsa/simlex999_rho"] = float(emb_b["simlex999"]["real_embedding_spearman_rho"])
                    if "sts_benchmark" in emb_b:
                        feedback_scores["rsa/stsb_rho"] = float(emb_b["sts_benchmark"]["real_embedding_spearman_rho"])
                    ma = report["human_rsa"].get("multi_agent_rsa", {}).get("stsb_multi_agent_split", {})
                    if "human_trained_weights_rho" in ma:
                        feedback_scores["rsa/multi_agent_panel_rho"] = float(ma["human_trained_weights_rho"])

                if "dream_rsi" in report:
                    dr = report["dream_rsi"]
                    if "final_scores" in dr and "mean" in dr["final_scores"]:
                        feedback_scores["judge/dream_rsi_mean"] = float(dr["final_scores"]["mean"])
                    if "delta_mean" in dr:
                        feedback_scores["dream_rsi/delta_mean"] = float(dr["delta_mean"])

                if "ablations" in report and isinstance(report["ablations"], dict):
                    full_row = next((r for r in report["ablations"].get("rows", []) if r.get("variant") == "full" and r.get("ok")), None)
                    if full_row and "scores" in full_row:
                        feedback_scores["judge/full_mean"] = float(full_row["scores"]["mean"])

                if "realworld_benchmark" in report and isinstance(report["realworld_benchmark"], dict):
                    rw = report["realworld_benchmark"]
                    if "single_llm_accuracy_pct" in rw:
                        feedback_scores["medmcqa/single_llm_acc_pct"] = float(rw["single_llm_accuracy_pct"])
                    if "ael_wake_accuracy_pct" in rw:
                        feedback_scores["medmcqa/ael_wake_acc_pct"] = float(rw["ael_wake_accuracy_pct"])
                    if "ael_dream_rsi_accuracy_pct" in rw:
                        feedback_scores["medmcqa/ael_dream_rsi_acc_pct"] = float(rw["ael_dream_rsi_accuracy_pct"])
                    if "gain_dream_rsi_pct" in rw:
                        feedback_scores["medmcqa/gain_pct"] = float(rw["gain_dream_rsi_pct"])

            import warnings
            pushed_count = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for f_key, f_val in feedback_scores.items():
                    try:
                        client.create_feedback(
                            run_id=run_id,
                            key=f_key,
                            score=f_val,
                            comment=f"Ael-Agents automated cortical telemetry ({project})",
                        )
                        pushed_count += 1
                    except Exception:
                        pass

            public_url: Optional[str] = None
            if share_public:
                try:
                    public_url = client.share_run(run_id)
                except Exception:
                    try:
                        public_url = client.read_run_shared_link(run_id)
                    except Exception:
                        public_url = None

            self.console.print(Panel(
                f"[bold green]LangSmith Online Dashboard Synced![/bold green]\n"
                f"  * [cyan]Project:[/cyan] {project}\n"
                f"  * [cyan]Root Run ID:[/cyan] {run_id}\n"
                f"  * [cyan]Feedback Badges Pushed:[/cyan] {pushed_count} metrics ({', '.join(list(feedback_scores.keys())[:6])}...)\n"
                f"  * [bold yellow]Public Shareable Trace URL:[/bold yellow] {public_url or 'Available in LangSmith project dashboard'}",
                title="[bold magenta]LangSmith Cloud Showcase[/bold magenta]",
                border_style="green",
            ))

            return {
                "enabled": True,
                "project": project,
                "run_id": run_id,
                "public_url": public_url,
                "feedback_metrics_pushed": feedback_scores,
            }
        except Exception as e:
            return {
                "enabled": False,
                "project": project,
                "run_id": run_id,
                "error": f"{type(e).__name__}: {e}",
            }

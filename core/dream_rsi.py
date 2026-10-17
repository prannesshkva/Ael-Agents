from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

CRITERIA = ["accuracy", "rigor", "integration", "falsifiability", "responsiveness"]
from analysis.monte_carlo import mct
from core.domain_registry import DomainSpec, domain_prior_distance
from core.framework import JacobianRSACortexFramework
from core.llm_utils import extract_json, llm_call

console = Console(force_terminal=True, legacy_windows=False)

class ToolRegistry:
    _tools: Dict[str, Dict[str, Callable]] = {}
    _sources: Dict[str, str] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        domain: str,
        name: str,
        func: Callable,
        description: str = "",
        source: str = "",
    ) -> None:
        if description:
            func.__doc__ = description
        elif not getattr(func, "__doc__", None):
            func.__doc__ = f"Quantitative helper tool {name} for {domain}."
        cls._tools.setdefault(domain, {})[name] = func
        cls._sources[name] = source
        cls._descriptions[name] = func.__doc__

    @classmethod
    def get_domain_tools(cls, domain: str) -> List[Callable]:
        return list(cls._tools.get(domain, {}).values())

    @classmethod
    def list_tools(cls) -> Dict[str, List[str]]:
        return {dom: list(fns.keys()) for dom, fns in cls._tools.items()}

def safe_str_strip(s: str = "") -> str:
    return s.strip() if isinstance(s, str) else str(s)

ToolRegistry.register(
    "global",
    "safe_str_strip",
    safe_str_strip,
    description="Strips outer whitespace from a string input.",
    source="def safe_str_strip(s: str = '') -> str: return s.strip()",
)

TOOL_SYNTH_SYSTEM = (
    "You are the Axiom-RSI Tool Synthesizer. Given a research question, a cross-domain bridge "
    "(Domain A <-> Domain B), and the weakest evaluation criterion, write ONE pure-Python quantitative "
    "helper function that computes a concrete dimensionless ratio, stability bound, or rate equation "
    "connecting the two domains.\n"
    "Rules:\n"
    "1. Function signature MUST have default float arguments and return a string summary, e.g.:\n"
    "   def compute_bridge_metric(param_a: float = 1.0, param_b: float = 0.5) -> str:\n"
    "2. Use ONLY standard `math` operations (math.exp, math.log, math.sqrt, etc.). No external imports.\n"
    "3. Return ONLY JSON: "
    '{"tool_name": "compute_...", "description": "one sentence description", "code": "def compute_..."}'
)

DREAM_REPLAY_SYSTEM = (
    "You are the Default Mode Network (DMN) Offline Dream Replay Engine. "
    "Given a research question, an unexplored high-surprise cross-domain bridge (Domain A <-> Domain B), "
    "their Wake-phase hypotheses, and the weakest evaluation criterion, formulate:\n"
    "1. `counterfactual_question`: a sharp sub-question probing the exact mathematical isomorphism between "
    "Domain A and Domain B that addresses the weak criterion.\n"
    "2. `consolidated_insight`: a 2-3 sentence rigorous mechanistic bridge equation/principle combining "
    "Domain A and Domain B.\n"
    'Return ONLY JSON: {"counterfactual_question": "...", "consolidated_insight": "..."}'
)

ROSTER_EVOLVE_SYSTEM = (
    "You are the Recursive Self-Improvement (RSI) Cortical Architect. "
    "A specialist agent underperformed in the previous Wake cycle. Refine its domain focus so it "
    "contributes directly to the research question and fixes the weakest evaluation criterion, while "
    "incorporating the offline Dream Replay insight.\n"
    'Return ONLY JSON: {"domain_title": "...", "focus": "..."}'
)

def synthesize_and_verify_tool(
    key_queue,
    question: str,
    domain_a: str,
    domain_b: str,
    weak_criterion: str,
) -> Optional[Dict[str, Any]]:
    user = (
        f"RESEARCH QUESTION:\n{question}\n\n"
        f"CROSS-DOMAIN BRIDGE: {domain_a} <-> {domain_b}\n"
        f"WEAKEST CRITERION TO IMPROVE: {weak_criterion}\n"
    )
    try:
        raw = llm_call(key_queue, TOOL_SYNTH_SYSTEM, user, name="rsi_tool_synth", temperature=0.2, max_tokens=1536)
        data = extract_json(raw)
        tool_name = str(data["tool_name"]).strip()
        description = str(data.get("description", f"Quantitative bridge calculator for {domain_a} and {domain_b}.")).strip()
        code = str(data["code"]).strip()

        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "sum": sum,
                "len": len,
                "range": range,
                "float": float,
                "int": int,
                "str": str,
                "pow": pow,
            },
            "math": math,
        }
        local_ns: Dict[str, Any] = {}
        exec(code, safe_globals, local_ns)
        fn = local_ns.get(tool_name) or next((v for v in local_ns.values() if callable(v)), None)
        if fn is None:
            return None

        test_output = str(fn())
        if not test_output.strip():
            return None

        ToolRegistry.register(domain_a, tool_name, fn, description=description, source=code)
        ToolRegistry.register(domain_b, tool_name, fn, description=description, source=code)

        return {
            "tool_name": tool_name,
            "domains": [domain_a, domain_b],
            "description": description,
            "code": code,
            "sandbox_output": test_output[:240],
        }
    except Exception:
        return None

def _clone_state(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_state(x) for x in obj]
    return copy.deepcopy(obj)

class DreamRSILoop:
    def __init__(
        self,
        framework: JacobianRSACortexFramework,
        judge: Optional[Any] = None,
    ):
        self.fw = framework
        if judge is None:
            from analysis.judge import Judge
            self.judge = Judge(framework.key_queue)
        else:
            self.judge = judge

    def _select_dream_bridge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        bridges = state.get("dmn_spontaneous_bridges", [])
        wm = set(state.get("working_memory_domains", []))
        for b in bridges:
            if b["domain_a"] not in wm or b["domain_b"] not in wm:
                return b
        return bridges[0] if bridges else {"domain_a": self.fw.domain_names[0], "domain_b": self.fw.domain_names[-1], "dmn_imagination_score": 0.5}

    def dream_replay(
        self,
        question: str,
        state: Dict[str, Any],
        weak_criterion: str,
        tool_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        bridge = self._select_dream_bridge(state)
        dom_a, dom_b = bridge["domain_a"], bridge["domain_b"]
        hyp_a = state["domain_solutions"].get(dom_a, "")[:900]
        hyp_b = state["domain_solutions"].get(dom_b, "")[:900]
        tool_note = (
            f"\nSYNTHESIZED TOOL OUTPUT ({tool_result['tool_name']}): {tool_result['sandbox_output']}\n"
            if tool_result else ""
        )
        user = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"UNEXPLORED DMN BRIDGE: {dom_a} <-> {dom_b} "
            f"(epistemic surprise={bridge.get('epistemic_surprise', 0.0):.3f}, "
            f"DMN score={bridge.get('dmn_imagination_score', 0.0):.3f})\n"
            f"WEAKEST CRITERION IN WAKE PHASE: {weak_criterion}\n"
            f"{tool_note}\n"
            f"--- {dom_a} Wake Hypothesis ---\n{hyp_a}\n\n"
            f"--- {dom_b} Wake Hypothesis ---\n{hyp_b}\n"
        )
        raw = llm_call(
            self.fw.key_queue,
            DREAM_REPLAY_SYSTEM,
            user,
            name="dmn_dream_replay",
            temperature=0.3,
            max_tokens=1536,
        )
        try:
            parsed = extract_json(raw)
            cf_q = str(parsed.get("counterfactual_question", "")).strip()
            insight = str(parsed.get("consolidated_insight", "")).strip()
        except Exception:
            cf_q = f"How do {dom_a} and {dom_b} jointly constrain {weak_criterion}?"
            insight = raw.strip()[:600]
        dream_vec = self.fw.encoder.encode(f"{cf_q}\n{insight}")

        return {
            "bridge": f"{dom_a} <-> {dom_b}",
            "domain_a": dom_a,
            "domain_b": dom_b,
            "counterfactual_question": cf_q,
            "consolidated_insight": insight,
            "dream_embedding": dream_vec,
        }

    def evolve_roster_and_weights(
        self,
        question: str,
        state: Dict[str, Any],
        usefulness: Dict[str, float],
        stability: Dict[str, float],
        dream_out: Dict[str, Any],
        weak_criterion: str,
    ) -> Dict[str, Any]:
        names = self.fw.domain_names
        combined_rank = {
            n: 0.65 * (usefulness.get(n, 5.0) / 10.0) + 0.35 * (stability.get(n, 50.0) / 100.0)
            for n in names
        }
        weakest_domain = min(names, key=lambda n: combined_rank[n])
        weak_idx = names.index(weakest_domain)
        old_spec = self.fw.roster[weak_idx]

        user = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"UNDERPERFORMING DOMAIN: {weakest_domain} ({old_spec.domain_title})\n"
            f"CURRENT FOCUS: {old_spec.description}\n"
            f"WEAKEST CRITERION: {weak_criterion}\n"
            f"OFFLINE DREAM INSIGHT ({dream_out['bridge']}): {dream_out['consolidated_insight']}\n"
        )
        raw = llm_call(
            self.fw.key_queue,
            ROSTER_EVOLVE_SYSTEM,
            user,
            name="rsi_roster_evolve",
            temperature=0.25,
            max_tokens=1536,
        )
        try:
            ev = extract_json(raw)
            new_title = str(ev.get("domain_title", old_spec.domain_title)).strip()
            new_focus = str(ev.get("focus", old_spec.description)).strip()
        except Exception:
            new_title = f"{old_spec.domain_title} & {dream_out['bridge']} Integration"
            new_focus = f"{old_spec.description} Focus on {weak_criterion}."

        new_spec = DomainSpec(
            name=weakest_domain,
            domain_title=new_title,
            focus=f"{new_focus} Incorporate verified bridge insight: {dream_out['consolidated_insight']}",
        )
        self.fw.roster[weak_idx] = new_spec
        agent = self.fw.agents[weak_idx]
        agent.domain_title = new_title
        agent.system_prompt = new_spec.prompt_for(weak_idx + 1)

        for idx, dom_name in enumerate(names):
            dom_tools = ToolRegistry.get_domain_tools(dom_name)
            if dom_tools:
                self.fw.agents[idx].domain_tools = dom_tools

        self.fw.prior_dist = domain_prior_distance(self.fw.roster, self.fw.encoder)
        self.fw.graph_builder.prior_dist = self.fw.prior_dist

        with torch.no_grad():
            for idx, n in enumerate(names):
                u_norm = (usefulness.get(n, 5.0) - 5.0) * 0.12
                dream_bonus = 0.22 if n in (dream_out["domain_a"], dream_out["domain_b"], weakest_domain) else 0.0
                self.fw.param_registry.weights[idx] = torch.clamp(
                    self.fw.param_registry.weights[idx] + u_norm + dream_bonus,
                    min=0.15,
                    max=2.50,
                )

        return {
            "evolved_domain": weakest_domain,
            "old_title": old_spec.domain_title,
            "new_title": new_title,
            "new_focus": new_focus,
        }

    def _replay_agents_and_resynthesize(
        self,
        question: str,
        state: Dict[str, Any],
        target_domains: List[str],
        dream_out: Dict[str, Any],
        tool_out: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        new_state = _clone_state(state)
        agents_by_name = {a.name: a for a in self.fw.agents}

        tool_context = (
            f"\n[Verified RSI Tool {tool_out['tool_name']}: {tool_out['sandbox_output']}]"
            if tool_out else ""
        )
        augmented_task = (
            f"{question}\n\n"
            f"[Offline DMN Dream Consolidation ({dream_out['bridge']})]: "
            f"{dream_out['consolidated_insight']}{tool_context}"
        )

        for d_name in target_domains:
            if d_name in agents_by_name:
                res = agents_by_name[d_name].execute_parallel(
                    task=augmented_task,
                    key_queue=self.fw.key_queue,
                    param_registry=self.fw.param_registry,
                )
                if not res.get("failed"):
                    new_state["domain_solutions"][d_name] = res["hypothesis"]
                    new_state["domain_embeddings"][d_name] = res["raw_embedding"]
                    new_state["domain_weighted_tensors"][d_name] = res["weighted_tensor"]

        sim_matrix, rdm_matrix, dmn_bridges, dmn_metrics = self.fw.dmn_engine(
            domain_embeddings=new_state["domain_embeddings"],
            domain_names=self.fw.domain_names,
            prior_dist=self.fw.prior_dist,
        )
        new_state["rsa_similarity_matrix"] = sim_matrix
        new_state["rdm_matrix"] = rdm_matrix
        new_state["dmn_spontaneous_bridges"] = dmn_bridges
        new_state["metadata"].update(dmn_metrics)

        self.fw._last_state = new_state
        self.fw.optimize_parameters(epochs=15)
        sn_out = self.fw.refresh_salience()
        new_state = self.fw.last_state

        final_solution = self.fw.cen_network.synthesize(
            task=augmented_task,
            domain_solutions=new_state["domain_solutions"],
            gating_weights=sn_out["gating_weights"],
            jacobian_sensitivities=sn_out["jacobian_sensitivities"],
            salient_matches=sn_out["salient_beautiful_matches"],
            working_memory_domains=sn_out["working_memory_domains"],
            key_queue=self.fw.key_queue,
            name="CEN_DreamRSI",
        )
        new_state["final_solution"] = final_solution
        new_state["metadata"]["cen_target_embedding"] = self.fw.encoder.encode(final_solution)
        return new_state

    def dream_consolidate_and_evolve(
        self,
        question: str,
        state: Dict[str, Any],
        weak_criterion: str,
        stability: Dict[str, float],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
        names = self.fw.domain_names
        salience = state.get("domain_salience", {})
        max_sal = max(salience.values()) if salience else 1.0
        combined_rank = {
            n: 0.65 * (salience.get(n, 0.1) / max(max_sal, 1e-8)) + 0.35 * (stability.get(n, 50.0) / 100.0)
            for n in names
        }
        weakest_domain = min(names, key=lambda n: combined_rank[n])
        weak_idx = names.index(weakest_domain)
        old_spec = self.fw.roster[weak_idx]

        bridge = self._select_dream_bridge(state)
        dom_a, dom_b = bridge["domain_a"], bridge["domain_b"]
        hyp_a = state["domain_solutions"].get(dom_a, "")[:600]
        hyp_b = state["domain_solutions"].get(dom_b, "")[:600]

        fused_system = (
            "You are the Default Mode Network (DMN) Offline Dream & Recursive Self-Improvement (RSI) Engine "
            "for Computational Drug Discovery & Therapeutics. Given the research question, an unexplored "
            "cross-piece bridge (Domain A <-> Domain B), and the underperforming piece, return ONE JSON object with:\n"
            '1. "counterfactual_question": sharp sub-question coupling Domain A and Domain B to fix the weak criterion.\n'
            '2. "consolidated_insight": 2-3 sentence quantitative pharmacological/biophysical mechanism.\n'
            '3. "domain_title": refined title for the underperforming domain.\n'
            '4. "focus": refined quantitative focus for the underperforming domain.\n'
            '5. "tool_name": name of a pure-Python function (e.g. "compute_pkpd_bound").\n'
            '6. "code": "def compute_...(a: float = 1.0, b: float = 0.5) -> str: ..." using only standard math.'
        )
        user = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"UNEXPLORED DMN BRIDGE: {dom_a} <-> {dom_b}\n"
            f"UNDERPERFORMING PIECE: {weakest_domain} ({old_spec.domain_title})\n"
            f"WEAKEST CRITERION TO IMPROVE: {weak_criterion}\n"
            f"--- {dom_a} ---\n{hyp_a}\n"
            f"--- {dom_b} ---\n{hyp_b}\n"
        )
        raw = llm_call(
            self.fw.key_queue,
            fused_system,
            user,
            name="dmn_fused_dream_rsi",
            temperature=0.25,
            max_tokens=1200,
        )
        try:
            parsed = extract_json(raw)
        except Exception:
            parsed = {}

        cf_q = str(parsed.get("counterfactual_question") or f"How do {dom_a} and {dom_b} jointly constrain {weak_criterion}?").strip()
        insight = str(parsed.get("consolidated_insight") or raw.strip()[:500]).strip()
        new_title = str(parsed.get("domain_title") or old_spec.domain_title).strip()
        new_focus = str(parsed.get("focus") or old_spec.description).strip()

        tool_out: Optional[Dict[str, Any]] = None
        code = str(parsed.get("code") or "").strip()
        tool_name = str(parsed.get("tool_name") or "compute_therapeutic_metric").strip()
        if code and "def " in code:
            try:
                safe_globals: Dict[str, Any] = {
                    "__builtins__": {
                        "abs": abs, "min": min, "max": max, "round": round,
                        "sum": sum, "len": len, "range": range, "float": float,
                        "int": int, "str": str, "pow": pow,
                    },
                    "math": math,
                }
                local_ns: Dict[str, Any] = {}
                exec(code, safe_globals, local_ns)
                fn = local_ns.get(tool_name) or next((v for v in local_ns.values() if callable(v)), None)
                if fn is not None:
                    test_output = str(fn())
                    if test_output.strip():
                        ToolRegistry.register(dom_a, tool_name, fn, description=cf_q, source=code)
                        ToolRegistry.register(dom_b, tool_name, fn, description=cf_q, source=code)
                        tool_out = {
                            "tool_name": tool_name,
                            "domains": [dom_a, dom_b],
                            "description": cf_q,
                            "code": code,
                            "sandbox_output": test_output[:240],
                        }
            except Exception:
                tool_out = None

        dream_vec = self.fw.encoder.encode(f"{cf_q}\n{insight}")
        dream_out = {
            "bridge": f"{dom_a} <-> {dom_b}",
            "domain_a": dom_a,
            "domain_b": dom_b,
            "counterfactual_question": cf_q,
            "consolidated_insight": insight,
            "dream_embedding": dream_vec,
        }

        new_spec = DomainSpec(
            name=weakest_domain,
            domain_title=new_title,
            focus=f"{new_focus} Incorporate verified bridge insight: {insight}",
        )
        self.fw.roster[weak_idx] = new_spec
        agent = self.fw.agents[weak_idx]
        agent.domain_title = new_title
        agent.system_prompt = new_spec.prompt_for(weak_idx + 1)

        for idx, dom_name in enumerate(names):
            dom_tools = ToolRegistry.get_domain_tools(dom_name)
            if dom_tools:
                self.fw.agents[idx].domain_tools = dom_tools

        self.fw.prior_dist = domain_prior_distance(self.fw.roster, self.fw.encoder)
        self.fw.graph_builder.prior_dist = self.fw.prior_dist

        with torch.no_grad():
            for idx, n in enumerate(names):
                s_bonus = 0.20 if n in (dom_a, dom_b, weakest_domain) else 0.0
                self.fw.param_registry.weights[idx] = torch.clamp(
                    self.fw.param_registry.weights[idx] + s_bonus,
                    min=0.15,
                    max=2.50,
                )

        evolve_out = {
            "evolved_domain": weakest_domain,
            "old_title": old_spec.domain_title,
            "new_title": new_title,
            "new_focus": new_focus,
        }
        return dream_out, tool_out, evolve_out

    def run(
        self,
        question: str,
        generations: int = 1,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        console.print(Panel.fit(
            f"[bold magenta]DREAM-RSI: WAKE -> DMN OFFLINE DREAM -> RECURSIVE SELF-IMPROVEMENT[/bold magenta]\n"
            f"[dim]Generations: {generations} | ToolRegistry + DMN Replay + Roster Evolution + Paired Judge Gate[/dim]",
            border_style="magenta",
        ))

        if initial_state is None:
            state = self.fw.run(question)
            self.fw.optimize_parameters(epochs=15)
            state = self.fw.last_state
        else:
            state = initial_state
            self.fw._last_state = state

        base_mct = mct(self.fw, state)
        best_state = _clone_state(state)
        best_weights = self.fw.param_registry.weights.detach().clone()
        best_roster = copy.deepcopy(self.fw.roster)

        best_scores: Optional[Dict[str, float]] = None
        history: List[Dict[str, Any]] = []

        for gen in range(1, generations + 1):
            cur_mct = mct(self.fw, best_state)
            stability = cur_mct["working_memory_stability"]
            weak_criterion = min(CRITERIA, key=lambda c: best_scores[c]) if best_scores else "rigor"

            dream_out, tool_out, evolve_out = self.dream_consolidate_and_evolve(
                question=question,
                state=best_state,
                weak_criterion=weak_criterion,
                stability=stability,
            )

            target_domains = [evolve_out["evolved_domain"]]
            cand_state = self._replay_agents_and_resynthesize(
                question=question,
                state=best_state,
                target_domains=target_domains,
                dream_out=dream_out,
                tool_out=tool_out,
            )
            cand_mct = mct(self.fw, cand_state)

            if best_scores is None:
                base_scores, cand_scores = self.judge.score_pair(
                    question, best_state["final_solution"], cand_state["final_solution"]
                )
                best_scores = dict(base_scores)
                weak_criterion = min(CRITERIA, key=lambda c: best_scores[c])
                history.append({
                    "generation": 0,
                    "phase": "wake_baseline",
                    "accepted": True,
                    "scores": base_scores,
                    "working_memory": list(best_state["working_memory_domains"]),
                    "rdm_mc_p": base_mct["rdm_structure_mc_test"]["mc_p_value"],
                })
            else:
                cand_scores = self.judge.score_answer(question, cand_state["final_solution"])

            improved_mean = cand_scores["mean"] > best_scores["mean"]
            improved_weak = (
                cand_scores["mean"] >= best_scores["mean"]
                and cand_scores[weak_criterion] >= best_scores[weak_criterion]
            )
            accepted = bool(improved_mean or improved_weak)

            if accepted:
                best_state = _clone_state(cand_state)
                best_scores = dict(cand_scores)
                best_weights = self.fw.param_registry.weights.detach().clone()
                best_roster = copy.deepcopy(self.fw.roster)
                self.fw._last_state = best_state
            else:
                with torch.no_grad():
                    self.fw.param_registry.weights.copy_(best_weights)
                self.fw.roster = copy.deepcopy(best_roster)
                for idx, spec in enumerate(self.fw.roster):
                    self.fw.agents[idx].domain_title = spec.domain_title
                    self.fw.agents[idx].system_prompt = spec.prompt_for(idx + 1)
                self.fw.prior_dist = domain_prior_distance(self.fw.roster, self.fw.encoder)
                self.fw.graph_builder.prior_dist = self.fw.prior_dist
                self.fw._last_state = best_state

            history.append({
                "generation": gen,
                "phase": "dream_rsi",
                "weak_criterion_targeted": weak_criterion,
                "dream_bridge": dream_out["bridge"],
                "counterfactual_question": dream_out["counterfactual_question"],
                "consolidated_insight": dream_out["consolidated_insight"],
                "synthesized_tool": tool_out["tool_name"] if tool_out else None,
                "sandbox_output": tool_out["sandbox_output"] if tool_out else None,
                "evolved_domain": evolve_out["evolved_domain"],
                "new_domain_title": evolve_out["new_title"],
                "candidate_scores": cand_scores,
                "accepted": accepted,
                "active_scores": best_scores,
                "working_memory": list(best_state["working_memory_domains"]),
                "rdm_mc_p": cand_mct["rdm_structure_mc_test"]["mc_p_value"],
            })

        summary_table = Table(title="Dream-RSI Wake-Dream-Improve Trajectory (Judge Acceptance Gate)")
        summary_table.add_column("Gen", justify="right", style="cyan")
        summary_table.add_column("Dream Bridge / Evolved Domain", style="yellow")
        summary_table.add_column("Synthesized Tool", style="magenta")
        summary_table.add_column("Cand Mean", justify="right")
        summary_table.add_column("Acc / Rigor / Integ / Falsif / Resp", justify="right")
        summary_table.add_column("Gate Status", justify="center", style="bold green")

        for h in history:
            if h["generation"] == 0:
                sc = h["scores"]
                summary_table.add_row(
                    "0 (Wake)",
                    f"Initial WM: {', '.join(h['working_memory'][:3])}",
                    "-",
                    f"{sc['mean']:.2f}",
                    f"{sc['accuracy']:.0f}/{sc['rigor']:.0f}/{sc['integration']:.0f}/{sc['falsifiability']:.0f}/{sc['responsiveness']:.0f}",
                    "BASELINE",
                )
            else:
                sc = h["candidate_scores"]
                gate_str = "[bold green]COMMITTED[/bold green]" if h["accepted"] else "[bold red]ROLLED BACK[/bold red]"
                summary_table.add_row(
                    str(h["generation"]),
                    f"{h['dream_bridge']} -> {h['evolved_domain']}",
                    str(h["synthesized_tool"] or "none"),
                    f"{sc['mean']:.2f}",
                    f"{sc['accuracy']:.0f}/{sc['rigor']:.0f}/{sc['integration']:.0f}/{sc['falsifiability']:.0f}/{sc['responsiveness']:.0f}",
                    gate_str,
                )
        console.print(summary_table)

        result_payload = {
            "baseline_scores": base_scores,
            "final_scores": best_scores,
            "delta_mean": round(best_scores["mean"] - base_scores["mean"], 3),
            "registered_tools": ToolRegistry.list_tools(),
            "generations": history,
            "final_solution": best_state["final_solution"],
        }
        if initial_state is None and self.fw.last_run_id:
            result_payload["langsmith"] = self.fw.telemetry.sync_to_langsmith(
                run_id=self.fw.last_run_id,
                state=best_state,
                param_rows=self.fw.param_registry.get_parameter_table(),
                report={"dream_rsi": result_payload, "mct": mct(self.fw, best_state)},
                share_public=True,
            )
        return result_payload

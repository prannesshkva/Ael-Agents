from typing import Any, Dict, List, Optional

from core.api_queue import APIKeyQueue
from core.llm_utils import llm_call

class CentralExecutiveNetwork:

    CEN_SYSTEM_PROMPT = (
        "You are the CENTRAL EXECUTIVE NETWORK (CEN) of a multi-domain research panel. "
        "Several domain experts have each given a hypothesis about the research question. "
        "Your job is to integrate ONLY the supplied working-memory contributions into one rigorous, "
        "internally consistent answer.\n"
        "Rules: maintain formal/logical consistency; make assumptions explicit; state where the "
        "domains disagree and how you resolve it; separate established knowledge from speculation; "
        "do not invent citations or data; finish with a concrete, falsifiable prediction or test. "
        "Keep the answer under 350 words."
    )

    DIRECT_SYSTEM_PROMPT = (
        "You are an expert research scientist. Answer the research question rigorously and "
        "concisely (under 350 words): make assumptions explicit, separate established knowledge "
        "from speculation, do not invent citations or data, and finish with a falsifiable test."
    )

    def synthesize(
        self,
        task: str,
        domain_solutions: Dict[str, str],
        gating_weights: Dict[str, float],
        jacobian_sensitivities: Dict[str, float],
        salient_matches: List[Dict[str, Any]],
        working_memory_domains: List[str],
        key_queue: APIKeyQueue,
        name: str = "CEN_Executive_Synthesizer",
    ) -> str:
        wm_blocks = []
        per_domain_char_cap = max(550, 4800 // max(1, len(working_memory_domains)))
        for d_name in working_memory_domains:
            gate_pct = gating_weights.get(d_name, 0.0) * 100.0
            jac_s = jacobian_sensitivities.get(d_name, 0.0)
            wm_blocks.append(
                f"--- WORKING MEMORY SLOT: {d_name} (gate {gate_pct:.1f}%, Jacobian sensitivity {jac_s:.4f}) ---\n"
                f"{domain_solutions.get(d_name, '')[:per_domain_char_cap]}"
            )
        wm_text = "\n\n".join(wm_blocks)

        bridge_text = ""
        bridge_task = ""
        if salient_matches:
            lines = [
                f"Bridge #{k}: {m['domain_a']} <--> {m['domain_b']} "
                f"(embedding similarity {m['cosine_similarity']:.3f}, salience score {m['salience_beauty_score']:.4f})"
                for k, m in enumerate(salient_matches, 1)
            ]
            bridge_text = (
                "CROSS-DOMAIN BRIDGES flagged by the Salience Network (candidate structural analogies "
                "to examine; they may be spurious, so test them rather than assume them):\n"
                + "\n".join(lines) + "\n\n"
            )
            bridge_task = "1. [BRIDGE ASSESSMENT]: is the top bridge a genuine structural correspondence? Why or why not.\n"
        n0 = 2 if bridge_task else 1
        user_prompt = (
            f"RESEARCH QUESTION:\n{task}\n\n{bridge_text}"
            f"WORKING-MEMORY CONTRIBUTIONS:\n{wm_text}\n\n"
            f"Write the integrated answer with these sections:\n{bridge_task}"
            f"{n0}. [INTEGRATED ANSWER]: the unified mechanism/explanation, with key equations if relevant.\n"
            f"{n0 + 1}. [FALSIFIABLE TEST]: a concrete experiment or analysis that could refute it."
        )
        return llm_call(
            key_queue, self.CEN_SYSTEM_PROMPT, user_prompt,
            name=name, temperature=0.25, max_tokens=1536,
        )

    def direct_answer(self, task: str, key_queue: APIKeyQueue) -> str:
        return llm_call(
            key_queue, self.DIRECT_SYSTEM_PROMPT, f"RESEARCH QUESTION:\n{task}",
            name="baseline_direct", temperature=0.25, max_tokens=1536,
        )

CorticalSynthesizer = CentralExecutiveNetwork

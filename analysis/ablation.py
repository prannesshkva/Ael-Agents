import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from analysis.judge import Judge
from core.embeddings import get_encoder
from core.jacobian_lens import SalienceJacobianNetwork
from core.parameter_registry import DomainParameterRegistry
from training.loss_functions import UnifiedCortexLoss
from training.trainer import CortexTrainer

ALL_VARIANTS = [
    "full", "no_dmn", "no_sn", "no_cen", "no_jacobian_scoring", "no_jacobian_loss",
    "untrained_weights", "unweighted_ensemble", "random_working_memory", "single_prompt",
]

class AblationRunner:
    def __init__(self, framework, judge: Judge, reference: Optional[str] = None, seed: int = 0):
        self.fw = framework
        self.judge = judge
        self.reference = reference
        self.seed = seed

    def _sn(self, weights, state, *, use_similarity=True, use_jacobian=True,
            capacity=None, enforce=True, flat_bridges=False) -> Dict[str, Any]:
        base = self.fw.salience_network
        sn = SalienceJacobianNetwork(
            temperature=base.temperature, interaction_gamma=base.interaction_gamma,
            working_memory_capacity=base.working_memory_capacity, top_matches_k=base.top_matches_k,
            use_similarity=use_similarity, use_jacobian=use_jacobian,
        )
        names = self.fw.domain_names
        bridges = state["dmn_spontaneous_bridges"]
        if flat_bridges:
            bridges = [
                {**b, "cosine_similarity": 0.0, "dmn_imagination_score": 1.0} for b in bridges
            ]
        return sn.compute_jacobian_and_salience(
            weights, state["domain_embeddings"], names, state["rsa_similarity_matrix"], bridges,
            create_graph=False, working_memory_capacity=capacity, enforce_match_domains=enforce,
        )

    def _synth(self, question, state, sn_out, wm, matches, tag) -> str:
        return self.fw.cen_network.synthesize(
            task=question,
            domain_solutions=state["domain_solutions"],
            gating_weights=sn_out["gating_weights"],
            jacobian_sensitivities=sn_out["jacobian_sensitivities"],
            salient_matches=matches,
            working_memory_domains=wm,
            key_queue=self.fw.key_queue,
            name=f"CEN_{tag}",
        )

    def _retrain_without_jacobian(self, state) -> torch.Tensor:
        reg = DomainParameterRegistry(domain_names=self.fw.domain_names, init_value=1.0, temperature=0.5)
        trainer = CortexTrainer(
            param_registry=reg, dmn_engine=self.fw.dmn_engine, salience_network=self.fw.salience_network,
            loss_fn=UnifiedCortexLoss(task_loss_weight=1.0, rsa_salience_weight=0.45, jacobian_loss_weight=0.0),
            learning_rate=0.08,
        )
        target = trainer.build_salience_guided_target(state, self.fw.domain_names)
        for _ in range(15):
            trainer.train_step_on_state(state, self.fw.domain_names, target)
        return reg.weights.detach().clone()

    def _build_variant(self, name: str, question: str, state: Dict[str, Any]) -> Dict[str, Any]:
        trained = self.fw.param_registry.weights.detach().clone()
        ones = torch.ones_like(trained)
        names = self.fw.domain_names
        n = len(names)
        full_sn = self._sn(trained, state)
        k = len(full_sn["working_memory_domains"])

        if name == "full":
            wm, m = full_sn["working_memory_domains"], full_sn["salient_beautiful_matches"]
            return {"sn": full_sn, "wm": wm, "matches": m, "text": self._synth(question, state, full_sn, wm, m, name)}
        if name == "no_dmn":
            sn = self._sn(trained, state, use_similarity=False, enforce=False, flat_bridges=True)
            wm = sn["working_memory_domains"]
            return {"sn": sn, "wm": wm, "matches": [], "text": self._synth(question, state, sn, wm, [], name)}
        if name == "no_sn":
            wm = list(names)
            return {"sn": full_sn, "wm": wm, "matches": [], "text": self._synth(question, state, full_sn, wm, [], name)}
        if name == "no_cen":
            wm = full_sn["working_memory_domains"]
            text = "\n\n".join(f"[{d}]\n{state['domain_solutions'][d]}" for d in wm)
            return {"sn": full_sn, "wm": wm, "matches": full_sn["salient_beautiful_matches"], "text": text}
        if name == "no_jacobian_scoring":
            sn = self._sn(trained, state, use_jacobian=False, capacity=k)
            wm, m = sn["working_memory_domains"], sn["salient_beautiful_matches"]
            return {"sn": sn, "wm": wm, "matches": m, "text": self._synth(question, state, sn, wm, m, name)}
        if name == "no_jacobian_loss":
            w = self._retrain_without_jacobian(state)
            sn = self._sn(w, state, capacity=k)
            wm, m = sn["working_memory_domains"], sn["salient_beautiful_matches"]
            return {"sn": sn, "wm": wm, "matches": m, "text": self._synth(question, state, sn, wm, m, name)}
        if name == "untrained_weights":
            sn = self._sn(ones, state, capacity=k)
            wm, m = sn["working_memory_domains"], sn["salient_beautiful_matches"]
            return {"sn": sn, "wm": wm, "matches": m, "text": self._synth(question, state, sn, wm, m, name)}
        if name == "unweighted_ensemble":
            sn = self._sn(ones, state)
            wm = list(names)
            return {"sn": sn, "wm": wm, "matches": [], "text": self._synth(question, state, sn, wm, [], name)}
        if name == "random_working_memory":
            rng = random.Random(self.seed)
            wm = rng.sample(names, min(k, n))
            return {"sn": full_sn, "wm": wm, "matches": [], "text": self._synth(question, state, full_sn, wm, [], name)}
        if name == "single_prompt":
            return {"sn": None, "wm": [], "matches": [],
                    "text": self.fw.cen_network.direct_answer(question, self.fw.key_queue)}
        raise ValueError(f"unknown variant {name}")

    def run(self, question: str, state: Dict[str, Any], variants: Optional[List[str]] = None,
            max_workers: int = 2) -> Dict[str, Any]:
        variants = variants or ALL_VARIANTS
        enc = get_encoder()
        ref_emb = enc.encode(self.reference) if self.reference else None

        def work(v: str) -> Dict[str, Any]:
            try:
                built = self._build_variant(v, question, state)
                scores = self.judge.score_answer(question, built["text"], self.reference)
                row = {
                    "variant": v, "ok": True, "scores": scores, "working_memory": built["wm"],
                    "bridges": [f"{m['domain_a']}<->{m['domain_b']}" for m in built["matches"]],
                    "answer": built["text"], "chars": len(built["text"]),
                }
                if ref_emb is not None:
                    row["reference_cosine"] = round(float(enc.encode(built["text"]) @ ref_emb), 4)
                return row
            except Exception as e:
                return {"variant": v, "ok": False, "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(work, variants))

        full = next((r for r in rows if r["variant"] == "full" and r["ok"]), None)
        for r in rows:
            if r["ok"] and full:
                r["delta_vs_full"] = round(r["scores"]["mean"] - full["scores"]["mean"], 3)
        return {"rows": rows, "reference_used": bool(self.reference)}

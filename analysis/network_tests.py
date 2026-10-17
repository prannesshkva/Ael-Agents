from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch
from scipy.stats import spearmanr

from analysis.ablation import AblationRunner
from analysis.judge import Judge
from core.embeddings import get_encoder

def working_memory_sweep(framework, judge: Judge, question: str, state: Dict[str, Any],
                         ks: Optional[List[int]] = None, reference: Optional[str] = None,
                         max_workers: int = 5) -> Dict[str, Any]:
    n = len(framework.domain_names)
    ks = sorted({max(1, min(k, n)) for k in (ks or [1, 2, 3, 5, 8, n])})
    runner = AblationRunner(framework, judge, reference)
    weights = framework.param_registry.weights.detach().clone()

    def _eval_k(k: int) -> Dict[str, Any]:
        try:
            sn = runner._sn(weights, state, capacity=k, enforce=False)
            wm = sn["working_memory_domains"]
            text = runner._synth(question, state, sn, wm, [], f"wm_k{k}")
            sc = judge.score_answer(question, text, reference)
            return {"k": k, "score_mean": sc["mean"], "scores": sc, "working_memory": wm, "ok": True}
        except Exception as e:
            return {"k": k, "ok": False, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(ks))) as ex:
        rows = list(ex.map(_eval_k, ks))

    ok = [r for r in rows if r["ok"]]
    shape = "insufficient data"
    if len(ok) >= 3:
        vals = [r["score_mean"] for r in ok]
        peak = max(range(len(vals)), key=lambda i: vals[i])
        if 0 < peak < len(vals) - 1:
            shape = "non-monotonic (interior peak)"
        elif peak == 0:
            shape = "best at smallest capacity (monotonic decrease)"
        else:
            shape = "best at largest capacity (monotonic increase)"
    return {"rows": rows, "shape": shape,
            "best_k": max(ok, key=lambda r: r["score_mean"])["k"] if ok else None}

def salience_validity(framework, judge: Judge, question: str, state: Dict[str, Any]) -> Dict[str, Any]:
    names = framework.domain_names
    runner = AblationRunner(framework, judge)
    useful = judge.rate_domain_usefulness(question, state["domain_solutions"])

    trained = framework.param_registry.weights.detach().clone()
    sn_trained = runner._sn(trained, state)
    sn_untrained = runner._sn(torch.ones_like(trained), state)

    q = get_encoder().encode(question)
    relevance = {n: float(state["domain_embeddings"][n] @ q) for n in names}

    predictors = {
        "SN salience (trained weights)": sn_trained["domain_salience"],
        "SN salience (untrained weights, Jacobian only effectively)": sn_untrained["domain_salience"],
        "gating only (trained)": sn_trained["gating_weights"],
        "Jacobian sensitivity only (trained)": sn_trained["jacobian_sensitivities"],
        "baseline: embedding relevance to question": relevance,
    }
    target = [useful[n] for n in names]
    top_k = min(2, len(names))
    judged_top = set(sorted(names, key=lambda n: useful[n], reverse=True)[:top_k])
    out = []
    for label, pred in predictors.items():
        vals = [pred[n] for n in names]
        if len(set(vals)) < 2 or len(set(target)) < 2:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(vals, target)
        top_pred = set(sorted(names, key=lambda n: pred[n], reverse=True)[:top_k])
        out.append({"predictor": label, "spearman_rho": None if rho != rho else round(float(rho), 3),
                    "p_analytic_uncorrected": None if p != p else round(float(p), 4),
                    "top5_overlap_with_judged": len(top_pred & judged_top), "n_domains": len(names)})
    return {"judged_usefulness": useful, "predictors": out}

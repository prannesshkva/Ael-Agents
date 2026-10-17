from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.stats import bootstrap, monte_carlo_test, permutation_test, spearmanr
from torch.distributions import Dirichlet

def _spearman_stat(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)

def mct(
    framework,
    state: Dict[str, Any],
    report: Optional[Dict[str, Any]] = None,
    n_resamples: int = 4999,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {"n_resamples": n_resamples}

    rdm = state["rdm_matrix"].detach().cpu().numpy()
    triu_i, triu_j = np.triu_indices(rdm.shape[0], k=1)
    obs_rdm_pairs = rdm[triu_i, triu_j]

    mc_rdm = monte_carlo_test(
        obs_rdm_pairs,
        rvs=lambda size: rng.normal(loc=1.0, scale=1.0 / np.sqrt(framework.encoder.info.dim), size=size),
        statistic=lambda x, axis=-1: np.std(x, axis=axis),
        n_resamples=n_resamples,
        alternative="greater",
    )
    results["rdm_structure_mc_test"] = {
        "observed_rdm_dispersion": round(float(mc_rdm.statistic), 4),
        "mc_p_value": round(float(mc_rdm.pvalue), 6),
    }

    with torch.no_grad():
        alpha_gate = framework.param_registry.get_gating_distribution().cpu()
        k = len(state.get("working_memory_domains", [])) or min(2, len(framework.domain_names))
        concentration = torch.clamp(alpha_gate * 25.0, min=0.2)
        samples = Dirichlet(concentration).sample((n_resamples,))
        topk_indices = torch.topk(samples, k=k, dim=1).indices
        inclusion_counts = torch.zeros(len(framework.domain_names), dtype=torch.float32)
        for idx in range(len(framework.domain_names)):
            inclusion_counts[idx] = (topk_indices == idx).any(dim=1).float().mean()

    results["working_memory_stability"] = {
        name: round(float(inclusion_counts[i].item()) * 100.0, 2)
        for i, name in enumerate(framework.domain_names)
    }

    if report and "human_rsa" in report:
        lc = report["human_rsa"].get("multi_agent_rsa", {}).get("live_concept_rdm")
        if lc and "agent_rdm" in lc and "reference_rdm" in lc:
            a_rdm = np.array(lc["agent_rdm"])
            r_rdm = np.array(lc["reference_rdm"])
            ti, tj = np.triu_indices(a_rdm.shape[0], k=1)
            x_vec, y_vec = a_rdm[ti, tj], r_rdm[ti, tj]

            perm = permutation_test(
                (x_vec, y_vec),
                statistic=_spearman_stat,
                permutation_type="pairings",
                n_resamples=n_resamples,
                alternative="greater",
                random_state=rng,
            )
            boot = bootstrap(
                (x_vec, y_vec),
                statistic=_spearman_stat,
                n_resamples=min(n_resamples, 1999),
                paired=True,
                confidence_level=0.95,
                method="percentile",
                random_state=rng,
            )
            results["concept_rdm_permutation"] = {
                "spearman_rho": round(float(perm.statistic), 4),
                "permutation_p_value": round(float(perm.pvalue), 6),
                "ci95_low": round(float(boot.confidence_interval.low), 4),
                "ci95_high": round(float(boot.confidence_interval.high), 4),
                "bootstrap_se": round(float(boot.standard_error), 4),
            }

    if report and "network_tests" in report:
        sal_val = report["network_tests"].get("salience_validity", {})
        useful = sal_val.get("judged_usefulness")
        if useful and state.get("domain_salience"):
            names = framework.domain_names
            sn_vec = np.array([state["domain_salience"][n] for n in names], dtype=float)
            ju_vec = np.array([useful[n] for n in names], dtype=float)
            if len(set(sn_vec)) >= 2 and len(set(ju_vec)) >= 2:
                perm_sal = permutation_test(
                    (sn_vec, ju_vec),
                    statistic=_spearman_stat,
                    permutation_type="pairings",
                    n_resamples=n_resamples,
                    alternative="two-sided",
                    random_state=rng,
                )
                results["salience_permutation_p"] = round(float(perm_sal.pvalue), 4)

    if report and "ablations" in report:
        rows = report["ablations"].get("rows", []) if isinstance(report["ablations"], dict) else report["ablations"]
        criteria = ["accuracy", "rigor", "integration", "falsifiability", "responsiveness"]
        full_row = next((r for r in rows if r.get("variant") == "full" and "scores" in r), None)
        if full_row:
            full_vec = np.array([full_row["scores"][c] for c in criteria], dtype=float)
            ablation_pvals: Dict[str, float] = {}
            for r in rows:
                if r.get("variant") != "full" and "scores" in r:
                    var_vec = np.array([r["scores"][c] for c in criteria], dtype=float)
                    if np.allclose(full_vec, var_vec):
                        ablation_pvals[r["variant"]] = 1.0
                    else:
                        p_res = permutation_test(
                            (full_vec, var_vec),
                            statistic=lambda a, b, axis=-1: np.mean(a - b, axis=axis),
                            permutation_type="samples",
                            n_resamples=n_resamples,
                            alternative="two-sided",
                            random_state=rng,
                        )
                        ablation_pvals[r["variant"]] = round(float(p_res.pvalue), 4)
            results["ablation_paired_permutation_p"] = ablation_pvals

    return results

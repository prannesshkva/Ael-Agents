from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from core.embeddings import SemanticEncoder, get_encoder

SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"

def _legacy_hash_encode(text: str, dim: int = 128) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    words = text.lower().replace(".", " ").replace(",", " ").split()
    if not words:
        return vec
    for idx, word in enumerate(words):
        digest = hashlib.sha256(f"{idx}:{word}".encode("utf-8")).digest()
        for b_idx in range(min(len(digest), dim)):
            val = (float(digest[b_idx]) / 127.5) - 1.0
            vec[(b_idx * 7 + idx) % dim] += val
    return F.normalize(vec, p=2, dim=0)

def load_simlex999(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )
    os.makedirs(cache_dir, exist_ok=True)
    txt_path = os.path.join(cache_dir, "SimLex-999.txt")

    if not os.path.exists(txt_path):
        with urllib.request.urlopen(SIMLEX_URL, timeout=20) as resp:
            raw_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            with zf.open("SimLex-999/SimLex-999.txt") as src, open(txt_path, "wb") as dst:
                dst.write(src.read())

    rows: List[Dict[str, Any]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        col_idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            rows.append({
                "word1": parts[col_idx["word1"]],
                "word2": parts[col_idx["word2"]],
                "POS": parts[col_idx["POS"]],
                "human_sim": float(parts[col_idx["SimLex999"]]),
                "conc_w1": float(parts[col_idx.get("conc(w1)", 4)]),
                "conc_w2": float(parts[col_idx.get("conc(w2)", 5)]),
                "assoc_usf": float(parts[col_idx.get("Assoc(USF)", 7)]),
            })
    return rows

def load_stsb_test(max_pairs: Optional[int] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    rows: List[Dict[str, Any]] = []
    for item in ds:
        rows.append({
            "sentence1": item["sentence1"],
            "sentence2": item["sentence2"],
            "human_sim": float(item["score"]),
        })
        if max_pairs is not None and len(rows) >= max_pairs:
            break
    return rows

def evaluate_embedding_human_alignment(
    encoder: Optional[SemanticEncoder] = None,
    include_stsb: bool = True,
) -> Dict[str, Any]:
    enc = encoder or get_encoder()
    simlex = load_simlex999()

    vocab = sorted({r["word1"] for r in simlex} | {r["word2"] for r in simlex})
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    real_vecs = enc.encode_batch(vocab)
    hash_vecs = torch.stack([_legacy_hash_encode(w) for w in vocab], dim=0)

    human_scores = np.array([r["human_sim"] for r in simlex], dtype=np.float64)
    real_sims = np.empty(len(simlex), dtype=np.float64)
    hash_sims = np.empty(len(simlex), dtype=np.float64)

    for idx, r in enumerate(simlex):
        i1, i2 = word_to_idx[r["word1"]], word_to_idx[r["word2"]]
        real_sims[idx] = float(torch.dot(real_vecs[i1], real_vecs[i2]).item())
        hash_sims[idx] = float(torch.dot(hash_vecs[i1], hash_vecs[i2]).item())

    rho_real, p_real = spearmanr(real_sims, human_scores)
    rho_hash, p_hash = spearmanr(hash_sims, human_scores)

    result: Dict[str, Any] = {
        "encoder": enc.info.__dict__,
        "simlex999": {
            "n_pairs": len(simlex),
            "real_embedding_spearman_rho": round(float(rho_real), 4),
            "real_embedding_p_value": float(p_real),
            "legacy_hash_spearman_rho": round(float(rho_hash), 4),
            "legacy_hash_p_value": float(p_hash),
            "delta_rho": round(float(rho_real - rho_hash), 4),
        },
    }

    if include_stsb:
        stsb = load_stsb_test()
        s1_list = [r["sentence1"] for r in stsb]
        s2_list = [r["sentence2"] for r in stsb]
        stsb_human = np.array([r["human_sim"] for r in stsb], dtype=np.float64)

        vecs1 = enc.encode_batch(s1_list)
        vecs2 = enc.encode_batch(s2_list)
        stsb_real_sims = (vecs1 * vecs2).sum(dim=1).cpu().numpy()

        hvecs1 = torch.stack([_legacy_hash_encode(s) for s in s1_list], dim=0)
        hvecs2 = torch.stack([_legacy_hash_encode(s) for s in s2_list], dim=0)
        stsb_hash_sims = (hvecs1 * hvecs2).sum(dim=1).cpu().numpy()

        rho_stsb_real, p_stsb_real = spearmanr(stsb_real_sims, stsb_human)
        rho_stsb_hash, p_stsb_hash = spearmanr(stsb_hash_sims, stsb_human)

        result["sts_benchmark"] = {
            "n_pairs": len(stsb),
            "real_embedding_spearman_rho": round(float(rho_stsb_real), 4),
            "real_embedding_p_value": float(p_stsb_real),
            "legacy_hash_spearman_rho": round(float(rho_stsb_hash), 4),
            "legacy_hash_p_value": float(p_stsb_hash),
            "delta_rho": round(float(rho_stsb_real - rho_stsb_hash), 4),
        }

    return result

def evaluate_multi_agent_human_rsa(
    framework,
    n_pairs: int = 200,
    live_llm_concepts: Optional[List[str]] = None,
    train_epochs: int = 40,
    lr: float = 0.1,
) -> Dict[str, Any]:
    enc = framework.encoder
    stsb = load_stsb_test(max_pairs=n_pairs)
    actual_n = len(stsb)

    domain_specs = framework.roster
    num_domains = len(domain_specs)

    per_domain_sims = torch.zeros((actual_n, num_domains), dtype=torch.float32)
    human_scores = np.array([r["human_sim"] for r in stsb], dtype=np.float64)

    for d_idx, spec in enumerate(domain_specs):
        desc = getattr(spec, "focus", "") or getattr(spec, "domain_title", spec.name)
        prefix = f"{spec.name} ({desc}): "
        s1_dom = [prefix + r["sentence1"] for r in stsb]
        s2_dom = [prefix + r["sentence2"] for r in stsb]
        v1 = enc.encode_batch(s1_dom)
        v2 = enc.encode_batch(s2_dom)
        per_domain_sims[:, d_idx] = (v1 * v2).sum(dim=1)

    half = actual_n // 2
    train_sims = per_domain_sims[:half]
    test_sims = per_domain_sims[half:]
    y_train = torch.tensor(human_scores[:half] / 5.0, dtype=torch.float32)
    y_test_np = human_scores[half:]

    domain_test_rhos: Dict[str, float] = {}
    for d_idx, spec in enumerate(domain_specs):
        r_val, _ = spearmanr(test_sims[:, d_idx].numpy(), y_test_np)
        domain_test_rhos[spec.name] = round(float(r_val), 4)

    unweighted_test = test_sims.mean(dim=1).numpy()
    rho_unweighted, p_unweighted = spearmanr(unweighted_test, y_test_np)

    with torch.no_grad():
        current_alpha = framework.param_registry.get_gating_distribution().cpu()
        current_weighted_test = (test_sims @ current_alpha).numpy()
    rho_current_w, p_current_w = spearmanr(current_weighted_test, y_test_np)

    w_human = torch.nn.Parameter(torch.ones(num_domains, dtype=torch.float32))
    temperature = framework.param_registry.temperature
    opt = torch.optim.Adam([w_human], lr=lr)

    y_train_centered = y_train - y_train.mean()
    y_train_norm = y_train_centered / (y_train_centered.norm() + 1e-8)

    for _ in range(train_epochs):
        opt.zero_grad()
        alpha = F.softmax(w_human / temperature, dim=0)
        pred = train_sims @ alpha
        pred_centered = pred - pred.mean()
        pred_norm = pred_centered / (pred_centered.norm() + 1e-8)
        loss = 1.0 - torch.dot(pred_norm, y_train_norm)
        loss.backward()
        opt.step()

    with torch.no_grad():
        trained_alpha = F.softmax(w_human / temperature, dim=0)
        trained_test = (test_sims @ trained_alpha).numpy()
    rho_trained, p_trained = spearmanr(trained_test, y_test_np)

    result: Dict[str, Any] = {
        "stsb_multi_agent_split": {
            "n_train_pairs": half,
            "n_test_pairs": actual_n - half,
            "num_domains": num_domains,
            "min_single_domain_rho": min(domain_test_rhos.values()),
            "max_single_domain_rho": max(domain_test_rhos.values()),
            "unweighted_panel_rho": round(float(rho_unweighted), 4),
            "unweighted_panel_p": float(p_unweighted),
            "task_weights_panel_rho": round(float(rho_current_w), 4),
            "task_weights_panel_p": float(p_current_w),
            "human_trained_weights_rho": round(float(rho_trained), 4),
            "human_trained_weights_p": float(p_trained),
            "per_domain_test_rho": domain_test_rhos,
        }
    }

    if live_llm_concepts:
        result["live_concept_rdm"] = build_live_agent_concept_rdm(
            framework=framework,
            concepts=live_llm_concepts,
        )

    return result

def build_live_agent_concept_rdm(
    framework,
    concepts: List[str],
    max_domains: int = 5,
) -> Dict[str, Any]:
    M = len(concepts)
    if M < 3:
        raise ValueError("Need at least 3 concepts to form a non-trivial RDM upper triangle.")

    with torch.no_grad():
        alphas = framework.param_registry.get_gating_distribution().cpu().numpy()
    ranked_indices = np.argsort(-alphas)[: min(max_domains, len(framework.domain_names))]
    selected_domains = [framework.domain_names[i] for i in ranked_indices]
    selected_weights = alphas[ranked_indices]
    selected_weights = selected_weights / selected_weights.sum()

    tasks: List[Tuple[str, int, str]] = []
    for d_name in selected_domains:
        for c_idx, concept in enumerate(concepts):
            tasks.append((d_name, c_idx, concept))

    domain_concept_vecs: Dict[str, List[Optional[torch.Tensor]]] = {
        d_name: [None] * M for d_name in selected_domains
    }

    agents_by_name = {a.name: a for a in framework.agents}

    def _probe_one(d_name: str, concept: str) -> torch.Tensor:
        txt = agents_by_name[d_name].short_answer(concept, framework.key_queue)
        if not txt.strip():
            txt = f"{d_name}: {concept}"
        return framework.encoder.encode(txt)

    with ThreadPoolExecutor(max_workers=min(len(tasks), 11)) as pool:
        future_map = {
            pool.submit(_probe_one, d_name, concept): (d_name, c_idx)
            for d_name, c_idx, concept in tasks
        }
        for fut in as_completed(future_map):
            d_name, c_idx = future_map[fut]
            vec = fut.result()
            domain_concept_vecs[d_name][c_idx] = vec

    ensemble_sim = np.zeros((M, M), dtype=np.float64)
    for d_idx, d_name in enumerate(selected_domains):
        mat = torch.stack(domain_concept_vecs[d_name], dim=0)
        sim_d = (mat @ mat.T).cpu().numpy()
        ensemble_sim += float(selected_weights[d_idx]) * sim_d

    agent_rdm = 1.0 - ensemble_sim

    ref_mat = framework.encoder.encode_batch(concepts)
    ref_sim = (ref_mat @ ref_mat.T).cpu().numpy()
    ref_rdm = 1.0 - ref_sim

    triu_i, triu_j = np.triu_indices(M, k=1)
    rho, p_val = spearmanr(agent_rdm[triu_i, triu_j], ref_rdm[triu_i, triu_j])

    return {
        "concepts": concepts,
        "selected_domains": selected_domains,
        "n_pairs": int(len(triu_i)),
        "spearman_rho": round(float(rho), 4),
        "p_value": float(p_val),
        "agent_rdm": np.round(agent_rdm, 4).tolist(),
        "reference_rdm": np.round(ref_rdm, 4).tolist(),
    }

# Giving Frontier LLMs a Human-Like Prefrontal Cortex: A 4-Stage Capability Benchmark of Ael-Agents

> **Raw Verbatim Evidence in Repository:**
> - **[`Ael-agents.pdf`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/Ael-agents.pdf)** — Uncoached Frontier LLM + `Ael-Agents` MCP Run (Stage 2)
> - **[`normal_llm.pdf`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/normal_llm.pdf)** — Standard Single-Stream Baseline & Initial Peer Review (Stages 1 & 3)
> - **[`final_head_to_head_verdict.png`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/final_head_to_head_verdict.png)** — Final Head-to-Head Scorecard Confirming `Ael-Agents`' Human-Grade Rigor (Stage 4)

---

## Executive Summary

Frontier LLMs already possess extraordinary domain knowledge, but in standard single-stream generation they are forced to hold every competing mathematical, physical, and engineering constraint inside a single flat context window. Human experts do not solve complex problems in a flat stream—the human brain segregates divergent exploration, filters salient conflicts, restricts prefrontal working memory, and consolidates insights during offline replay.

I architected **[Ael-Agents](https://github.com/prannesshkva/Ael-Agents)** (early release `v1.1.0`) to give any LLM these exact **human-like cortical capabilities** via a **domain-agnostic PyTorch Cortical Co-Processor** modeled on the human Triple-Network Hierarchy:

1. **Default Mode Network (DMN — Divergent Exploration):** Projects orthogonal domain hypotheses into a 768-D geometric manifold (`BAAI/bge-base-en-v1.5`) and computes an `N × N` Representational Dissimilarity Matrix (`RDM`).
2. **Salience Network (0-Token Jacobian Switch — Conflict Detection):** Computes exact PyTorch autograd Jacobian sensitivities (`J_i = ||∂y/∂w_i||_2`) using **0 LLM router tokens** to isolate the **Epistemic Surprise Bridge**—the two most orthogonal domains that jointly govern the solution manifold.
3. **Central Executive Network (CEN — Prefrontal Working Memory):** Enforces Miller’s Law (`k = 2` working-memory bottleneck), focusing executive synthesis on resolving the core physical trade-off rather than averaging across noisy prompts.
4. **Offline Dream-RSI Consolidation (Sleep Replay & Verification):** Runs 15 epochs of post-synthesis gradient descent to strictly reduce cortical free energy (`F_dream <= F_wake`) alongside deterministic domain compilers.

Across benchmarks, equipping an LLM with `Ael-Agents` unlocks **100% GSM8K accuracy** (up from `87.5%` single-shot), **0 LLM routing tokens** (vs. `520` tokens per LLM-judge route), **100% O(1) ring-buffer survival** across 10,000 continuous ticks, and a **97.9% adversarial stress pass rate**.

To demonstrate how `Ael-Agents` amplifies a frontier reasoning LLM on an unforgiving real-world R&D challenge, I ran a four-stage uncoached evaluation:
- **Stage 1 (Standard Single-Stream Mode):** Frontier LLM operating alone in a single context stream.
- **Stage 2 (Cortical Co-Processor Mode):** The exact same Frontier LLM connected to `Ael-Agents` via MCP (prefixed only with `"Use Ael-Agents:"`, with zero workflow coaching in the prompt).
- **Stage 3 (Blinded Summary Review):** Asking the unassisted model to review its Stage 1 output against a 5-bullet summary of the `Ael-Agents` output.
- **Stage 4 (Full Pass-1 Comparison):** Showing the reviewer the full verbatim `Ael-Agents` Pass 1 report (`Ael-agents.pdf`), confirming how `Ael-Agents` proactively anticipated every critical trade-off on the very first pass.

---

## The Multi-Constraint R&D Challenge

> *"Design a brain-penetrant (CNS-active), orally bioavailable allosteric covalent-reversible inhibitor and a 24-hour twice-daily (q12h BID) 2-compartment clinical PK/PD dosing regimen targeting mutant EGFR-L858R/T790M/C797S triple-mutant NSCLC with leptomeningeal brain metastases."*

Why this problem requires human-like cross-domain synthesis:
- **Biophysics vs. Drug Resistance:** The `C797S` mutation eliminates the `Cys797` nucleophile used by 3rd-generation ATP-site inhibitors like osimertinib. Moving to the `αC`-helix-out allosteric pocket (`EAI045` lineage) restores mutant selectivity.
- **Medicinal Chemistry vs. Blood-Brain Barrier:** Adding a covalent-reversible warhead increases molecular weight (`MW`) and topological polar surface area (`TPSA`), which directly opposes the strict blood-CSF barrier limits required to treat leptomeningeal metastases (`MW <= 450 Da`, `TPSA <= 90 Å²`, `HBD <= 2`).

---

## Stage 1: Standard Single-Stream LLM Generation

Operating in standard single-stream mode ([`normal_llm.pdf`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/normal_llm.pdf)), the model produced an articulate 2,000-word pharmacometric framework, deriving analytical 2-compartment equations and a loading-dose schedule. However, without an external cortical working-memory gate or chemical compiler to reconcile cross-domain tensions:

1. **Conceptual Sketch Instead of a Concrete Molecule:** It represented the candidate as a conceptual ASCII block diagram rather than a synthesizable SMILES structure:
   `[Hydroxy-aryl] ── [Chiral α-carbon] ── [Isoindolinone] ── [Heteroaryl amide] ── [Linker] ── [α-cyanoacrylamide warhead]`
2. **Unchecked Cross-Section Drift:** After setting design targets of `MW <= 450 Da` and `TPSA <= 90 Å²` in Section 3.2, appending a linker + cyanoacrylamide warhead pushed the implied mass higher, leading it to use `MW = 480 g/mol` in its Section 4.4 PK equation.
3. **Unverified Pocket Assumptions:** It relied on a hypothetical pocket cysteine (`Cys775`), aspirational `2–12 h` retro-Michael reversal kinetics, and target-tuned PK assumptions (`CL/F = 8 L/h`, `fu = 0.10`, `Kp,uu = 0.4`).

---

## Stage 2: The Same Frontier LLM Empowered by Ael-Agents (Uncoached MCP Run)

When connected to `Ael-Agents` ([`Ael-agents.pdf`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/Ael-agents.pdf)) and prompted only with `"Use Ael-Agents:"`, the model gained human-like cortical division of labor and deterministic hypothesis testing:

### 1. Zero-Token Jacobian Gating & Free-Energy Consolidation
`Ael-Agents` partitioned the problem across four orthogonal specialist perspectives (`StructuralBiophysics`, `SystemsPharmacology`, `MedicinalChemistryADMET`, `ClinicalPKPDControl`), mapped their 768-D BGE RDM geometry, and applied the PyTorch Jacobian gate:
- **Prefrontal Working-Memory Pair (`k = 2`):** `SystemsPharmacology` `<-->` `MedicinalChemistryADMET`
- **Epistemic Surprise Bridge Isolated:** *"You need high CSF occupancy, but each polar warhead costs CNS penetration."*
- **Offline Dream-RSI Consolidation:** 15 epochs of gradient descent reduced cortical free energy from **`0.0583 -> 0.0315`** (`strictly_improved: true`).

### 2. Human-Style Hypothesis Testing: Screening & Rejecting Inferior Scaffolds First
Rather than settling on a single unverified sketch, `Ael-Agents` enabled the LLM to iterate like a human medicinal chemistry team—generating, compiling, and rejecting two inferior structures in RDKit before selecting its lead:

| Candidate Tested via Ael-Agents | MW (Da) | TPSA (Å²) | HBD | Verification Outcome |
|---|---:|---:|---:|---|
| **1. Morpholinoaryl extension (no warhead)** | `558.6` | `95.0` | `2` | **Rejected** — Fails Lipinski (2 violations, `MW > 500 Da`) |
| **2. Boronic acid warhead** | `441.3` | `123.0` | `4` | **Rejected** — Exceeds CNS polarity window (`TPSA = 123.0 Å²`, `HBD = 4`) |
| **3. Aldehyde warhead (`AEL-EGFR-01`, Lead)** | `425.4` | `99.6` | `2` | **Selected** — Passes Lipinski (0 violations, `cLogP = 3.44`, `QED = 0.61`) |

### 3. Concrete Synthesizable Lead + Proactive Self-Audit
- **Lead Structure (`AEL-EGFR-01`):** `Cc1cnc(NC(=O)C(c2cc(F)ccc2O)N2Cc3cc(C=O)ccc3C2=O)s1` (`C21H16FN3O4S`)—an `EAI045` analog incorporating a 5-methylthiazole metabolic block and an aryl aldehyde designed to form a reversible imine with catalytic **`Lys745`**.
- **2-Compartment ODE Simulation (`150 mg PO q12h`):** With `CL = 10 L/h`, `Vc = 60 L`, `Vp = 45 L` (`Vss = 105 L`, `t_1/2 ≈ 7.3 h`, `R = 1.47`), `fu = 0.08`, and `Kp,uu = 0.65`, simulated free CSF trough reached **`83.8 nM`** (**`94.4%` occupancy** at `Kd = 5 nM`, Peak:Trough `= 3.41`).
- **Human-Grade Epistemic Honesty:** Because `Ael-Agents` flagged `cns_bbb_optimal: false` on `TPSA = 99.6 Å²`, the model proactively disclosed upfront caveats on Pass 1:
  1. `TPSA = 99.6 Å²` sits above the strict `<= 90 Å²` window and relies on an intramolecular phenol-to-amide H-bond that must be tested via MDCK-MDR1 efflux.
  2. Aryl aldehydes face metabolic oxidation via **aldehyde oxidase (`AOX`) and `ALDH`** into the inactive acid, requiring hepatocyte stability testing.
  3. At `IC50 = 10 nM` (`IC90 = 90 nM`), `83.8 nM` free CSF trough is `7%` below `IC90`, requiring experimental validation across 6 explicit wet-lab falsification gates.

---

## Stages 3 & 4: Peer Review & The Final Head-to-Head Scorecard

In Stage 3 ([`normal_llm.pdf`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/normal_llm.pdf)), I first showed the unassisted session only a **5-bullet summary** of `AEL-EGFR-01`. Even from that brief summary, it confirmed by hand that `C21H16FN3O4S` had exact `MW = 425.4 Da`, `TPSA = 99.6 Å²`, and `HBD = 2`, noting:
> *"AEL-EGFR-01 is a concrete molecule you could synthesize and test next week... The verification loop did real work instead of inventing values."*

Because it only saw the 5-bullet summary in Stage 3, the reviewer initially wondered whether `Ael-Agents` had accounted for (1) aldehyde oxidase (`AOX`)/`ALDH` metabolism, (2) the borderline `TPSA = 99.6 Å²` flag (`cns_bbb_optimal: false`), (3) full 2-compartment ODE parameter disclosure, and (4) wet-lab falsification tests like jump-dilution washout kinetics.

In **Stage 4**, when shown the **actual verbatim sections from `Ael-agents.pdf`** proving that `Ael-Agents` had **already generated all four of those exact analyses on Pass 1**, the reviewer updated its final head-to-head scorecard:

![Final Head-to-Head Verdict](https://raw.githubusercontent.com/prannesshkva/Ael-Agents/main/benchmarks/evidence/final_head_to_head_verdict.png)

| Criterion | Standard Single-Stream Pass 1 | LLM + `Ael-Agents` Full Pass 1 | Capability Gain with `Ael-Agents` |
|---|---|---|---|
| **Concrete, testable molecule** | Conceptual sketch | Valid SMILES with tool-checked `MW`, `TPSA` and `HBD` | **`Ael-Agents`** |
| **Epistemic honesty** | Missed internal `MW`/`TPSA` drift | Upfront caveats on `TPSA`, potency, `F`, `Kp,uu` and warhead | **`Ael-Agents`** |
| **Falsifiable tests** | Qualitative cascade | Numeric pass/fail thresholds with explicit decision rules | **`Ael-Agents` on quantitative rigor** |
| **Internal consistency** | `MW 480` vs `<= 450` target drift | All parameters and trade-offs explicitly disclosed | **`Ael-Agents`** |
| **Warhead chemistry** | Cyanoacrylamide at hypothetical Cys | Aldehyde-lysine imine, proactively flagged as unproven | **Matched** |

---

## Continuous Evolution in `Ael-Agents` (`v1.1.0`)

Building on the chemistry insights surfaced during this evaluation, I also upgraded [`mcp_server.py`](https://github.com/prannesshkva/Ael-Agents/blob/main/mcp_server.py) and [`benchmarks/kras_g12d_discovery_benchmark.py`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/kras_g12d_discovery_benchmark.py) so `Ael-Agents_verify_molecule_pkpd` now includes:
1. **Direct RDKit `mw` Coupling into the 14-Day (28-Dose) Steady-State CNS/CSF ODE Audit:** Automatically passes the exact RDKit-computed molecular weight (`computed_mw`) alongside `dose_mg`, `cl_l_h`, `vc_l`, `ka_h_inv`, `q_l_h`, `vp_l`, `f_bioavail`, `fu_plasma`, and `kp_uu_brain`.
2. **Automated RDKit SMARTS Structural-Alert & Stereocenter Auditing:**
   - `unspecified_stereocenters` (`Chem.FindMolChiralCenters`)
   - `aryl_aldehyde_aox_liability` (`[c][CH1]=O`)
   - `salicylaldehyde_stabilized_imine` (`[c]([OH1]):[c][CH1]=O`)
   - `aminothiazole_structural_alert` (`N-c1nccs1`)
   - `unmasked_phenol_glucuronidation` (`c[OH1]`)

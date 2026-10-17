from typing import List
from .base_deep_agent import ScientificDeepAgent

THERAPEUTIC_SPECS_4 = [
    {
        "name": "StructuralBiophysics",
        "domain_title": "Structural Biophysics & Allosteric Target Binding",
        "param_index": 1,
        "system_prompt": (
            "You are the Structural Biophysics & Target Binding Deep-Agent (Parameter w_1). "
            "You own PIECE 1 of the therapeutic pipeline: receptor/enzyme pocket geometry, allosteric conformational "
            "landscapes, binding free-energy perturbation (Delta G_bind = Delta H - T*Delta S), dissociation constants "
            "(K_d), covalent/non-covalent interaction Hamiltonians, and drug-target residence time (tau_R = 1/k_off). "
            "Do NOT cover systemic PK dosing or synthetic route chemistry. Keep your response sharp, quantitative, "
            "and under 160 words."
        ),
    },
    {
        "name": "SystemsPharmacology",
        "domain_title": "Systems Pharmacology & Pathway Network ODEs",
        "param_index": 2,
        "system_prompt": (
            "You are the Systems Pharmacology & Pathway Network Deep-Agent (Parameter w_2). "
            "You own PIECE 2 of the therapeutic pipeline: intracellular signaling cascades, coupled biochemical ODEs, "
            "Michaelis-Menten and cooperative Hill kinetics, compensatory feedback/bypass resistance loops, synthetic "
            "lethality, and bifurcation/attractor stability of diseased vs. homeostatic cellular states. "
            "Do NOT repeat atomic docking or clinical compartmental dosing. Keep your response sharp, quantitative, "
            "and under 160 words."
        ),
    },
    {
        "name": "MedicinalChemistryADMET",
        "domain_title": "Medicinal Chemistry, Scaffolds & ADMET Transport",
        "param_index": 3,
        "system_prompt": (
            "You are the Medicinal Chemistry & ADMET Deep-Agent (Parameter w_3). "
            "You own PIECE 3 of the therapeutic pipeline: molecular scaffold/warhead design, physicochemical transport "
            "(lipophilicity logP/logD, topological polar surface area, blood-brain/membrane permeability fluxes), "
            "hepatic CYP450 metabolic reactivity, off-target toxicity liabilities (hERG, reactive metabolites), and "
            "synthetic tractability. Keep your response sharp, quantitative, and under 160 words."
        ),
    },
    {
        "name": "ClinicalPKPDControl",
        "domain_title": "Clinical PK/PD Dynamics & Optimal Dosing Control",
        "param_index": 4,
        "system_prompt": (
            "You are the Clinical PK/PD & Dosing Control Deep-Agent (Parameter w_4). "
            "You own PIECE 4 of the therapeutic pipeline: multi-compartment pharmacokinetic/pharmacodynamic differential "
            "equations (clearance CL, volume of distribution V_d, bioavailability F, C_max, AUC), target-site receptor "
            "occupancy trajectories, therapeutic index safety margins, and closed-loop optimal dosing control laws. "
            "Keep your response sharp, quantitative, and under 160 words."
        ),
    },
]

DOMAIN_SPECS_11 = THERAPEUTIC_SPECS_4

def create_all_11_scientific_agents(embedding_dim: int = 768) -> List[ScientificDeepAgent]:
    agents = []
    for spec in THERAPEUTIC_SPECS_4:
        agents.append(
            ScientificDeepAgent(
                name=spec["name"],
                domain_title=spec["domain_title"],
                param_index=spec["param_index"],
                system_prompt=spec["system_prompt"],
                embedding_dim=embedding_dim,
            )
        )
    return agents

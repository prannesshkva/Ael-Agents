import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ask_scientist import run_scientific_inquiry

def run_demonstration():
    task_prompt = (
        "Design a precision allosteric covalent-reversible inhibitor and closed-loop PK/PD dosing regimen "
        "that selectively targets mutant KRAS-G12D signaling, prevents compensatory MAPK/PI3K feedback "
        "reactivation, and maintains hepatic/cardiac ADMET safety margins."
    )

    live_concepts = [
        "affinity",
        "allostery",
        "clearance",
        "permeability",
        "homeostasis",
        "toxicity",
    ]

    ablation_variants = [
        "full",
        "no_dmn",
        "no_sn",
        "no_cen",
        "no_jacobian_scoring",
        "untrained_weights",
        "random_working_memory",
        "single_prompt",
    ]

    output_json = os.path.join(current_dir, "evaluation_results.json")
    save_plot = os.path.join(current_dir, "cortex_diagnostics.png")

    return run_scientific_inquiry(
        question=task_prompt,
        roster_mode="default",
        epochs=15,
        run_network_tests=True,
        run_ablations=True,
        ablation_variants=ablation_variants,
        run_human_rsa=True,
        live_concepts=live_concepts,
        output_json=output_json,
        save_plot=save_plot,
    )

if __name__ == "__main__":
    run_demonstration()

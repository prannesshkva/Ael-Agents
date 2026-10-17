from __future__ import annotations

import argparse
import json
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ask_scientist import resolve_roster
from core.api_queue import APIKeyQueue
from core.config import FrameworkConfig
from core.dream_rsi import DreamRSILoop, ToolRegistry, synthesize_and_verify_tool
from core.embeddings import get_encoder
from core.framework import JacobianRSACortexFramework

__all__ = [
    "DreamRSILoop",
    "ToolRegistry",
    "synthesize_and_verify_tool",
    "run_dream_rsi",
]

def run_dream_rsi(
    question: str,
    roster_mode: str = "default",
    num_auto_domains: int = 4,
    generations: int = 1,
    output_json: str | None = None,
):
    encoder = get_encoder()
    key_queue = APIKeyQueue()
    roster = resolve_roster(roster_mode, question, key_queue, num_auto_domains=num_auto_domains)
    config = FrameworkConfig(latent_dim=encoder.info.dim)
    framework = JacobianRSACortexFramework(
        config=config,
        roster=roster,
        encoder=encoder,
        key_queue=key_queue,
    )
    loop = DreamRSILoop(framework=framework)
    res = loop.run(question=question, generations=generations)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return res

def main():
    parser = argparse.ArgumentParser(description="Execute the Dream-RSI Wake-Dream-Improve loop.")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=(
            "Design a precision allosteric covalent-reversible inhibitor and closed-loop PK/PD dosing regimen "
            "that selectively targets mutant KRAS-G12D signaling, prevents compensatory MAPK/PI3K feedback "
            "reactivation, and maintains hepatic/cardiac ADMET safety margins."
        ),
    )
    parser.add_argument("--roster", "-r", type=str, default="default")
    parser.add_argument("--num-auto-domains", type=int, default=4)
    parser.add_argument("--generations", "-g", type=int, default=1)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    run_dream_rsi(
        question=args.question,
        roster_mode=args.roster,
        num_auto_domains=args.num_auto_domains,
        generations=args.generations,
        output_json=args.output_json,
    )

if __name__ == "__main__":
    main()

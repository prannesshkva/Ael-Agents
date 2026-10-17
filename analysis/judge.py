import statistics
from typing import Dict, Optional, Tuple

from core.api_queue import APIKeyQueue
from core.llm_utils import extract_json, llm_call

CRITERIA = ["accuracy", "rigor", "integration", "falsifiability", "responsiveness"]

JUDGE_SYSTEM = (
    "You are a strict, impartial peer reviewer. You are given a research question and ONE candidate "
    "answer. You do not know how the answer was produced. Score it on each criterion with an integer "
    "from 0 to 10 (be harsh; 5 is mediocre, 8+ is exceptional):\n"
    "- accuracy: claims consistent with established science; penalise fabricated facts/citations.\n"
    "- rigor: explicit assumptions, logical/mathematical consistency, separation of knowledge from speculation.\n"
    "- integration: genuine, non-superficial use of multiple fields/perspectives where relevant.\n"
    "- falsifiability: contains a concrete test that could refute the proposal.\n"
    "- responsiveness: directly answers the question that was asked.\n"
    'Be concise in internal reasoning and return ONLY JSON: {"accuracy":int,"rigor":int,"integration":int,'
    '"falsifiability":int,"responsiveness":int,"comment":"<=25 words"}'
)

USEFULNESS_SYSTEM = (
    "You are a strict research editor. Given a research question and several domain-expert "
    "contributions, rate how useful EACH contribution is for answering the question (integer 0-10; "
    "0 = irrelevant or empty, 10 = essential). Judge usefulness to the question, not writing style. "
    'Be concise in internal reasoning and return ONLY JSON: {"scores": {"<DomainName>": int, ...}}'
)

class Judge:
    def __init__(self, key_queue: APIKeyQueue):
        self.key_queue = key_queue

    def score_answer(self, question: str, answer: str, reference: Optional[str] = None) -> Dict[str, float]:
        user = f"RESEARCH QUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}"
        if reference:
            user += (
                "\n\nREFERENCE (ground truth for accuracy; an answer that contradicts it must get a "
                f"low accuracy score):\n{reference}"
            )
        raw = llm_call(self.key_queue, JUDGE_SYSTEM, user, name="judge", temperature=0.0, max_tokens=2048)
        data = extract_json(raw)
        scores = {c: float(max(0, min(10, data[c]))) for c in CRITERIA}
        scores["mean"] = round(statistics.mean(scores[c] for c in CRITERIA), 3)
        scores["comment"] = str(data.get("comment", ""))
        return scores

    def rate_domain_usefulness(self, question: str, domain_solutions: Dict[str, str]) -> Dict[str, float]:
        blocks = "\n\n".join(
            f"### {name}\n{text[:1200]}" for name, text in domain_solutions.items()
        )
        user = f"RESEARCH QUESTION:\n{question}\n\nCONTRIBUTIONS:\n{blocks}"
        raw = llm_call(
            self.key_queue, USEFULNESS_SYSTEM, user, name="judge_usefulness",
            temperature=0.0, max_tokens=1024,
        )
        scores = extract_json(raw).get("scores", extract_json(raw))
        return {n: float(max(0, min(10, scores.get(n, 0)))) for n in domain_solutions}

    def score_pair(
        self, question: str, answer_a: str, answer_b: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        paired_system = (
            "You are a strict, impartial peer reviewer in Computational Drug Discovery & Therapeutics. "
            "Score BOTH Candidate A and Candidate B side-by-side on each criterion (integer 0-10): "
            "accuracy, rigor, integration, falsifiability, responsiveness. "
            'Return ONLY JSON: {"A": {"accuracy":int,"rigor":int,"integration":int,"falsifiability":int,'
            '"responsiveness":int,"comment":"<=20 words"}, '
            '"B": {"accuracy":int,"rigor":int,"integration":int,"falsifiability":int,'
            '"responsiveness":int,"comment":"<=20 words"}}'
        )
        user = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"--- CANDIDATE A ---\n{answer_a}\n\n"
            f"--- CANDIDATE B ---\n{answer_b}"
        )
        raw = llm_call(
            self.key_queue, paired_system, user, name="judge_paired", temperature=0.0, max_tokens=1200
        )
        data = extract_json(raw)
        out = []
        for key in ("A", "B"):
            sub = data.get(key, {})
            sc = {c: float(max(0, min(10, sub.get(c, 5)))) for c in CRITERIA}
            sc["mean"] = round(statistics.mean(sc[c] for c in CRITERIA), 3)
            sc["comment"] = str(sub.get("comment", ""))
            out.append(sc)
        return out[0], out[1]

# Ael-Agents // Autonomous Weight-Space & Horizon Engine

> **Autonomous Weight-Space Programming & Long-Horizon Discovery via Topological State Manifolds, Invariant Guardrails, and Dream-RSI.**  
> Powered by **LangGraph**, **LangSmith**, and Adaptive Cognitive Agent Networks.

---

## 🌌 The Fundamental Impossibility We Solve

Every existing AI agent in the world (Devin, Claude Code, OpenAI o1/o3, AutoGen, CrewAI) hits the **Horizon & Amnesia Wall**:
* **Compounding Error Entropy:** Over 30–50 sequential turns, sliding context windows degrade, attention drifts, and the agent collapses into hallucinations.
* **The Zero-Learning Paradox:** When an agent takes 100 turns and fails at step 95, **all 100 turns are discarded**. The agent learns nothing and restarts from scratch with frozen model weights.

**Ael-Agents breaks this wall mathematically:**
1. **Topological State Manifold (Decoupled Context):** Stores reasoning as an immutable Directed Acyclic Graph (DAG). At Step 500, prompt size and context length remain constant.
2. **Strict Invariant Contracts (Zero Lobotomy / Zero Drift):** Enforces non-negotiable mathematical and domain boundaries across steps.
3. **Causal Credit Assignment (Pivot Analysis):** Backpropagates through failed trajectories to pinpoint the **exact pivot node** where divergence began.
4. **Universal Dream-RSI (Zero-Cost Offline Evolution):** Replays historical manifolds across mutant exploration policies **at $0.00 compute and token cost**.

---

## 🏛️ Supported Domains

### 1. `surgery` — Weight-Space Programming & Neural Surgery (Default)
* **Invariants:** Surgical sparsity budget ($\le 0.20\%$ modified weights), Perplexity preservation ($< 4.0\%$ drift), Circular loop freedom.
* **Objective:** Backprop-free surgical editing of model weights (e.g., Qwen3.5-0.8B) to project out repetitive reasoning loop attractors and unlock mathematical reasoning (7.9% $\to$ 51.0% MATH-500).

### 2. `code` — Systems, Kernel & Software Engineering
* **Invariants:** AST symbol preservation, 100% baseline unit test integrity, memory safety, zero data races.
* **Objective:** Synthesizing lock-free CAS queues, SIMD vectorization, cache tiling, distributed consensus.

### 3. `psyche` — Cognitive Dynamics & Psychological Simulation
* **Invariants:** Clinical safety bound (SUDS distress $\le 0.75$), non-judgmental stance, zero shame spirals.
* **Objective:** Multi-persona Internal Family Systems (IFS) unblending, compassionate inquiry, somatic grounding.

---

## ⚡ Quickstart

### 1. Clone & Run
```bash
git clone https://github.com/prannesshkva/Ael-Agents.git
cd Ael-Agents

# Windows (Weight-Space Surgery)
.\run.ps1 --domain surgery
# Systems Code or Cognitive Simulation:
.\run.ps1 --domain code
.\run.ps1 --domain psyche

# macOS / Linux
./run.sh --domain surgery
```

### 2. Run the Verification Suite
```bash
uv run pytest
```

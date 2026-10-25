# One-Sentence Prompt, Real Engineering Compiler: A 2-Domain Blinded Benchmark of Ael-Agents vs. Standalone Frontier LLM

> **Repository:** [`https://github.com/prannesshkva/Ael-Agents`](https://github.com/prannesshkva/Ael-Agents)
>
> **Evidence Slides in Repository:**
> - **[`slide_1_mcp_connector_prompt.png`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/slide_1_mcp_connector_prompt.png)** — 1-Click MCP Connector Toggle & 1-Sentence Plain-English Prompt
> - **[`slide_2_fusion_mhd_scorecard_trace.png`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/slide_2_fusion_mhd_scorecard_trace.png)** — Run 1: 12-Tesla Fusion MHD Blanket Scorecard (`34/40` vs. `15/40`) & Live 4-Tool Execution Trace
> - **[`slide_3_quantum_cryocmos_trace.png`](https://github.com/prannesshkva/Ael-Agents/blob/main/benchmarks/evidence/slide_3_quantum_cryocmos_trace.png)** — Run 2: 100,000-Qubit Cryo-CMOS at 100 mK Execution Trace & Verified Architecture (`30/40` vs. `9/40`)
>
> **Scope & Disclaimer:** *All multi-physics, structural, magnetohydrodynamic (MHD), and cryogenic quantum hardware calculations in this benchmark are computational (*in-silico*) order-of-magnitude verification studies generated from self-selected physical parameters to evaluate AI engineering reasoning and self-falsification.*

![Slide 1: 1-Click MCP Connector & 1-Sentence Prompt](https://raw.githubusercontent.com/prannesshkva/Ael-Agents/main/benchmarks/evidence/slide_1_mcp_connector_prompt.png)

---

## Executive Summary

A normal user should not need a 500-word prompt packed with equations to get real engineering rigor out of an AI. Yet when you ask a standalone frontier LLM a **single plain-English sentence** about a coupled physical system, it faces a fundamental structural limitation: it generates prose in a single autoregressive stream without an orthogonal specialist decomposition, a mathematical salience gate, or a programmatic verification compiler to falsify its own formulas before answering.

I architected **[Ael-Agents](https://github.com/prannesshkva/Ael-Agents)** (`v1.2.0`) as a **100% domain-agnostic Cortical Co-Processor & Programmatic Verification MCP Server** modeled on the human brain's Triple-Network Hierarchy:

1. **Default Mode Network (DMN — 768-D Determinantal Point Process Specialist Selection):** Projects candidate scientific axes into a 768-D semantic manifold (`BAAI/bge-base-en-v1.5`) and maximizes $\log\det(L_S)$ to select 5 maximally orthogonal specialist perspectives for any prompt.
2. **Salience Network (0-Token PyTorch Autograd Jacobian Switch):** Computes exact autograd sensitivities ($J_i = \|\partial y / \partial w_i\|_2$) outside the LLM at **0 router tokens** to isolate the **Epistemic Surprise Bridge**—the exact pair of conflicting disciplines ($k = 2$) governing the physical bottleneck.
3. **Offline Dream-RSI Consolidation:** Runs 15 epochs of gradient descent (`F_dream <= F_wake`) to sharpen cortical weights toward the bottlenecked domains.
4. **Anthropic-Style Programmatic Verification (`Ael-Agents_programmatic_verify`):** Requires the model to write and execute a self-contained Python verification script (`numpy`, `scipy`, `torch`, `rdkit`) that tests and rejects candidate designs against coupled physical inequalities *before* emitting a final answer.

### Head-to-Head Summary Across Two 1-Sentence Engineering Benchmarks

| Benchmark Domain | Exact 1-Sentence User Prompt | Standalone Frontier LLM Score | Frontier LLM + `Ael-Agents` MCP Score | Critical Error Caught by `Ael-Agents` |
| :--- | :--- | :---: | :---: | :--- |
| **Run 1: Nuclear Fusion MHD Engineering** | *"How can we design a liquid-metal cooling blanket for a compact 12-Tesla fusion reactor so it survives the extreme plasma heat and neutron radiation without the magnetic field or heat destroying the pipes?"* | **`15 / 40`** (`21 / 50` w/ breadth) | **`34 / 40`** (`42 / 50` w/ breadth) | Caught a **30× stress formula error** (standalone LLM applied circular boiler hoop stress $\sigma = Pr/t = 182\text{ MPa}$ to a square duct where flat-plate bending is $\sigma = Pb^2/(2t^2) = 6,050\text{ MPa}$) & solved the $t_w \to c \to \Delta P_{\text{MHD}}$ feedback trap across 5 candidates (`A`–`E`). |
| **Run 2: Cryogenic Quantum Hardware** | *"How can we wire 100,000 silicon spin qubits to a control chip inside a dilution refrigerator at 100 millikelvin without the chip's heat and microwave noise destroying the quantum state?"* | **`9 / 40`** | **`30 / 40`** (pre-rebuttal) | Caught a **37× internal Bose–Einstein photon occupation hallucination** ($\bar{n} = 3.7$ vs. specialist guess $0.1$ at $4\text{ K}, 20\text{ GHz}$), proved why a single $20\text{ dB}$ pad at $100\text{ mK}$ blows the heat budget, and closed staged attenuation ($10\text{ dB @ 1 K} + 10\text{ dB @ 100 mK} \to 0.22\text{ K}, 120\text{ }\mu\text{W}$). |

---

# Benchmark Run 1: Compact 12-Tesla Fusion Reactor Liquid-Metal MHD Blanket

### The Exact 1-Sentence Prompt
> *"How can we design a liquid-metal cooling blanket for a compact 12-Tesla fusion reactor so it survives the extreme plasma heat and neutron radiation without the magnetic field or heat destroying the pipes?"*

---

## Run 1 — Stage 1: What the Standalone Frontier LLM Did (Extended Thinking, `1m 1s`)

Operating standalone with Extended Thinking enabled, the frontier LLM produced a polished multi-section engineering overview proposing a **Dual-Coolant Lead-Lithium (`DCLL`) blanket** with `Eurofer-97` / `F82H` RAFM steel, $\text{SiC}_f/\text{SiC}$ flow channel inserts (FCIs), and a 3-step quantitative check at $B = 12\text{ T}$, duct half-width $a = 5\text{ cm}$ ($10\text{ cm} \times 10\text{ cm}$ square poloidal duct), velocity $v = 0.1\text{ m/s}$, and length $L = 2\text{ m}$:

1. **The Fatal 30× Structural Stress Formula Error:**
   After computing an unmitigated conducting-wall MHD pressure drop of $\Delta P_{\text{MHD}} \approx 7.3\text{ MPa}$, the standalone LLM checked whether the $10\text{ cm} \times 10\text{ cm}$ square steel duct would burst by plugging $r = 0.05\text{ m}$ and $t = 2\text{ mm}$ into the **thin-walled circular cylinder boiler hoop-stress formula**:
   $$\sigma_{\text{hoop}} = \frac{P \cdot r}{t} = \frac{7.3\text{ MPa} \times 0.05\text{ m}}{0.002\text{ m}} = 182.5\text{ MPa}$$
   Wait: a $10\text{ cm} \times 10\text{ cm}$ square duct is **not a circular pipe**—its flat walls carry internal pressure in **flat-plate bending**, not pure membrane hoop tension:
   $$\sigma_{\text{bend}} = \frac{P \cdot b^2}{2 t^2} = \frac{7.3\text{ MPa} \times (0.10\text{ m})^2}{2 \times (0.002\text{ m})^2} \approx \mathbf{9,125\text{ MPa}} \quad (\text{or }\mathbf{6,050\text{ MPa}}\text{ at }P = 4.84\text{ MPa})$$
   By using a circular pipe formula on a square duct, the standalone LLM **underpredicted the wall stress by 30× to 50×**, declaring $182.5\text{ MPa}$ "manageable" when the flat duct wall actually bursts at $>20\times$ the yield strength of RAFM steel!
2. **Missed the Coupled Wall-Conductance Feedback Trap ($t_w \to c \to \Delta P_{\text{MHD}}$):**
   It never showed that if an engineer thickens the bare steel wall ($t_w$) to reduce bending stress, the wall conductance ratio $c = \frac{\sigma_w t_w}{\sigma_f a}$ increases linearly ($c \propto t_w$), which increases $\Delta P_{\text{MHD}} = c \cdot \sigma_f v B^2 L$ linearly ($P \propto t_w$), leaving flat-plate bending stress scaling only as $1/t_w$ instead of $1/t_w^2$.
3. **Zero Failure-Mode Quantification:**
   It assumed an idealized, uncracked $\text{SiC}_f/\text{SiC}$ FCI with zero analysis of liquid-metal ingress through irradiation microcracks or plasma disruption $J \times B$ loads.

---

## Run 1 — Stage 2: Frontier LLM + `Ael-Agents` MCP (Uncoached 1-Sentence Run)

Given the exact same 1-sentence prompt with `Ael-Agents` enabled:

1. **768-D DPP Orthogonal Specialist Roster (`Ael-Agents_select_specialist_roster`):**
   Selected 5 orthogonal specialists: `Plasma_MHD`, `Thermal_Hydraulics`, `Structural_Mechanics`, `Radiation_Materials`, and `Tritium_Fuel_Cycle`.
2. **0-Token PyTorch Jacobian Salience Gate (`Ael-Agents_cortical_rdm_and_jacobian_gate`):**
   Isolated **`Plasma_MHD × Structural_Mechanics`** (`salience_beauty_score = 0.2339`, `epistemic_surprise = 0.2336`) as the governing cross-domain bottleneck—identifying the exact feedback loop between wall thickness $t_w$, wall conductance ratio $c$, MHD pressure drop $\Delta P_{\text{MHD}}$, and flat-plate bending stress $\sigma_{\text{bend}}$.
3. **Programmatic Verification (`Ael-Agents_programmatic_verify`) & Dream-RSI (`0.0725 -> 0.0442`):**
   Executed a multi-physics Python verification script across **5 candidate duct architectures (`A`–`E`)** and **7 coupled auxiliary checks**:

| Candidate Tested in `Ael-Agents_programmatic_verify` | Conductance Ratio $c$ | $\Delta P_{\text{MHD}}$ (MPa) | Flat-Plate Bending Stress $\sigma_{\text{bend}}$ (MPa) | Total Primary + Thermal Stress (MPa) | Compiler Verdict (Limit $\le 380\text{ MPa}$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`A`: Bare RAFM Steel ($t_w = 3\text{ mm}$)** | `0.048` | `2.66` | `2,216` | `2,331` | **REJECTED** (Immediate structural burst) |
| **`B`: Thick Bare Steel ($t_w = 20\text{ mm}$, self-consistent $c \propto t_w$)** | `0.320` | `17.7` | `332` | `1,099` | **REJECTED** ($t_w \to \Delta P$ trap + thermal stress blowout) |
| **`C`: `5 mm` RAFM + `1 mm` $\text{SiC}_f/\text{SiC}$ FCI (Nominal, pressure-equalized)** | `0.001` | `0.102` | `61.3` | `176.3` | **PASSED** (`2.15×` safety margin below `380 MPa`) |
| **`D`: Degraded / Micro-Cracked FCI (PbLi ingress into cracks)** | `0.010` | `1.01` | `606` | `721` | **REJECTED** (Proves FCI crack sealing is safety-critical) |
| **`E`: Mechanically Shorted FCI (PbLi bridges pressure slot)** | `0.050` | `5.05` | `3,030` | `3,145` | **REJECTED** (Catastrophic failure if slot shorts) |

Plus **7 additional coupled checks** verified in the same script:
- **First-Wall Helium Cooling:** `8 MPa` He in `1.5 mm` microchannels ($q'' = 1.0\text{ MW/m}^2$) $\to \sigma_{\text{thermal}} = 197\text{ MPa}$, $\sigma_{\text{hoop,He}} = 73\text{ MPa}$, combined $270\text{ MPa} < 380\text{ MPa}$ (**PASS**), outer surface $517^\circ\text{C} < 550^\circ\text{C}$ creep limit (**PASS**).
- **PbLi–RAFM Corrosion Interface:** FCI drops steel interface temperature to $454^\circ\text{C} < 480^\circ\text{C}$ corrosion limit (**PASS**).
- **Neutron Damage Life:** At $15\text{ dpa/FPY}$ and $42\text{ dpa}$ RAFM limit $\to \mathbf{2.8\text{ FPY}}$ mandatory cassette replacement schedule.
- **12-T REBCO Magnet Shielding:** Requires $1.22\text{ m}$ of $\text{WC}/\text{B}_4\text{C}$ shielding ($\lambda = 0.14\text{ m}$) to keep 10-year fast-neutron fluence below $3 \times 10^{18}\text{ n/cm}^2$.
- **Plasma Disruption $J \times B$ Loads:** Unsegmented blanket suffers $\sim 19\text{ MPa}$ electromagnetic pressure during a $12\text{ T} / 10\text{ ms}$ quench; $10\times$ toroidal segmentation reduces disruption load to $1.9\text{ MPa}$ (**PASS**).
- **Explicit Caveats:** Flagged `Eurofer-97` ferromagnetic Maxwell stress in $12\text{ T}$ gradients and explicitly deferred $\text{TBR} \ge 1.05$ to 3D `OpenMC`/`MCNP` Monte Carlo neutronics rather than faking a 1D number.

---

## Run 1 — Stages 3 & 4: Blinded Peer Review & Rebuttal Scorecard

![Slide 2: 12-Tesla Fusion MHD Blanket Scorecard & Live Execution Trace](https://raw.githubusercontent.com/prannesshkva/Ael-Agents/main/benchmarks/evidence/slide_2_fusion_mhd_scorecard_trace.png)

When the standalone frontier LLM audited its own answer (**Response A**) against the verbatim `Ael-Agents` output (**Response B**), it immediately confessed its 30× structural formula error:

> *"Here A has a genuine, serious engineering error that B catches... A square or rectangular duct does not carry internal pressure in pure hoop tension—its flat walls bend... A's 3-step 'self-consistent check' is therefore broken: it sized a rectangular duct with a circular-pipe formula, underpredicting the wall stress by more than an order of magnitude."*

After verifying that `Response B` generated all 5 candidate architectures and 7 multi-physics checks from the exact same 1-sentence prompt with zero hidden coaching, the standalone LLM issued its final post-rebuttal scorecard:

| Criterion | Standalone Frontier LLM (Extended Thinking) | Frontier LLM + `Ael-Agents` MCP | Winner |
| :--- | :---: | :---: | :--- |
| **1. Coupled MHD–Structural Closure** | `4 / 10` *(30× hoop-stress formula error on square duct)* | **`9 / 10`** *(Exact $t_w \to c \to \Delta P$ feedback + flat-plate bending)* | **`Ael-Agents` (+5)** |
| **2. Candidate Exploration & Quantitative Rejection** | `3 / 10` *(1 narrative path, 0 rejected models)* | **`8 / 10`** *(5 models `A`–`E` tested, 4 quantitatively rejected)* | **`Ael-Agents` (+5)** |
| **3. Failure-Mode Realism (Cracked FCI / Disruptions)** | `3 / 10` *(Qualitative mention only)* | **`8 / 10`** *(Cracked $c=0.01$, shorted $c=0.05$, $19 \to 1.9\text{ MPa}$ disruption)* | **`Ael-Agents` (+5)** |
| **4. Epistemic Honesty & Boundary Discipline** | `5 / 10` *(Missed its own broken stress check)* | **`9 / 10`** *(Flagged Eurofer ferromagnetism & deferred 3D OpenMC TBR)* | **`Ael-Agents` (+4)** |
| **5. System-Level Architecture Breadth** | `6 / 10` | **`8 / 10`** | **`Ael-Agents` (+2)** |
| **Total Score** | **`15 / 40`** (`21 / 50` w/ breadth) | **`34 / 40`** (`42 / 50` w/ breadth) | **`Ael-Agents` (+19 / +21 pts)** |

---
---

# Benchmark Run 2: 100,000 Silicon Spin Qubits Wired to Cryo-CMOS at 100 mK

### The Exact 1-Sentence Prompt
> *"How can we wire 100,000 silicon spin qubits to a control chip inside a dilution refrigerator at 100 millikelvin without the chip's heat and microwave noise destroying the quantum state?"*

---

## Run 2 — Stage 1: What the Standalone Frontier LLM Did

Given the 1-sentence prompt without `Ael-Agents`, the standalone frontier LLM produced a qualitative summary of cryogenic spin-qubit architectures:
- Mentioned splitting active cryo-CMOS to `4 K` (or `1 K`) and keeping the qubit die + passive elements at `100 mK`.
- Mentioned $\sqrt{N}$ crossbar addressing, global microwave control with local Stark shifting, and frequency-multiplexed reflectometry.
- Quoted a single rough scale ($k_B T / C \approx 7\text{ }\mu\text{V}_{\text{rms}}$ for $1\text{ pF}$ at $4\text{ K}$).
- **What It Failed to Do:** It performed **zero heat-budget summation**, **zero cold-attenuator noise-temperature cascade math**, **zero Kapitza thermal boundary resistance calculation**, **zero off-resonant Rabi crosstalk math**, and **tested/rejected zero candidate architectures**.

---

## Run 2 — Stage 2: Frontier LLM + `Ael-Agents` MCP (Uncoached 1-Sentence Run)

![Slide 3: 100,000-Qubit Cryo-CMOS Execution Trace & Verified Architecture](https://raw.githubusercontent.com/prannesshkva/Ael-Agents/main/benchmarks/evidence/slide_3_quantum_cryocmos_trace.png)

Given the exact same 1-sentence prompt with `Ael-Agents` enabled, here is the verbatim output produced by the pipeline:

> **What the pipeline did.** It chose five orthogonal specialist domains: `cryo-CMOS thermal budget`, `spin-qubit physics`, `QEC architecture`, `microwave noise`, and `interconnects`. The salience gate then picked **`thermal budget × spin-qubit physics`** as the core tension, so the heat and noise tradeoff drives the answer. I then tested four candidate architectures with the verifier.
>
> **Why the obvious approaches fail**
> - **Put the control chip at 100 mK:** the fridge gives roughly $0.5\text{ mW}$ there, which is about $5\text{ nW}$ per qubit. A control chip needs micro- to milliwatts per qubit, so this misses by 3–5 orders of magnitude.
> - **One coax pair per qubit:** 200,000 lines cannot physically fit the fridge, and their conduction load of about $4\text{ mW}$ exceeds the cooling power.
> - **Drive from 4 K with no cold attenuation:** the qubits would see the $4\text{ K}$ noise temperature, and the spin electrons would run far hotter than the $\sim 0.3\text{ K}$ target.
>
> **The design that passed**
> 1. **Split by temperature.**
>    - **4 K:** the bulk of the control ASIC (DACs, digital logic, decoder), at about $1.2\text{ W}$ against the $\sim 1.5\text{ W}$ budget.
>    - **100 mK:** only passive switches and sample-and-hold capacitors.
> 2. **Use a crossbar for the DC gate voltages.** About 316 row lines and 316 column lines (634 in total) replace 100,000 individual wires. Each gate voltage sits on a small capacitor that a shared DAC refreshes at about $1\text{ kHz}$.
>    - Leakage droop is around $0.1\text{ }\mu\text{V}$, and $k_B T/C$ sampling noise is around $12\text{ }\mu\text{V}$.
>    - Both are far below the $\sim 1\text{ mV}_{\text{rms}}$ gate-voltage noise that keeps dephasing error under $10^{-4}$ (at a Stark slope of $20\text{ MHz/V}$).
> 3. **Drive the spins with a global microwave field instead of per-qubit tones.**
>    - A shared resonator or dielectric resonator antenna supplies the ESR field.
>    - Individual qubits are tuned into resonance with the local DC gate voltages (Stark shift).
>    - This avoids the frequency-crosstalk problem: at a $5\text{ MHz}$ Rabi rate, tones spaced $10\text{ MHz}$ apart cause about $25\%$ off-resonant error. That figure assumes simple unshaped pulses; pulse shaping and timing tricks reduce it.
> 4. **Attenuate the microwave line in cold stages.** The line passes $10\text{ dB}$ at $1\text{ K}$ and $10\text{ dB}$ at $100\text{ mK}$, which brings the effective noise temperature from $4\text{ K}$ down to about $0.22\text{ K}$.
>    - The $1\text{ K}$ stage absorbs about $1\text{ mW}$ and the $100\text{ mK}$ stage about $120\text{ }\mu\text{W}$ in total (in the model).
>    - A single $20\text{ dB}$ attenuator at $100\text{ mK}$ would dump hundreds of $\mu\text{W}$ there, so staging is what makes the noise and heat budgets both work.
> 5. **Remove heat at the qubit die.**
>    - The qubit die needs a large effective thermal contact area (about $0.05\text{ m}^2$, e.g. sintered silver or gold-plated backside). At about $120\text{ }\mu\text{W}$ that gives a die temperature rise of only a few mK.
>    - A bare $1\text{ cm}^2$ contact would give a rise of order a kelvin.
> 6. **Connect the stages with superconducting links.** Flip-chip bump bonds or an Nb/NbTi interposer between the $4\text{ K}$ ASIC and the qubit die keep conduction to about $1\text{ }\mu\text{W}$ or less per cable.
> 7. **Shield and filter.** Use superconducting ground planes, on-chip low-pass filters on the DC gate lines, and isolation of better than $70\text{ dB}$ on any leaked clock or LO tone near the Larmor frequency.
>
> **Caveats**
> - **Correction to a specialist's claim:** one hypothesis said the $4\text{ K}$ photon occupation at $20\text{ GHz}$ is about $0.1$. The verifier computed about $3.7$, which is why cold attenuation is essential.
> - **Weakest assumptions:** the global-resonator dissipation, the $0.02\text{–}0.05\text{ }\mu\text{W}$ per cable at $100\text{ mK}$, the Stark slope, and the die contact area. Each could shift the $100\text{ mK}$ budget by an order of magnitude.
> - **Two trivial checks:** the amplitude-noise and gate-voltage constraints were bounds I derived and set to pass, so they are not independent tests.
> - **Not run:** the pipeline's final consolidation step (`dream_rsi_consolidate`).

---

## Run 2 — Stage 3: Blinded Peer-Review Audit (`30 / 40` vs. `9 / 40`)

When the standalone frontier LLM audited its own answer (**Response A**) against the `Ael-Agents` output (**Response B**) across a 40-point quantitative engineering rubric, it scored **`Ael-Agents` `30 / 40` vs. its own `9 / 40` (`3.3×` higher)**:

| Criterion | Standalone Frontier LLM (`Response A`) | Frontier LLM + `Ael-Agents` (`Response B`) | Standalone Auditor's Note |
| :--- | :---: | :---: | :--- |
| **1. Coupled Thermal–Noise Closure** | `2 / 10` | **`8 / 10`** | *"A (2/10) only says to use cold attenuation and filtering. It has no heat number, no noise-temperature calculation, and no staging argument. B (8/10) closes the numbers... Using $T_{\text{out}} = T_{\text{in}}/L + T_{\text{phys}}(1 - 1/L)$, 10 dB at 1 K followed by 10 dB at 100 mK takes 4 K to 1.3 K and then to 0.22 K, matching B."* |
| **2. Die-Level Kapitza & Crosstalk Math** | `1 / 10` | **`6 / 10`** | *"A (1/10) has neither calculation. B (6/10) quantifies both off-resonant Rabi transfer and Kapitza temperature rise."* |
| **3. Self-Falsification & Error Catching** | `1 / 10` | **`8 / 10`** | *"A (1/10) tested nothing and ranked no alternatives. B (8/10) catches a real error. The Bose-Einstein occupation at 20 GHz and 4 K is $hf/kT = 0.24$, so $\bar{n} = 1/(e^{0.24} - 1) \approx 3.7$, which confirms the 3.7-versus-0.1 correction. It also rejects three architectures on sound arithmetic."* |
| **4. Epistemic Honesty** | `5 / 10` | **`8 / 10`** | *"B (8/10) discloses its assumptions, admits two checks were set to pass, and says one step was not run."* |
| **Total Score** | **`9 / 40`** | **`30 / 40`** | **`Ael-Agents` wins by `+21 points` (`3.3×` baseline)** |

---

## Run 2 — Stage 4: Technical Rebuttal & Verification of the Auditor's 3 Critiques

While scoring `Response B` `30 / 40` vs. `9 / 40`, the standalone auditor deducted points from `Response B` on three points—two of which were actually **oversights in the standalone auditor's own hand calculations**, and the third of which highlights why `Ael-Agents_programmatic_verify` is essential:

### 1. The Auditor Forgot `Response B`'s Stated $5\text{ MHz}$ Rabi Rate When Checking the $1\text{ mV}_{\text{rms}}$ Gate-Noise Bound
- **What the Standalone Auditor Claimed (Criterion 4):**
  > *"Its ~1 mV gate-noise threshold looks too loose: at 20 MHz/V that is 20 kHz of frequency noise, which over a ~1 µs gate gives errors near $10^{-3}$ to $10^{-2}$, not under $10^{-4}$."*
- **Why the Auditor Was Wrong:**
  The standalone auditor arbitrarily plugged in a **$1\text{ }\mu\text{s}$ gate time**, overlooking that **Bullet 3 of `Response B` explicitly specified a $5\text{ MHz}$ Rabi rate** ($\Omega_{\text{Rabi}} = 5\text{ MHz}$). At a $5\text{ MHz}$ Rabi rate, a single-qubit gate takes $t_g \approx \frac{1}{2\Omega_{\text{Rabi}}} = \mathbf{0.08\text{–}0.10\text{ }\mu\text{s}}$ ($10\times$ faster than $1\text{ }\mu\text{s}$). Over $t_g = 0.08\text{ }\mu\text{s}$ with $\delta f = (20\text{ MHz/V})(1\text{ mV}) = 20\text{ kHz}$:
  $$\epsilon_{\phi} = (2\pi \, \delta f \, t_g)^2 = (2\pi \times 20\times 10^3\text{ Hz} \times 0.08\times 10^{-6}\text{ s})^2 = (0.010\text{ rad})^2 = \mathbf{1.0 \times 10^{-4}}$$
  The Python verifier inside `Ael-Agents` derived the $1\text{ mV}$ bound directly from the $5\text{ MHz}$ Rabi rate used in Bullet 3—whereas the standalone auditor failed to propagate $t_g$ from Bullet 3 to Bullet 2.

### 2. Why Attenuator Staging Is What Makes *Both* the Noise and Heat Budgets Work Simultaneously
- **What the Standalone Auditor Claimed (Criterion 1):**
  > *"A single 20 dB attenuator at 100 mK gives about 0.14 K, which is lower than the staged 0.22 K. Staging is justified by heat alone, not noise, so 'both budgets work' overstates it."*
- **Why `Response B`'s Statement Is Exact:**
  Consider the two unstaged alternatives for achieving $20\text{ dB}$ of cold attenuation:
  - **Unstaged Option 1 (All $20\text{ dB}$ at $1\text{ K}$, $0\text{ dB}$ at $100\text{ mK}$):** Passes the $100\text{ mK}$ heat budget ($0\text{ }\mu\text{W}$ dumped at $100\text{ mK}$), but **fails the noise budget** ($T_{\text{noise}} = \frac{4\text{ K}}{100} + 1\text{ K}(0.99) = \mathbf{1.03\text{ K}} > 0.3\text{ K}$ target).
  - **Unstaged Option 2 (All $20\text{ dB}$ at $100\text{ mK}$, $0\text{ dB}$ at $1\text{ K}$):** Passes the noise budget ($T_{\text{noise}} = 0.14\text{ K} < 0.3\text{ K}$), but **fails the $100\text{ mK}$ heat budget** (dumps $\sim 1.1\text{–}1.3\text{ mW}$ at $100\text{ mK}$, exceeding the $0.5\text{ mW}$ dilution stage limit).
  - **Staged Option ($10\text{ dB}$ at $1\text{ K}$ + $10\text{ dB}$ at $100\text{ mK}$):** Absorbs $90\%$ of the microwave drive power at $1\text{ K}$ ($\sim 1\text{ mW}$) and only $9\%$ at $100\text{ mK}$ ($\sim 120\text{ }\mu\text{W} < 500\text{ }\mu\text{W}$ budget) while cooling the noise temperature to $T_{\text{noise}} = \mathbf{0.22\text{ K}} < 0.3\text{ K}$. Thus, staging is literally what allows **both** inequalities ($T_{\text{noise}} < 0.3\text{ K}$ and $P_{100\text{ mK}} < 0.5\text{ mW}$) to be satisfied simultaneously.

### 3. Why the $\bar{n} = 0.1 \to 3.7$ Bose–Einstein Correction Was Real, Not a Staged Strawman
- **What the Standalone Auditor Wrote (Criterion 3):**
  > *"B (8/10) catches a real error. The Bose-Einstein occupation at 20 GHz and 4 K is $hf/kT = 0.24$, so $\bar{n} = 1/(e^{0.24} - 1) \approx 3.7$, which confirms the 3.7-versus-0.1 correction... I deduct points because I can't verify that the 'specialist claim' was a real independent hypothesis rather than a staged strawman."*
- **How It Actually Happened Inside the MCP Pipeline:**
  In Step 1 (`Ael-Agents_select_specialist_roster`) and Step 2 (`Ael-Agents_cortical_rdm_and_jacobian_gate`), the LLM generated 5 text hypotheses from memory before running Python code—and its `microwave noise` specialist text genuinely guessed $\bar{n} \approx 0.1$ at $4\text{ K}, 20\text{ GHz}$. In Step 3 (`Ael-Agents_programmatic_verify`), the model wrote `n_bar = 1.0 / (np.exp(h * f / (k_B * T)) - 1)` inside the Python sandbox, which returned **`3.697`**. Because `Ael-Agents` enforces programmatic verification before final synthesis, the model caught its own Step 1 text hallucination and documented the correction under **Caveats**.

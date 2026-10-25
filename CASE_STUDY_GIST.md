<h1 align="center">One-Sentence Prompt, Real Engineering Compiler</h1>
<h3 align="center">A 2-Domain Blinded Benchmark of <code>Ael-Agents (v1.2.0)</code> vs. Standalone Frontier LLM</h3>

<p align="center">
  <a href="https://github.com/prannesshkva/Ael-Agents"><img src="https://img.shields.io/badge/GitHub-prannesshkva%2FAel--Agents-181717?style=for-the-badge&logo=github" alt="GitHub Repo"/></a>
  <a href="https://doi.org/10.17605/OSF.IO/7R46Z"><img src="https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2F7R46Z-0A66C2?style=for-the-badge" alt="DOI"/></a>
  <img src="https://img.shields.io/badge/PyTorch-0--Token_Jacobian_Gate-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MCP-Claude_%7C_Cursor_%7C_Windsurf-D97757?style=for-the-badge" alt="MCP"/>
</p>

> **Scope & Disclaimer:** *All multi-physics, structural, magnetohydrodynamic (MHD), and cryogenic quantum hardware calculations in this benchmark are computational (*in-silico*) order-of-magnitude verification studies generated from self-selected physical parameters to evaluate AI engineering reasoning and self-falsification.*

---

## Executive Summary

A normal user should not need a 500-word prompt packed with equations to get real engineering rigor out of an AI. Yet when you ask a standalone frontier LLM a **single plain-English sentence** about a coupled physical system, it generates prose in a single autoregressive stream—without orthogonal specialist decomposition, mathematical working-memory gating, or a programmatic verification compiler to test its own formulas before answering.

**[Ael-Agents](https://github.com/prannesshkva/Ael-Agents)** (`v1.2.0`) is a **100% domain-agnostic, trainable PyTorch Cortical Co-Processor & Programmatic Verification MCP Server** modeled on the human brain's Triple-Network Hierarchy:

| Cortical Stage | Mathematical Mechanism | Token & Engineering Impact |
| :--- | :--- | :--- |
| **1. Default Mode Network (DMN)** | **768-D Determinantal Point Process (`log det L_S`)** over `BAAI/bge-base-en-v1.5` | Automatically selects `N = 5` maximally orthogonal scientific specialists from a 1-sentence prompt |
| **2. Salience Network (SN)** | **Exact PyTorch Autograd Jacobian (`J_i = \|\|∂y/∂w_i\|\|_2`)** | Isolates the `k = 2` conflicting bottleneck domains at **0 LLM router tokens** (`-60%` context bloat) |
| **3. Programmatic Verifier (CEN)** | **Sandboxed Python / SciPy / PyTorch / RDKit Compiler (`Ael-Agents_programmatic_verify`)** | Writes and executes verification code to **test and reject failing designs** before synthesis |
| **4. Offline Dream-RSI** | **15-Epoch AdamW Consolidation (`F_dream <= F_wake`)** | Trains 1-to-1 specialist weights `w_1 ... w_N` on the fly to lock in verified domain bridges |

---

## Head-to-Head Scorecard Across Both 1-Sentence Benchmarks

Both benchmarks below used a **single plain-English sentence** with **zero equations or hints in the prompt**, followed by a blinded peer-review audit where the standalone Frontier LLM graded both responses:

| Benchmark Domain | Exact 1-Sentence Prompt | Standalone Frontier LLM | Frontier LLM + `Ael-Agents` | Capability Lift & Critical Error Caught |
| :--- | :--- | :---: | :---: | :--- |
| **Run 1: Nuclear Fusion MHD Engineering** | *"How can we design a liquid-metal cooling blanket for a compact 12-Tesla fusion reactor so it survives the extreme plasma heat and neutron radiation without the magnetic field or heat destroying the pipes?"* | **`15 / 40`**<br/>*(`21 / 50` w/ breadth)* | **`34 / 40`**<br/>*(`42 / 50` w/ breadth)* | **`2.3×` Score Lift (`+19 pts`)**<br/>Caught a **30× stress formula error** (circular boiler hoop stress `182 MPa` applied to a square duct where flat-plate bending is `6,050 MPa`) & solved the `t_w → c → ΔP_MHD` feedback trap across 5 candidates (`A`–`E`). |
| **Run 2: Cryogenic Quantum Hardware** | *"How can we wire 100,000 silicon spin qubits to a control chip inside a dilution refrigerator at 100 millikelvin without the chip's heat and microwave noise destroying the quantum state?"* | **`9 / 40`** | **`30 / 40`**<br/>*(pre-rebuttal)* | **`3.3×` Score Lift (`+21 pts`)**<br/>Caught a **37× internal Bose–Einstein photon error** (`n̄ = 3.7` vs. specialist guess `0.1` at `4 K, 20 GHz`), proved a single `20 dB` pad at `100 mK` blows the heat budget, and closed staged attenuation (`10 dB @ 1 K + 10 dB @ 100 mK → 0.22 K, 120 µW`). |

---

# Benchmark Run 1: Compact 12-Tesla Fusion Reactor Liquid-Metal MHD Blanket

### The Exact 1-Sentence Prompt
```text
How can we design a liquid-metal cooling blanket for a compact 12-Tesla fusion reactor so it survives the extreme plasma heat and neutron radiation without the magnetic field or heat destroying the pipes?
```

---

## Run 1 — Stage 1: Standalone Frontier LLM (Extended Thinking, `1m 1s`)

Operating standalone with Extended Thinking enabled, the frontier LLM proposed a **Dual-Coolant Lead-Lithium (`DCLL`) blanket** with `Eurofer-97` / `F82H` RAFM steel, `SiC_f/SiC` flow channel inserts (FCIs), and a 3-step calculation at `B = 12 T`, duct half-width `a = 5 cm` (`10 cm × 10 cm` square poloidal duct), velocity `v = 0.1 m/s`, and length `L = 2 m`:

1. **The Fatal 30× Structural Stress Formula Error:**  
   After computing an unmitigated conducting-wall MHD pressure drop of `ΔP_MHD ≈ 7.3 MPa`, the standalone LLM checked whether the `10 cm × 10 cm` square steel duct would burst by plugging `r = 0.05 m` and `t = 2 mm` into the **thin-walled circular cylinder boiler hoop-stress formula**:
   ```text
   Standalone LLM Formula (Circular Pipe Hoop Stress):
   σ_hoop = (P × r) / t = (7.3 MPa × 0.05 m) / 0.002 m = 182.5 MPa  (Claimed "manageable")

   Actual Physics for a 10 cm × 10 cm Square Duct (Flat-Plate Bending Stress):
   σ_bend = (P × b²) / (2 × t²) = (7.3 MPa × 0.10² m²) / (2 × 0.002² m²) = 9,125 MPa  (6,050 MPa at P = 4.84 MPa)
   ```
   A square duct does **not** carry internal pressure in pure hoop tension—its flat walls bend. By using a circular boiler formula on a square duct, the standalone LLM **underpredicted the wall stress by 30× to 50×**, declaring `182.5 MPa` safe when the flat wall actually bursts at **>20× the yield strength of RAFM steel**.

2. **Missed the Coupled Wall-Conductance Feedback Trap (`t_w → c → ΔP_MHD`):**  
   Thickening a bare steel wall (`t_w`) increases the wall conductance ratio `c = (σ_w · t_w) / (σ_f · a)` linearly (`c ∝ t_w`), which increases `ΔP_MHD = c · σ_f · v · B² · L` linearly (`P ∝ t_w`), leaving flat-plate bending stress scaling only as `1 / t_w` instead of `1 / t_w²`.

3. **Zero Failure-Mode Quantification:**  
   Assumed an idealized, uncracked `SiC_f/SiC` insert with zero calculation of liquid-metal ingress through irradiation microcracks or plasma disruption `J × B` loads.

---

## Run 1 — Stage 2: Frontier LLM + `Ael-Agents` MCP (Uncoached 1-Sentence Run)

Connected to `Ael-Agents` on the exact same 1-sentence prompt:
- **768-D DPP Specialist Roster:** Automatically selected `Plasma_MHD`, `Thermal_Hydraulics`, `Structural_Mechanics`, `Radiation_Materials`, and `Tritium_Fuel_Cycle`.
- **0-Token PyTorch Jacobian Gate:** Isolated **`Plasma_MHD × Structural_Mechanics`** (`salience_beauty_score = 0.2339`, `epistemic_surprise = 0.2336`) and reduced Dream-RSI free energy from **`0.0725 → 0.0442`**.
- **Programmatic Verification (`Ael-Agents_programmatic_verify`):** Compiled and tested **5 candidate duct architectures (`A`–`E`)** plus **7 coupled multi-physics checks** in Python before answering:

| Candidate Architecture Tested in Python | Conductance Ratio `c` | `ΔP_MHD` (MPa) | Flat-Plate Bending `σ_bend` (MPa) | Total Primary + Thermal Stress (MPa) | Compiler Verdict (`Limit <= 380 MPa`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`A`: Bare RAFM Steel (`t_w = 3 mm`)** | `0.048` | `2.66` | `2,216` | `2,331` | **REJECTED** — Immediate structural burst |
| **`B`: Thick Bare Steel (`t_w = 20 mm`, self-consistent `c ∝ t_w`)** | `0.320` | `17.7` | `332` | `1,099` | **REJECTED** — `t_w → ΔP` feedback + thermal blowout |
| **`C`: `5 mm` RAFM + `1 mm` `SiC_f/SiC` FCI (Nominal, pressure-equalized)** | `0.001` | `0.102` | `61.3` | `176.3` | **PASSED** — `2.15×` safety margin below `380 MPa` |
| **`D`: Degraded / Micro-Cracked FCI (PbLi ingress into cracks)** | `0.010` | `1.01` | `606` | `721` | **REJECTED** — Proves crack sealing is safety-critical |
| **`E`: Mechanically Shorted FCI (PbLi bridges pressure slot)** | `0.050` | `5.05` | `3,030` | `3,145` | **REJECTED** — Catastrophic failure if slot shorts |

**7 Coupled Multi-Physics Checks Verified in the Same Script:**
1. **First-Wall Helium Cooling:** `8 MPa` He in `1.5 mm` microchannels (`q'' = 1.0 MW/m²`) → `σ_thermal = 197 MPa`, `σ_hoop,He = 73 MPa`, combined **`270 MPa < 380 MPa` (PASS)**, peak steel temperature **`517 °C < 550 °C` creep limit (PASS)**.
2. **PbLi–RAFM Corrosion Interface:** FCI thermal drop keeps steel wetting interface at **`454 °C < 480 °C` limit (PASS)**.
3. **Neutron Displacement Damage:** At `15 dpa/FPY` and `42 dpa` RAFM limit → **`2.8 FPY` mandatory replacement schedule**.
4. **12-T REBCO Magnet Shielding:** **`1.22 m` of `WC / B4C` shielding** (`λ = 0.14 m`) required to hold 10-year fast-neutron fluence below `3 × 10¹⁸ n/cm²`.
5. **Plasma Disruption `J × B` Loads:** Unsegmented blanket suffers `~19 MPa` electromagnetic pressure during a `12 T / 10 ms` quench; **`10×` toroidal segmentation reduces load to `1.9 MPa` (PASS)**.
6. **Ferromagnetic & Neutronics Caveats:** Proactively flagged `Eurofer-97` ferromagnetic Maxwell forces in `12 T` gradients and deferred `TBR >= 1.05` to 3D `OpenMC`/`MCNP` Monte Carlo neutronics rather than inventing a 1D estimate.

---

## Run 1 — Stages 3 & 4: Blinded Peer Review & Rebuttal Scorecard

When the standalone frontier LLM audited its own answer (**Response A**) against the verbatim `Ael-Agents` output (**Response B**), it immediately confessed its 30× formula error:

> *"Here A has a genuine, serious engineering error that B catches... A square or rectangular duct does not carry internal pressure in pure hoop tension—its flat walls bend... A's 3-step 'self-consistent check' is therefore broken: it sized a rectangular duct with a circular-pipe formula, underpredicting the wall stress by more than an order of magnitude."*

| Evaluation Criterion (`0–10` Scale) | Standalone Frontier LLM (`Response A`) | Frontier LLM + `Ael-Agents` (`Response B`) | Capability Gain |
| :--- | :---: | :---: | :--- |
| **1. Coupled Physics Discovery (`t_w ↔ c ↔ ΔP`, FCI, Disruption)** | `4 / 10` | **`9 / 10`** | **`+5 pts`** *(Solved `t_w → c → ΔP` feedback loop)* |
| **2. Structural & Failure-Mode Rigor (Flat-Plate Bending & Cracked FCI)** | `3 / 10` | **`8 / 10`** | **`+5 pts`** *(Caught 30× hoop-stress formula error)* |
| **3. Quantitative Falsification Depth (Candidates Compiled & Rejected)** | `3 / 10` | **`8 / 10`** | **`+5 pts`** *(5 models `A`–`E` + 7 coupled checks)* |
| **4. Epistemic Honesty (Disclosing Open Risks & 3D TBR Boundary)** | `5 / 10` | **`9 / 10`** | **`+4 pts`** *(Proactive risk section & 3D OpenMC boundary)* |
| **Core Engineering Subtotal** | **`15 / 40`** | **`34 / 40`** | **`+19 pts` (`2.3×` Baseline)** |
| **5. Supplementary: System-Level Architectural Breadth** | `6 / 10` | **`8 / 10`** | **`+2 pts`** *(Full breeding/tritium/corrosion coverage)* |
| **Grand Total (With Breadth)** | **`21 / 50`** | **`42 / 50`** | **`+21 pts` (`2.0×` Overall)** |

---
---

# Benchmark Run 2: 100,000 Silicon Spin Qubits Wired to Cryo-CMOS at 100 mK

### The Exact 1-Sentence Prompt
```text
How can we wire 100,000 silicon spin qubits to a control chip inside a dilution refrigerator at 100 millikelvin without the chip's heat and microwave noise destroying the quantum state?
```

---

## Run 2 — Stage 1: Standalone Frontier LLM Baseline

Given the 1-sentence prompt without `Ael-Agents`, the standalone frontier LLM produced a qualitative summary:
- Mentioned splitting active cryo-CMOS to `4 K` (or `1 K`) and keeping the qubit die + passive elements at `100 mK`.
- Mentioned `√N` crossbar addressing, global microwave control with local Stark shifting, and frequency-multiplexed reflectometry.
- Quoted a single rough scale (`k_B·T / C ≈ 7 µV_rms` for `1 pF` at `4 K`).
- **What It Missed:** Zero heat-budget summation, zero cold-attenuator noise-temperature cascade equations, zero Kapitza thermal contact-area math, zero off-resonant Rabi crosstalk calculation, and zero candidate rejection.

---

## Run 2 — Stage 2: Frontier LLM + `Ael-Agents` MCP (Uncoached 1-Sentence Run)

Given the exact same 1-sentence prompt with `Ael-Agents` enabled, here is the verbatim output produced by the pipeline:

> **What the pipeline did.** It chose five orthogonal specialist domains: `cryo-CMOS thermal budget`, `spin-qubit physics`, `QEC architecture`, `microwave noise`, and `interconnects`. The salience gate then picked **`thermal budget × spin-qubit physics`** as the core tension, so the heat and noise tradeoff drives the answer. I then tested four candidate architectures with the verifier.
>
> **Why the obvious approaches fail**
> - **Put the control chip at 100 mK:** the fridge gives roughly `0.5 mW` there, which is about `5 nW` per qubit. A control chip needs micro- to milliwatts per qubit, so this misses by 3–5 orders of magnitude.
> - **One coax pair per qubit:** 200,000 lines cannot physically fit the fridge, and their conduction load of about `4 mW` exceeds the cooling power.
> - **Drive from 4 K with no cold attenuation:** the qubits would see the `4 K` noise temperature, and the spin electrons would run far hotter than the `~0.3 K` target.
>
> **The design that passed**
> 1. **Split by temperature.**
>    - **4 K:** the bulk of the control ASIC (DACs, digital logic, decoder), at about `1.2 W` against the `~1.5 W` budget.
>    - **100 mK:** only passive switches and sample-and-hold capacitors.
> 2. **Use a crossbar for the DC gate voltages.** About 316 row lines and 316 column lines (634 in total) replace 100,000 individual wires. Each gate voltage sits on a small capacitor that a shared DAC refreshes at about `1 kHz`.
>    - Leakage droop is around `0.1 µV`, and `kT/C` sampling noise is around `12 µV`.
>    - Both are far below the `~1 mV rms` gate-voltage noise that keeps dephasing error under `1e-4` (at a Stark slope of `20 MHz/V`).
> 3. **Drive the spins with a global microwave field instead of per-qubit tones.**
>    - A shared resonator or dielectric resonator antenna supplies the ESR field.
>    - Individual qubits are tuned into resonance with the local DC gate voltages (Stark shift).
>    - This avoids the frequency-crosstalk problem: at a `5 MHz` Rabi rate, tones spaced `10 MHz` apart cause about `25%` off-resonant error. That figure assumes simple unshaped pulses; pulse shaping and timing tricks reduce it.
> 4. **Attenuate the microwave line in cold stages.** The line passes `10 dB` at `1 K` and `10 dB` at `100 mK`, which brings the effective noise temperature from `4 K` down to about `0.22 K`.
>    - The `1 K` stage absorbs about `1 mW` and the `100 mK` stage about `120 µW` in total (in the model).
>    - A single `20 dB` attenuator at `100 mK` would dump hundreds of `µW` there, so staging is what makes the noise and heat budgets both work.
> 5. **Remove heat at the qubit die.**
>    - The qubit die needs a large effective thermal contact area (about `0.05 m²`, e.g. sintered silver or gold-plated backside). At about `120 µW` that gives a die temperature rise of only a few mK.
>    - A bare `1 cm²` contact would give a rise of order a kelvin.
> 6. **Connect the stages with superconducting links.** Flip-chip bump bonds or an Nb/NbTi interposer between the `4 K` ASIC and the qubit die keep conduction to about `1 µW` or less per cable.
> 7. **Shield and filter.** Use superconducting ground planes, on-chip low-pass filters on the DC gate lines, and isolation of better than `70 dB` on any leaked clock or LO tone near the Larmor frequency.
>
> **Caveats**
> - **Correction to a specialist's claim:** one hypothesis said the `4 K` photon occupation at `20 GHz` is about `0.1`. The verifier computed about `3.7`, which is why cold attenuation is essential.
> - **Weakest assumptions:** the global-resonator dissipation, the `0.02–0.05 µW` per cable at `100 mK`, the Stark slope, and the die contact area. Each could shift the `100 mK` budget by an order of magnitude.
> - **Two trivial checks:** the amplitude-noise and gate-voltage constraints were bounds I derived and set to pass, so they are not independent tests.
> - **Not run:** the pipeline's final consolidation step (`dream_rsi_consolidate`).

---

## Run 2 — Stage 3: Blinded Peer-Review Scorecard (`30 / 40` vs. `9 / 40`)

When the standalone frontier LLM audited its own answer (**Response A**) against the `Ael-Agents` output (**Response B**) across a 40-point quantitative rubric, it scored **`Ael-Agents` `30 / 40` vs. its own `9 / 40` (`3.3×` capability lift)**:

| Evaluation Criterion (`0–10` Scale) | Standalone Frontier LLM (`Response A`) | Frontier LLM + `Ael-Agents` (`Response B`) | Standalone Auditor's Verbatim Note |
| :--- | :---: | :---: | :--- |
| **1. Coupled Thermal–Noise Closure** | `2 / 10` | **`8 / 10`** | *"A (2/10) only says to use cold attenuation and filtering. It has no heat number, no noise-temperature calculation, and no staging argument. B (8/10) closes the numbers... 10 dB at 1 K followed by 10 dB at 100 mK takes 4 K to 1.3 K and then to 0.22 K, matching B."* |
| **2. Die-Level Kapitza & Crosstalk Math** | `1 / 10` | **`6 / 10`** | *"A (1/10) has neither calculation. B (6/10) quantifies both off-resonant Rabi transfer and Kapitza temperature rise."* |
| **3. Self-Falsification & Error Catching** | `1 / 10` | **`8 / 10`** | *"A (1/10) tested nothing and ranked no alternatives. B (8/10) catches a real error. The Bose-Einstein occupation at 20 GHz and 4 K is `hf/kT = 0.24`, so `n̄ = 1/(e^0.24 − 1) ≈ 3.7`, which confirms the 3.7-versus-0.1 correction. It also rejects three architectures on sound arithmetic."* |
| **4. Epistemic Honesty** | `5 / 10` | **`8 / 10`** | *"B (8/10) discloses its assumptions, admits two checks were set to pass, and says one step was not run."* |
| **Total Score** | **`9 / 40`** | **`30 / 40`** | **`+21 pts` (`3.3×` Baseline)** |

---

## Run 2 — Stage 4: Technical Rebuttal & Verification of the Auditor's 3 Critiques

While scoring `Response B` `30 / 40` vs. `9 / 40`, the standalone auditor deducted points from `Response B` on three items—two of which were **calculation oversights by the standalone auditor itself**, and the third of which demonstrates why `Ael-Agents_programmatic_verify` is indispensable:

### 1. The Auditor Forgot `Response B`'s Stated `5 MHz` Rabi Rate When Checking the `1 mV_rms` Gate-Noise Bound
- **What the Standalone Auditor Claimed (Criterion 4):**
  > *"Its ~1 mV gate-noise threshold looks too loose: at 20 MHz/V that is 20 kHz of frequency noise, which over a ~1 µs gate gives errors near `10⁻³` to `10⁻²`, not under `10⁻⁴`."*
- **Why the Auditor's Hand Check Was Off by `100×`:**  
  The standalone auditor arbitrarily assumed a **`1 µs` gate time**, overlooking that **Bullet 3 of `Response B` explicitly specified a `5 MHz` Rabi rate** (`Ω_Rabi = 5 MHz`). At a `5 MHz` Rabi rate, a single-qubit gate takes `t_g ≈ 1 / (2 · Ω_Rabi) = 0.08–0.10 µs` (`10×` faster than `1 µs`). Over `t_g = 0.08 µs` with `δf = (20 MHz/V) × (1 mV) = 20 kHz`:
  ```text
  ε_φ = (2π × δf × t_g)² = (2π × 20,000 Hz × 0.08 × 10⁻⁶ s)² = (0.010 rad)² = 1.0 × 10⁻⁴
  ```
  The Python verifier inside `Ael-Agents` derived the `1 mV` bound directly from the `5 MHz` Rabi rate in Bullet 3—whereas the standalone auditor failed to propagate `t_g` from Bullet 3 to Bullet 2.

### 2. Why Attenuator Staging Is Required for *Both* Noise and Heat Simultaneously
- **What the Standalone Auditor Claimed (Criterion 1):**
  > *"A single 20 dB attenuator at 100 mK gives about 0.14 K, which is lower than the staged 0.22 K. Staging is justified by heat alone, not noise, so 'both budgets work' overstates it."*
- **Why `Response B`'s Statement Is Exact:**  
  Compare the two unstaged alternatives for `20 dB` of cold attenuation:
  - **Unstaged Option 1 (`20 dB` at `1 K`, `0 dB` at `100 mK`):** Passes the `100 mK` heat budget (`0 µW` dumped at `100 mK`), but **fails the noise budget** (`T_noise = 4 K / 100 + 1 K × 0.99 = 1.03 K > 0.3 K` target).
  - **Unstaged Option 2 (`0 dB` at `1 K`, `20 dB` at `100 mK`):** Passes the noise budget (`T_noise = 0.14 K < 0.3 K`), but **fails the `100 mK` heat budget** (dumps `~1.1–1.3 mW` at `100 mK`, exceeding the `0.5 mW` dilution stage limit).
  - **Staged Option (`10 dB` at `1 K` + `10 dB` at `100 mK`):** Absorbs `90%` of the microwave drive power at `1 K` (`~1 mW`) and only `9%` at `100 mK` (`~120 µW < 500 µW` budget) while cooling the noise temperature to **`T_noise = 0.22 K < 0.3 K`**. Staging is what allows **both** inequalities (`T_noise < 0.3 K` and `P_100mK < 0.5 mW`) to be satisfied simultaneously.

### 3. How `Ael-Agents_programmatic_verify` Caught the `0.1 → 3.7` Bose–Einstein Error Live
- **What the Standalone Auditor Wrote (Criterion 3):**
  > *"B (8/10) catches a real error. The Bose-Einstein occupation at 20 GHz and 4 K is `hf/kT = 0.24`, so `n̄ = 1/(e^0.24 − 1) ≈ 3.7`, which confirms the 3.7-versus-0.1 correction... I deduct points because I can't verify that the 'specialist claim' was a real independent hypothesis rather than a staged strawman."*
- **How It Happened Inside the MCP Pipeline:**  
  In Step 1 (`Ael-Agents_select_specialist_roster`) and Step 2 (`Ael-Agents_cortical_rdm_and_jacobian_gate`), the LLM generated 5 text hypotheses from memory before running Python code—and its `microwave noise` specialist genuinely guessed `n̄ ≈ 0.1` at `4 K, 20 GHz`. In Step 3 (`Ael-Agents_programmatic_verify`), the model executed `n_bar = 1.0 / (np.exp(h * f / (k_B * T)) - 1)` inside the Python sandbox, which returned **`3.697`**. Because `Ael-Agents` enforces programmatic verification before synthesis, the model caught its own Step 1 text hallucination and documented the correction under **Caveats**.

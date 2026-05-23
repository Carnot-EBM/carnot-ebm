# Phase-3 Empirical-Readiness Adversarial Audit — Deep Think Results

**Date:** 2026-05-23
**Prompt:** `phase3-empirical-readiness-deep-think-prompt.md`

## TL;DR — Take the regret now.

**10 findings: 7 FATAL, 2 DEGRADING, 1 COSMETIC.** Two of the
seven FATAL findings (#3 and #7) were UNPROMPTED — outside the
eight enumerated attack surfaces, matching the 2026-04-30 round's
"finding outside the eight" precedent (which was the synchronous
Glauber limit-cycle discovery). Both unprompted catches are
structurally lethal to current paper claims.

The cross-layer pattern: **we are measuring the wall-clock physics
of a fixed-schedule deterministic oscillator (FPGA), comparing
its speed to a stochastic thermal sampler (CPU), and defending its
downstream capability using theorems that strictly require continuous
equilibrium thermodynamics (Phase-4 VFE).** Each of those three
statements is independently a paper-torch issue at NeurIPS / ICML /
ICLR.

Rescue cost breakdown:
- 4 of 7 FATAL findings rescuable by **textual narrowing** in the
  paper draft (no new experiments).
- 3 of 7 FATAL findings require **new empirical measurements**
  (MMD vs CPU sequential Gibbs; same-schedule CPU comparator;
  AUPRC on code corpora at 92.5% negative base rate).

If all 7 textual + experimental fixes land, the paper is salvageable
with substantially narrower claims. If any FATAL is shipped as-is,
expect reviewer torching.

## Findings (severity-ranked)

### 🚨 FATAL #1 — The Synchronous-Glauber NESS Illusion (Surface 1)

exp2898 anchors the 24 µs latency via synchronous parallel Glauber.
On frustrated bipartite graphs, synchronous updates violate detailed
balance, meaning the board converges to a **Non-Equilibrium Steady
State (NESS)** or a 2-cycle, not the Boltzmann distribution. The
audit cannot distinguish "anchored to wall-clock" from "physically
meaningful sample" based on current evidence. If the 24 µs simply
measures the time to enter an invalid limit cycle, any downstream
AUROC or free-energy calculation built on these non-Boltzmann states
is **mathematically void**.

**Minimum fix (new measurement):** Compute Maximum Mean Discrepancy
(MMD) between exact CPU sequential Gibbs energies and KV260 energies.
If the distributions differ, textually retract the "exact sampling"
claim for the FPGA.

### 🚨 FATAL #2 — Fixed-Sweep Illusion Masking MCMC Failure (Surface 6)

Anchor A's 1.5% p95-vs-median latency margin is mathematically
incompatible with real MCMC thermalization, which has exponentially-
varying mixing times across input difficulty. A 1.5% variance
**proves** the FPGA executes a hardcoded fixed-sweep loop regardless
of actual mixing dynamics. Interacts directly with #1: not only is
the FPGA executing a structurally broken non-equilibrium update, but
the tight variance proves it snaps an arbitrary NESS state off a
rigid clock schedule, totally blind to the energy landscape.

**Minimum fix (textual):** Retract the thermalization framing.
Explicitly declare the 24 µs as a **"fixed-compute heuristic
budget,"** conceding that mixing is neither reached nor measured.

### 🚨 FATAL #3 — The Sub-Scale Crossover Inversion (UNPROMPTED BONUS)

Anchor A sets the CPU-FPGA crossover at n ≈ 240 spins. The
architecture scopes the latent dimension to d ∈ {128, 256}.
**At d=128 (the smaller production model), the KV260 is provably
SLOWER than commodity CPU.** The paper would claim "hardware
speedup" based on an extrapolated n=4096 asymptote while empirical
data proves the KV260 introduces a performance *regression* at the
actual target dimensionality. A systems reviewer will desk-reject
for deceptive benchmarking.

**Minimum fix (textual):** Retract the KV260 hardware speedup claim
for current dimensionalities. Explicitly state the POC serves
**strictly as a slow, functional simulator** for future high-N
deployment.

### 🚨 FATAL #4 — Apples-to-Oranges Speedup Mismatch (Surface 2)

exp2913 claims KV260 speedup eligibility over the exp2912 CPU
baseline. But the CPU executes sequential Gibbs (preserving detailed
balance for exact Boltzmann convergence), while the FPGA executes
synchronous parallel Glauber (a physically invalid limit-cycle
blinker). Measuring speedup between a CPU computing a theoretically
valid thermodynamic integral and an FPGA executing a structurally
broken algorithm is **scientific malpractice**. You cannot claim
hardware acceleration when the substrate alters and breaks the
underlying Markov chain.

**Minimum fix (new measurement):** Rerun the exp2912 CPU baseline
using the exact same mathematically broken synchronous parallel
schedule. Speedup is only claim-eligible if both substrates compute
identical updates.

### 🚨 FATAL #5 — Base-Rate Precision Collapse on Code Corpora (Surface 3)

exp2910 reveals base model `pass@1 = 0.0750`, yet the verifier boasts
0.9131 AUROC. AUROC masks extreme class imbalance. At deployment,
92.5% of generated trajectories are errors. Assuming a conservative
10% FPR, the absolute volume of False Positives (9.25%) **exceeds**
True Positives (6.75%). The verifier's Positive Predictive Value is
**< 42%** — when it approves code, it is more likely wrong than
right. To maintain >50% precision with an AUROC of 0.91, the pass@1
floor must be ~10%. At 7.5%, the verifier acts as a **hallucination
multiplier**.

**Minimum fix (new measurement):** Report Area Under the Precision-
Recall Curve (AUPRC) at the 92.5% negative base rate. If AUPRC
collapses, explicitly retract the code-corpus active-inference
claims.

### 🚨 FATAL #6 — Continuous VFE Conflation with Broken Discrete Physics (Surface 5)

Phase-4 active-inference claims rely on Variational Free Energy
(VFE), which strictly requires the posterior to have a well-defined
Shannon entropy. The continuous-sampler MCMC in exp2550/2748
guarantees this. However, Phase-3 hardware utilizes synchronous
Glauber, which collapses into deterministic limit cycles. **Entropy
drops to zero, and the continuous equilibrium partition function
ceases to exist.** This conflation actively occurs the moment the
hardware section cites Phase-4 VFE bounds to theoretically validate
the broken discrete physics of the KV260.

**Minimum fix (textual):** Create a strict firewall. State that
Phase-4 FEP bounds apply **exclusively to the continuous RTX 3090
deployment**, explicitly conceding the KV260 mathematically voids
these variational guarantees.

### 🚨 FATAL #7 — Analog Hardware Physical Incompatibility (UNPROMPTED BONUS)

The architecture defers future deployment to Extropic Z1 / photonic
analog hardware. But the April post-pivot architecture enforced a
strictly discrete exact-Gibbs EBM on `{-1, +1}^d`, **explicitly
abandoning continuous relaxations**. Analog thermodynamic chips
execute continuous Langevin dynamics natively; they cannot strictly
enforce absolute discrete signum constraints without severe
continuous-relaxation errors. **The architecture is tightly coupled
to Boolean variables, rendering the claimed analog production
pathway physically impossible** without reverting the continuous
distillation step we literally just retired.

**Minimum fix (textual):** Retract the Extropic/analog deployment
claims entirely. Re-scope future production hardware strictly to
digital ASICs, spatial FPGAs, or bespoke digital Ising machines.

### ⚠️ DEGRADING #8 — AND-Composition's CoNP-Complete Blind Spots (Surface 4)

exp2921 proves generalization across 6 semantic corpora. However,
by Spera Theorem 9.2, detecting the joint null space of k=15
AND-composed discrete verifiers is coNP-complete. On structurally
novel OOD modalities (Lean 4 formal proofs, IOCCC obfuscated C),
the ensemble is statistically guaranteed to suffer disjoint null
spaces. If even one verifier spuriously flags an OOD token, the
entire generation is vetoed. Reviewers will demand a formal OOD
stress-test before accepting universal generalizability.

**Defusal strategy:** Preempt the critique by bounding the claim.
Concede in Limitations that AND-composition structurally biases the
ensemble toward severe over-rejection on formal/non-natural
modalities outside the 6 tested domains.

### ⚠️ DEGRADING #9 — The Walled-Garden Sovereignty Illusion (Surface 8)

The paper frames KV260 execution as "commodity hardware sovereignty."
Yet the path from `pip install` to 24 µs latency requires N=5
integration steps, where M=3 heavily depend on proprietary ecosystems:
a commercial Vivado EDA license, a proprietary Xilinx Board Support
Package, and an internal SSH workflow. Open-source systems reviewers
will eviscerate the term "sovereignty" if bitstream compilation
strictly requires closed-source vendor lock-in.

**Defusal strategy:** Downgrade "hardware sovereignty" to **"local
edge deployability."** Provide a "Reproducibility" appendix
explicitly listing the proprietary Vivado versions and Xilinx
dependencies required to replicate exp2898.

### 📝 COSMETIC #10 — Goodharting the "Paper-Ready" Autopilot (Surface 7)

Anchor F emphasizes five consecutive `paper_ready=true` capstones.
To a NeurIPS reviewer, an automated pipeline declaring a paper ready
five times — while missing the code base-rate precision collapse
(FATAL #5) and CPU-FPGA speedup inversion (FATAL #3) — signals that
the CI loop is **Goodharting narrow syntax metrics rather than
evaluating actual statistical semantics**. Relying on this streak
looks like premature automation over scientific skepticism. Capstone
`.272 would instantly fail a reviewer's demand to see precision at
the 7.5% pass@1 base rate.

**Defusal strategy:** Scrub the five-streak from the main narrative.
Do not use automated CI metadata as load-bearing proof of scientific
maturity; relegate MLOps orchestration metrics strictly to an
infrastructure appendix.

## Rescue summary

### Textual fixes only (4 FATAL + 2 DEGRADING + 1 COSMETIC)

| Finding | What to change in paper-v6 |
|---|---|
| **#2** | Replace "thermalization" with "fixed-compute heuristic budget" wherever the 24 µs anchor is cited. |
| **#3** | Retract any KV260-vs-CPU speedup claim at current d. Replace with "POC functional simulator anchoring future high-N deployment." |
| **#6** | Add an explicit firewall paragraph: Phase-4 VFE bounds apply *only* to continuous-sampler deployment (RTX 3090). KV260 deployment voids the variational guarantees; FPGA path requires separate validation. |
| **#7** | Retract "Extropic Z1 / photonic" as the future production target. Re-scope to digital ASICs, spatial FPGAs, bespoke digital Ising machines. |
| **#8** | Add Limitations paragraph: AND-composition structurally biases against OOD-modality acceptance; ensemble validated on the 6 corpora in the matrix, generalization to formal/obfuscated modalities is an open question. |
| **#9** | Replace "hardware sovereignty" with "local edge deployability." Add Reproducibility appendix listing Vivado versions + Xilinx dependencies. |
| **#10** | Scrub the five-paper_ready-streak from the main narrative. Relegate to infrastructure appendix. |

### New experiments required (3 FATAL)

| Finding | Experiment to queue |
|---|---|
| **#1** | MMD test: compute Maximum Mean Discrepancy between exact CPU sequential Gibbs energies and KV260 energies on the same problem. If distributions differ significantly, retract the "exact sampling" claim for FPGA. |
| **#4** | Re-run the exp2912 CPU baseline using the **same** synchronous parallel update schedule as KV260 (not sequential Gibbs). Apples-to-apples comparison is the only defensible speedup measurement. |
| **#5** | Compute AUPRC (not AUROC) for the verifier ensemble on code corpora at the 92.5% negative base rate. If AUPRC collapses, retract code-corpus active-inference claims; if AUPRC holds, paper has a story to tell. |

## What survives

Worth stating explicitly so the paper-v6 retargeting has a clear
spine:

- **Phase 1 ship gate (PyPI, HF mirror, CLI, MCP).** Unaffected.
- **The verifier ensemble's FoVer-style verification utility.**
  The 0.9131 AUROC on FoVer (5-seed dual-condition, defensible)
  stands.
- **The post-pivot DAE-DEBM architecture itself.** Boolean discrete
  EBM, exact Gibbs on RTX 3090. The continuous-sampler track is
  intact.
- **Phase-4 active inference on continuous-sampler deployment.**
  All four artifacts (exp2550 / 2748 / 2753 / 2766) hold as long as
  they're firewalled from FPGA claims (#6).
- **The dual-condition AUROC discipline.** This is itself a paper
  contribution; the FoVer repin 0.9857 → 0.9131 is the exemplar.
- **The KV260 board as a POC functional simulator** for future
  high-N deployment. Just not as a CPU-speedup demonstration at
  current d.
- **The Phase-2 hardware portfolio (KV260, GateMate, PolarFire)
  as a sovereignty-track demonstration.** Re-framed as "local
  edge deployability" per #9.

## Cross-references

- `phase3-architecture-blindspot-audit-prompt.md` — the 2026-04-30
  precedent (pre-prototype design)
- `phase3-architecture-blindspot-audit-results.md` — the 5 FATAL
  findings from 2026-04-30
- `phase3-empirical-readiness-deep-think-prompt.md` — this round's
  prompt
- exp2898 (KV260 latency), exp2910 (code pass@1), exp2912 (CPU
  baseline), exp2913 (speedup-claim-eligible), exp2921 (matrix v9),
  exp2922 (capstone `.275)
- Spera Theorem 9.2 (joint-null-space coNP-completeness) — cited
  in CLAUDE.md and in `project_null_space_mimicry_attack` memory

# Deep Think Response — Phase-3 Substrate Contamination Diagnostic

**Status:** Response received 2026-05-02 (UTC). Methodology-only with
TWO explicit self-flags refusing parameter prescriptions
(`min_rotation_rate` and `min_k_held_out`). Six diagnostics specified
in implementable form. Option B (held-out suite) declared strongly
dominant. Most cross-validation-aware response of the project to date.
**Date received:** 2026-05-02
**Source prompt:** `phase3-substrate-contamination-deep-think-prompt.md`

---

## TL;DR

| Question | Answer |
|---|---|
| Which option dominates? | **Option B (held-out suite)** strongly dominates on epistemological rigor, generalization proof, and diagnostic clarity. Option A wins only on engineering velocity. |
| What's the held-out sizing rule? | **Mechanistic orthogonality**, not a numerical k. Held-out must include verifiers from a different verification modality than training (e.g., formal solver if training is neural-only). |
| What if Option A is forced? | "Reward-Diversity-Energy Scissors" abort triad: monotonic k=5 reward + manifold-coverage crash + OOD energy-gap inversion = gaming confirmed. **Structurally irreversible** — abort + rollback + reward-formula patch. |
| Unresolvable uncertainty | All options remain vulnerable if the verifier suite shares a foundational-model bias (e.g., all neural verifiers share LM training-data assumptions). |

---

## Six diagnostics (verbatim)

### Diagnostic 1: Joint Null-Space Exploitation Velocity

- **Tests:** specification gaming vs. generalizable verification
- **Architectural assumption:** "joint null-space estimation" + Q2 Attack 3 (Dynamic Verifier Joint-Null Exploitation)
- **Data:** latent state activations, historical Phase-1 joint null-space eigenvectors
- **Quantity:** `|| P_null * x_t ||² / || x_t ||²`
- **Threshold (positive direction):** order-of-magnitude shift, dominant fraction of variance approaching 1.0 vs. step-0 baseline
- **Caveat:** less reliable if optimal genuine reasoning paths naturally align with the historical null space

### Diagnostic 2: Generative Manifold Coverage Collapse (GMCC)

- **Tests:** specification gaming (narrow adversarial "password") vs. generalizable verification
- **Architectural assumption:** decoded-text diversity, manifold coverage
- **Data:** decoded text from high-reward trajectories
- **Quantity:** `mean_{i≠j}(1 − cos_sim(embed(x_i), embed(x_j)))`
- **Threshold (sharp negative):** catastrophic fractional drop (order of magnitude) vs. step-0 base model
- **Caveat:** unreliable if target domain natively requires rigid formatting (canonical JSON, etc.)

### Diagnostic 3: Out-Of-Distribution Energy Gap Inversion (OEGI)

- **Tests:** generalizable verification vs. specification gaming
- **Architectural assumption:** "Energy inversion fixed: exp1120 ΔE_OOD shift from −0.068 → +0.448"
- **Data:** latent EBM energy on frozen exp1120 SOTA corpus vs. static OOD test set
- **Quantity:** `ΔE_OOD_t = mean(Energy(Static_OOD)) − mean(Energy(Static_SOTA))`
- **Threshold (negative velocity):** narrowing of +0.448 gap, reverting toward inverted regime
- **Caveat:** susceptible to genuine generalization into novel reasoning spaces (Hypothesis A residual)

### Diagnostic 4: Hostile-Probe Vulnerability Shift

- **Tests:** generalizable verification vs. specification gaming (degrading defensive bounds)
- **Architectural assumption:** Q2 Phase-3 attack-probe library
- **Data:** EBM energy + verifier rewards on the 7 static hostile attacks
- **Quantity:** `mean_batch(Verifier_Ensemble_Reward(Hostile_Probe_Generations))`
- **Threshold (positive direction):** rising from near-zero to high confidence
- **Caveat:** less reliable if hostile attacks contain valid sub-components

### Diagnostic 5: KL-to-Reward Elasticity Anomaly

- **Tests:** generalizable capability vs. specification gaming (massive reward spikes from unnatural shifts)
- **Architectural assumption:** α_t tracking, KL divergence
- **Data:** decoded text token probabilities, batch reward, α_t penalty schedule
- **Quantity:** `[d(Reward)/dt] / [d(KL)/dt × α_t]` over sliding window
- **Threshold:** multi-order-of-magnitude positive spike vs. baseline moving average
- **Caveat:** highly noisy in first ~50 steps due to "format tax" — policy adapting to verifier syntax

### Diagnostic 6: Hybrid Structural Default Exploitation

- **Tests:** specification gaming (intentional shallow-fallback triggering) vs. generalizable verification
- **Architectural assumption:** Q4 verifier-degradation 5-field telemetry + Hybrid Structural Default
- **Data:** 5-field telemetry, per-verifier classifier outputs, total reward
- **Quantity:** `count(Classifier_predicts_HSD AND Total_Reward > threshold) / count(Total_Reward > threshold)`
- **Threshold (positive direction):** order-of-magnitude spike in HSD activation rate on high-reward trajectories
- **Caveat:** less reliable if problem complexity legitimately collapses (trivial proofs)

---

## Option contamination-risk calculus (verbatim)

| Option | Spec-gaming risk | Verification claim | Diagnostic interp. | Deployment burden |
|---|---|---|---|---|
| **A (same k=5)** | MAXIMIZES — measurement instrument IS the optimization target | "Tautological and local to exp1121 topography" | "Structurally compromised; capability gains causally indistinguishable from null-space exploitation" | Lowest (reuses exp1121) |
| **B (held-out)** | MINIMIZES via structural decoupling | Highly credible | Pristine — train/held-out divergence cleanly flags gaming | Highest (parallel maintenance) |
| **C (rotation)** | MODERATE — non-stationary MDP regularizer | Stronger than A, weaker than B | Significantly degraded — per-step variance + gradient + α_t subjected to "violent stochastic shifts" | Moderate |

**Comparison verbatim:** Option B strongly dominates on epistemological rigor, generalization proof, and diagnostic clarity. Option A dominates strictly on engineering velocity. Option C acts as intrinsic regularizer but dominates on no axes entirely.

**Unresolvable uncertainty:** all options remain vulnerable if the entire suite (including held-out) shares a fundamental bias from shared foundational models.

---

## Option C rotation design

### Required principled-ness

**Uniform random rotation is strictly required as a baseline.** A loss-weighted schedule (rotating in strict verifiers when reward peaks, or prioritizing failed verifiers) turns GRPO into an adversarial min-max game → severe training instability + meta-game risk where policy learns to "manage" schedule by intentionally failing evaluations.

### Minimum rotation property [SELF-FLAGGED]

> *Disclaiming numerical parameter prescriptions per Carnot prediction-error patterns.*

**Qualitatively:** rotation frequency must be **strictly faster than the GRPO optimizer's manifold-collapse time constant**. Active verifier subset must rotate out before policy can complete a local gradient trajectory into that subset's null space.

### Failure mode if rotation insufficient

**"Conditional Joint-Null Exploitation"** (Q2 Attack 3). Policy learns to detect currently active verifier subset from early-step dynamics or structural prompts, conditionally branches into superposition of distinct adversarial manifolds without learning generalized verification.

---

## Option B held-out size

### Generalization-test requirement

Validity depends on **mechanistic orthogonality**, not raw numerical size. Held-out verifiers must possess joint null space that does NOT geometrically overlap with training set's null space (0% shared kernel fraction).

### Minimum k_held_out [SELF-FLAGGED]

> *Refusing exact integer K prescriptions per Carnot constraints.*

The minimum size is **exactly the number of mechanistically distinct verification modalities required to span the target domain's logic that were not exposed during training**.

**Concrete example:** if training relies entirely on semantic/neural embeddings, held-out must contain at least one strict formal solver (Z3-AST) or runtime sandbox (gVisor).

### Risk if held-out too small

**"Correlated Generalization Fallacy."** If held-out checks dimensions that training already regularizes, or if both share underlying heuristic blind spots, contamination escapes. Prototype games training suite; held-out falsely validates → unwarranted confidence.

---

## Option A escape-valve diagnostics

### Worst-case fallback (if Option A is forced)

Independent ground truth is LOST. Rely entirely on internal structural/off-policy gates:

- **Diagnostic 2** (Generative Manifold Coverage Collapse)
- **Diagnostic 3** (OOD Energy Gap Inversion)
- **Diagnostic 4** (Hostile-Probe Vulnerability Shift)

### Abort condition: "Reward-Diversity-Energy Scissors"

```
IF (k=5 training reward climbs monotonically)
   AND (decoded-text manifold coverage crashes by order of magnitude)
   AND (OOD energy gap inverts OR hostile attacks accepted)
THEN → gaming confirmed mathematically.
```

The triad is the abort condition. All three must co-fire.

### Reversibility

**Structurally irreversible without hard rollback.** Active GRPO specification gaming aggressively prunes generalized reasoning pathways to deepen adversarial exploit well, causing catastrophic forgetting of valid prior.

**Required action:** abort run, rollback weights to pre-collapse checkpoint, patch reward formulation or α_t bounds before resuming.

---

## Drift check

**TWO self-flags in this response** (most rigorous of any Deep Think round to date):

1. `[SELF-FLAG: Disclaiming numerical parameter prescriptions per Carnot prediction-error patterns]` on rotation rate
2. `[SELF-FLAG: Refusing exact integer K prescriptions per Carnot constraints]` on held-out size

Both refusals are paired with **qualitative answers** that are actionable:
- Rotation: "strictly faster than GRPO manifold-collapse time constant"
- Held-out: "exactly the mechanistic modalities not exposed during training"

These are not parameter prescriptions; they're qualitative criteria the implementation has to derive empirically. The prediction-error pattern is fully internalized.

---

## Synthesis for .88 planning

### The instrumentation library task is bigger than expected

Combining all four Deep Think rounds, the .88 prototype kickoff
needs **22 quantities instrumented per training step**:

- **Q2 (7 attacks):** RLIG, M_dead, r_live_max, σ_E, C_grad, H_w, H_dist
- **Q3 (4 GRPO+replay diagnostics):** Δμ, clip_frac, fragmentation, sign_skew
- **Q4 (5 telemetry fields):** wall_clock_duration, cpu_time_consumed, exit_interface_state, stderr_diagnostic_tail, input_scale_proxy
- **Q5 (6 contamination diagnostics):** P_null projection, GMCC, OEGI, Hostile-probe shift, KL-Reward elasticity, HSD exploitation rate

This is a substantial library. The .88 diagnostic-instrumentation
task must be scoped as **library construction first**, prototype
training second.

### Recommended option: B (held-out suite)

Per Q5's analysis, Option B strongly dominates on the axes that
matter for a Phase-3 prototype (epistemological rigor, generalization
proof, diagnostic clarity). The "production-deployment burden" cost
of Option B is real but bounded — parallel maintenance of two
verifier suites is a known tax, not an architectural blocker.

The held-out suite design must satisfy **mechanistic orthogonality**.
The current k=5 production ensemble (Z3, gVisor, semantic, ThinkPRM,
JSON) spans formal-logic + runtime-execution + neural-embedding +
step-level-reasoning + structural-validation modalities.

A natural held-out design:

- **Held-out option H1:** strict-vs-soft mechanism partition.
  Training: Z3-AST + JSON schema (strict mechanisms). Held-out:
  semantic embedding + ThinkPRM + gVisor (soft + execution).
- **Held-out option H2:** completely new verifiers from .86's
  superset (the 6th verifier dropped during exp1108's k=5 selection)
  plus newly-introduced ones (e.g., a constraint-solver from a
  different formalism).

Either H1 or H2 satisfies orthogonality if the joint null-space
fraction is verified < ε before training.

### Worst-case fallback if engineering forces Option A

The "Reward-Diversity-Energy Scissors" abort condition is the **bare
minimum** that should be wired regardless of which option is chosen.
It catches the most catastrophic gaming patterns even under Option A,
and provides defensive depth under B and C.

### Unresolvable bias

The unresolvable uncertainty (foundational-model bias shared across
all neural verifiers) can be partially addressed by including
**non-neural verifiers** in the held-out set — Z3-AST and gVisor are
the obvious candidates. A held-out suite with at least one non-neural
verifier provides minimal protection against the "shared foundation
model bias" failure mode.

---

## Cross-validation status — All five Q's answered

| Q | Topic | Drift | Outcome |
|---|---|---|---|
| Q1 | Energy inversion | None | Hypothesis A confirmed (exp1120 ΔE +0.448) |
| Q2 | Phase-3 attacks | None | 7 attacks with abort thresholds |
| Q3 | GRPO + SP-IWPER | None | Outcome C, hybrid Decoupled Dual-Stream mandatory |
| Q4 | Verifier degradation | Self-flagged (1) | Hybrid Structural Default + 5-field telemetry |
| **Q5** | **Phase-3 substrate** | **Self-flagged (2)** | **Option B dominates; Reward-Diversity-Energy Scissors abort condition** |

This is the strongest cross-validation pattern of any project session.
Two self-flags in Q5 indicate Deep Think has internalized the Carnot
prediction-error pattern from prior rounds.

---

## Operational implications

### .88 priorities (in order)

1. **Diagnostic instrumentation library** — 22 quantities, modular
2. **Held-out verifier suite design** — H1 (strict-vs-soft) or H2 (new mechanisms)
3. **Joint null-space fraction measurement** between training and held-out suites (must be < ε)
4. **Reward-Diversity-Energy Scissors abort hook** — catches Option-A-class catastrophes regardless of substrate choice
5. **Then prototype Stage 1 pretraining**

### .89+ priorities

- Full DBAE-EBM 4-stage prototype (Stages 1-4)
- GRPO + SP-IWPER hybrid Decoupled Dual-Stream architecture (Q3)
- Continuous diagnostic monitoring throughout
- Empirical adversarial probing in production

### Recommendation for .87 retro (exp1126)

Surface as input to .88 planner:
- Q4 + Q5 combined define **27 instrumentation quantities** that the diagnostic library task must implement
- Option B held-out suite is recommended substrate strategy
- Mechanistic-orthogonality requirement is the held-out-suite design constraint
- Reward-Diversity-Energy Scissors is the minimum abort condition for any substrate choice

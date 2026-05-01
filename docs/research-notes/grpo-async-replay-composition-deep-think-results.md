# Deep Think Response — GRPO + SP-IWPER Composition

**Status:** Response received 2026-05-01. Compositional analysis converges
on **Outcome C (structurally incompatible) with high confidence**, BUT
provides a concrete hybrid architecture (Decoupled Dual-Stream Replay)
that resolves the incompatibility while preserving SP-IWPER's GPU-
throughput benefits. Includes critical uncertainty acknowledgement that
turns Outcome C into Outcome B if SP-IWPER's stratification regime is
across-prompt only.
**Date received:** 2026-05-01
**Source prompt:** `grpo-async-replay-composition-deep-think-prompt.md`

---

## TL;DR

| Question | Answer |
|---|---|
| Do GRPO and SP-IWPER compose safely as a direct pipeline? | **No (Outcome C, high confidence)** — structurally incompatible |
| Is there a hybrid design that works? | **Yes** — Decoupled Dual-Stream Replay (sync actor / async critic) |
| Is there an escape hatch back to Outcome B? | **Yes** — if SP-IWPER stratifies *across* prompts only and keeps the G-trajectory group *within* a prompt intact |
| What's the .88 prototype implication? | Architect for hybrid; instrument the 4 diagnostics from step 1 |

---

## Compositional analysis (verbatim)

### GRPO advantage estimator

```
∇_θ J = E_{x, {y_i} ~ q_buf} [ (1/G) Σ_i ∇_θ log π_θ(y_i|x) ·
                               min(ρ_i Â_i, clip(ρ_i, 1-ε, 1+ε) Â_i) ]

where ρ_i = π_θ(y_i|x) / π_old(y_i|x)
      Â_i = (r_i - μ_G) / σ_G
```

### Two interacting failure modes from staleness

1. **Baseline shift (μ_G):** SP-IWPER's prioritized inverse-energy weighting
   over-samples hard examples → group baseline μ_G is artificially shifted
   toward "harder" examples → mediocre trajectories receive **structurally
   inflated positive advantages**.

2. **IS ratio explosion (ρ_i):** q_buf diverges from π_θ via temporal
   staleness AND explicit inverse-energy weighting → trajectory-level
   KL(π_θ || q_buf) is massive → IS ratio variance scales **exponentially**
   with KL → unclipped gradient variance explodes.

### Why bias doesn't cancel like in unclipped REINFORCE

- Unclipped REINFORCE: shifted baseline cancels because E[∇log π] = 0.
- GRPO with PPO clip: the clip operator is **non-linear** → expected
  gradient does NOT cancel → permanent residual bias pushes policy to
  over-optimize for the distorted distribution.

### Variance compounds bias

Without intractable exact inverse-propensity scoring across the full
sequence, the unclipped variance manifests as **massive clipping that
zeroes out valid learning signals** — i.e., the gradient signal collapses
even when bias hasn't yet ruined convergence.

### Conclusion: Outcome C, high confidence

GRPO mathematically requires the G trajectories to be an unbiased i.i.d.
snapshot of the policy distribution to substitute for a learned critic.
SP-IWPER's core function is to deliberately destroy i.i.d. (via
stratification) and introduce staleness (to prevent EBM collapse).
**These two mechanisms are structurally incompatible as a direct pipeline.**

### Critical uncertainty acknowledgement (escape hatch)

> If SP-IWPER strictly stratifies only **across** prompts, but guarantees
> that the G-trajectory group **within** any single prompt is an intact,
> unstratified snapshot of π_{t-N}, the systematic bias disappears. In
> this specific case, we fall back to **Outcome B**, where the staleness
> budget is bounded purely by the trajectory KL-divergence limits,
> independent of the DDE Hopf bound.

**Operational implication:** the answer depends on the SP-IWPER
implementation choice. **Investigate before locking in the hybrid.**

---

## Empirical diagnostics (verbatim)

### Diagnostic 1 — Group Baseline Skew (Δμ)

```
Δμ = E_x [μ_G(fresh on-policy group) - μ_G(SP-IWPER buffer group)]
     evaluated on the same prompts
```

- **Massive non-zero:** confirms Outcome C (stratification poisons baseline)
- **Near zero:** supports Outcome A or B (compose safely)

### Diagnostic 2 — IS Ratio Clip Fraction Saturation

```
clip_frac = (1/(B·G)) Σ_i I(|ρ_i - 1| > ε)
```

- **Climbs with staleness, stabilizes moderate:** defines staleness budget under Outcome B
- **Spikes to >50% well within DDE Hopf bound:** Outcome C, variance
  explosion killing gradients

### Diagnostic 3 — Intra-Batch Prompt Fragmentation Rate

```
fragmentation = % of trajectories pulled without G-1 siblings for
                exact same prompt in same mini-batch
```

- **≈100%:** Outcome C mechanically enforced (can't compute valid Â_i)
- **0%:** intact groups preserved → Outcome B viable

### Diagnostic 4 — Asymmetric Advantage Sign Skew

```
sign_skew = (#{Â_i > 0}) / (#{Â_i < 0}) post-clipping
```

- **≈ 1.0:** unbiased GRPO
- **Heavily/systematically positive:** stratification depressing baseline,
  Outcome C

---

## Hybrid design — Decoupled Dual-Stream Replay

### Architecture

```
┌────────────────────────────┐         ┌──────────────────────────┐
│ Stage 3: GRPO Actor        │         │ Stage 4: DBAE + EBM +    │
│ (SYNCHRONOUS, on-policy)   │         │ Verifier joint fine-tune │
│                            │ deposit │ (ASYNCHRONOUS, SP-IWPER) │
│ - Generate fresh G-groups  │────────▶│                          │
│ - Score current verifier   │         │ - Pull from buffer       │
│ - Unbiased Â_i             │         │ - Stratified, weighted   │
│ - Policy update            │         │ - DDE Hopf bounded       │
└────────────────────────────┘         └──────────────────────────┘
```

### Tradeoff vs. pure SP-IWPER

Re-introduces a GPU-throughput bottleneck for the Actor — policy gradient
must pause to wait for synchronous rollouts. Sacrifices pure async speed.

### Tradeoff vs. pure GRPO

Preserves the Phase-3 Round 3 fix for Stage 4 joint fine-tune. EBM +
verifier still get continuous high-throughput stratified hard-negative
mined data → avoids mode collapse + satisfies DDE Hopf. GRPO preserves
on-policy mathematical integrity.

---

## Group size dependency

| G size | Outcome C severity | Hybrid viability |
|---|---|---|
| Small (4-8) | Worsens variance — single hard buffer sample warps advantage | **Highly viable** — sync generation of small G minimizes Actor bottleneck |
| Large (64+) | Locks in bias deterministically | **Cripples** — sync generation of 64 traj/prompt = massive latency |

**Pareto frontier:** sharp tradeoff between GRPO advantage stability
(favors large G) and hybrid throughput viability (favors small G).
**Architecture demands operating at the smallest G capable of sustaining
reliable policy convergence.**

---

## Drift check (per `feedback_carnot_prediction_pattern.md`)

- ✅ **No parameter prescriptions** — no specific G value, no ε, no KL
  threshold, no learning rate.
- ✅ **Compositional analysis is mathematical**, not numerical (closed-form
  bias term identified, variance scaling identified).
- ✅ **Diagnostics specify formulas**, not thresholds — directional only
  ("massive non-zero", "≈ 100%", "spikes to >50%").
- ✅ **Group size is qualitative** with named Pareto tradeoff.
- ✅ **Uncertainty acknowledgement** provides explicit escape hatch
  (Outcome B if stratification regime differs).

---

## Operational implications for .88

### Pre-prototype investigation (block .88 kickoff until resolved)

**Investigate the SP-IWPER stratification regime** as documented in
`docs/research-notes/async-replay-buffer-schema-deep-think-results.md`.
Specifically: does stratification operate across prompts (escape hatch
to Outcome B) or within prompts (locks in Outcome C)?

This is a **load-bearing investigation** — if across-prompt, the simpler
Outcome B regime applies (just need KL staleness budget). If within-
prompt, the hybrid Decoupled Dual-Stream is mandatory.

### .88 prototype architecture

**Default to the hybrid Decoupled Dual-Stream design** unless the
stratification investigation demonstrates Outcome B applies. The hybrid:

- Stage 3 GRPO Actor: synchronous on-policy
- Stage 4 DBAE + EBM + Verifier: asynchronous SP-IWPER
- Bridge: Actor deposits fresh trajectories into buffer

### Diagnostic library expansion

Combined with Q2's 7 hostile-reviewer attacks, the .88 diagnostic
library now needs **11 quantities instrumented per training step**:

- **Q2 attacks (7):** RLIG, M_dead, r_live_max, σ_E, C_grad, H_w, H_dist
- **Q3 diagnostics (4):** Δμ, clip_frac, fragmentation, sign_skew

Plus the abort thresholds from Q2 (direction + order of magnitude).

### Group size sweep in .88

Per the Pareto-frontier discussion, .88 should run a **G sweep** at
{4, 8, 16, 32} early in training — measuring sign_skew and clip_frac
to find the smallest G that keeps both within bounds. **Do not lock in
a specific G value before this sweep.** This guards against the prior
Carnot prediction-error pattern (numerical prescriptions wrong even
when methodology is right).

### Q1+Q2+Q3 = .88 prototype kickoff specification

The three Deep Think responses today combine into a complete .88
prototype kickoff specification:

- **Q1 (Energy Inversion):** corpus is the cause (validated by exp1120's
  +0.448 ΔE swing today). Phase 1 production wiring proceeds (exp1121).
- **Q2 (Phase-3 attacks):** 7 hostile-reviewer attacks define the abort
  conditions for the prototype's first 1000 steps.
- **Q3 (GRPO + SP-IWPER):** hybrid Decoupled Dual-Stream architecture
  resolves the structural incompatibility; 4 diagnostics monitor the
  composition; group size sweep before committing.

---

## Cross-validation status

This is the third Deep Think response of the day staying entirely in the
methodology / compositional-analysis lane. Pattern confirmed:
**questions framed as "what to measure / what's the structure" get well-
calibrated answers; questions framed as "what value to use" get
systematically-wrong prescriptions.**

The .88 prototype kickoff now has:
- Empirical answer to the energy inversion blocker (Q1)
- Pre-flight adversarial gates with abort thresholds (Q2)
- Compositional analysis + hybrid architecture for the RL+replay layer (Q3)

**Next blocker is the SP-IWPER stratification regime investigation**
to settle Outcome B vs. Outcome C definitively. That investigation is
read-the-spec, not Deep Think — should be ~15 min of reading.

---

## SP-IWPER stratification regime — INVESTIGATED 2026-05-01

Per `async-replay-buffer-schema-deep-think-results.md` line 205:

> **Buffer Parameters:** 50,000 max capacity. Verifier-stratified
> (4 buckets of 12.5k). Sampled via SP-IWPER (α=0.6, c=2.0).
> Evicted via lowest current TD-error.

**Stratification is verifier-based, NOT prompt-based.** Each of the 4
buckets holds samples from many prompts, segregated by which verifier
returned the verdict. When the buffer is sampled for a training batch,
trajectories are pulled from each stratum — but those trajectories will
generally come from DIFFERENT prompts.

For GRPO's relative advantage to be valid, the G trajectories in a group
must share the SAME prompt. Under the current SP-IWPER design, pulling a
G-trajectory group from the buffer would be **fragmented across prompts**
(Deep Think's Diagnostic 3 ≈ 100%).

**Conclusion: Outcome C is locked in. Escape hatch unavailable.** The
Decoupled Dual-Stream Hybrid architecture is **mandatory** for .88.

---

## Recommended next steps

1. ✅ **DONE:** SP-IWPER stratification regime investigated. Outcome C
   confirmed; escape hatch unavailable.
2. **Now:** Commit this response document for the planner.
3. **In .87 retro (exp1126):** Q1+Q2+Q3 all answered; .88 prototype
   spec is now complete with concrete architecture. Surface as input
   to .88 planning.
4. **In .88 kickoff:** Architect for **hybrid Decoupled Dual-Stream
   from day 1**. Implement the 11-diagnostic library FIRST. Run a G-
   sweep at {4, 8, 16, 32} early to find the smallest viable group
   size. Do NOT skip diagnostic 3 (fragmentation rate) — it's the
   sentinel that catches buffer regressions.

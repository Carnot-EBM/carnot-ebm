# Deep Think Q10 Response — NRGPT Non-Monotonicity Interpretation

**Status:** Response received 2026-05-03 ~20:30Z. Methodology-level interpretation with concrete distinguishing signatures.
**Source prompt:** `nrgpt-non-monotonicity-interpretation-deep-think-prompt.md`
**Pattern compliance:** clean — interpretation chosen, distinguishing signatures named with lane-checks, honest unresolvable surfaced.

---

## TL;DR

| Interpretation | Verdict | Mechanism |
|---|---|---|
| (a) Engineering — step-size overshoot | **RULED OUT** | exp1172 token-level AUROC improvement would not occur with numerical instability |
| (b) Architectural — causal-mask "apples-to-oranges" | **DOMINANT** | Causal attention + parallel updates mathematically break monotonicity |
| (c) Theoretical — non-conservative surrogate | LIKELY COMPOUND | NRGPT paper §2.3 confirms; learned inference rate trades monotonicity for AUROC |

**The category error:** expecting parallel updates in a causal sequence model to yield global monotonic descent is a measurement category error, NOT an architecture failure.

**Phase-3 scale-up:** NRGPT SURVIVES without architectural revision. The fix is in the **framing** of paper-v6.

---

## The dominant interpretation (b) — Architectural / Causal Shifting

**Mechanism (Deep Think verbatim):**

> "Inference is not a monolithic global optimization. Due to the causal attention mask, the energy landscape for any token T is parameterized by the changing representations of its preceding tokens (< T). When the recurrence block updates all tokens simultaneously, the causal boundary conditions shift. Evaluating E_{i+1}(x_{i+1}) vs E_i(x_i) for token T is comparing apples-to-oranges because the landscape itself has moved beneath the token. Monotonicity is mathematically broken by the architecture's concurrent sequence updates."

**Published evidence:** NRGPT paper §2.3 explicitly proves that only the FIRST token is mathematically guaranteed asymptotic stability (monotonic energy decrease). Subsequent tokens experience "sequential thermalization" — their energy derivatives contain cross-terms from shifting prefixes, breaking monotonicity by design until earlier tokens settle.

This is not a bug. This is what the NRGPT paper claims to do.

---

## Distinguishing signatures (Q10.1)

For the small chance that (a) is contributing or that (c) dominates over (b), Deep Think specified empirical signatures:

### (a) — Fractional Step-Size Sweep
Force inference rate α ≪ 1. If sequence-level trace resolves to strict monotonicity, prior behaviour was discrete-time integration overshoot.
**Lane-check:** theoretical argument (Lipschitz step-size bound 2/L; uniform step in heterogeneous-Lipschitz Regime C guarantees overshoot in high-curvature dimensions).

### (b) — Frozen-Prefix Evaluation
Re-run NRGPT energy recurrence for token T but freeze prefix updates. Or simpler: measure E_1's trace alone (no prefix). If isolated trace is strictly monotonic, non-monotonicity is definitively a context-shifting artifact.
**Lane-check:** PUBLISHED EVIDENCE (NRGPT paper §2.3 proves first-token monotonicity).

### (c) — Vector Field Alignment Check
Cosine similarity between Δx (representation update) and -∇E (negative energy gradient). Non-zero angle = non-conservative flow.
**Lane-check:** PUBLISHED EVIDENCE (NRGPT authors explicitly note the learned inference rate matrix can be CONSTRAINED to monotonic descent but "doesn't lead to best performing models" — they trade monotonicity for AUROC).

---

## Why exp1172 (Q10.2) rules out (a) and supports (b)+(c)

**Against (a):** If pathological numerical instability were present, per-token unrolling would AMPLIFY chaotic noise and DEGRADE AUROC. exp1172 produced AUROC > batch baseline. The dynamics are constructive, not broken. Strong evidence against (a).

**Supports (b):** Token-level evaluation BYPASSES the apples-to-oranges measurement error. Per-token energy inference evaluates each token independently at its own optimal stabilization depth, aligning with the cascaded multi-agent convergence the architecture actually performs.

**Supports (c):** Different tokens semantically require different algorithmic depths. Detaching from uniform batch schedule lets each traverse its non-conservative vector field for optimal duration.

The Q10.2 sub-question challenge ("does exp1172's positive AUROC argue against (b)?") is decisively answered: NO. Token-level success is consistent with (b)/(c), not against them.

---

## Honest unresolvable

> "From the aggregate boolean n_iters_monotone: False flag alone, Carnot cannot definitively rule out whether the non-monotonicity is purely the causal context shifting (b), or if the learned preconditioner actively abandoned the conservative gradient field (c). Resolving this degeneracy requires running the Frozen-Prefix Evaluation on the exp1163 artifact."

The Frozen-Prefix Evaluation is bounded compute (likely <1 GPU-hour) — small enough to file as a follow-on experiment but not strictly required for paper-v6 framing. The paper-v6 fix can proceed with the (b)+(c) joint interpretation; resolving which dominates is research, not framing.

---

## Phase-4 framing fix for paper-v6 (Q10.3 — the load-bearing finding)

**Carnot must PLURALIZE inference regimes in paper-v6.** Instead of claiming "all Phase-4 milestones exhibit active-inference monotonic descent," distinguish two regimes:

### Regime 1 — Monolithic Global Inference

**Milestones:** exp1156 Phase-4 sampler, exp1165 ARC-AGI pilot

**Mechanism:**
- Wraps Phase-3 substrate in rigorous mathematical integrators (MCMC, Langevin, HMC)
- Operates over globally symmetric / non-causal states
- Enforces strict physical transition kernels on stationary goal-state
- Native Lyapunov free-energy minimization
- Predicts: `energy_trace_monotone_fraction = 1.0` (CONFIRMED by exp1165)

### Regime 2 — Cascaded Multi-Agent Inference

**Milestones:** exp1163 NRGPT batch, exp1172 NRGPT per-token

**Mechanism:**
- Causally-masked architecture: each token is an individual active-inference agent
- Updates conditioned on Markov blanket (prefix tokens)
- Unrolled into fast-path neural surrogate
- Explicitly trades thermodynamic monotonicity for algorithmic speed
- Predicts: `n_iters_monotone = False` for non-first tokens (CONFIRMED by exp1163)
- AUROC improvement at per-token granularity (CONFIRMED by exp1172)

**The framing:** both regimes are valid forms of active inference. They differ in inference regime, not in active-inference status. The first builds rigorous integrators; the second amortizes inference into a learned surrogate that abandons monotonicity for compute efficiency.

**Conclusion (Deep Think verbatim):**

> "NRGPT survives Phase-3 scale-up without architectural revision. You can confidently cite it as positive evidence for Phase-4, provided the section explicitly distinguishes the rigid global free-energy minimization of the sampler from the sequential, surrogate 'thermalization' of the NRGPT causal forward pass. Expecting parallel updates in a causal sequence model to yield global monotonic descent is a category error in measurement, not a failure of the architecture."

---

## What changes in paper-v6 eAI section

The drafted paper-v6 eAI section (`docs/research-notes/paper-v5-decentralization-section-draft.md`) currently treats Phase-4 evidence as homogeneous (sampler + NRGPT + ARC pilot all cited as "active inference"). After Q10:

1. Section must DISTINGUISH the two inference regimes (monolithic vs cascaded multi-agent)
2. Cite exp1156 + exp1165 under Regime 1 (monolithic), with monotonic energy traces as positive evidence
3. Cite exp1163 + exp1172 under Regime 2 (cascaded multi-agent), with the non-monotonicity correctly characterized as "sequential thermalization" per NRGPT §2.3
4. The honest disclosure (currently "n_iters_monotone=False" treated as a finding to flag) becomes a CONFIRMING measurement of the cascaded-multi-agent regime, not a problem
5. ISSUE-13 in the audit punch-list is REFRAMED — it was originally framed as "NRGPT non-monotonicity is a finding to disclose"; the actual fix is "distinguish inference regimes so non-monotonicity is correctly framed as architectural-by-design, not a failure"

---

## Cross-validation status

```
Q1-Q5     historical                          (clean)
Q7        HMC compatibility                   (clean, self-corrected)
Q8        action representation               (clean, self-corrected)
Q9        in-situ training adversarial review (clean, 3 explicit unresolvables)
Q10       NRGPT non-monotonicity              (clean, dominant interpretation named,
                                              published evidence cited from
                                              NRGPT §2.3, honest unresolvable
                                              between (b) and (c) flagged)
```

Pattern across rounds: methodology-level interpretation, distinguishing signatures with lane-checks, honest unresolvables. Q10 is the cleanest yet — Deep Think cited the NRGPT paper directly to ground the dominant interpretation.

---

## Recommended next steps

1. ✅ **Save this response** (done — this file).
2. **Update paper-v6 draft** to pluralize Phase-4 inference regimes (Regime 1 monolithic / Regime 2 cascaded).
3. **Update ISSUE-13 framing** in the paper integrity audit — reframe from "NRGPT non-monotonicity needs disclosure" to "Phase-4 framing must distinguish inference regimes."
4. **File Frozen-Prefix Evaluation as optional .95-.96 experiment** — bounded compute (<1 GPU-hour), would resolve (b)/(c) ambiguity. Not required for paper-v6 but a clean follow-on.
5. **Update Q10 memory entry** — Phase-4 architecture is now understood as having two regimes, not one.

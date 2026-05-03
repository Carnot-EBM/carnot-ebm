# Deep Think Q10 — NRGPT Non-Monotonicity Interpretation

**Status:** PROMPT DRAFT — not yet sent to Deep Think
**Drafted:** 2026-05-03 ~20:10Z
**Strategic role:** Decide whether NRGPT survives Phase-3 substrate scale-up; determines paper-v6 framing
**Predecessors:** Q1-Q5, Q7 (HMC compatibility), Q8 (action representation), Q9 (in-situ training adversarial)
**Subject:** interpreting `n_iters_monotone=False` in exp1163 NRGPT energy-recurrence prototype

---

## Context

Carnot is an open-source energy-based-model framework. Phase-4 (active inference, committed 2026-05-02) is a parallel research track to Phase-3 (foundation-model substrate). Phase-4 wraps an external sampler around the same substrate the Phase-3 endgame targets, allowing inference-time exploration of energy landscapes via blocked-Gibbs / Langevin / surrogate-gradient HMC (per Q7's regime classification — Carnot is in Regime C, "vanilla HMC inappropriate for heterogeneous-Lipschitz ∇E").

NRGPT (arXiv 2512.16762, "Energy-based GPT Alternative") was integrated into Phase-4 as one of the architectural anchors. The integration is a small-scale prototype with two milestones:

- **exp1163** (.90, batch-level): NRGPT with N energy-recurrence blocks, batch-evaluated
- **exp1172** (.91, token-level): NRGPT extended to per-token energy inference

Both produced positive accuracy results (AUROC 0.92 batch, AUROC > batch baseline at token level). But exp1163's artifact contains a finding that has not yet been interpreted:

**`n_iters_monotone: False`** — the energy iteration that NRGPT's whole architecture is built around does NOT decrease monotonically across recurrence blocks.

The Phase-4 "active inference" framing predicts monotonic energy decrease as the inference loop converges toward a goal-state. exp1165's ARC-AGI pilot reported `energy_trace_monotone_fraction = 1.0` (consistent with the framing). But exp1163 NRGPT contradicts this for the recurrence blocks specifically.

The interpretation gap is the question. It directly determines:
- Whether NRGPT survives Phase-3 substrate scale-up (1B-7B params)
- Whether the active-inference framing of Phase-4 is correct or needs revision
- Whether paper-v6's eAI section can cite NRGPT as positive evidence
- Whether subsequent NRGPT-class work (per-token, GRPO-coupled) is well-founded

---

## What NRGPT does (concrete architecture under interpretation)

NRGPT introduces N **energy-recurrence blocks** that operate sequentially on a representation. At each block i, the model computes E_i(x) based on the input/state and produces an updated representation. The architectural intuition is that successive blocks should refine the energy estimate, with the energy converging toward a low-energy goal-state as iterations proceed. The "energy recurrence" framing positions this as analogous to Hopfield-network-style settling, gradient-descent-on-energy, or active-inference belief updates.

Concrete numbers from exp1163:

```
nrgpt_auroc_n1     = 0.9209   (single-block)
nrgpt_auroc_n3     = 0.9158   (three-block — slightly LOWER than single-block)
nrgpt_above_baseline = True   (DBAE batch baseline beaten)
n_iters_monotone   = False    (the headline interpretation gap)
nrgpt_phase3_prototype_honest_result = True
```

The `n_iters_monotone=False` field reports whether E_1(x) > E_2(x) > E_3(x) holds across the recurrence iteration. It does not.

exp1172 then extended this to per-token energy inference (AUROC > batch baseline) but did NOT track per-token monotonicity. So exp1172's positive-AUROC finding is *consistent with* but does not *resolve* the monotonicity gap.

---

## The interpretation gap

The non-monotonicity could mean any of three structurally distinct things:

### Interpretation (a) — Engineering issue

**Mechanism:** The energy iteration uses an implicit step size / damping coefficient that is mis-calibrated. With wrong step size, gradient-descent-like dynamics overshoot and oscillate, producing energy traces that increase then decrease (non-monotone) instead of decreasing monotonically. The architecture is sound; the implementation has a tunable bug.

**Implication if true:** NRGPT survives Phase-3 scale-up unchanged. Engineering fix produces monotonic convergence. Paper-v6 framing is correct as-is.

### Interpretation (b) — Architectural issue

**Mechanism:** The recurrence blocks compute energy on representations that are themselves changing (because the previous block transformed them). Energy comparison E_i vs E_{i+1} compares energies of DIFFERENT input representations, so monotonicity is not even well-defined across blocks. The non-monotonicity is an artifact of comparing apples-to-oranges; the architecture is doing something other than what the active-inference framing predicts.

**Implication if true:** NRGPT requires architectural revision before Phase-3 scale-up — either the recurrence blocks need to share a fixed input representation (so energy comparison is meaningful), or the framing as "energy iteration" is mistaken and NRGPT is doing something else (e.g., learning a multi-scale ensemble, not iterating). Paper-v6 framing must change.

### Interpretation (c) — Theoretical issue

**Mechanism:** Active inference / free-energy minimization predicts monotonic decrease ONLY under specific conditions (e.g., a Lyapunov function whose minimum is the goal state). NRGPT's recurrence blocks don't satisfy these conditions; the architecture is mathematically distinct from active inference. The positive AUROC arises from some other mechanism (multi-step feature ensembling? implicit residual learning?). Cataloguing NRGPT under "active inference" is a category error.

**Implication if true:** NRGPT's positive AUROC is REAL but its FRAMING is wrong. Phase-4 active-inference architecture cannot cite NRGPT as evidence; the actual mechanism producing the AUROC needs to be identified and named correctly. Could re-cement Phase-4 around exp1156 sampler + exp1165 ARC pilot (which DO show monotonic energy traces) and treat NRGPT as a separate, unframed-as-active-inference workstream.

---

## The question for Deep Think

**Which of (a), (b), (c) is the correct interpretation, and what empirical signature would distinguish them?**

### Q10.1 — Distinguishing (a) from (b) from (c)

For each interpretation, what specific measurement on the existing exp1163 / exp1172 artifacts (or a small follow-on experiment with bounded compute) would distinguish it from the others?

- For (a): if step size is the issue, varying it across a sweep should produce a regime where monotonicity holds. Has any energy-recurrence literature reported step-size-dependent monotonicity transitions?
- For (b): if input representation changes between blocks, then E_i and E_{i+1} are functions of different inputs. What signature in the loss landscape or activation patterns would surface this?
- For (c): if the framing is wrong, the architecture's behaviour should match a non-active-inference mechanism (multi-scale ensemble, residual-like learning). What signature would identify the actual mechanism?

### Q10.2 — Does exp1172 (per-token AUROC > batch baseline) inform the interpretation?

The per-token extension produced a positive-AUROC result. What does this tell us about (a)/(b)/(c)?

- Under (a), the engineering fix would presumably help token-level too — but exp1172's improvement is REAL even with the engineering bug present. Does this argue against (a)?
- Under (b), the architectural mismatch would manifest at both batch and token level — but token-level still works. Does this argue against (b)?
- Under (c), the underlying mechanism (whatever it actually is) would need to explain WHY token-level extension improves AUROC even more. What mechanism plausibly produces both behaviours?

### Q10.3 — What does this tell us about the broader Phase-4 active-inference framing?

If interpretation (c) holds, NRGPT is misframed but exp1156 (sampler) and exp1165 (ARC pilot) might be genuinely active-inference. Is there an architectural commitment Carnot can hold across all three (sampler + pilot + NRGPT) that interprets them all coherently, or does NRGPT belong in a different category?

---

## Format constraints (methodology level only)

Per the project's Deep Think prediction-error pattern (`memory/feedback_carnot_prediction_pattern.md`):

> "Cross-validate every architectural prescription with Deep Think; qualitative survival claims well-calibrated, specific prescriptions systematically wrong."

Therefore Deep Think should:

- **Name the interpretation that's most likely correct**, but at a methodology level (not "the bug is on line 47"; rather "the failure mode is implicit step-size mis-calibration in iterative energy refinement").
- **For each interpretation, name an empirical signature that would distinguish it.** Each signature should be measurable on existing artifacts or a bounded follow-on experiment.
- **Lane-check** for each signature: published evidence / theoretical argument / speculation. Distinguish.
- **Honest unresolvable**: if the interpretation cannot be determined from the data Carnot has, say so. Specify what additional measurement WOULD resolve it.

What we DO NOT want:
- Specific architectural prescriptions (e.g., "you should use AdamW with β1=0.95"). The previous Deep Think rounds have shown specific prescriptions are systematically wrong.
- A claim that all three interpretations are "partially correct" without distinguishing which dominates.
- Speculation untethered from published or theoretical foundation.

---

## Cross-references for Deep Think

If Deep Think can read these in context:

- exp1163 NRGPT batch result artifact:
  `results/experiment_1163_nrgpt_energy_native_prototype.json`
  fields: nrgpt_auroc_n1, nrgpt_auroc_n3, n_iters_monotone, nrgpt_phase3_prototype_honest_result
- exp1172 NRGPT per-token extension result:
  `results/experiment_1172_nrgpt_per_token_energy_inference.json`
- NRGPT paper: arXiv 2512.16762, "Energy-based GPT Alternative"
- Phase-4 commitment context: `memory/feedback_active_inference_phase4_committed.md`
- exp1165 ARC pilot (which DOES show monotonic energy):
  `results/experiment_1165_phase4_active_inference_pilot.json`
  fields: energy_trace_monotone_fraction, phase4_solved_rate, action_count_ratio
- Q9 prior round flagged this dynamics class (failure mode #2 MCMC mixing paralysis):
  `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-results.md`
- Paper integrity ISSUE-13:
  ops/known-issues.md — "NRGPT n_iters_monotone=False — needs interpretation in paper v6"

---

## Why this prompt now (decision-leverage)

paper-v6 eAI section (drafted 2026-05-02 ~22:00Z, queued for integration after paper-v5 critical fixes ship in .94) currently cites NRGPT as one of Carnot's Phase-4 architectural anchors. The eAI section makes the load-bearing claim that Carnot is "directionally consistent with active-inference principles" using exp1156 + exp1163 + exp1165 as the empirical triad.

If interpretation (c) holds, NRGPT's framing is wrong. Citing it as Phase-4 active-inference evidence would be the kind of overclaim the paper integrity audit was designed to catch (cf. ISSUE-13). The paper revision would need to either:
- Drop NRGPT from the Phase-4 evidence triad
- Reframe NRGPT as a separate workstream (multi-scale energy ensemble?) with honest cataloguing
- Identify the actual mechanism producing NRGPT's AUROC and cite that

If interpretation (a) or (b) holds, the path forward differs:
- (a): ship engineering fix in .95-.96 (sweep step sizes), report monotonicity restored, keep current framing
- (b): redesign recurrence input handling in .95-.96 before next NRGPT scale-up

The question is decision-leverage NOW because paper-v6 should not ship with a misframing the audit would catch later. Deep Think's interpretation directly determines what paper-v6 says about NRGPT.

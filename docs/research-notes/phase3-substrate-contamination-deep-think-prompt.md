# Deep Think Prompt — Phase-3 Substrate Contamination Diagnostic

**Status:** Time-sensitive. Phase 1 production wiring landed today
(exp1121, k=5 ensemble in VerifyRepairPipeline default). The .88 planner
runs in ~75 minutes and will draft `research-roadmap-next.yaml` with
candidate Phase-3 prep tasks. Without this question answered, the
diagnostic instrumentation library task may be designed against the
production k=5 ensemble by default, baking in a substrate-contamination
risk before .88's training begins.
**Date drafted:** 2026-05-02 (UTC)
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project completed Phase 1 production wiring on 2026-05-01:

- **k=5 verifier ensemble** — Z3-AST formal, gVisor runtime,
  semantic embedding, ThinkPRM step probe, JSON schema — wired into
  `VerifyRepairPipeline` as the production default (exp1121,
  `k5_deployed_and_benchmarked`).
- **Empirically validated diversity:** max pairwise r = 0.462 on
  exp1108's 6-verifier superset (k=5 subset is the deployed default).
- **Energy inversion fixed:** exp1120 retrained on SOTA corpus,
  ΔE_OOD shifted from −0.068 (inverted) to +0.448 (healthy).
  Hypothesis A (corpus narrowness) confirmed via partial Test A on
  summary statistics.

Phase 3's DBAE-EBM 4-stage prototype is scoped for milestones .89-.90
(2-4 days from now), with prerequisite tasks in .88:

- **Phase-3 attack-probe library** — Q2 Deep Think (today) specified
  7 hostile-reviewer attacks the prototype must survive in the first
  1000 training steps.
- **Phase-3 diagnostic instrumentation library** — α_t tracking, joint
  null-space estimation, KL divergence, decoded-text diversity,
  manifold coverage. This is the .88 candidate task at risk.
- **Verifier degradation gate** — Q4 Deep Think (today) specified
  5-field telemetry + per-verifier classifiers + Hybrid Structural
  Default behavior.

### The contamination concern

Phase 3 trains a deep EBM on a verifier reward signal. The natural
default for the .88 diagnostic library is to use **the same k=5
production ensemble** that exp1121 wired. This bakes in a
contamination risk:

> **Specification gaming:** if the Phase-3 prototype trains against
> the same k=5 verifiers that the Phase-1 pipeline uses in production,
> the prototype could learn to satisfy *those specific verifiers' blind
> spots* without learning generalizable verification.

Phase 1 already documented (exp1093, .85) that the verifier suite has
a non-zero joint null space — even after dropping the rogue pair
(ThinkPRMProbe × Z3MathVerifier), max r = 0.462 with shared kernel
fraction = 0% on the static evaluation set. Under active GRPO
optimization (Stage 3 of the prototype), the policy could discover
adversarial text manifolds that exploit this kernel — a failure mode
Q2's Attack 3 (Dynamic Verifier Joint-Null Exploitation) explicitly
calls out.

The question: **how should the Phase-3 prototype's verifier suite
relate to the Phase-1 production k=5 ensemble?**

### Three plausible options

#### Option A: Same k=5 ensemble

Train Phase 3 against the production k=5. Pros: maximum signal density,
simplest infrastructure (one ensemble, one wiring path). Cons: highest
specification-gaming risk; the prototype's verification claims will
not generalize to *new* verifier suites — only to the specific ones
it was trained against.

#### Option B: Held-out verifier suite (different from k=5)

Train Phase 3 against a separate verifier suite (e.g., k=4 with
different mechanism mix, or k=6 with extra verifiers from the
.86 superset). Production deployment uses the original k=5; Phase 3's
training uses the held-out set. Pros: minimal contamination — the
production k=5 acts as an independent test set for the prototype's
generalization. Cons: lower training-time signal density; need to
maintain two verifier configurations; risk of held-out being so
different that training-time signal is uninformative for production.

#### Option C: Mixed schedule with verifier rotation

Train Phase 3 against a randomly-selected subset of verifiers per
training step (or per epoch), drawn from a larger pool (k=8 or k=10
total). Production deployment uses the k=5 default. Pros: prevents
overfitting to any specific subset's blind spots; the rotation
schedule itself is a regularizer. Cons: training instability if the
rotation is too aggressive; harder to measure per-verifier
contribution; the rotation introduces a new hyperparameter (rotation
rate) that violates Carnot's prediction-error pattern (specific
numerical values systematically wrong).

### What we want from Deep Think

We are NOT asking which option to pick — that's a value-laden
project decision dependent on engineering capacity, infrastructure
maturity, and the project's risk tolerance for specification gaming.

We ARE asking for **diagnostics that empirically distinguish "Phase-3
prototype is learning generalizable verification" from "Phase-3
prototype is specification-gaming the training-time verifier suite"**.
These diagnostics need to be measurable in the first 1000 training
steps (per the .88 prototype kickoff schedule), runnable on existing
artifact data where possible, and stable enough to act as abort
gates.

### Specific questions

1. **What diagnostics distinguish generalizable verification from
   specification gaming?** List quantities that can be logged per
   training step. For each diagnostic, specify:
   - The hypothesis it tests (generalization vs. gaming)
   - The data it requires (training-time verifier outputs, held-out
     verifier outputs, decoded text, latent activations)
   - The computed quantity (formula or pseudocode)
   - The decision threshold (direction + order of magnitude — no
     specific numerical prescriptions)

2. **Is one of the three options (A/B/C) clearly better than the
   others on theoretical grounds?** Walk through the contamination-
   risk calculus for each:
   - Specification-gaming risk magnitude
   - Verification claim generalizability
   - Diagnostic interpretability
   - Production-deployment burden

3. **For Option C (mixed rotation):** does the rotation schedule
   itself need to be principled (e.g., rotate based on recent
   per-verifier loss) or can it be uniform? What's the minimum rotation
   rate that prevents specification gaming, qualitatively?

4. **What's the minimum size of the held-out verifier suite (Option B)
   that would provide a credible generalization test?** k=2? k=3?
   The same k=5 with different specific verifiers? The answer depends
   on how independent the held-out verifiers are from the training
   set — characterize the dependency rather than prescribe a number.

5. **If Option A is chosen anyway** (for engineering simplicity), what
   are the **escape-valve diagnostics** that would catch specification
   gaming early enough to abort the training run before scaling? This
   is the worst-case scenario; we want a fallback even if the riskiest
   choice is taken.

### Constraints on output

- **NO parameter prescriptions.** Don't recommend specific verifier
  counts, rotation rates, or KL thresholds. Carnot's prediction-error
  pattern (memory: `feedback_carnot_prediction_pattern.md`) makes
  these systematically wrong.
- **DO provide diagnostic-quantity definitions** precise enough to
  implement (formulas, pseudocode, or log-line examples).
- **DO link each diagnostic to a Phase-3 architectural assumption**
  named in the synthesis above. Diagnostics that don't trace to a
  specific assumption are speculative.
- **DO acknowledge uncertainty** — if any of the three options has
  unresolvable risk that no diagnostic can catch, name it explicitly
  rather than offering false confidence.

### Output format request

```
DIAGNOSTICS:
  Diagnostic 1: <name>
    Hypothesis tested: <generalization vs. gaming>
    Architectural assumption: <quote from synthesis>
    Data required: <verifier outputs, latent state, decoded text>
    Computed quantity: <formula or pseudocode>
    Decision threshold: <direction + order of magnitude + rationale>
    Confidence calibration: <when this diagnostic is less reliable>
  Diagnostic 2: ...
  ... (4-7 diagnostics)

OPTION CONTAMINATION-RISK CALCULUS:
  Option A (same k=5): <theoretical analysis>
  Option B (held-out): <theoretical analysis>
  Option C (rotation): <theoretical analysis>
  Comparison: <which option dominates on which axes>

OPTION C ROTATION DESIGN:
  Required principled-ness: <uniform vs. loss-weighted vs. hybrid>
  Minimum rotation property: <qualitative — what makes rotation work>
  Failure mode if rotation insufficient: <which gaming pattern survives>

OPTION B HELD-OUT SIZE:
  Generalization-test requirement: <independence properties needed>
  Minimum k_held_out: <derive from independence requirement>
  Risk if held-out too small: <which contamination escapes the test>

OPTION A ESCAPE-VALVE DIAGNOSTICS:
  Worst-case fallback: <2-3 diagnostics that catch gaming under Option A>
  Abort condition: <specific signal pattern justifying training-run abort>
  Reversibility: <whether spec-gaming damage is recoverable post-abort>
```

### Cross-validation reminder

Per `feedback_carnot_prediction_pattern.md`: prior Deep Think rounds
have qualitative survival claims well-calibrated, but specific
numerical prescriptions systematically wrong. This question is in the
diagnostic-design / contamination-risk-analysis lane. If your answer
drifts toward parameter prescriptions (rotation rate = N, k_held_out
= K, threshold = T), please flag the drift explicitly.

Note: today's earlier Q4 response included a self-flag on
`min_transient_confidence` parameter, demonstrating Deep Think's
awareness of the pattern. The same self-discipline is welcome here —
disclaim parameter prescriptions inline rather than removing them.

---

## End of prompt

# Deep Think Prompt — Verifier Ensemble Degradation Gate Methodology

**Status:** Ready to send. Time-sensitive: .87 exp1121 (currently
running) is wiring AND-composition k=5 into VerifyRepairPipeline as
the production default. The verifier-failure handling policy is being
decided in production code RIGHT NOW. If hardcoded wrong, refactor will
happen under pressure when the first verifier failure occurs in
deployment.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's production verify-repair pipeline is wiring an
AND-composed k=5 verifier ensemble as the default acceptance gate for
LLM outputs. The ensemble combines five topologically-distinct
verifiers (mechanism families): Z3-AST formal verification, gVisor
runtime execution, semantic embedding probe, ThinkPRM step-level
reasoning probe, and JSON schema validation. AND-composition means an
output is accepted **only if all five verifiers return PASS**.

### Empirical validation grounding the k=5 choice

- **exp1108 (.86):** measured pairwise correlations on a 6-verifier
  superset. The k=5 subset (dropping ThinkPRMProbe OR Z3MathVerifier,
  whichever forms the rogue pair) achieves max pairwise r = 0.462,
  satisfying the < 0.5 architectural threshold from the Welch-Rankin
  Simplex bound.
- **exp1093 (.85):** measured the joint null-space fraction at 0.0%
  for the chosen k=5 subset (no shared kernel across verifiers).
- **exp1120 (.87, today):** corpus retraining flipped the energy
  inversion (ΔE_OOD: −0.068 → +0.448), validating the verifier
  ensemble's energy ordering on SOTA outputs.

### The problem

In production, **verifiers will fail intermittently** for reasons
unrelated to the input's correctness:

- **Z3-AST:** can time out on pathological constraint sets; can run
  out of memory on very deep ASTs.
- **gVisor runtime:** can crash if the candidate code triggers an
  unrecoverable syscall pattern; can time out on infinite loops.
- **Semantic embedding probe:** can fail if the embedding model has
  a network failure (HuggingFace hub outage, GPU OOM during forward
  pass).
- **ThinkPRM step probe:** can time out on outputs longer than its
  context window; can return garbage if the model checkpoint isn't
  loaded.
- **JSON schema validation:** can fail if the schema file itself is
  malformed or missing.

Under AND-composition, a verifier returning anything other than PASS
is interpreted as "this output is not verified." But this conflates
two *very different* failure modes:

1. **Transient infrastructure failure** — verifier crashed for a
   reason unrelated to the input (network blip, GPU OOM from another
   process, timeout under load). The output might be perfectly
   correct; we just couldn't check.

2. **Systematic blind spot** — verifier deterministically refuses to
   return PASS for a class of inputs (e.g., Z3 returns UNKNOWN on
   non-linear arithmetic, semantic probe times out on >2K-token
   reasoning chains). The output may or may not be correct, but the
   verifier provides no signal.

In the AND framework, both modes look identical to the pipeline:
"verifier i didn't return PASS, so reject."

### The two principles in tension

Carnot's CLAUDE.md mandates two principles that disagree on the
correct response to verifier failure:

- **Decentralization principle (Rule 5 in CLAUDE.md):** "Hardware
  portability as a political requirement, not just an engineering
  one. Nation-states, institutions, and individuals subject to
  compute-resource sanctions or supply-chain constraints must still
  be able to run Carnot." The decentralization-respecting
  interpretation is **fail-open with degraded tier**: if one
  verifier is unavailable, the user runs with k=4 ensemble and a
  clearly-labelled "decentralization-degraded" output flag. They
  still get protection from 4 verifiers, just slightly less
  redundancy.

- **Verification principle (energy is ground truth, project vision):**
  Carnot's value proposition is "second-pair-of-eyes verification
  grounded in objective energy." The verification interpretation is
  **fail-closed**: if any verifier in the AND ensemble doesn't
  return PASS, do not accept the output. Don't lie to the user about
  whether it's been verified.

These two principles can be reconciled by a per-call operator choice
(let the user pick fail-open vs fail-closed at call time), but ONLY
if the production pipeline can distinguish transient from systematic
failure modes empirically. Otherwise the operator is choosing between
"trust me" and "don't trust me" without information.

### The gap we want Deep Think to close

We are about to ship the production wiring (exp1121, completing in
~30 min). What we have:

- An ensemble that is empirically validated as k=5 viable
- Two competing principles (decentralization vs verification)
- A roughly-named idea that "transient" and "systematic" are
  different and should be handled differently
- No methodology for distinguishing them at runtime

What we DO NOT have:

- A specification of the **telemetry** the production pipeline must
  collect on each verifier failure
- A **classification rule** that separates transient from systematic
- An **operator-facing API** that lets the user make the fail-open
  vs fail-closed choice based on this classification

### Specific question

For the AND-composed k=5 verifier ensemble described above, design a
methodology for production verifier-failure handling that lets
operators choose between fail-open (decentralization-degraded tier)
and fail-closed (strict verification) at call time, **based on
runtime telemetry that empirically distinguishes transient from
systematic failures**.

Specifically:

1. **What runtime telemetry must the production pipeline collect on
   each verifier invocation?** List specific quantities (latency,
   error code, return value distribution, input characteristics)
   that downstream classification needs. Specify the data with
   enough precision that we can implement the schema without
   ambiguity.

2. **What classification rule** would distinguish transient from
   systematic failure for each of the five verifier types listed
   above? For each verifier, describe the empirical signature of
   transient failure (e.g., latency spike pattern, error code class)
   versus systematic blind spot (e.g., consistent UNKNOWN return on
   inputs with property X).

3. **How should the operator-facing API be structured** to let the
   caller make the fail-open vs fail-closed choice, ideally with
   the system providing a confidence score about whether failure
   was transient or systematic? Should the choice be per-call
   (`verify(output, on_failure="open"|"closed")`)? Per-deployment
   (a config flag)? Per-verifier (different defaults for different
   failure classes)?

4. **What's the right default behavior in the production wiring**
   that exp1121 is finalizing tonight? Specifically: in the absence
   of operator override, what happens when a verifier returns
   non-PASS? The CLAUDE.md decentralization mandate suggests
   fail-open by default; the verification mandate suggests fail-
   closed. Is there a third path (e.g., fail-closed by default but
   with mandatory telemetry that lets the operator switch on a
   per-input basis after the fact)?

### What NOT to recommend

- **Specific timeout values, error code lists, or threshold numbers.**
  The Carnot prediction-error pattern (memory:
  `feedback_carnot_prediction_pattern.md`) says specific numerical
  prescriptions are systematically wrong. Stay in the methodology /
  classification lane.
- **Generic distributed-systems advice** ("use circuit breakers",
  "implement bulkheads"). The question is specific to this 5-verifier
  AND ensemble and its principles, not a general "how to handle
  service failures" question.
- **Recommendations that require new verifier instrumentation that
  doesn't exist yet.** The question is about telemetry from the
  *existing* verifier interfaces (Z3 returns SAT/UNSAT/UNKNOWN,
  gVisor returns exit codes, etc.), not about modifying the
  verifiers themselves.

### Output format request

```
TELEMETRY SCHEMA:
  Per verifier invocation, collect:
    - Field 1: <name + type + what it captures>
    - Field 2: ...
    ...

CLASSIFICATION RULES (per-verifier signature):
  Z3-AST:
    Transient failure signature: <pattern>
    Systematic blind spot signature: <pattern>
    Discriminator: <how to tell them apart from telemetry>
  gVisor: ...
  Semantic embedding: ...
  ThinkPRM: ...
  JSON schema: ...

OPERATOR-FACING API:
  Recommended structure: <per-call / per-deployment / per-verifier>
  Confidence-score signal: <quantity the system reports + interpretation>
  Sample call signature: <Python pseudocode>

DEFAULT BEHAVIOR (production wiring):
  Recommendation: <fail-open / fail-closed / hybrid>
  Rationale: <which principle wins in the absence of operator override + why>
  Migration path: <how this default can be changed if empirical data
                   shows the chosen default is causing harm>
```

### Cross-validation reminder

Per `feedback_carnot_prediction_pattern.md`: prior Deep Think rounds
have qualitative survival claims well-calibrated, but specific
numerical prescriptions systematically wrong. This question is in the
methodology / classification-design lane (which telemetry, which
discriminator, which API shape). If your answer drifts toward specific
numerical thresholds or specific timeout values, please flag the
drift explicitly and provide the qualitative answer alongside.

Note: the methodology must respect the Carnot decentralization
mandate (CLAUDE.md): "Carnot's value proposition — second-pair-of-eyes
verification grounded in objective energy — must survive any [vendor
/ infrastructure] failures." The fail-open default cannot be a stub;
it must produce output the user can act on with full disclosure of
the degraded-verification tier.

---

## End of prompt

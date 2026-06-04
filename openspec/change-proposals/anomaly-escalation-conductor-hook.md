# Anomaly-Escalation Conductor Hook Proposal

**Status:** Proposed advisory hook only.

## Purpose

The 2026-06-03 Deep Think P3 review identified the Verification Trap: a nascent
paradigm can begin in a valley of disappointment, where a strict verifier reads
"higher error" as a dead-end and the autonomous loop auto-reconciles it away.
The requested upgrade is not verifier relaxation. It is an advisory distinction
between:

- `clean_bounded_negative`: an expected negative from a declared kill-gate or
  known bounded lineage.
- `frame_violating_anomaly`: an unexpected result that breaks the experiment
  frame, such as a failed load-bearing positive control, a contradicted
  assumption, or a measurement outside its declared prediction envelope.
- `clean_positive`: a positive terminal result with no anomaly signal.

## Proposed Operator Wiring

Use `carnot.autoresearch.anomaly_escalation_advisory.classify_negative` as a
recommend-only advisory call in the operator-controlled reconciliation path.
The operator applies the recommendation after an experiment artifact exists and
after the existing adversarial verification pass has preserved fabrication
discipline.

Recommended conductor-side behavior for the operator to implement:

1. Load the experiment artifact and any task metadata that declares prior
   expectations, kill-gates, positive controls, assumptions, or prediction
   envelopes.
2. Call `classify_negative(artifact_or_verdict)` at the start of the
   conductor's negative-result reconcile branch, after the artifact has passed
   `scripts/adversarial_verify.py` and before automatic pruning/reconciliation
   records the negative as closed.
3. Record the advisory dictionary (`recommendation`, `reason`,
   `frame_violation`) in the reconciliation log.
4. If the advisory returns `recommendation="auto_reconcile"`, continue the
   existing auto-reconciliation path. For a non-negative terminal result, keep
   the existing positive reconciliation path.
5. If the advisory returns `recommendation="escalate_to_human"`, pause pruning
   for that line and escalate to a human reviewer with the advisory reason.

## Non-Negotiable Anti-Fabrication Caveat

The hook MUST NOT auto-relax verification, suspend verifiers, lower acceptance
thresholds, incubate a paradigm automatically, edit artifacts, or prune research
state. Valley funding remains human-gated. The only anomaly action is:

`pause pruning + ask a human`

The classifier is therefore complementary to `scripts/adversarial_verify.py`.
The existing verifier detects fabrication and methodology risk; this proposal
adds a frame-audit signal over honest verdict plus prior expectation metadata.

## Conductor Scope

This proposal does not modify `scripts/research_conductor.py`. It describes the
operator wiring point for a future change. Exp 3809 wires the tuned Exp 3802
classifier through a standalone advisory wrapper, validates it with offline
replay, and emits this proposal for the operator to apply. The classifier and
wrapper are advisory only; neither can relax verification.

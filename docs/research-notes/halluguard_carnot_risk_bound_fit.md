# HalluGuard Carnot Risk-Bound Fit Audit

Run date: `20260507`

## Verdict

Full HalluGuard reproduction is not claimed.
`claim_allowed`: `false`

## Data-Driven Evidence-Availability Risk

Carnot data-driven hallucination risk means risk from missing, stale, or confounded evidence about whether the generated answer is grounded.

Available fields: `live_sota_model_inference_used`, `topk_logprobs_available`, `logits_available`, `known_verifier_label`, `balanced_label_counts`, `telemetry_adversarial_validity_verdict`, `missing_evidence_caveats`

## Reasoning-Step Risk

Carnot reasoning-driven hallucination risk means risk that a reasoning step or bounded certificate path is invalid even when evidence is present.

Available fields: `bound_is_sound`, `unsafe_mass_bounds`, `empirical_violation_rates`, `prefix_closed_constraints`, `reasoning_step_validity_limitations`

## Implemented Assumptions

- Live local SOTA telemetry artifacts report top-k logprobs and logits availability.
- FoVer-style verifier labels are present in the balanced telemetry manifest.
- BEAVER-lite artifacts report sound prefix-bound checks over live logprob provenance.
- Adversarial telemetry validity audit blocks headline telemetry claims under superficial-confound checks.
- Paper-v6 claim boundaries already avoid broad universal verifier and reproduction claims.

## Missing Assumptions

- HalluGuard NTK feature construction is not implemented in Carnot.
- Formal HalluGuard DHRB/RHRB theorem assumptions are not checked locally.
- No calibrated data-driven hallucination risk bound is certified over the deployment distribution.
- No full reasoning-step certificate proves every chain-of-thought or latent reasoning step valid.
- Current BEAVER-lite bounds cover terminal/prefix-closed constraints, not arbitrary hallucination semantics.
- The telemetry signal is known to be vulnerable to superficial or mechanical confounds from Exp 1473.
- The .114 balanced telemetry run used one live model family, not a complete HalluGuard reproduction suite.

## Allowed Wording

- Carnot has a HalluGuard-style fit audit that separates evidence-availability risk from reasoning-step risk using existing telemetry, FoVer-style labels, BEAVER-lite bounds, and verifier outcomes.
- Full HalluGuard reproduction is not claimed: NTK/certification assumptions and complete DHRB/RHRB formal checks are unimplemented.

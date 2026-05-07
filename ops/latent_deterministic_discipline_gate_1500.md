# Latent vs Deterministic Discipline Gate 1500

Spec: REQ-VERIFY-1500, SCENARIO-VERIFY-1500.

Run date: 20260507.

## Purpose

Exp 1500 sets the claim discipline for latent, energy-like, probabilistic, and
LLM-derived signals after the .114 retirements. The rule is simple:
deterministic validators dominate whenever they apply to the same decision.
Latent signals may rank, triage, or localize only after the acceptance checks
below are present.

## Gated Inputs

| Input | Finding used by this policy |
|---|---|
| `results/experiment_1499_verifier_ensemble_dry_orthogonality_v2.json` | Orthogonality matrix exists and recommends deterministic validators before generative pairwise self-verification. |
| `results/experiment_1481_semantic_energy_feasibility_audit.json` | Semantic Energy headline telemetry is retired because the best semantic proxy did not beat a superficial lexical baseline. |
| `results/experiment_1487_v1_pairwise_self_verification_vs_energy.json` | V_1 pairwise self-verification is not an active gate because it did not beat deterministic energy ranking or superficial baselines. |

## Headline Evidence

| Signal | Allowed use | Gate |
|---|---|---|
| `deterministic_executable_validators` | Headline accept/reject evidence for claims that can be checked by executable constraints, tool-result consistency, final-answer validity, or equivalent validators. | Validator applies directly to the claim and false accepts are counted on the same surface. |
| `conservative_deterministic_bounds` | Headline safety evidence for bounded unsafe-mass or prefix-closure claims. | Bound is conservative, deterministic, and does not rely on pass-only calibration rows as independent fail evidence. |

## Auxiliary Ranking Evidence

| Signal | Allowed use | Gate |
|---|---|---|
| `carnot_energy_ranking_after_validator_comparison` | Rank candidates that deterministic validators have already accepted or rejected. | Must be compared against deterministic validator decisions and matched superficial baselines. |
| `partial_trace_energy_localization_for_repair` | Localize repair spans after a deterministic failure is known. | Cannot headline answer quality by itself. |
| `query_time_memory_policy_zero_soundness_gated` | Opt-in routing or replay assistance. | Memory hits cannot bypass deterministic validators and must remain zero-soundness-gated. |
| `calibrated_probabilistic_verifiers_after_all_checks` | Secondary routing or ranking. | Requires held-out calibration, superficial-baseline comparison, and false-accept accounting. |

## Triage Evidence

| Signal | Allowed use | Gate |
|---|---|---|
| `llm_or_latent_uncertainty_for_manual_review_priority` | Queue ordering and manual-review priority. | Cannot accept, reject, or headline a claim. |
| `uncalibrated_energy_like_scores_for_debugging_only` | Debugging and experiment design. | Must remain no-claim until calibrated and compared against baselines. |
| `structured_verdict_records_for_auditability_only` | Provenance and auditability. | Schema records are not independent verifier votes. |

## Retired / No-Claim Evidence

| Signal | Status | Reason |
|---|---|---|
| `semantic_energy_headline_telemetry` | Retired from headline claims. | Exp 1481 found it confounded by a superficial lexical baseline. |
| `semantic_energy_logit_telemetry_headline` | Retired from headline claims. | .114 retired logit telemetry as a headline signal without stronger anti-confound evidence. |
| `v1_pairwise_self_verification_active_gate` | Retired from active gates. | Exp 1487 found it worse than deterministic energy ranking on matched pairs. |
| `uncalibrated_latent_or_llm_scores_without_required_checks` | No-claim evidence. | Missing one or more required acceptance checks below. |

## Required Checks Before Latent Influence

- deterministic validator comparison
- superficial-baseline comparison
- held-out calibration
- false-accept accounting

## Deterministic-First Rules

- Run applicable deterministic validators before latent, energy-like, probabilistic, or LLM-derived signals.
- When deterministic validators address the same claim, their reject decision dominates any latent accept or rank.
- Use energy ranking only after deterministic validity is known, or for repair localization rather than answer acceptance.
- Do not let memory hits, schema records, or LLM judge votes bypass deterministic validator failures.

## Superficial-Baseline Rules

- Any latent, energy-like, probabilistic, or LLM-derived signal must beat matched superficial baselines before supporting a claim.
- Matched superficial baselines must include response length, lexical overlap, format validity, or another task-specific cheap confound when applicable.
- Held-out calibration must be measured before a probabilistic or latent score can leave triage status.
- False accepts must be counted on the same decision surface before a latent signal can influence ranking or routing.

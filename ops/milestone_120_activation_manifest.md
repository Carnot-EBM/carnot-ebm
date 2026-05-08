# Milestone .120 Activation Manifest

Predecessor milestone: `2026.04.119`
Target milestone: `2026.05.120`
Run date: `20260508`
.119 criteria: `12` of `13` met

## Allowed .120 Tracks

| track | guardrail |
|---|---|
| kinetic-defense-in-depth validation | Validate THRML block-Gibbs plateau friction before sampler security claims. |
| BRAIN+Linear-AR rescue | Treat BRAIN-as-published as ruled out; benchmark only the Linear-AR rescue. |
| SpecAnn rejection record | Record HUBO-to-QUBO and level-crossing rejection without relitigating it. |
| THRML vendoring + candidate-warm-start | Fix KL mismatch by vendoring THRML and initialize inference at the candidate. |
| Soft-Gibbs Residual implementation + coverage bound | Use the n=8 prototype track before claiming paper-v6 coverage behavior. |
| ρ(C) measurement | Measure compute-dependent adversarial FPR on the k=6 corpus before headlines. |
| FR-11 v14 retention audit | Audit retained policies for mode collapse before v14/v15 claims. |
| paper-v6 §3 sampler drafting | Draft around THRML block-Gibbs, candidate warm-start, and explicit caveats. |
| AR-REINFORCE step-wise baseline | Reduce Linear-AR score-function variance before noisy-hardware claims. |
| .120 retro | Close the milestone from source artifacts and preserve claim boundaries. |

## Same-Roadmap Gates

- kinetic_defense_validation_ready: True
- brain_linear_ar_validation_ready: True
- thrml_vendoring_ready: True
- soft_gibbs_residual_ready: True
- rho_C_measurement_ready: True
- paper_v6_drafting_ready: True

## Deep Think Verdicts

- DT-BRAIN-CORRELATIONS: DT-BRAIN-CORRELATIONS response (received 2026-05-08, ~21:00Z) — **VERDICT: BRAIN-AS-PUBLISHED RULED OUT; LINEAR AR RESCUE IS A 4TH NOVEL PAPER-V6 CONTRIBUTION**
- DT-COMPOSITION: DT-COMPOSITION response (received 2026-05-08, ~20:30Z) — **MIXED VERDICT: VALUABLE FAILURE-MODE FINDINGS; TOP-LINE RECOMMENDATION CONTRADICTS PRIOR VERDICTS**
- DT-OT-RESIDUAL: DT-OT-RESIDUAL response (received 2026-05-08, ~20:00Z) — **VERDICT: LEMMA 3.4 SURVIVES BUT THEOREM 3.10 FAILS IN PRACTICE; CARNOT'S SOFT-GIBBS RESIDUAL IS A NOVEL EXTENSION**
- DT-MCMC-STATELESS: DT-MCMC-STATELESS response (received 2026-05-08, ~19:30Z) — **VERDICT: WARM-START AT THE CANDIDATE; DECOUPLE TRAINING SAMPLER FROM INFERENCE**
- DT-MCMC-NULL: DT-MCMC-NULL response (received 2026-05-08, ~19:00Z) — **VERDICT: STICK WITH GIBBS — ITS SLUGGISHNESS IS A SECURITY FEATURE**
- DT-MCMC-K1: DT-MCMC-K1 response (received 2026-05-08, ~18:30Z) — **VERDICT: K=1 PCD DIVERGES ON NON-CONVEX ISING; NEED ADAPTIVE K + SA/PT**
- DT-2: DT-2 response (received 2026-05-08, ~18:00Z) — **VERDICT: HYPOTHESIS INVERTED — RETENTIONS ARE THE BUG, NOT RETIREMENTS**
- DT-5: DT-5 response (received 2026-05-08, ~17:30Z) — **VERDICT: PAPER-V6 PUBLISHES THE C-PARAMETERIZED VERSION**
- DT-7: DT-7 response (received 2026-05-08, ~17:00Z) — **VERDICT: VENDOR THRML DIRECTLY**

## Preserved Headline Blocks

- Semantic Energy/logit headline claims
- pairwise LLM verifier headline claims
- arbitrary generated-Python verifier trust
- TSU hardware claims
- KV260 board claims
- KAN synthesis claims
- legacy small-model headline results

## THRML Scaling Sweep Retirement

- retired: True
- reason: Retire the THRML scaling sweep lineage (exp1526-1531, 1543-1544 patterns). Once THRML is vendored (exp1564), parity is constructive (KL=0 by definition); the scaling sweep becomes a paper-v6 retrospective entry, not active research.

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1560_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1560_activation_workflow

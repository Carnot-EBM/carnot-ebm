# Milestone .119 Activation Manifest

Predecessor milestone: `2026.04.118`
Target milestone: `2026.04.119`
Run date: `20260508`
.118 criteria: `13` of `14` met

## Allowed .119 Tracks

| track | guardrail |
|---|---|
| THRML independent-RNG audit | Reject byte-identical stochastic summaries before any further parity headline. |
| SATQuest oracle repair | Proof and witness evidence must remove solver-oracle false accepts. |
| SATQuest SOTA re-eval | Runs only after repaired oracle evidence reports zero false accepts. |
| unified automata/SAT/runtime contract gate | Automata masks are generation support; deterministic validators remain authority. |
| residual-drift repair | Localize repairs to forgotten commitments with deterministic replay. |
| claim-isolation scale | Budget savings must not hide deterministic failures. |
| product-line scale | Scale behind parser, feasibility, oracle, and false-accept gates. |
| FR-11 positive-utility-or-retire | Positive utility requires utility_delta > 0; no model-weight mutation. |
| ARM/EBT telemetry repair | Soft values and logprobs stay diagnostic below deterministic validators. |
| Weaver-style verification routing | Route verification compute across weak and deterministic signals with cost metrics. |
| THRML/Extropic packet update | Update readiness only after RNG evidence; no hardware execution claim. |
| milestone retro | Close .119 from source artifacts with carry-forward gates for .120. |

## Same-Roadmap Gates

- thrml_independent_rng_audit: prior_thrml_n256_ready == True; thrml_independent_rng_required == True
- satquest_oracle_repair: prior_satquest_benchmark_ready == True
- satquest_sota_reeval: prior_satquest_zero_solver_false_accepts == True
- unified_contract_gate: prior_automata_ready == True; prior_satquest_zero_solver_false_accepts == True
- fr11_positive_utility_or_retire: prior_fr11_safe_only == True; prior_fr11_positive_utility == False
- thrml_extropic_packet_update: thrml_independent_rng_required == False; prior_extropic_packet_ready == True

## Prior Readiness

- prior_automata_ready: True
- prior_satquest_benchmark_ready: True
- prior_satquest_solver_oracle_false_accepts: 3
- prior_satquest_zero_solver_false_accepts: False
- prior_residual_drift_ready: True
- prior_fr11_safe_only: True
- prior_fr11_positive_utility: False
- prior_product_line_ready: True
- prior_claim_router_ready: True
- prior_arm_ebm_diagnostic_ready: True
- prior_thrml_n256_ready: True
- prior_thrml_diverse_n64_ready: True
- thrml_independent_rng_required: True
- prior_extropic_packet_ready: True
- research_complete_has_118_entry: True

## Retired Headline Signals And Blocks

- legacy small-model headline claims
- SATQuest acceptance before oracle repair
- ARM/EBT soft-value acceptance authority
- Extropic TSU/Z1/XTR-0 hardware execution claims
- KV260 board claims
- model-weight mutation

## Mandated Local SOTA Models

- unsloth/Qwen3.6-35B-A3B-GGUF
- unsloth/gemma-4-31B-it-GGUF
- unsloth/gemma-4-26B-A4B-it-GGUF

## Continuous Self-Learning

- continuous_self_learning_required: True
- required .119 task: exp1555 FR-11 positive-utility-or-retire

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1547_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1547_activation_workflow

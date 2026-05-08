# Milestone .117 Activation Manifest

Predecessor milestone: `2026.04.116`
Target milestone: `2026.04.117`
Run date: `20260508`
.116 criteria: `13` of `13` met

## Allowed .117 Tracks

| track | guardrail |
|---|---|
| runtime-contract E2E | Integrate .116 contracts into one deterministic false-accept ledger. |
| live SOTA contract-guided repair | Use mandated local GGUF models; deterministic contracts remain authoritative. |
| CDG root-cause repair | Use dependency graphs for repair localization, not LLM-judge acceptance. |
| product-line rescue/retirement | One parser/feasibility rescue is allowed before retiring the weak branch. |
| FR-11 live policy promotion | Promote only rollback-passing query-time policies; no model-weight mutation. |
| MARCH-style claim isolation | Claim isolation is an ablation; deterministic validators are the trust boundary. |
| THRML/Carnot parity scaling | Software/simulator parity only; no TSU, synthesis, bitstream, or board claim. |

## Gated .117 Tracks

| task | track | gate |
|---|---|---|
| exp1520 | runtime_contract_e2e | exp1519.prior_runtime_contract_ready == true |
| exp1521 | live_sota_contract_guided_repair | exp1520.runtime_contract_e2e_ready == true |
| exp1522 | cdg_root_cause_repair | exp1520.runtime_contract_e2e_ready == true |
| exp1523 | product_line_rescue_retirement | exp1519.prior_product_line_benchmark_ready == true |
| exp1524 | fr11_live_policy_promotion | exp1519.prior_fr11_rollback_ready == true; exp1520.runtime_contract_e2e_ready == true |
| exp1525 | march_claim_isolation | exp1524.live_policy_promotion_ready == true |
| exp1526 | thrml_carnot_parity_scaling | exp1519.prior_thrml_conformance_ready == true |
| exp1527 | thrml_carnot_parity_scaling | exp1526.thrml_parity_n8_passed == true |
| exp1528 | thrml_carnot_parity_scaling | exp1527.thrml_parity_n16_passed == true |
| exp1529 | thrml_carnot_parity_scaling | exp1528.thrml_parity_n32_passed == true |
| exp1530 | thrml_carnot_parity_scaling | exp1529.thrml_parity_n64_passed == true |
| exp1531 | thrml_carnot_parity_scaling | exp1528.thrml_parity_n32_passed == true |

## Same-Roadmap Gates

- exp1520: exp1519.prior_runtime_contract_ready == true
- exp1521: exp1520.runtime_contract_e2e_ready == true
- exp1522: exp1520.runtime_contract_e2e_ready == true
- exp1523: exp1519.prior_product_line_benchmark_ready == true
- exp1524: exp1519.prior_fr11_rollback_ready == true; exp1520.runtime_contract_e2e_ready == true
- exp1525: exp1524.live_policy_promotion_ready == true
- exp1526: exp1519.prior_thrml_conformance_ready == true
- exp1527: exp1526.thrml_parity_n8_passed == true
- exp1528: exp1527.thrml_parity_n16_passed == true
- exp1529: exp1528.thrml_parity_n32_passed == true
- exp1530: exp1529.thrml_parity_n64_passed == true
- exp1531: exp1528.thrml_parity_n32_passed == true

## Prior Readiness

- prior_runtime_contract_ready: True
- prior_fr11_rollback_ready: True
- prior_product_line_benchmark_ready: True
- prior_thrml_conformance_ready: True
- prior_kan_shape_manifest_ready: True
- prior_kv260_property_pack_ready: True

## Retired Headline Signals And Blocks

- Semantic Energy/logit telemetry headline claims
- pairwise LLM verifier headline claims
- arbitrary generated-Python verifier trust
- TSU hardware claims
- KV260 board claims
- KAN synthesis claims
- legacy small-model headline results

## Mandated Local SOTA Models

- unsloth/Qwen3.6-35B-A3B-GGUF
- unsloth/gemma-4-31B-it-GGUF
- unsloth/gemma-4-26B-A4B-it-GGUF

## Continuous Self-Learning

- continuous_self_learning_required: True
- required .117 task: exp1524 FR-11 live policy promotion

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1519_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1519_activation_workflow

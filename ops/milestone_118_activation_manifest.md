# Milestone .118 Activation Manifest

Predecessor milestone: `2026.04.117`
Target milestone: `2026.04.118`
Run date: `20260508`
.117 criteria: `14` of `14` met

## Allowed .118 Tracks

| track | guardrail |
|---|---|
| orphan-test guard | Audit generated tests for import targets before downstream work runs. |
| automata/XGrammar/ABS contract decoding | Runtime contracts remain acceptance authority; decoding masks are auxiliary. |
| SATQuest CNF benchmark | PySAT or an equivalent deterministic solver is the oracle. |
| BEAVER-lite prefix-risk audit | Prefix/logprob bounds route risk but do not accept answers. |
| residual-drift ledger | Separate contradictions from satisfiable forgotten commitments. |
| external-feedback FR-11 skill graph | No model-weight mutation; positive utility is required for headline claims. |
| product-line scale | Scale only against parser, feasibility, oracle, and false-accept metrics. |
| claim-isolation uncertainty routing | Claim isolation must report routed budget cost and deterministic outcomes. |
| ARM/EBT soft-value diagnostics | Soft-value signals stay below deterministic validators. |
| THRML n=256/n=64-diverse stress | Software/simulator parity only; no TSU, synthesis, bitstream, or board claim. |
| Extropic Z1 readiness packet | Write access and transcript requirements without claiming hardware execution. |
| milestone retro | Close .118 with criteria accounting and .119 carry-forward gates. |

## Gated .118 Tracks

| task | track | gate |
|---|---|---|
| exp1534 | orphan_test_guard | exp1533.prior_orphan_test_incident_recorded == true |
| exp1535 | automata_contract_decoding | exp1533.prior_runtime_contract_e2e_ready == true; exp1534.orphan_test_guard_ready == true |
| exp1536 | satquest_cnf_benchmark | exp1533.prior_runtime_contract_e2e_ready == true |
| exp1537 | beaver_prefix_risk_audit | exp1535.contract_decoder_adapter_ready == true |
| exp1538 | residual_drift_ledger | exp1536.satquest_benchmark_ready == true |
| exp1539 | external_feedback_fr11_skill_graph | exp1533.prior_fr11_promotion_ready == true; exp1538.residual_drift_ledger_ready == true |
| exp1540 | product_line_scale | exp1533.prior_product_line_ready == true; exp1535.contract_decoder_adapter_ready == true |
| exp1541 | claim_isolation_uncertainty_routing | exp1533.prior_claim_isolation_ready == true; exp1537.beaver_bound_ready == true |
| exp1542 | arm_ebt_soft_value_diagnostics | exp1536.satquest_benchmark_ready == true; exp1537.beaver_bound_ready == true |
| exp1543 | thrml_n256_n64_diverse_stress | exp1533.prior_thrml_n128_ready == true |
| exp1544 | thrml_n256_n64_diverse_stress | exp1533.prior_thrml_diverse_ready == true; exp1543.thrml_parity_n256_schedule_ready == true |
| exp1545 | extropic_z1_readiness_packet | exp1543.thrml_parity_n256_schedule_ready == true; exp1544.diverse_topology_parity_n64_ready == true |

## Same-Roadmap Gates

- exp1534: exp1533.prior_orphan_test_incident_recorded == true
- exp1535: exp1533.prior_runtime_contract_e2e_ready == true; exp1534.orphan_test_guard_ready == true
- exp1536: exp1533.prior_runtime_contract_e2e_ready == true
- exp1537: exp1535.contract_decoder_adapter_ready == true
- exp1538: exp1536.satquest_benchmark_ready == true
- exp1539: exp1533.prior_fr11_promotion_ready == true; exp1538.residual_drift_ledger_ready == true
- exp1540: exp1533.prior_product_line_ready == true; exp1535.contract_decoder_adapter_ready == true
- exp1541: exp1533.prior_claim_isolation_ready == true; exp1537.beaver_bound_ready == true
- exp1542: exp1536.satquest_benchmark_ready == true; exp1537.beaver_bound_ready == true
- exp1543: exp1533.prior_thrml_n128_ready == true
- exp1544: exp1533.prior_thrml_diverse_ready == true; exp1543.thrml_parity_n256_schedule_ready == true
- exp1545: exp1543.thrml_parity_n256_schedule_ready == true; exp1544.diverse_topology_parity_n64_ready == true

## Prior Readiness

- prior_runtime_contract_e2e_ready: True
- prior_live_sota_repair_ready: True
- prior_cdg_ready: True
- prior_product_line_ready: True
- prior_fr11_promotion_ready: True
- prior_claim_isolation_ready: True
- prior_thrml_n128_ready: True
- prior_thrml_diverse_ready: True
- prior_orphan_test_incident_recorded: True

## Retired Headline Signals And Blocks

- legacy small-model headline claims
- BEAVER/logprob acceptance authority
- ARM/EBT soft-value acceptance authority
- Extropic TSU/Z1 hardware execution claims
- KV260 board claims
- model-weight mutation

## Mandated Local SOTA Models

- unsloth/Qwen3.6-35B-A3B-GGUF
- unsloth/gemma-4-31B-it-GGUF
- unsloth/gemma-4-26B-A4B-it-GGUF

## Continuous Self-Learning

- continuous_self_learning_required: True
- required .118 task: exp1539 external-feedback FR-11 skill graph

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1533_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1533_activation_workflow

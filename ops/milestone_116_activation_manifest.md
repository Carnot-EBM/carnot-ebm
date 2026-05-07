# Milestone .116 Activation Manifest

Predecessor milestone: `2026.04.115`
Target milestone: `2026.04.116`
Run date: `20260507`
.115 criteria: `12` of `12` met

## Allowed .116 Tracks

| track | guardrail |
|---|---|
| safe-DSL verifier induction | Candidate verifiers must compile through the safe DSL. |
| trigger+grammar certificate decoding | Use grammar-bounded certificates; do not trust free-form parsing. |
| executable monitor runtime | Runtime events remain replayable and false-accept checked. |
| plan-graph structural contracts | Promote deterministic structural gates, not trained-GNN headlines. |
| product-line solver oracle | Anchor feature-model claims to deterministic solver feasibility. |
| FR-11 verifier-feedback replay | Bound updates to query-time policy replay with rollback evidence. |
| trace2skill portable pack | Promote only rollback-passing skill entries with provenance. |
| THRML SamplerBackend conformance | Simulator/software conformance only; no TSU hardware claim. |
| KAN shape normalization | Normalize proxy-vs-hardware shapes before any synthesis claim. |
| KV260 source-level RTL properties | Source-level lint/property work only; no board or bitstream claim. |

## Gated .116 Tracks

| track | task | gate |
|---|---|---|
| trigger_grammar_certificate_decoding | exp1508 | exp1507.verifier_induction_ready == true |
| executable_monitor_runtime | exp1509 | exp1507.verifier_induction_ready == true; exp1508.certificate_decoder_ready == true |
| fr11_policy_rollback_replay | exp1513 | exp1512.policy_cache_ready == true |
| trace2skill_portable_pack | exp1514 | exp1513.rollback_audit_passed == true |
| thrml_samplerbackend_conformance | exp1515 | exp1506.prior_thrml_parity_ready == true |
| kan_shape_normalization | exp1516 | exp1506.prior_kan_shape_blocker_recorded == true |
| kv260_source_level_rtl_properties | exp1517 | exp1506.prior_kv260_source_track_active == true |

## Prior Readiness

- prior_trigger_certificates_ready: True
- prior_validator_compiler_ready: True
- prior_monitor_replay_ready: True
- prior_fr11_daily_eval_ready: True
- prior_thrml_parity_ready: True
- prior_kan_shape_blocker_recorded: True
- prior_kv260_source_track_active: True

## Retired Headline Signals And Blocks

- Semantic Energy/logit telemetry headline claims
- V_1 pairwise headline claims
- decoded-quality claims from injected-failure localization
- arbitrary generated-Python verifier trust
- TSU hardware claims
- KV260 board claims
- synthesis claims
- legacy small-model headline results

## Continuous Self-Learning

- continuous_self_learning_required: True
- required task: exp1512 FR-11 verifier-feedback policy cache

## Mandated Local SOTA Models

- unsloth/Qwen3.6-35B-A3B-GGUF
- unsloth/gemma-4-31B-it-GGUF
- unsloth/gemma-4-26B-A4B-it-GGUF

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1506_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1506_activation_workflow

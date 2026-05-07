# Milestone .115 Activation Manifest

Predecessor milestone: `2026.04.114`
Target milestone: `2026.04.115`
Run date: `20260507`
.114 criteria: `12` of `13` met

## Allowed .115 Tracks

| track | guardrail |
|---|---|
| Trigger-Token Certificate Export | Switch to constrained certificates only after a trigger token. |
| Prompt-to-Validator Compilation | Compile prompt constraints into executable validators with false-accept accounting. |
| interwhen-Style Monitoring | Poll intermediate traces only after certificate and validator readiness gates pass. |
| HoVer Safe-Prefix Continuation | Continue from the last verified prefix without increasing false accepts. |
| FR-11 Trace2Skill Daily Eval | Mandatory continuous self-learning work must include rot and resolver checks. |
| Artifact Reachability | Learned skills must not point at missing, stale, or unreachable evidence. |
| Verifier Orthogonality | Measure redundancy before promoting verifier ensembles. |
| Graph-Energy Adapters | Treat graph risk as deterministic adapter evidence, not a Kona/TSU claim. |
| KAN Hardware Accounting | Run no-synthesis resource accounting only; no board or accelerator claim. |
| Gated THRML Import/Parity | Classify import readiness before any simulator parity run. |

## Gated .115 Tracks

| track | task | gate |
|---|---|---|
| interwhen_style_monitoring | exp1495 | exp1493.trigger_certificate_ready == true; exp1494.validator_compiler_ready == true |
| hover_safe_prefix_continuation | exp1496 | exp1495.monitor_intervention_ready == true |
| artifact_reachability | exp1498 | exp1497.daily_eval_manifest_ready == true |
| latent_vs_deterministic_discipline_gate | exp1500 | exp1499.orthogonality_matrix_written == true |
| thrml_carnot_simulator_parity | exp1504 | exp1503.thrml_import_ready == true |

## Retired Headline Signals

- Semantic Energy/logit telemetry headline claims
- V_1 pairwise self-verification headline claims

## Carry-Forward Guardrail Blocks

- Semantic Energy/logit telemetry headline claims
- V_1 pairwise headline claims
- decoded-quality claims from injected-failure localization
- THRML parity before import readiness
- KV260 board claims
- TSU hardware claims
- GRPO/VPRM reopenings
- WOPR puzzle cartridges
- legacy small-model headline results

## Continuous Self-Learning

- continuous_self_learning_required: True
- required task: exp1497 FR-11 trace2skill daily eval with rot and resolver checks

## Mandated Local SOTA Models

- unsloth/Qwen3.6-35B-A3B-GGUF
- unsloth/gemma-4-31B-it-GGUF
- unsloth/gemma-4-26B-A4B-it-GGUF

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1492_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1492_activation_workflow

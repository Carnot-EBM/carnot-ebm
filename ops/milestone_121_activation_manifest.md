# Milestone .121 Activation Manifest

Predecessor milestone: `2026.05.120`
Target milestone: `2026.05.121`
Run date: `20260508`

## What .120 Proved

- THRML vendoring reached KL=0 parity and candidate warm-start support in exp1564
- candidate warm-start beat cold/cached starts in exp1566
- Soft-Gibbs Residual and its Jensen coverage bound were operational in exp1565/exp1570
- step-wise AR-REINFORCE baseline reduced variance in exp1571

## What .120 Falsified

- kinetic defense-in-depth for THRML block-Gibbs was falsified in exp1561
- BRAIN+Linear-AR expressivity widening was falsified in exp1562
- one FR-11 v14 retained policy showed mode-collapse predictors in exp1568

## Carried Forward To .121

- exp1569 paper-v6 sampler draft must resume with corrected prior-failure metadata
- exp1573 Extropic Z1 packet must resume with corrected prior-failure metadata
- BRAIN REINFORCE training dynamics remain untested at k=15
- FR-11 lambda-GRPO retention reversal is required for the flagged v14 policy
- exp1569: paper_v6_section_3_finalization_after_exp1569_draft
- exp1563: specann_rejection_record_verification
- docs/research-notes/iclr26-deep-think-responses.md: phase5_pcd_divergence_audit
- exp1565+exp1570: soft_gibbs_residual_production_scale_n128
- exp1561+exp1562+exp1563: mcmc_layer_free_phase5_architecture

## Allowed .121 Tracks

| track | guardrail |
|---|---|
| prior-failure repair | Fix exp1569 and exp1573 prior_failures metadata before resuming them. |
| paper-v6 sampler drafting | Draft from .120 evidence, with blocked claims marked explicitly. |
| Extropic Z1 readiness update | Update the packet as simulator/readiness work only; no Z1 execution claim. |
| BRAIN REINFORCE training dynamics | Test the training axis that exp1562 did not cover. |
| OT verification framework adoption | Adopt terminology and conflict ledger without upgrading acceptance authority. |
| DCCD/JSONSchemaBench SOTA smoke | Smoke structured outputs on mandated SOTA GGUFs; tiny models are fallback only. |
| FR-11 lambda-GRPO retention repair | Reverse only replay-confirmed mode-collapsed v14 retentions. |
| Phase-1 ship readiness | Audit software ship readiness independent of paper and hardware. |
| Z1 drift correction | Treat analog drift correction as a prerequisite, not executed hardware evidence. |
| Tenstorrent/PolarFire/Strix/KV260 hardware portfolio correction | Correct portfolio scope without board, TSU, or latency claims. |
| retro | Close .121 from source artifacts and exact gate fields. |

## Structured Gate Fields

- prior_failure_autofill_ready: True
- paper_v6_sampler_resume_ready: True
- extropic_packet_resume_ready: True
- brain_reinforce_training_ready: True
- ot_framework_adoption_ready: True
- dccd_jsonschema_smoke_ready: True
- fr11_v15_patch_ready: True
- phase1_ship_readiness_ready: True
- hardware_eval_ready: True

## Preserved Claim Blocks

- TSU/Z1 hardware execution claims
- KV260 board claims without transcripts
- legacy-small-model headline results
- soft energy/logprob scores as acceptance authority

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1574_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1574_activation_workflow

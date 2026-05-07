# GRPO/VPRM Lineage Retirement

Run date: `20260507`
Lineage: `GRPO/VPRM`
Artifact: `results/experiment_1456_grpo_vprm_lineage_consolidation_retirement.json`

## Experiments Reviewed

| experiment | verdict | measured positives | repeated blockers | retained lesson |
|---|---|---|---|---|
| exp1084 | 7349-step corpus generated | large process-reward corpus created | corpus generation alone was not a GRPO result | Process-level labels are useful infrastructure. |
| exp1111 | retrain attempted | ThinkPRM reward source available for v1/v2 | 300-sample retrain quality was not production-grade | Reward model quality must be verified on held-out slices. |
| exp1118 | positive_improvement | +4pp on 25 live-GPU eval questions | small eval slice | Energy-shaped rewards can move in-distribution slices. |
| exp1129 | positive_improvement | +8.51pp on 47 completed eval questions | evaluation wall budget hit | DRA diversity and proxy reuse were useful but budget-sensitive. |
| exp1146 | positive_below_exp1129 | +2.86pp | below Exp 1129 and reflection reward mean stayed 0.0 | Reflection reward did not add a stronger signal by itself. |
| exp1159 | structural_warmup_above_0851 | +10pp, +1.49pp over Exp 1129 | still small and lineage-specific | Structural warm-up was the best GRPO-only historical result. |
| exp1173 | training_wall_hit | none | llama.cpp runtime lacked GPU offload | Runtime prerequisites must gate training variants. |
| exp1184 | gpu_offload_prerequisite_not_met | none | CPU-only llama.cpp build | Do not score blocked runtime setup as science. |
| exp1187 | latent_grpo_no_delta | none | 0.0pp delta on the proxy | Invalid-sample masking needs actual invalid samples or a nonzero target. |
| exp1195 | missing | none | missing artifact in .93 retro | Missing gated artifacts must not be treated as successful retries. |
| exp1196 | blocked_gate_check_failed | none | prior_failures metadata incomplete | Prior-failure hygiene must be complete before reruns. |
| exp1208 | improvement_below_v4 | none | TinyV abstained on 62.5% of rewards and regressed -35pp | False-negative correction can suppress valid rewards if over-calibrated. |
| exp1209 | step_supervision_improves_over_outcome | +24pp over outcome-only baseline | process-supervision result, not a reason for GRPO v15 churn | Preserve VPS as process-supervision evidence. |
| exp1219 | root_cause_identified | diagnosed TinyV abstention root cause | diagnosis confirmed v5 regression | Abstention thresholds need calibration and reward-mass checks. |
| exp1220 | vps_training_beats_v4 | +15pp over v4 floor | VPS is the retained lesson, not open-ended GRPO expansion | Step rewards are more promising than more variant labels. |
| exp1221 | insufficient_logprob_coverage | none | -6.11pp on only 9 questions | FSPO needs logprob coverage before claims. |
| exp1235 | in_progress | none | no terminal improvement artifact | In-progress skeletons are not evidence. |
| exp1236 | missing | none | planned artifact absent | Execution-grounded credit never produced terminal evidence. |
| exp1247 | in_progress | none | no measured outcome | Simplification did not close the lineage. |
| exp1259 | in_progress | none | no measured outcome | PROGRS did not produce terminal evidence. |
| exp1272 | prime weights selected | process/outcome alignment weights selected | audit only | Verifier weighting is useful as an input to process supervision. |
| exp1273 | smoke_only_not_headline | smoke delta +83.798pp | headline_result_allowed=false | Smoke-only deltas must not become headline claims. |
| exp1289 | gated/missing | none | SOTA certificate path did not open | SOTA/DVI gates must precede headline GRPO/VPRM. |
| exp1304 | gated/missing | none | absent SOTA certificate headline result | No downstream GRPO launch without live certificate evidence. |
| exp1317 | grpo_vprm_v11_positive_headline_gate | +0.45 score delta on 40 replay cases | deterministic replay audit; no large training job or new generation | Keep as micro-audit evidence, not a mandate for variants. |
| exp1330 | missing/gated | none | DVI lossless claim gate absent | DVI acceptance gates correctly block downstream claims. |
| exp1346 | missing/gated | none | DVI lossless claim gate absent | Repeating the same gate does not add evidence. |
| exp1360 | blocked_gate_check_failed | none | missing exp1359 lossless_acceptance_claim_allowed | Final v14 attempt stayed downstream of a closed DVI gate. |
| exp1383 | grpo_v7_jury_rl_no_improvement | none | 0pp held-out improvement on all-UNKNOWN rewards | Formal-verifier reward policies need non-UNKNOWN diversity. |
| exp1388 | dvi_only_headline_allowed | DVI path integrated 59 fresh verified cases | grpo_cases_integrated=0 | Self-learning should stay DVI-only until GRPO produces positive evidence. |
| exp1393 | grpo_v8_ngrpo_no_improvement_all_unknown_retired | none | 0pp improvement, UNKNOWN rollout rate 1.0, retire_if_same_verdict=true | NGRPO calibration did not fix the zero-reward root cause. |

## Measured Positives

- Energy-shaped process rewards produced real early signal in small slices (Exp 1118 +4pp, Exp 1129 +8.51pp, Exp 1159 +10pp), but those results do not justify more variant churn after the later blocked and no-improvement path.
- Step-level process supervision is the useful lesson to keep: GRPO-VPS showed +24pp in Exp 1209 and +15pp over the v4 floor in Exp 1220.

## Repeated Blockers

- TinyV false-negative correction must be calibrated before use; Exp 1208 over-abstained on 62.5% of rewards and regressed by -35pp.
- Candidate-pool saturation is a hard blocker. Future selector work must change the candidate pool or target false-acceptance reduction instead of repeating best-of-N selection.
- Formal verifier rewards need non-UNKNOWN candidate diversity. Exp 1383 and Exp 1393 both produced 0pp held-out improvement because rewards stayed all zero or UNKNOWN.
- Gate discipline worked: SOTA certificate, DVI lossless acceptance, parse, and non-forgetting gates should block downstream GRPO/VPRM instead of launching placeholder variants.

## Future Reopen Conditions

- An operator explicitly reopens the GRPO/VPRM scope and names the prior failure mode being addressed.
- The proposal identifies a root cause not already tested by TinyV, VPS, FSPO, PROGRS, PRIME/VPRM, JURY-RL, or NGRPO variants.
- The proposal changes a prerequisite that failed before, such as non-UNKNOWN reward diversity, calibrated false-negative correction, an unsaturated candidate pool, live SOTA certificate parse/truthfulness gates, or DVI lossless acceptance.
- The proposal states a falsifiable acceptance gate: at least +10pp over the best retained v4/VPS baseline on 50 or more evaluation cases, with headline eligibility and no missing upstream artifacts.

## Final Decision

GRPO/VPRM is retired as active research scope. GRPO v15 and VPRM v15 variant proposals are blocked unless an operator explicitly reopens the scope under the conditions above.

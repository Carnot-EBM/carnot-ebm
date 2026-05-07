# HardNet++/DSP Repair Stack Retirement

Run date: `20260507`
Artifact: `results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json`

## Experiments Reviewed

| experiment | title | verdict | evidence | lesson |
|---|---|---|---|---|
| exp1147 | HardNet++-Style Projection Repair Layer for Arithmetic Constraints | projection_accurate_and_fast | 20 violations tested, projection_repair_accuracy=1.0, and projection_repair_latency_us=117.33625. | Hard projection can certify feasibility in explicit numeric domains. |
| exp1275 | FSNet Feasibility Step for Continuous EBM | feasibility_step_viable | feasibility_delta_overall=4.58324736985576 and violation_count_mean dropped from 5.0 to 0.0. | Feasibility seeking should be an operator inside repair, not a headline claim. |
| exp1276 | SnareNet Repair Layer - Gated on FSNet Feasibility Delta | adaptive_repair_improves_fsnet | final_constraint_satisfaction=0.9895926247512347 and repair_delta_over_fsnet=0.2199604492292856. | Adaptive repair improves local behavior but does not justify open-ended variants. |
| exp1291 | HardNet++ Nonlinear Repair Benchmark | hardnetpp_nonlinear_repair_viable | hardnetpp_delta_over_snarenet=1.2207222442957435 with nonlinear_repair_viable=true. | Route nonlinear residual cases to HardNet++ rather than repeated local repair. |
| exp1292 | DSP Feasibility-Channel Diagnostic | feasibility_channel_predictive_marginal | n_cases=156, feasibility_channel_auc=0.6604651162790698, repair_help_prediction_accuracy=0.6538461538461539, and false_continue_rate=0.7714285714285715. | DSP phi is useful telemetry but marginal as a learned stop signal. |
| exp1305 | HardNet++ + DSP Feasibility Stop Policy | conservative_replay_policy_useful_dsp_marginal | policy_stop_accuracy=1.0, stop_policy_precision=1.0, and baseline_dsp_continue_precision=0.6142857142857143. | Conservative replay is the retained operator gate. |
| exp1318 | HardNet++/DSP Learned Stop Policy Generalization | learned_policy_matched_conservative_replay | held_out_count=36, dsp_feasibility_auc=0.640625, stop_policy_precision=1.0, stop_policy_recall=1.0, and hardnetpp_delta_over_replay_policy=0.0. | The learned policy did not prove general value beyond replay distribution. |

## Cited Recent Constraint Papers

- HardNet++ (arXiv:2604.19669): Differentiable nonlinear projection supports Carnot's hard feasibility-first repair lesson.
- KKT-Hardnet (arXiv:2507.08124): KKT projection remains a possible future mechanism if a reopened scope needs machine-precision equality/inequality feasibility.
- SnareNet (arXiv:2602.09317): Adaptive repair layers validate feasibility repair, while Carnot's lineage shows they should not proliferate without new evidence.
- Differentiable Symbolic Planning with Feasibility Channels (arXiv:2604.02350): Feasibility channels are worth retaining as signals, but Carnot's DSP replay tests did not establish a broad learned stop rule.

## Hard Constraint Lesson

- Hard projection and repair layers are valuable when the domain has an explicit feasible set; they should remain available for continuous numeric repair and Phase-3 substrate work.
- FSNet and SnareNet-style feasibility steps reduce hard violations, but their local-linear repair behavior should stop when nonlinear residuals remain instead of spawning more variants.
- HardNet++ is the retained route for residual nonlinear feasibility cases because it reached hard feasibility where repeated local-linear repairs left violations.
- DSP feasibility channels are useful telemetry, but the measured AUC and false-continue behavior were marginal as a learned general stop rule.
- Conservative replay is the retained operator gate: stop once hard feasibility is reached, continue only when hard violations remain, and route nonlinear residuals to the certifying repair operator.
- Hard projection can certify feasibility in explicit numeric domains.
- Feasibility seeking should be an operator inside repair, not a headline claim.
- Adaptive repair improves local behavior but does not justify open-ended variants.
- Route nonlinear residual cases to HardNet++ rather than repeated local repair.
- DSP phi is useful telemetry but marginal as a learned stop signal.
- Conservative replay is the retained operator gate.
- The learned policy did not prove general value beyond replay distribution.

## Why It Is Not Active Headline Scope

The hard-constraint result is retained, but the active lineage is closed. Exp 1305 showed the useful policy was conservative replay: stop after hard feasibility, continue only while violations remain, and route nonlinear residuals to HardNet++. Exp 1318 then showed a learned policy matched that conservative replay policy with hardnetpp_delta_over_replay_policy=0.0 on the held-out split. That does not justify another HardNet++/DSP variant during .112.

## Future Reopen Conditions

- An operator explicitly reopens the line and names the new root cause that was not addressed by Exps 1292, 1305, and 1318.
- A proposal shows non-replay held-out evidence that a learned DSP or HardNet++/DSP policy improves over the conservative replay gate by a predeclared margin.
- The proposal ties the repair layer to a production verifier failure or Phase-3 continuous-latent substrate gate, not just another variant of the same repair-stack benchmark.
- The proposal includes a falsifiable acceptance gate, a fresh corpus or OOD split, and an explicit retire-if-same-verdict rule.

## Final Decision

The HardNet++/DSP repair stack is retired as active headline scope. The hard-constraint lesson stays in the project, while new HardNet++/DSP, FSNet/SnareNet, KKT-Hardnet, or DSP stop-policy variants are blocked unless the reopen conditions above are met.

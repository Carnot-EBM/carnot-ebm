# V657 ARC opportunity audit

Date: 2026-09-22

The enlarged evidence has 36 completed episodes across 12 games. Each game has
three seeds. No episode reached new progress, so the evidence does not establish
transfer beyond the observed games.

The supervisor call site was reached 6,480 times. The recorded candidate set
kept every arm's eligibility as `null` with
`call_site_exposes_only_selected_redirect_not_all_arm_eligibility`. The runtime
recorded 36 firings of `force_exploration_diversity`, one in each episode. Every
firing had `applied_redirection=false`. Therefore, these are shadow firings, not
applied supervisor choices with causal outcomes.

The next live-path bottleneck is reachability evidence. A future authorized live
capture must expose explicit per-arm eligibility and whether the chosen arm was
applied, then join that decision to a later progress outcome. Until that exists,
supervisor efficacy tuning is retired. This note does not propose or implement a
new supervisor policy.

Cost support reaches the planned 36 episodes and 12 games with matching model,
quantization, native runtime, policy, observer, budgets, and interval protocol.
However, native-forward, sampling, parsing, planner, environment, and update
time have no exclusive timers. They remain unknown. The audit therefore emits
no cost share or Amdahl bound.

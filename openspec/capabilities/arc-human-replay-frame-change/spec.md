# ARC Human Replay Frame-Change Predictor Capability Specification

**Capability:** arc-human-replay-frame-change
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines the ARC-AGI-3 human replay frame-change predictor used to rank live
exploration candidates without reading private environment internals. The
predictor consumes rendered frames only, learns or loads an action-effect
scorer when replay data is available, and falls back to deterministic legacy
candidate order when the corpus or weights are absent.

## Requirements

### REQ-ARC-FCP-4490: Frame-Only Replay Feature Contract

The repository SHALL provide a frame-change predictor module for experiment
4490 that recomputes model inputs from raw rendered replay frames, not from
mirror-supplied opaque feature vectors or `env._game` internals. The module
SHALL expose a small torch CNN with a click heatmap head and a directional
action head, plus helpers that normalize ARC frames into fixed-size tensors.

The terminal artifact at
`results/experiment_4490_human_replay_frame_change_predictor.json` SHALL record
whether the ARC Public Demo replay corpus was locally available, whether any
weights were trained or bundled, and the exact preconditions checked.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies the cached-candidate duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.
- `trained_on_human_corpus`: bare bool: true only when raw replay frames were locally available and used for training.
- `weights_bundled`: bare bool: true only after official CC0/MIT-0-compatible licensing is verified for bundled weights.
- `heldout_median_actions_before`: baseline median actions-to-first-level-up on held-out frame-only evaluation.
- `heldout_median_actions_after`: predictor-ranked median actions-to-first-level-up on the same held-out frame-only evaluation.
- `implied_efficiency_delta`: score-relevant delta in min(human/agent,1)^2 efficiency, never inferred when held-out data is missing.
- `positive_control`: synthetic clickability sanity check proving the ranking harness can detect a known win.
- `solve_rate_dropped`: guardrail bool: efficiency wins must not come from reducing solve rate.

### REQ-ARC-FCP-4491: Behavior Prior and Candidate Ranking

The repository SHALL expose a behavior-cloning action prior that combines
marginal action frequencies with optional state-conditioned click cells. The
ARC graph explorer SHALL use the prior and/or learned frame-change scorer to
rank `rich_action_candidates` while preserving the previous salience/raster
order as the stable tie-break and as the default when no scorer is supplied.

### REQ-ARC-FCP-4492: Honest Efficiency Artifact

Experiment 4490 SHALL report held-out median actions-to-first-level-up before
and after predictor ranking when a real human replay corpus is present. If the
corpus is not locally present, the artifact SHALL use a terminal-prefixed
`honest_verdict` that records the blocked resource instead of fabricating
training, weights, or efficiency deltas. The artifact SHALL include
`inference_substrate=verifier_ensemble_against_cached_candidates`,
`preconditions_checked`, and field principles for all required artifact fields.

### REQ-ARC-FCP-4493: Energy-Augmentation LOO Gate

Experiment 4492 SHALL write
`results/experiment_4492_energy_augmentation_loo_gate.json` from the
discriminative `cross_game_features_v3` leave-one-game-out verifier run. The
artifact SHALL compare `v3_loo_auroc` against the `0.503` v2 baseline and the
`0.600` deployment gate, include feature-class leave-one-game-out AUROCs and
material movement labels, and use
`inference_substrate=verifier_ensemble_against_cached_candidates`.

If `v3_loo_auroc > 0.600`, the frame-change ranking surface SHALL expose a
structural-feature energy term scored as `P(change) * (-delta_E)` while
preserving the previous stable tie-break ordering. If the gate does not pass,
the artifact SHALL honestly report which v3 feature classes moved the AUROC and
which did not, without enabling the structural energy ranking hook.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

### REQ-ARC-FCP-4495: Human Replay Corpus Staging

Experiment 4495 SHALL stage ARC Public Demo replay training shards under a
gitignored data directory so A1 loaders can reuse them without repeating a
cold upstream download. Each staged example SHALL be derived from replay
frames and actions and expose the local training contract
`frame`, `action`, `frame_delta`, and `level_progress`. The staging artifact at
`results/experiment_4495_human_replay_corpus_staging.json` SHALL record the
upstream URL, reachable mirror URL, source checksum metadata, shard checksums,
license status, attribution text when a CC BY mirror is used, and whether any
weights were committed.

If an official ARC source with CC0/MIT-0-compatible terms is reachable, the
artifact MAY record `official_license_verified=true`. If only the CC BY mirror
or an otherwise non-CC0/MIT-0 mirror is reachable, the artifact SHALL still
stage the format and attribution but SHALL record `weights_committed=false`,
`official_license_verified=false`, and a terminal-prefixed honest verdict that
does not imply bundled weights.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

### REQ-ARC-FCP-4496: Submitted-Agent Scoreboard Tracks Headline Signals

Experiment 4496 SHALL write
`results/experiment_4496_submitted_agent_scoreboard.json` as a milestone
scoreboard for the exact submitted-default ARC agent. Each scoreboard row SHALL
record the `SUBMITTED_AGENT_CONFIG` snapshot, the frame-only held-out generic
solve-rate with `env._game` blocked, and the variant-transfer rate. The artifact
SHALL keep `reproducible_total_levels` only as context and SHALL explicitly mark
it as not the headline signal.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

The artifact SHALL declare
`inference_substrate=verifier_ensemble_against_cached_candidates`, record the
precondition smoke checks for `arc_solver_kit.offline_arcade()` and Torch, and
reference the submitted-agent parity test so the reported generic solve-rate
cannot silently drift away from what ships.

### REQ-ARC-FCP-4505: Submitted-Agent Scoreboard Refresh Tracks Real Leaderboard Signal

Experiment 4505 SHALL write
`results/experiment_4505_submitted_agent_scoreboard.json` as the refreshed
.415 B2 submitted-default ARC scoreboard. The artifact SHALL report the current
`SUBMITTED_AGENT_CONFIG` snapshot, the frame-only held-out generic solve-rate
with `env._game` blocked, the variant-transfer solve-rate, and the A1
value-weight verdict. The artifact SHALL keep `reproducible_total_levels` only
as context and SHALL explicitly mark it as not the leaderboard headline.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_.
- `inference_substrate`: explicit substrate so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

The artifact SHALL declare
`inference_substrate=verifier_ensemble_against_cached_candidates`, record the
precondition smoke checks for `arc_solver_kit.offline_arcade()` and Torch,
include a focused parity gate for
`tests/python/test_arc_submitted_agent_parity.py` that records the CURRENT
`SUBMITTED_AGENT_CONFIG["value_weight"]` (originally frozen as
`value_weight==0.0` at the .415 B2 milestone; commit `0fad75f38`, PHASE A1 /
REQ-LEARN-4652, later deliberately moved it to a tiny bounded-positive value
once a compute-cost fix made that route affordable -- a legitimate policy
evolution the scoreboard now tracks dynamically rather than re-asserting the
stale literal), and make the real leaderboard signal the pair
`submitted_default_heldout_generic_solve_rate` plus `variant_transfer_rate`
rather than `reproducible_total_levels`.

### REQ-ARC-FCP-4501: Frame-Only Predictor Rerun From Staged Replay Shards

Experiment 4501 SHALL write
`results/experiment_4501_frame_change_predictor_rerun.json` from the locally
staged ARC Public Demo human replay corpus. The rerun SHALL consume only
raw-frame shard fields (`frame`, normalized `action_id`, optional click
coordinates, `frame_delta`, and `level_progress`) and SHALL NOT consume mirror
`feature_keys`, opaque state vectors, bundled third-party weights, or
`env._game` internals.

The rerun SHALL train or smoke-train the repository's small torch CNN action
effect model and emit a behavior-cloning action prior that can be passed to
`rich_action_candidates`. It SHALL record how many local examples were actually
loaded versus the 14,672-example `action_effect_dict.npz` target. If the exact
14,672-example NPZ corpus is absent but staged frame-only shards are available,
the artifact SHALL keep the run complete but mark the corpus shortfall and use
the FALSE_NEGATIVE_RISK null guard instead of fabricating a headline efficiency
win.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies the cached-candidate duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.
- `expected_action_effect_examples`: the external action-effect corpus target count, fixed at 14,672 for this rerun.
- `corpus_examples_loaded`: the exact number of frame-only shard examples loaded locally for this run.
- `feature_source`: proves features were recomputed from raw frames, not mirror feature vectors.
- `behavior_prior_emitted`: bare bool showing the behavior-cloning prior was built and can rank candidates.
- `heldout_median_actions_before`: baseline median actions-to-first-level-up on held-out frame-only evaluation.
- `heldout_median_actions_after`: predictor/prior-ranked median actions-to-first-level-up on the same held-out frame-only evaluation.
- `implied_efficiency_delta`: score-relevant delta in min(human/agent,1)^2 efficiency.
- `solve_rate_dropped`: guardrail bool; efficiency wins must not come from reducing solve rate.
- `false_negative_risk_guard`: records whether the null is interpretable because the positive control passed.

### REQ-ARC-FCP-4502: Energy-Augmented Candidate Ranking Measurement

Experiment 4502 SHALL write
`results/experiment_4502_energy_augmented_ranking.json` after the structural
energy gate has passed. The experiment SHALL combine the A2 frame-change
predictor score with an objective candidate energy over the v3 structural
feature classes and SHALL rank held-out cached candidates by
`P(frame_change) * (-delta_E)`. The predictor-only baseline SHALL rank the
same cached candidates by `P(frame_change)` alone, using the previous stable
candidate order as the tie-break.

The measurement SHALL report held-out solve-rate and action-efficiency for
both arms on the same candidate groups. If the energy-augmented arm does not
improve over predictor-only, the artifact SHALL report an honest null without
fabricating a gain. The artifact SHALL identify the v3 feature classes used for
energy, including the frame-delta and object-relational classes that moved the
4492 leave-one-game-out AUROC.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies the cached-candidate duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.
- `predictor_only_solve_rate`: held-out solve-rate for the same cached candidates ranked by P(frame_change) alone.
- `energy_augmented_solve_rate`: held-out solve-rate for the same cached candidates ranked by P(frame_change) * (-delta_E).
- `predictor_only_median_actions`: median actions-to-first-heldout-solve under predictor-only ranking.
- `energy_augmented_median_actions`: median actions-to-first-heldout-solve under energy-augmented ranking.
- `efficiency_delta_vs_predictor_only`: difference in min(human/agent,1)^2 efficiency versus predictor-only, never inferred from mismatched candidate groups.
- `energy_term_added_value`: bare bool indicating whether the energy term improved solve-rate or efficiency without reducing solve-rate.

### REQ-ARC-FCP-4511: Self-Supervised No-Op Pruning From Offline Arcade Transitions

Experiment 4511 SHALL write
`results/experiment_4511_frame_change_prune_predictor.json` from a
self-supervised offline-arcade corpus. The corpus SHALL be collected from
locally executed `(frame, action, next_frame)` transitions across the 25 public
offline games, deriving `changed` and `magnitude` only from rendered frame
differences. The optional human-replay corpus MAY bootstrap counts, but the
experiment SHALL remain complete without network or external quota.

The frame-change predictor SHALL train a small torch action-effect model and
wire it into `rich_action_candidates` as a PRUNING gate. Candidates with
`P(frame_change)` below the swept threshold SHALL be dropped before explorer
expansion, while default behavior without a threshold SHALL preserve the legacy
candidate set. The experiment SHALL compare the fixed local submission-gate
baseline of 7760 median actions against the pruned measurement and SHALL report
an honest null if pruning does not reduce median actions or reduces solve-rate.

Required field principles:

- `honest_verdict`: principle "terminal prefix complete:/success:/passed:/shipped_; e.g. success: frame_change_prune_median_actions_<n>_below_7760 OR complete: prune_no_action_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search + small-predictor scoring, NO GGUF/LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control so the delta is auditable, not a moving baseline."
- `median_actions_with_prune`: principle "the headline -- pruning's whole point is to cut this number."
- `solve_rate_baseline`: principle "pruning MUST NOT drop solve-rate; a faster agent that solves less is not a win."
- `solve_rate_with_prune`: principle "the no-regression check on solve-rate."
- `heldout_noop_precision`: principle "the predictor must generalize cross-game (pooled training) -- the StochasticGoose persist-across-games idea, measured held-out."
- `positive_control_passed`: principle "proves the harness can detect a real reduction (guards against a silently-broken metric)."
- `false_negative_risk_checked`: principle "a null result is only valid if a positive control passed -- per CLAUDE.md FALSE_NEGATIVE_RISK."
- `random_seed`: principle "determinism is the precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."

### REQ-ARC-FCP-4512: Imitation Action-Type and Sequence Prior

Experiment 4512 SHALL write
`results/experiment_4512_imitation_action_prior.json` from a behavior-cloning
prior over action types and action sequences. The prior SHALL estimate
`P(action | frame-class)` from the locally staged human replay corpus when it is
loadable. If the human replay corpus is absent or empty, the experiment SHALL
fall back to a self-supervised marginal derived from offline-arcade
`(frame, action, next_frame)` transitions, so the prior cannot null merely
because the replay corpus is unavailable.

The explorer SHALL accept the action prior as an expansion-ordering signal and
an opt-in low-likelihood pruning gate. Candidate expansion SHALL preserve the
legacy candidate order as the stable tie-break, SHALL drop only the bottom
quantile requested by the prior-prune setting, and SHALL retain at least one
candidate. Experiment 4512 SHALL compare the fixed 8-game local submission-gate
baseline of 7760 median actions against the prior-guided measurement on the
same gate games, report per-game rows, solve-rate, and median
actions-to-first-level-up, and use an honest null if the prior does not reduce
actions or reduces solve-rate.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: imitation_prior_median_actions_<n>_below_7760 OR complete: imitation_prior_no_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control, fixed."
- `median_actions_with_prior`: principle "the headline -- did the human prior cut exploration."
- `solve_rate_baseline`: principle "no-regression reference."
- `solve_rate_with_prior`: principle "the prior must not drop solve-rate."
- `prior_source`: principle "honest declaration of whether the human replays or the self-supervised fallback supplied the prior (no silent corpus dependency)."
- `positive_control_passed`: principle "proves the harness detects a real reduction."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4513: ACT-Style Adaptive Per-Step Explorer Budget

Experiment 4513 SHALL write
`results/experiment_4513_adaptive_per_step_budget.json` from an ACT-style
adaptive budget gate for the live `StepwiseExplorer`. The gate SHALL use only
already-computed, frame-local signals: value-head/candidate margin when
available, predicted no-op fraction from an already-wired frame-change or
induced-model scorer when available, and frame novelty from the rendered-state
hash. It SHALL NOT train a new model, load an LLM, submit to the leaderboard, or
claim to implement LoopWM; it SHALL frame the method as an ACT/PonderNet-style
budget controller applied to the existing explorer.

The explorer SHALL preserve legacy behavior when no adaptive threshold is
configured. When configured, each newly materialized frame SHALL compute an
ambiguity score from those signal components; scores below the swept threshold
SHALL commit the single top-ranked candidate as budget 1, while scores at or
above threshold SHALL retain the normal candidate width. The gate SHALL retain
at least one candidate and SHALL record diagnostics sufficient to audit how many
frames committed versus expanded.

Experiment 4513 SHALL sweep thresholds on the fixed 8-game local submission
gate, compare against the fixed 7760 median-actions baseline, report per-game
rows, solve-rate, and median actions-to-first-level-up, and reproduce solved
first-level-up action segments through `arc_solver_kit.reproduce`. If no swept
threshold reduces median actions without dropping solve-rate, the artifact SHALL
report an honest null. A positive control SHALL prove the harness can detect a
known reduction before a null may be interpreted.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: adaptive_budget_median_actions_<n>_below_7760 OR complete: adaptive_budget_no_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control, fixed."
- `median_actions_with_adaptive`: principle "the headline -- did skipping expansion on easy frames cut actions."
- `solve_rate_baseline`: principle "no-regression reference."
- `solve_rate_with_adaptive`: principle "the gate must not drop solve-rate."
- `ambiguity_signal_components`: principle "names the already-computed signals used (no new model/training) -- the zero-cost claim is auditable."
- `positive_control_passed`: principle "proves the harness detects a real reduction."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4514: Lazy Best-First Value-Weight Remeasure

Experiment 4514 SHALL write
`results/experiment_4514_lazy_best_first_value_weight.json` by sweeping the
submitted-default E3 explorer with `search_mode="best_first"` and lazy
frame-hash-cached value-head scoring at `value_weight` in `{0.0, 0.5, 1.0,
2.0}`. The sweep SHALL keep `value_weight=0.0` as the explicit control and
SHALL NOT rerun the known-regressed `value_weight=5.0` arm.

The live `StepwiseExplorer` SHALL preserve the full frontier: lazy top-K
selection MAY decide which candidate nodes pay the expensive value-head cost,
but SHALL NOT drop unscored tail nodes. Frontier ordering SHALL remain
depth-primary using the existing `depth + value_weight * value, depth,
-on_path` blend for nodes with value scores, while unscored tail nodes retain
their cheap priority and remain expandable. Value scores SHALL be cached by
frame hash.

The experiment SHALL measure the local submission-gate game set, report
held-out solve-rate, median actions-to-first-level-up on the fixed core
`{lp85,m0r0,sp80,vc33}`, and median per-game wall seconds for every swept
weight. A positive weight SHALL be selected only when it preserves every core
solve, reduces median core actions versus the `0.0` control, and keeps median
per-game wall time within the approximately 390-second budget. Otherwise the
artifact SHALL report an honest null and keep `SUBMITTED_VALUE_WEIGHT=0.0`.
Solved first-level-up segments SHALL be checked through
`arc_solver_kit.reproduce()`.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: lazy_value_weight_<w>_beats_0 OR complete: lazy_value_weight_null_keep_0."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade + value-head scoring, no GGUF load (1s floor)."
- `per_weight_results`: principle "the full {weight -> (solve_rate, median_actions, median_wall_s) table so the decision is auditable, not asserted."
- `control_value_weight_0`: principle "the explicit baseline -- a weight only wins if it BEATS 0 (guards the FALSE_NEGATIVE_RISK null)."
- `chosen_submitted_value_weight`: principle "the new SUBMITTED_VALUE_WEIGHT (0 if null); must keep test_arc_submitted_agent_parity.py consistent."
- `lazy_eval_speedup_confirmed`: principle "confirms the cheap-eval cost regime that distinguishes this from the .416 full-cost null."
- `false_negative_risk_checked`: principle "a null is valid only with the value_weight=0 control present."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4516: Submitted Integration Gate And Forward Navigation Loop

Experiment 4516 SHALL write
`results/experiment_4516_integration_8game_gate.json` by reading the landed
A1-A4 artifacts (`experiment_4511` through `experiment_4514`) and integrating
only non-flagged levers that pass the CORE set-containment gate: every baseline
core solve in `{lp85,m0r0,sp80,vc33}` SHALL be preserved and
`median_actions_on_core` SHALL be lower than the no-lever/value-weight-0
control. Any artifact carrying `flagged_adversarial: true` SHALL be skipped.
If no A1-A4 lever passes, the submitted configuration SHALL keep the bare
explorer and report an honest null for those levers.

The live `StepwiseExplorer` SHALL expose forward-navigation diagnostics that
count frontier navigation attempts, exact `_shortest_path` hits, partial
forward-walk hits, RESET replay fallbacks, recorded forward edges, and the
resulting `forward_walk_hit_rate`. When exact forward navigation to a selected
frontier node is unavailable, the explorer MAY walk forward to the deepest
known reachable ancestor of that node and replay only the suffix; if no
ancestor is reachable, it SHALL retain the existing RESET-replay fallback. This
navigation change SHALL NOT alter which frontier nodes are reachable and SHALL
NOT corrupt the stored `node["path"]` reconstruction. The artifact SHALL
separately call out the deepest known reachable ancestor behavior in its
nav-loop finding when that partial walk engages.

Experiment 4516 SHALL remeasure the fixed 8-game local submission gate
end-to-end on the submitted configuration after the accepted levers and
navigation fix are wired. The artifact SHALL report median actions, solve-rate,
held-out solve-rate, and an explicit finding that explains why the prior
forward-edge fix did or did not move actions. An honest null is valid only when
the same 7760 baseline is present and measured on the same gate.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: integrated_median_actions_<n>_below_7760 OR complete: no_lever_beats_7760_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end, no LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control."
- `median_actions_integrated`: principle "the HEADLINE -- the SUBMITTED-config median after wiring the winners + the nav fix."
- `levers_integrated`: principle "names which of A1-A4 (and the nav fix) were wired -- traceable to their measured deltas."
- `solve_rate_integrated`: principle "integration must not drop solve-rate (and ideally keeps >13 reproducible levels for the submission gate)."
- `heldout_solve_rate`: principle "the real transfer signal (was 0.143); integration should not regress it."
- `nav_loop_finding`: principle "the answer to why the .416 nav-edge fix did not move actions (closes candidate 5)."
- `false_negative_risk_checked`: principle "an honest null only valid with the 7760 baseline measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4518: Canonical Local Submission Metric Harness

Experiment 4518 SHALL write
`results/experiment_4518_metric_harness_canonical.json` and make
`scripts/kaggle/arc_local_submission_gate.py` the canonical A1-A4/A6 local
metric dashboard. The gate SHALL pin the fixed eight-game set
`{lp85,m0r0,sp80,vc33,cd82,ft09,su15,ls20}`, derive CORE from the verified
baseline solves `{lp85,m0r0,sp80,vc33}`, and guard the baseline median of
`7760.0` against silent movement. The canonical action metric SHALL be total
`actions` on solved CORE games for both baseline and treatment; a treatment
that reports a different action field than the baseline SHALL fail the guard
instead of mixing `actions_to_first_levelup` with total actions.

The gate SHALL expose `--lever <name>` and include a per-lever attribution row
with `median_actions_on_core`, `core_solves_preserved`, `bonus_solves`, and
the uniform delta versus the fixed baseline. The CORE set-containment verdict
SHALL be retained: every baseline CORE solve must be preserved, fringe bonus
solves SHALL be reported but never netted against a CORE loss, and the old raw
solved-count verdict SHALL NOT return. The fixture suite SHALL keep the A1
lost-CORE failure, A2 CORE-for-fringe failure, positive-control improvement,
neutral non-inferior pass, bonus reporting, and legacy baseline fallback.

The harness SHALL measure the headroom budget `B*` as the smallest candidate
budget in `{8000,12000,16000,24000}` whose baseline solved set equals the
baseline solved set at `1.5B`. The CLI default budget SHALL remain `8000`
until that measurement is available, and after measurement SHALL use `B*` as
the canonical default. The experiment artifact SHALL record the measured
headroom table and the selected default so cap changes cannot hide a slowdown.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. shipped: metric_harness_canonical_ci_guarded OR complete: metric_harness_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
- `canonical_game_set`: principle "the fixed 8 games -- pins the metric so no A-task can cherry-pick an easier subset."
- `canonical_baseline`: principle "7760, guarded against silent movement (raising-a-cap-to-hide-drift is forbidden)."
- `positive_control_passed`: principle "proves the harness can detect a real reduction (guards a silently-broken metric)."
- `tests_added_pass`: principle "Tests Must Run and Assert."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4523: Forward-Walk Navigation Cost And Frontier Batch Sweep

Experiment 4523 SHALL write
`results/experiment_4523_forward_walk_navigation.json` by sweeping the live
`StepwiseExplorer` over frontier batch sizes `{1,3,8,all}` and the
navigation-cost frontier tie-break `{false,true}` on the fixed eight-game local
submission gate. The control SHALL be exactly `k=1` with no navigation-cost
tie-break and SHALL measure the same canonical total `actions` field as every
treatment.

The live `StepwiseExplorer` SHALL expose opt-in controls for both levers while
preserving the submitted default when those controls are left at their control
values. Frontier ordering SHALL keep depth as the primary priority; navigation
cost MAY only break ties among equal-depth eligible frontier nodes and SHALL
score exact forward reachability by `_shortest_path` length before falling back
to root replay path length. When the explorer has already paid navigation to a
frontier node, the frontier batch lever MAY queue up to `k` of that node's
untested salient actions before global frontier selection is allowed to move
elsewhere; `k=1` SHALL remain byte-for-byte equivalent to the previous single
probe behavior.

Experiment 4523 SHALL report, for every swept config,
`median_actions_on_core`, `core_solves_preserved`, `reset_replay_fallbacks`,
`reset_replay_steps`, and `forward_walk_hit_rate`. The submitted config SHALL
be wired only if the gate verdict tag is `IMPROVED`, every baseline CORE solve
in `{lp85,m0r0,sp80,vc33}` is preserved, and the treatment median actions on
CORE is strictly lower than the `k=1`/no-tie-break control. Otherwise the
artifact SHALL report an honest null and leave the submitted config unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: forward_walk_median_actions_on_core_<n>_below_<control> OR complete: forward_walk_no_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search, no GGUF/LLM load (1s floor)."
- `median_actions_on_core_control`: principle "the k=1/no-tie-break baseline measured the SAME way -- the apples-to-apples control (A3 false-win guard)."
- `median_actions_on_core_best`: principle "the headline -- did amortizing the replay cut TOTAL actions on the CORE games."
- `core_solves_preserved`: principle "HARD empirical gate on {lp85,m0r0,sp80,vc33} -- under the fixed explore_budget, batch/reorder CAN drop a knife-edge solve (the .417 m0r0 mechanism); a dropped CORE solve FAILS the lever regardless of action savings."
- `nav_diagnostics_before_after`: principle "reset_replay_fallbacks + reset_replay_steps + forward_walk_hit_rate WITH vs WITHOUT -- the causal mechanism witness (did replay actually drop, or is any action change incidental)."
- `action_field_used`: principle "names the SINGLE action field both conditions were measured on (total actions on solved) -- the A3 metric-mismatch guard."
- `config_sweep`: principle "the full {k, tie_break -> (median_actions_on_core, core_solves_preserved, reset_replay_steps)} table so the decision is auditable, not asserted."
- `chosen_submitted_config`: principle "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); must keep test_arc_submitted_agent_parity.py consistent."
- `positive_control_passed`: principle "proves the harness can detect a real replay reduction (guards a silently-broken metric)."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4527: Nav Metric First-Class Submission Gate Guard

Experiment 4527 SHALL write
`results/experiment_4527_nav_metric_harness.json` after validating that the
local submission gate records the authoritative per-game score levers and the
secondary navigation diagnostics as first-class fields. The gate SHALL keep
CORE set-containment and per-level efficiency as the verdict metric; it SHALL
track each game's deepest level reached and per-level efficiency, because
solving more or deeper levels is the real score lever. The gate SHALL also
track each game's `reset_replay_steps` and `forward_walk_hit_rate` from the
live navigation diagnostics, but these diagnostics SHALL only produce a
wall-clock warning and SHALL NOT replace or demote the per-level score verdict.

The `--update-baseline` path SHALL validate the candidate baseline with
`validate_canonical_baseline` before persisting it. An efficiency-bearing
baseline SHALL include per-level efficiency for every CORE game and SHALL keep
the `lp85` per-level efficiency at or above the canonical floor, so a deflated
baseline cannot later disarm the SF-3 per-game efficiency guard.

Required field principles:

- `honest_verdict`: principle "terminal prefix; shipped: nav_metric_first_class_ci_guarded OR complete: nav_metric_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
- `nav_metric_added`: principle "names the per-game nav fields the gate now tracks -- the nav-regression early warning."
- `tests_added_pass`: principle "Tests Must Run and Assert."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4528: .417 B-Track Infra Carryforward Audit

Experiment 4528 SHALL write
`results/experiment_4528_infra_carryforward.json` by reading the upstream
`.417 B2` artifact
`results/experiment_4518_metric_harness_canonical.json` and reconciling the
canonical local submission-gate state without rerunning the live headroom
measurement unless the upstream artifact is missing required evidence. The
audit SHALL confirm whether the CORE set-containment verdict is canonical,
whether the four-or-more verdict fixtures are CI-guarded, and whether the
headroom-budget table measured `B*` as the smallest candidate whose baseline
solved set equals the baseline solved set at `1.5B`. If no candidate plateau is
present in the upstream table, the audit SHALL record `b_star_measured=false`
and keep the default budget unchanged instead of raising it blindly.

Required field principles:

- `honest_verdict`: principle "terminal prefix; shipped: infra_carryforward_complete OR complete: infra_audit_already_done."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (reads the .417 B2 artifact) unless it measures B* live (then verifier_ensemble_against_cached_candidates)."
- `b_track_status`: principle "what landed in .417 B2 vs what this task completed -- the audit trail."
- `cited_upstream_artifacts`: principle "traceability of the .417 B2 numbers this reconciles."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4524: Stop After Scored Target Level-Up

Experiment 4524 SHALL write
`results/experiment_4524_stop_after_levelup.json` by measuring the fixed
eight-game local submission gate under a run-to-completion control and a
stop-at-scored-target treatment. The control SHALL use the submitted
run-to-completion target, log actions to reach each observed level-up for every
gate game, and report the total run actions on the same `actions` field used by
the local submission gate. The treatment SHALL stop through `is_done` once the
measured scored target is reached; it SHALL NOT compare
`actions_to_first_levelup` from one arm against total `actions` from another.

The treatment SHALL be accepted only when every baseline CORE solve in
`{lp85,m0r0,sp80,vc33}` is preserved, every gate game's banked level depth is
preserved (`best_level` before versus after), and median CORE `actions` is
strictly lower than the measured control. If the treatment drops a CORE solve,
sheds any game's banked level depth, or shows no total-action overrun, the
artifact SHALL report an honest null and leave the submitted target unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: stop_after_levelup_core_actions_<n>_below_control OR complete: no_overrun_or_drops_solve_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
- `core_solves_preserved`: principle "stopping earlier MUST NOT drop a CORE solve (set-containment over {lp85,m0r0,sp80,vc33})."
- `levels_per_game_preserved`: principle "HARD gate: per-game best_level before vs after -- stopping early MUST NOT shed any game's banked level depth (the CORE gate checks the game set only; the competition scores total LEVELS, so a level-depth regression that the gate would PASS must be caught here)."
- `median_actions_on_core_control`: principle "the run-to-completion baseline, same action field."
- `median_actions_on_core_best`: principle "the headline -- did stopping at the scored target cut total actions."
- `action_field_used`: principle "single action field both conditions measured on (A3 metric-mismatch guard)."
- `positive_control_passed`: principle "proves the harness detects a real reduction."
- `false_negative_risk_checked`: principle "a null is valid only with the control present."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4547: Cached Human-Replay CNN Ranker Measurement

Experiment 4547 SHALL write
`results/experiment_4547_frame_change_predictor.json` after training the
repository's small frame-only CNN on the locally cached ARC Public Demo human
replay `(frame, action, frame_delta)` corpus. The training and scoring path
SHALL use CPU or integrated-device execution only, recompute features from
rendered frames, and SHALL NOT consume mirror `feature_keys`, bundled third
party weights, `env._game` internals, network resources, or 3090-class GPUs.

The trained CNN SHALL be exposed through `FrameChangeScorer` and passed into
`rich_action_candidates`/`StepwiseExplorer` as an ordering signal that ranks
candidate clicks and ACTION1-5 by predicted frame change while preserving the
legacy blind-BFS/salience order as the stable fallback tie-break. Experiment
4547 SHALL compare the CNN-ranked arm against the matched blind-BFS control on
the same held-out games or cached candidate groups, reporting median
actions-to-first-level-up and solve-rate for both arms. A success verdict is
allowed only when the CNN median is strictly lower than the blind median and
solve-rate is not dropped; otherwise the artifact SHALL report an honest null.

The experiment SHALL also compute a held-out transition-delta AUROC for the CNN
against the trivial `0.5` baseline. A median-actions null SHALL be considered
interpretable only when this positive control passes; otherwise the artifact
SHALL mark the false-negative risk as unchecked instead of claiming that the
ranker was fairly ruled out. If no action reduction is found after the positive
control passes, the artifact SHALL include the secondary recommended input:
hidden-field probing in the state hash for the `ka59`/`ar25` L2 stall.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: frame_change_cnn_median_actions_reduced_<n> OR complete: frame_change_cnn_no_action_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- CNN trains on cached transitions + scores offline candidates, no headline LLM load."
- `median_actions_to_first_levelup_cnn`: principle "the HEADLINE -- held-out median actions-to-first-levelup with the CNN ranker (the score-metric lever)."
- `median_actions_to_first_levelup_blind`: principle "the matched blind-BFS control measured the SAME way -- the apples-to-apples comparison."
- `solve_rate_preserved`: principle "HARD gate -- the action-efficiency win must NOT drop solve-rate (a faster agent that solves fewer games is worse)."
- `cnn_held_out_delta_auroc`: principle "the POSITIVE CONTROL -- the CNN predicts held-out transition deltas above a trivial baseline; guards a silently-broken predictor."
- `positive_control_passed`: principle "the CNN learned the action-effect signal; a median-actions null is valid only if this passed."
- `false_negative_risk_checked`: principle "a null is valid only with the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, human-replay corpus cached); pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4568: Pooled Clickability Action-Effect Predictor

Experiment 4568 SHALL write
`results/experiment_4568_clickability_action_effect_predictor.json` after
loading the local ARC Public Demo human replay shards and the local
`data/arc_transition_corpus/*.npz` self-captured transitions into a pooled
frame/action/frame-change corpus. The training path SHALL use the small
frame-only CNN action-effect model, CPU or integrated-device inference, rendered
frames only, and SHALL NOT depend on mirror feature vectors, third-party
weights, live LLM calls, or `env._game` internals.

The experiment SHALL expose the trained model through `FrameChangeScorer` and
wire that scorer into `rich_action_candidates` so candidate clicks and
ACTION1-5 are ordered by predicted frame-change probability with the existing
salience order as the no-regression tie-break. It SHALL compare the
predictor-ranked arm against the matched blind-BFS order on held-out games or
cached candidate groups, report median actions-to-first-level-up for both arms,
emit `actions_delta = baseline - with_predictor`, and gate any success verdict
on a strictly positive action delta, a bootstrap CI excluding zero, and no
solve-rate drop.

The artifact SHALL include the generic-transfer measurement shape, the
leaderboard efficiency term `min(human/agent,1)^2` from replay-derived human
action counts, a positive-control clickability check, `verifier_is_oracle=false`,
`offline_reproduced`, `chosen_submitted_config`, a `null_delta_methodology_note`
when the action delta is exactly zero, and a checksum over corpus/model/metric
inputs.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: clickability_predictor_actions_to_levelup_<n>_below_blind OR complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- the CNN trains/scores against cached transitions (no LLM load, 1s floor); fast CPU forward pass declared."
- `verifier_is_oracle`: principle "MUST be false -- learned action-effect model is oracle-DISTINCT from executable win-check."
- `median_actions_to_first_levelup_with_predictor`: principle "the HEADLINE -- held-out median actions-to-first-levelup with the predictor-ranked explorer."
- `median_actions_to_first_levelup_baseline`: principle "the blind-BFS baseline, measured the SAME way."
- `actions_delta`: principle "baseline - with_predictor; positive = fewer actions."
- `actions_delta_ci`: principle "bootstrap CI on the actions delta; efficiency claim requires the CI to exclude zero."
- `efficiency_score_min_human_agent_sq`: principle "min(human/agent,1)^2 with the human baseline from replay corpus."
- `generic_transfer_rate_with_predictor`: principle "held-out variant transfer WITH the predictor vs the 0.04 baseline."
- `solve_rate_preserved`: principle "HARD gate -- efficiency win must NOT drop solve-rate."
- `positive_control_passed`: principle "learnable-clickability control where predictor-ranking must beat blind."
- `false_negative_risk_checked`: principle "a no-value null is valid only if the positive control passed."
- `null_delta_methodology_note`: principle "present when actions_delta==0.0 -- honest no-gain null, not a measurement bug."
- `chosen_submitted_config`: principle "recommend enable predictor-ranker when successful; unchanged if null."
- `missing_verifier_gaps`: principle "if no gain, record residual generation/ranking gap."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, torch, LOCAL corpora)."

### REQ-ARC-FCP-4575: Learned-CNN Substrate Guard

The adversarial artifact reader SHALL recognize a learned CNN action-effect
artifact that declares `inference_substrate=verifier_ensemble_against_cached_candidates`
and references CNN or torch markers but no LLM/GGUF invocation as an offline
cached-candidate scoring artifact. Such an artifact SHALL use the one-second
verifier-scoring duration floor, not the sixty-second live-model floor, so a
fast CPU or integrated-device forward pass is not quarantined as
`DURATION_TOO_SHORT`.

The guard SHALL remain strict for real live-model claims: an artifact that
declares `inference_substrate=live_llm_inference` or names a GGUF/live LLM
model and reports `duration_s < 60` SHALL still emit a critical
`DURATION_TOO_SHORT` flag. The summary reader SHALL surface the substrate floor
it applied so reviewers can see why a CNN artifact used the offline floor and
why a fake live-LLM artifact used the live floor.

Experiment 4575 SHALL write
`results/experiment_4575_learned_cnn_substrate_guard.json` with required
principle-annotated fields for `honest_verdict`, `inference_substrate`,
`guard_mechanism`, `cnn_artifact_not_flagged`, `fake_llm_still_flagged`,
`tests_added_pass`, and `preconditions_checked`.

Required field principles:

- `honest_verdict`: principle "terminal prefix; shipped: learned_cnn_substrate_guard_added OR complete: learned_cnn_substrate_guard_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the guard against fixtures, no model load (1s floor)."
- `guard_mechanism`: principle "names the recognized substrate/floor + where it fires -- the fix that stops a fast-but-real CNN being quarantined."
- `cnn_artifact_not_flagged`: principle "a fast learned-CNN action-model fixture is NOT DURATION_TOO_SHORT-flagged -- the .422 A1 headline protection."
- `fake_llm_still_flagged`: principle "a live_llm_inference fixture at <60s IS still flagged -- guards against weakening the real fabrication check."
- `tests_added_pass`: principle "Tests Must Run and Assert -- both the not-flagged-on-CNN and still-flagged-on-fake-LLM cases."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4629: Graduate Action-Effect Predictor Into The Scored Live Explorer

Experiment 4629 SHALL write
`results/experiment_4629_graduate_action_effect_predictor_live.json` after
assembling the self-supervised small CNN action-effect predictor and the
PersistentAEM cross-game action-effect memory from the local
`(frame, action, next_frame)` transition corpus. The graduated scorer SHALL be
reachable from the SCORED `E3AgentPolicy` path and consumed by
`arc_graph_explore.rich_action_candidates` as a frame-only action-effect ranker,
with the bare explorer order retained as the matched control and stable
tie-break.

When both a candidate router and the action-effect ranker are supplied, the
candidate router MAY provide a stable tie-break order, but the action-effect
ranker SHALL be the final ordering pass so predicted no-op actions are
deprioritized on the submitted live path. The scorer SHALL use only rendered
frames, action ids, click coordinates, and locally cached transition effects;
it SHALL NOT inspect `env._game`, use an executable win-check as a scorer, call
live LLMs, or require 3090-class hardware.

Experiment 4629 SHALL compare the predictor-ranked live candidate ordering
against the bare explorer order on the same held-out cached public-game
candidate groups, reporting median actions-to-first-levelup, the
`min(human/agent,1)^2` efficiency term, first-win-rate, solve-rate, and a
bootstrap confidence interval on `actions_delta = bare - predictor`. A success
verdict is allowed only when the action delta is positive with a CI excluding
the bare baseline and solve-rate is preserved. If the matched bare control
passes but the live efficiency delta is zero, the artifact SHALL report the
honest null and leave `chosen_submitted_config` unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: action_effect_predictor_graduated_live_efficiency_up_<n> OR complete: action_effect_predictor_graduated_no_live_efficiency_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); the CNN is a small conv net (CPU/iGPU), declared so a fast forward-pass is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the action-effect predictor is a learned action-pruner, oracle-DISTINCT from the executable win-check (north-star §5 action-pruner role)."
- `solve_provenance`: principle "live_agent_self_discovery -- this improves the SCORED live agent's OWN action selection (arc_graph_explore/E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the predictor module is imported by arc_graph_explore (rich_action_candidates) AND reachable from E3AgentPolicy; arc_orphan_solver_lint passes (NOT orphaned)."
- `median_actions_to_first_levelup_predictor`: principle "the HEADLINE -- LIVE median actions-to-first-levelup WITH the action-effect predictor (lower = the score-term win)."
- `median_actions_to_first_levelup_bare`: principle "the matched bare-explorer actions on the SAME variants (today's no-op-burning baseline)."
- `actions_delta`: principle "bare - predictor (positive = fewer actions), emitted explicitly so a null (0) is annotated."
- `efficiency_score_term`: principle "the min(human/agent,1)^2 leaderboard efficiency term WITH the predictor (the score metric we have NONE of)."
- `actions_delta_ci`: principle "bootstrap CI on the actions delta; an efficiency claim requires the CI to exclude the bare baseline."
- `first_win_rate_delta`: principle "predictor - bare first-win-rate; emitted explicitly so a null is annotated (efficiency must not cost solves)."
- `solve_rate_preserved`: principle "HARD gate -- ranking candidates by predicted frame-change must NOT drop solve-rate vs bare."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the bare explorer ran on the SAME variants; an efficiency null is valid only then."
- `false_negative_risk_checked`: principle "true with the bare control run -- a no-efficiency null is valid only then."
- `null_delta_methodology_note`: principle "present when actions_delta==0 -- states the equality is an honest no-value null, not a bug."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the single source of truth."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (predictor on, ranking mode) -- the A6 input; 'unchanged' if null."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, E3AgentPolicy + rich_action_candidates importable); pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4641: Action-Effect Expansion Prior For Live Search

Experiment 4641 SHALL write
`results/experiment_4641_action_effect_expansion_prior_live.json` after
graduating the proven `persistent_aem_plus_optional_cnn` action-effect predictor
from a candidate ranker into a live search expansion prior. The implementation
SHALL be imported by `arc_graph_explore`, SHALL be reachable from the scored
`E3AgentPolicy` / `StepwiseExplorer` path, and SHALL bias frontier branch
expansion by predicted frame-change for each frontier node's remaining
candidate actions. The existing 4629 ranker-only behavior SHALL remain
available as the matched control.

The expansion prior SHALL treat the action-effect predictor as a learned
action-pruner, not an oracle: it may use rendered frames, candidate action ids,
click coordinates, local PersistentAEM evidence, and the optional small CNN
forward pass, but it SHALL NOT inspect `env._game`, call a win-check to score
branches, require live LLM inference, or require 3090-class hardware. Broken or
missing scorer calls SHALL fail closed to the ranker-only ordering.

Experiment 4641 SHALL compare the expansion-prior live search against the
ranker-only baseline on the same held-out public-game cached transition groups,
reporting live solve-rate, depth-of-live-solve, median actions-to-win,
first-win-rate, `solve_rate_delta`, `depth_of_live_solve_delta`,
`first_win_rate_delta`, and bootstrap confidence intervals for the solve-rate
and depth deltas. A deeper-solve success verdict is allowed only when solve-rate
or live depth improves over the ranker-only baseline with the relevant CI
excluding zero, first-win-rate does not regress, parity stays green, and
`scripts/arc_orphan_solver_lint.py` passes. If the matched ranker-only control
passes but deltas are zero, the artifact SHALL report an honest null with
`null_delta_methodology_note` and leave `chosen_submitted_config` unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: action_effect_expansion_prior_live_deeper_solve_<n> OR complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); the predictor is a small conv net (CPU/iGPU), declared so a fast forward-pass is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the action-effect expansion prior is a learned action-pruner, oracle-DISTINCT from the executable win-check (north-star section 5 action-pruner role)."
- `solve_provenance`: principle "live_agent_self_discovery -- this improves the SCORED live agent's OWN search expansion (arc_graph_explore/E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the expansion-prior module is imported by arc_graph_explore (graph_explore_solve_v2) AND reachable from E3AgentPolicy; arc_orphan_solver_lint passes (NOT orphaned)."
- `live_solve_rate_expansion`: principle "the HEADLINE -- LIVE solve-rate WITH the action-effect EXPANSION PRIOR on the SCORED agent."
- `live_solve_rate_ranker_baseline`: principle "the matched .427 ranker-only baseline solve-rate on the SAME variants (the no-regression control)."
- `solve_rate_delta`: principle "expansion - ranker_baseline (positive = the expansion prior deepened the live solve), emitted explicitly so a null (0) is annotated."
- `depth_of_live_solve_delta`: principle "max live level reached: expansion - ranker_baseline (the direct measure of converting first-win into a deeper solve -- the 2nd-level-up the wall sits at)."
- `first_win_rate_delta`: principle "expansion - ranker_baseline first-win-rate; emitted explicitly so a null is annotated (deepening must not cost first-wins)."
- `solve_rate_delta_ci`: principle "bootstrap CI on the solve-rate / depth delta; a deeper-solve claim requires the CI to exclude the ranker-only baseline."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the .427 ranker-only baseline ran on the SAME variants; a no-deeper-solve null is valid only then."
- `false_negative_risk_checked`: principle "true with the ranker-only baseline run + reachable-headroom confirmed -- a no-deeper-solve null is valid only then."
- `null_delta_methodology_note`: principle "present when a delta==0 -- states the equality is an honest no-value null, not a bug."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the single source of truth."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (expansion-prior mode) -- the A6 input; 'unchanged' if null."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, E3AgentPolicy + graph_explore_solve_v2 importable, .427 predictor artifact present); pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4715: Goal-Free Online Action Learning Driver Corrected Build

Experiment 4715 SHALL write
`results/experiment_4715_online_action_learning_driver_corrected.json` after
installing the corrected goal-free online action-learning driver into the live
cascade. The driver SHALL remain additive to the existing `cascade=True`
goal-induction path: it may propose and rank actions through the scored
`E3AgentPolicy` / `StepwiseExplorer` import closure, but it SHALL NOT remove
the existing LLM goal-induction planner or banked-solution replay path.

The online action-effect driver SHALL use self-supervised binary frame-change
labels from the agent's own transitions, perform one CPU-safe Adam/BCE update
on approximately every five observed actions, use hash-deduped examples, and
expose a coordinate head that proposes top-k ACTION6 click coordinates. Plain
dict candidate rows and ArcAction-like objects SHALL both receive the CNN term
so the coordinate head is not silently bypassed. On any observed level increase,
the driver SHALL reset per-level buffer state, optimizer state, and trainable
CNN weights to the scorer's initial cross-game prior snapshot; this reset-to-
prior behavior SHALL be measured separately from a scratch/random arm.

The corrected build SHALL also flip the cheap online dynamics floor by default:
`gated_engine_from_transitions` SHALL default to `trust_metric="cell_recall"`
while preserving explicit callers that request `"exact"`. The experiment SHALL
record the live preconditions from the operator prompt: CUDA available for
offline arms, Qwen3.5-9B-MTP GGUF cached, offline arcade and the bug-fixed
Go-Explore archive importable, and `/props` verification that the proposer
port serves Qwen rather than Gemma.

Experiment 4715 SHALL compare the frozen, online-scratch, and online-warm
arms on the experiment 4605 held-out harness or a content-addressed reuse of
the corresponding completed arm artifacts. A success verdict is allowed only
when online-warm first-win rate exceeds frozen by at least +0.05 and a
goal-free lp85/sc25 multi-level probe reaches L2 and offline-reproduces via
`arc_solver_kit.reproduce`. If the arm delta is flat and no goal-free L2 is
reproduced, the artifact SHALL report a terminal `complete:` null, name the
residual cause, keep `verifier_is_oracle=false`, and leave the submitted config
unchanged except for explicitly safe additive floors.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: online_warm_beats_frozen_<delta>_l2_<game> OR complete: online_action_learning_no_first_win_lift_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference for the live arm that loads the Qwen GGUF, or verifier_ensemble_against_cached_candidates for the offline held-out harness arm; model_specs name the GGUF."
- `online_warm_first_win`: principle "the +0.05 online-warm-over-frozen gate is the whole bet; the warm arm isolates online adaptation from scratch initialization."
- `online_scratch_first_win`: principle "the online-from-random arm isolates whether the win is online learning or warm start."
- `frozen_first_win`: principle "the frozen-prior baseline is the current submitted behavior and the no-online control."
- `online_warm_vs_frozen_delta`: principle "online_warm_first_win - frozen_first_win; >=+0.05 is the gate; emitted explicitly so a null is annotated."
- `cpu_train_step_ms`: principle "the Kaggle path is CPU under a 12h/600-RPM cap; an online step too slow to run every five actions makes the loop infeasible regardless of offline gains."
- `goal_free_l2_reached`: principle "a goal-free L2 deepening proves the wall is crossed by demoting goal-induction, not fixing it."
- `offline_reproduced`: principle "a goal-free L2 counts only if offline-reproduced via arc_solver_kit.reproduce."
- `reproduced_levels`: principle "the integer level the goal-free driver reached on the multi-level probe."
- `solve_provenance`: principle "live_agent_self_discovery for a generic goal-free L2; development_proxy if an adapter or cached arm artifact was needed."
- `verifier_is_oracle`: principle "MUST be false -- the online frame-change CNN is oracle-distinct and does not run the win-check."
- `live_path_reachable`: principle "HARD gate -- the changed E3AgentPolicy/StepwiseExplorer path is in the scored agent import closure and arc_orphan_solver_lint passes."
- `bare_control_passed`: principle "the positive control proves the held-out harness has reachable first-win headroom before accepting a flat null."
- `false_negative_risk_checked`: principle "true only when the three arms ran or were content-addressed and reachable headroom was confirmed."
- `null_methodology_note`: principle "present when online_warm_vs_frozen_delta is approximately zero; states the equality is an honest no-lift null, not a measurement bug."
- `chosen_submitted_config`: principle "the recommended submitted-agent config change: online driver on, reset-to-prior, cell_recall un-gate; unchanged if null."
- `proposer_served_model`: principle "the model served by /props; MUST be Qwen3.5-9B-MTP to guard the port-8919 Gemma confound."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records CUDA, Qwen cache, offline arcade, bug-fixed archive, and /props served Qwen."

### REQ-ARC-FCP-4726: Valid Online Driver Non-Degeneracy Gate

Experiment 4726 SHALL write
`results/experiment_4726_online_action_learning_driver_valid_test.json` before
interpreting any held-out lift from the goal-free online action-learning
driver. The experiment SHALL first verify the hard resource preconditions:
CUDA for offline training arms, cached Qwen3.5-9B-MTP GGUF weights, the offline
ARC arcade plus the bug-fixed Go-Explore archive, and a `/props`-verified Qwen
proposer on a non-8919 port. If any precondition is absent, the artifact SHALL
record `preconditions_checked` and a terminal blocked verdict instead of
fabricating an A/B measurement.

The non-degeneracy gate SHALL compare the frozen, online-scratch, and
online-warm arms before measuring lift. The gate SHALL prove that the three arms
produce distinct per-arm action distributions, that online Adam steps actually
execute with positive gradient norms, and that the online coordinate head
proposes click cells that differ from the frozen prior's click head. If this
gate fails, the artifact SHALL report
`complete: online_driver_arms_degenerate_confirmed_harness_bug` with diagnostic
evidence and SHALL NOT treat a flat first-win delta as a capability null.

When the gate passes, Experiment 4726 SHALL aggregate the held-out first-win
rates for `{frozen, online-scratch, online-warm}` on the experiment 4605
color-permuted harness, measure one CPU online train-step latency, run the
live-path orphan lint and submitted-agent parity checks, and include the
bounded lp85/sc25 goal-free L2 probe. A success verdict requires
`online_warm_vs_frozen_delta >= 0.05` or an offline-reproduced goal-free L2.
Otherwise the artifact SHALL report an honest no-lift residual only when the
non-degeneracy gate, parity, and reachable-headroom positive control pass.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: online_warm_beats_frozen_<delta>_or_l2_<game> OR complete: online_driver_arms_degenerate_confirmed_harness_bug OR complete: online_action_learning_no_first_win_lift_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference precondition for the Qwen GGUF plus verifier_ensemble_against_cached_candidates for the offline held-out arm artifacts."
- `arms_non_degenerate`: principle "first gate; true only when action distributions differ, online train steps have positive gradient norms, and online coordinate proposals differ from the frozen prior."
- `per_arm_action_distribution_distinct`: principle "explicit evidence the arm action histograms are not byte-identical."
- `online_train_steps_executed`: principle "positive-gradient Adam steps actually run; proves the online CNN trained."
- `online_warm_first_win`: principle "the +0.05 online-warm-over-frozen gate is the bet."
- `online_scratch_first_win`: principle "online-from-random arm isolates online learning from warm start."
- `frozen_first_win`: principle "frozen-prior baseline, the no-online control."
- `online_warm_vs_frozen_delta`: principle "online_warm_first_win - frozen_first_win; >=+0.05 is the first-win gate."
- `cpu_train_step_ms`: principle "CPU wall-clock for one online Adam/BCE step after about five actions."
- `goal_free_l2_reached`: principle "a goal-free L2 deepening proves the wall is crossed by demoting goal-induction."
- `offline_reproduced`: principle "a goal-free L2 counts only if offline-reproduced."
- `reproduced_levels`: principle "integer level reached by the goal-free multi-level probe."
- `solve_provenance`: principle "live_agent_self_discovery for a generic goal-free L2; development_proxy otherwise."
- `verifier_is_oracle`: principle "MUST be false; the online frame-change CNN does not run the win-check."
- `live_path_reachable`: principle "the changed E3AgentPolicy/StepwiseExplorer code is in the scored agent import closure."
- `bare_control_passed`: principle "positive control; held-out harness has reachable first-win headroom."
- `false_negative_risk_checked`: principle "true only with non-degenerate arms and reachable headroom."
- `null_delta_methodology_note`: principle "present when a flat non-degenerate delta is an honest no-lift null."
- `positive_control_passed`: principle "bool(parity_test_green AND arms_non_degenerate); gates the TAUTOLOGY null-delta exemption."
- `chosen_submitted_config`: principle "recommended submitted-agent config; unchanged for honest null."
- `proposer_served_model`: principle "the `/props`-reported model; MUST be Qwen3.5-9B-MTP."
- `parity_test_green`: principle "test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift."
- `preconditions_checked`: principle "records CUDA, Qwen cache, offline arcade, Go-Explore import, and `/props` verification."

### REQ-ARC-FCP-5360: Live-Reachable Color-Blob Salience Level-Up Attempt

Experiment 5360 SHALL write
`results/experiment_5360_arc_perception_salience_levelup_attempt_v488.json`
after a registry-prechecked ARC live-path attempt. The workflow SHALL audit the
live frame-diff/perception path for self-consistent but wrong reads, expose a
classical single-color connected-component/color-blob salience tier through the
live `E3AgentPolicy`/`StepwiseExplorer` candidate ordering path, and run a
bounded first-contact next-level attempt on a rotated shallow target such as
`re86`, `sb26`, `bp35`, or `lf52`.

The result artifact SHALL include `experiment_id`, `milestone`, `status`,
`honest_verdict`, `inference_substrate=live_arc_agent_policy`,
`solve_provenance=live_agent_self_discovery`,
`registry_precheck_completed`, `target_game`, `target_level_before`,
`perception_audit_completed`, `salience_policy_live_reachable`,
`offline_reproduced`, `reproduced_levels`, `new_level_banked`,
`actions_to_first_levelup`, `perception_error_classes`, `outer_loop_re_used`,
`registry_updated`, and `tests_run`. A credited level-up SHALL require
`offline_reproduced=true`, `reproduced_levels>=1`, and
`outer_loop_re_used=false`; otherwise the artifact SHALL report an honest null
with exact blockers.

Required field principles:

- `experiment_id`: principle "Stable id ties the artifact to this roadmap task."
- `milestone`: principle "Satisfies the per-milestone ARC standing floor for `.488`."
- `status`: principle "Lets capstone distinguish banked level, honest null, or blocked live path."
- `honest_verdict`: principle "Terminal prefix `complete:` or `blocked_` prevents ambiguous ARC progress."
- `inference_substrate`: principle "Expected value is live_arc_agent_policy for credited progress."
- `solve_provenance`: principle "Must be live_agent_self_discovery for a credited ARC solve path."
- `registry_precheck_completed`: principle "Bare boolean prevents duplicate re-solving of already banked levels."
- `target_game`: principle "Names the rotated target game for coverage auditing."
- `target_level_before`: principle "Bare integer records pre-attempt reproduced depth."
- `perception_audit_completed`: principle "Bare boolean proves the known-issues priority was exercised."
- `salience_policy_live_reachable`: principle "Bare boolean proves the live agent can reach the new mechanism."
- `offline_reproduced`: principle "Bare boolean is the ARC level-up lint and registry gate."
- `reproduced_levels`: principle "Bare integer records new live-path level count; success gate includes reproduced_levels>=1."
- `new_level_banked`: principle "Bare boolean separates real progress from diagnostics."
- `actions_to_first_levelup`: principle "Bare integer or null measures action efficiency."
- `perception_error_classes`: principle "Lists observed perception failure modes."
- `outer_loop_re_used`: principle "Bare boolean must be false for credited live-path progress."
- `registry_updated`: principle "Bare boolean records whether solve registry changed."
- `tests_run`: principle "Lists live-path, registry, and salience-policy checks."

### REQ-ARC-FCP-5373: Live-Reachable Salience Repair Re86 Level-Up Attempt

Experiment 5373 SHALL write
`results/experiment_5373_arc_salience_re86_levelup_v489.json` after a
registry-prechecked ARC live-path attempt. The workflow SHALL select `re86` L3
when the registry still records only two reproduced `re86` levels; if that depth
is already banked, it SHALL choose another non-duplicate live-path target and
record the reason. The repair SHALL be reachable from the live
`E3AgentPolicy`/`StepwiseExplorer` path and SHALL improve the Exp5360 salience
error classes by deprioritizing status bars and large flat blobs, ranking
button-like blobs ahead of flat distractors, and gating frame-diff action-effect
scores until at least one observed transition validates the scorer against real
frame change.

The result artifact SHALL include `status`, `solve_provenance`,
`registry_precheck_done`, `target_game`, `target_level_before`,
`attempted_level`, `salience_repair_live_reachable`,
`status_bar_deprioritization_enabled`,
`frame_diff_ground_truth_validated`, `button_like_blob_rank_delta`,
`offline_reproduced`, `reproduced_levels`, `new_level_banked`,
`registry_total_before`, `registry_total_after`, `live_attempt_count`,
`perception_error_classes`, `no_outer_loop_re`, `no_duplicate_solve`, and
`honest_verdict`. A credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`, `offline_reproduced=true`,
`reproduced_levels>=1`, `new_level_banked=true`, `no_outer_loop_re=true`, and
`no_duplicate_solve=true`; otherwise the artifact SHALL report an honest null
with residual perception/salience error classes.

Required field principles:

- `status`: principle "complete or honest_null; never claim a solve without registry-compatible evidence."
- `solve_provenance`: principle "must be live_agent_self_discovery for credited solves."
- `registry_precheck_done`: principle "must be true before target selection or attempts."
- `target_game`: principle "target game id after registry precheck."
- `target_level_before`: principle "reproduced level count before the attempt."
- `attempted_level`: principle "level attempted for +1 deeper progress."
- `salience_repair_live_reachable`: principle "true only if the repair is in the live agent path."
- `status_bar_deprioritization_enabled`: principle "whether the repair addresses the .488 status-bar error class."
- `frame_diff_ground_truth_validated`: principle "whether frame-diff salience is validated before committing probes."
- `button_like_blob_rank_delta`: principle "measured ranking change for button-like blobs if available."
- `offline_reproduced`: principle "true only when a new level is banked by the accepted registry/evidence path; include this exact field for ARC lint."
- `reproduced_levels`: principle "number of newly reproduced levels; success requires reproduced_levels>=1."
- `new_level_banked`: principle "true only if registry-compatible evidence banks a level not already present before this task."
- `registry_total_before`: principle "total reproduced levels before the attempt."
- `registry_total_after`: principle "total reproduced levels after the attempt."
- `live_attempt_count`: principle "number of live attempts made."
- `perception_error_classes`: principle "residual perception/salience errors observed."
- `no_outer_loop_re`: principle "must be true for credited solve."
- `no_duplicate_solve`: principle "must be true."
- `honest_verdict`: principle "one-line banked/no-bank verdict."

### REQ-ARC-FCP-5385: Geometric Salience Live-Path Level-Up Attempt

Experiment 5385 SHALL write
`results/experiment_5385_arc_geometric_salience_live_path_v490.json` after a
registry-prechecked ARC live-path attempt. The workflow SHALL prefer `re86` L3
when the registry still records fewer than three reproduced `re86` levels; if
that depth is already reproducible, it SHALL choose a non-duplicate live-path
target or emit a duplicate-blocked artifact. The salience mechanism SHALL be a
live-reachable geometric, hyperbolic, or geodesic ranking signal over the
agent-observed frame and its own observed transition stream, and SHALL NOT use
offline ground-truth BFS, public-game source inspection, per-game adapters, or
outer-loop reverse engineering for a credited solve.

The result artifact SHALL include `status`, `solve_provenance`,
`registry_precheck_done`, `target_game`, `target_level_before`,
`attempted_level`, `geometric_salience_live_reachable`,
`hyperbolic_or_geodesic_ranking_enabled`, `live_attempt_count`,
`offline_reproduced`, `no_outer_loop_re`, `no_per_game_adapter`,
`no_duplicate_solve`, `reproduced_levels`, `new_level_banked`,
`failure_mode`, and `honest_verdict`. A credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`, `offline_reproduced=true` from
replaying a live-agent-discovered solution, `no_outer_loop_re=true`,
`no_per_game_adapter=true`, `no_duplicate_solve=true`, and
`reproduced_levels >= target_level_before + 1`; otherwise the artifact SHALL
report an honest null or duplicate block with the live-path blocker.

Required field principles:

- `status`: principle "complete, honest_null, or duplicate_blocked with evidence."
- `solve_provenance`: principle "must equal live_agent_self_discovery for any credited solve."
- `registry_precheck_done`: principle "must be true."
- `target_game`: principle "selected game id."
- `target_level_before`: principle "reproduced level count before this task."
- `attempted_level`: principle "target level attempted."
- `geometric_salience_live_reachable`: principle "true only if the live agent can use the salience signal without outer-loop help."
- `hyperbolic_or_geodesic_ranking_enabled`: principle "whether the GeoWorld-inspired ranking was active."
- `live_attempt_count`: principle "number of live attempts."
- `offline_reproduced`: principle "true only if a live-agent-discovered solve was replayed/reproduced for banking; must not mean offline BFS or outer-loop reverse engineering."
- `no_outer_loop_re`: principle "must be true."
- `no_per_game_adapter`: principle "must be true."
- `no_duplicate_solve`: principle "must be true for credited deliverables."
- `reproduced_levels`: principle "level count after this task; for a credited first-contact solve this must satisfy reproduced_levels>=1, and for a deeper solve it must be at least target_level_before+1."
- `new_level_banked`: principle "true only if the live agent self-discovered a new reproducible level."
- `failure_mode`: principle "if no bank, concrete live-path blocker."
- `honest_verdict`: principle "one-line ARC outcome."

### REQ-ARC-FCP-5397: Blob Salience Generation-Stage Live-Path Level-Up Attempt

Experiment 5397 SHALL write
`results/experiment_5397_arc_blob_salience_live_path_v491.json` after a
registry-prechecked ARC live-path attempt. The workflow SHALL prefer `re86` L3
when the registry still records fewer than three reproduced `re86` levels; if
that depth is already reproducible, it SHALL choose the next unsolved reachable
target and SHALL NOT duplicate a prior solved level. The salience mechanism
SHALL segment rendered frames into single-color connected components, mask or
deprioritize status-bar regions, classify blobs into tiers from button
likelihood, salient color, size, and non-status evidence, and SHALL apply those
tiers in the live `E3AgentPolicy`/`StepwiseExplorer` generation stage before the
click candidate cap. It SHALL NOT use a per-game adapter, offline BFS,
outer-loop reverse engineering, or a hand-coded game model for a credited solve.

The result artifact SHALL include `status`, `milestone`, `target_game`,
`attempted_level`, `registry_precheck_done`, `duplicate_solve_avoided`,
`solve_provenance`, `live_agent_policy_modified`,
`connected_component_salience_enabled`, `salience_tiers_emitted`,
`per_game_adapter_used`, `offline_bfs_used`, `outer_loop_re_used`,
`live_attempt_count`, `offline_reproduced`, `reproduced_levels`,
`new_level_banked`, `failure_mode`, and `honest_verdict`. A credited level-up
SHALL require `status=complete`, `solve_provenance=live_agent_self_discovery`,
`offline_reproduced=true`, `reproduced_levels>=1`, `new_level_banked=true`,
`per_game_adapter_used=false`, `offline_bfs_used=false`, and
`outer_loop_re_used=false`; otherwise the artifact SHALL report `honest_null`
with a concise failure mode, or `blocked` only when harness access is missing.

Required field principles:

- `status`: principle "complete for a banked +1 level, honest_null for a real no-bank attempt, or blocked for missing harness access."
- `milestone`: principle "must equal 2026.07.491."
- `target_game`: principle "game selected after registry precheck."
- `attempted_level`: principle "level attempted after registry precheck."
- `registry_precheck_done`: principle "must be true."
- `duplicate_solve_avoided`: principle "must be true."
- `solve_provenance`: principle "must be live_agent_self_discovery for a credited solve."
- `live_agent_policy_modified`: principle "true only if E3AgentPolicy generation-stage action prioritization was changed."
- `connected_component_salience_enabled`: principle "true if the blob salience mechanism was active."
- `salience_tiers_emitted`: principle "true if action tiers were logged."
- `per_game_adapter_used`: principle "must be false."
- `offline_bfs_used`: principle "must be false."
- `outer_loop_re_used`: principle "must be false."
- `live_attempt_count`: principle "count of live harness attempts."
- `offline_reproduced`: principle "true only if the live-discovered new level is reproduced."
- `reproduced_levels`: principle "number of newly reproduced levels, success requires reproduced_levels>=1."
- `new_level_banked`: principle "true only for a +1 reproducible level."
- `failure_mode`: principle "null on success or concise no-bank reason."
- `honest_verdict`: principle "one-line summary starting with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5410: Live Trajectory Frontier Self-Discovery Level-Up Attempt

Experiment 5410 SHALL write
`results/experiment_5410_arc_live_trajectory_frontier_levelup_v492.json` after a
registry-prechecked bounded ARC live-agent level-up attempt. The workflow SHALL
prefer `re86` L3 when that level is not already live/reproduction reached; if
the registry already records the target level, it SHALL choose the next eligible
target or emit a blocked duplicate-solve artifact without replaying the solved
level. The mechanism SHALL reuse Exp5397's connected-component blob/color
salience only as a live perception route, SHALL generate action prefixes from
the live agent's own attempted actions and observed runtime transitions, and
SHALL reject or cautiously withhold low-support inferred dynamics through an
uncertainty gate. It SHALL NOT use offline BFS, game source inspection,
per-game adapters, hidden-source probes, or outer-loop reverse engineering for
credited progress.

The result artifact SHALL include `registry_precheck_done`, `target_game`,
`target_level`, `solve_provenance`, `offline_reproduced`, `attempt_count`,
`frontier_expansion_count`, `salience_routes_used`,
`uncertainty_rejections`, `reproduced_levels`, `arc_new_level_banked`,
`duplicate_solve_avoided`, `no_offline_bfs`, `no_per_game_adapter`,
`inference_substrate`, and `honest_verdict`. A credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`arc_new_level_banked=true`, `duplicate_solve_avoided=true`,
`no_offline_bfs=true`, and `no_per_game_adapter=true`; otherwise the artifact
SHALL report `honest_null:` with bounded live-attempt evidence or `blocked:`
when the duplicate/precondition gate prevents a valid attempt.

Required field principles:

- `registry_precheck_done`: principle "bare bool proving no duplicate solve attempt starts before reading the registry."
- `target_game`: principle "selected game id for reproducibility."
- `target_level`: principle "selected target level label for reproducibility."
- `solve_provenance`: principle "must be live_agent_self_discovery for credited progress."
- `offline_reproduced`: principle "legacy ARC lint field; true only for a live-agent self-discovered registry-compatible new level."
- `attempt_count`: principle "bounded live-agent effort count."
- `frontier_expansion_count`: principle "number of trajectory/frontier prefixes emitted by the new mechanism."
- `salience_routes_used`: principle "auditable blob/color salience routes used by prefix generation."
- `uncertainty_rejections`: principle "low-support inferred dynamics rejected by the gate."
- `reproduced_levels`: principle "registry-compatible new level count."
- `arc_new_level_banked`: principle "standing ARC floor success flag."
- `duplicate_solve_avoided`: principle "registry discipline prevents duplicate solved-level credit."
- `no_offline_bfs`: principle "must be true; forbidden solve path was not used."
- `no_per_game_adapter`: principle "must be true; no hand per-game shortcut was used."
- `inference_substrate`: principle "must be offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "terminal status starts with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5423: CoEx Landmark Live Self-Discovery Level-Up Attempt

Experiment 5423 SHALL write
`results/experiment_5423_arc_coex_landmark_levelup_v493.json` after a
registry-prechecked bounded ARC live-agent level-up attempt. The workflow SHALL
prefer `lf52` L3 when the registry records fewer than three reproduced `lf52`
levels; otherwise it SHALL choose the nearest eligible unbanked frontier from
`ops/arc_solve_registry.yaml` or emit a blocked duplicate-solve artifact. The
credited mechanism SHALL run through the live ARC agent runtime by feeding a
CoEx-style persistent frontier generator into the `E3AgentPolicy` action-prior
and QD sequence hooks. The generator SHALL learn only from the agent's own
runtime transitions, persist frontier states across resets, decompose observed
progress into hierarchical landmarks, cluster action histories, and emit
measurement-access receipts for the observations used to promote any prefix.
It SHALL NOT inspect hidden game source, run offline ground-truth BFS as the
credited solve path, or create a per-game adapter/calibration solver as the
headline path.

The result artifact SHALL include `registry_precheck`, `target_game`,
`target_level`, `duplicate_solve_avoided`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `arc_new_level_banked`,
`attempt_count`, `frontier_expansion_count`, `landmark_count`,
`action_sequence_receipts`, `no_offline_bfs`, `no_per_game_adapter`,
`arc_levelup_lint_passed`, `inference_substrate`, and `honest_verdict`. A
credited level-up SHALL require `solve_provenance=live_agent_self_discovery`,
`inference_substrate=live_arc_agent_runtime`, `offline_reproduced=true`,
`reproduced_levels>=1`, `arc_new_level_banked=true`,
`duplicate_solve_avoided=true`, `no_offline_bfs=true`,
`no_per_game_adapter=true`, at least one frontier expansion, at least one
landmark or explicit no-landmark receipt, and at least one replayable action
sequence receipt. Otherwise the artifact SHALL report `honest_null:` with
bounded live-attempt evidence or `blocked:` when the duplicate/precondition gate
prevents a valid attempt.

Required field principles:

- `registry_precheck`: principle "bare bool proving duplicate-solve avoidance ran before any live attempt."
- `target_game`: principle "selected game id with registry provenance."
- `target_level`: principle "selected target level label with registry provenance."
- `duplicate_solve_avoided`: principle "true only when the target level was not already banked."
- `solve_provenance`: principle "must be live_agent_self_discovery for credited progress."
- `offline_reproduced`: principle "true only after the live-discovered sequence passes the offline reproduction gate."
- `reproduced_levels`: principle "new reproduced level count; complete requires at least one."
- `arc_new_level_banked`: principle "north-star metric flag for a newly banked level."
- `attempt_count`: principle "bounded live-agent action count."
- `frontier_expansion_count`: principle "CoEx persistent frontier prefixes emitted by the live mechanism."
- `landmark_count`: principle "hierarchical landmark count discovered from runtime observations."
- `action_sequence_receipts`: principle "replayable live action sequences and measurement receipts used for reproduction."
- `no_offline_bfs`: principle "must be true; forbidden offline ground-truth BFS was not used."
- `no_per_game_adapter`: principle "must be true; no hand per-game adapter/calibration solver was used."
- `arc_levelup_lint_passed`: principle "roadmap guarantee lint result recorded when practical."
- `inference_substrate`: principle "must be live_arc_agent_runtime."
- `honest_verdict`: principle "terminal status starts with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5437: Registry-Guided Live Reinduction Level-Up Attempt

Experiment 5437 SHALL write
`results/experiment_5437_arc_live_reinduction_levelup_v494.json` after a
registry-prechecked bounded ARC live-agent level-up attempt. The workflow SHALL
prefer `cn04` L4 when the registry records exactly three reproduced `cn04`
levels and SHALL prefer `vc33` L3 when `cn04` is not eligible and the registry
records exactly two reproduced `vc33` levels. If neither preferred frontier is
eligible, it SHALL choose the nearest eligible unbanked next-level frontier from
`ops/arc_solve_registry.yaml` or emit a blocked duplicate-solve artifact before
spending live-agent budget. The credited mechanism SHALL run through the live
ARC agent runtime and use registry-guided per-level reinduction evidence,
runtime observation clustering, generic verifier routing, and
measurement-access receipts from the agent's own attempts. It SHALL NOT inspect
hidden game source, run offline ground-truth BFS as the credited solve path, or
create a per-game adapter/calibration solver as the headline path.

The result artifact SHALL include `registry_precheck`, `target_game`,
`target_level`, `duplicate_solve_avoided`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `arc_new_level_banked`,
`attempt_count`, `frontier_expansion_count`, `runtime_predicate_count`,
`action_sequence_receipts`, `no_offline_bfs`, `no_per_game_adapter`,
`arc_levelup_lint_passed`, `inference_substrate`, and `honest_verdict`. A
credited level-up SHALL require `solve_provenance=live_agent_self_discovery`,
`inference_substrate=live_arc_agent_runtime`, `offline_reproduced=true`,
`reproduced_levels>=1`, `arc_new_level_banked=true`,
`duplicate_solve_avoided=true`, `no_offline_bfs=true`,
`no_per_game_adapter=true`, at least one runtime predicate or frontier
transition from the live attempt, and at least one replayable action sequence
receipt. Otherwise the artifact SHALL report `honest_null:` with bounded
live-attempt evidence or `blocked:` when the duplicate/precondition gate
prevents a valid attempt.

Required field principles:

- `registry_precheck`: principle "duplicate-solve avoidance"
- `target_game`: principle "target provenance"
- `target_level`: principle "target provenance"
- `duplicate_solve_avoided`: principle "no already-banked headline"
- `solve_provenance`: principle "credited path"
- `offline_reproduced`: principle "reproducible solve gate"
- `reproduced_levels`: principle "level-up gate; must be >=1 for a banked solve"
- `arc_new_level_banked`: principle "north-star metric"
- `attempt_count`: principle "effort accounting"
- `frontier_expansion_count`: principle "mechanism evidence"
- `runtime_predicate_count`: principle "reinduction evidence"
- `action_sequence_receipts`: principle "reproducibility"
- `no_offline_bfs`: principle "live-path discipline"
- `no_per_game_adapter`: principle "live-path discipline"
- `arc_levelup_lint_passed`: principle "roadmap guarantee evidence"
- `inference_substrate`: principle "actual live agent"
- `honest_verdict`: principle "terminal status; start with complete: or honest_null: or blocked:"

### REQ-ARC-FCP-5450: Measurement-Access Live Level-Up Target Rotation

Experiment 5450 SHALL write
`results/experiment_5450_arc_measurement_access_live_levelup_v495.json` after a
registry-prechecked bounded ARC live-path +1 level-up attempt. The workflow
SHALL precheck all public-game frontier depths from
`ops/arc_solve_registry.yaml` and the current `results/arc_loop_solve*.json`
artifacts before selecting a target. It SHALL rotate away from stale recent
no-bank targets including `cn04` L4 and repeated `re86` L3 salience attempts
unless the selected live mechanism explicitly addresses their recorded residual
gap. The credited mechanism SHALL use only the live agent's own runtime
measurement access: frame-level measurements, action-effect observations,
state-change summaries, and verifier-routed predicates induced from attempted
transitions. It SHALL NOT read hidden game source, run offline ground-truth BFS
as the credited path, or credit a hand per-game `GameAdapter` as the solve path.

The result artifact SHALL include `solve_provenance`,
`registry_precheck_total_levels`, `selected_game`, `selected_target_level`,
`target_rotation_reason`, `live_attempt_count`,
`runtime_predicates_induced`, `offline_reproduced`, `reproduced_levels`,
`new_levels_banked`, `new_level_reproduced`, `no_offline_bfs`,
`no_source_reading`, `no_per_game_adapter_credited`,
`arc_new_level_banked`, `inference_substrate`, and `honest_verdict`. A
credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`new_levels_banked>=1`, `new_level_reproduced=true`,
`arc_new_level_banked=true`, `no_offline_bfs=true`,
`no_source_reading=true`, `no_per_game_adapter_credited=true`, a selected
target level deeper than the registry precheck, and at least one runtime
predicate induced from the live attempt. Otherwise the artifact SHALL report
`honest_null:` with frontier expansions, attempted predicates, and the residual
wall, or `blocked:` when preconditions prevent a valid attempt.

Required field principles:

- `solve_provenance`: principle "live_agent_self_discovery -- credited path is the live agent's own attempts."
- `registry_precheck_total_levels`: principle "duplicate-solve prevention."
- `selected_game`: principle "target audit."
- `selected_target_level`: principle "target audit."
- `target_rotation_reason`: principle "no stale rerun."
- `live_attempt_count`: principle "live effort evidence."
- `runtime_predicates_induced`: principle "credited mechanism evidence."
- `offline_reproduced`: principle "official reproduction gate."
- `reproduced_levels`: principle "level-up acceptance."
- `new_levels_banked`: principle "north-star movement."
- `new_level_reproduced`: principle "lint-readable solve gate."
- `no_offline_bfs`: principle "no outer-loop solve."
- `no_source_reading`: principle "no hidden source path."
- `no_per_game_adapter_credited`: principle "live mechanism only."
- `arc_new_level_banked`: principle "capstone field."
- `inference_substrate`: principle "explicit runtime path."
- `honest_verdict`: principle "terminal status; start with complete: or honest_null: or blocked:."

### REQ-ARC-FCP-5464: ARC Metric-Integrity And Perception Precheck

Experiment 5464 SHALL write
`results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json`
as a live-path precheck that does not claim a level solve. The workflow SHALL
read `ops/arc_solve_registry.yaml`, current reproduced public-game levels, and
recent ARC no-bank artifacts before selecting Exp5465 targets. It SHALL reject
duplicate-depth solve credit, reject off-path or source-derived provenance,
check whether one-step null-coordinate/no-coordinate trajectories contaminate
banked reproduced levels, and emit a target shortlist that avoids already
reached levels and recent duplicate no-bank lanes unless explicitly justified.

The workflow SHALL also exercise live-agent reachable perception diagnostics
for connected components, color blobs, changed pixels, salience tiers, and
action-effect observations through the submitted
`E3AgentPolicy`/`StepwiseExplorer` salience path. The perception receipts SHALL
be written as JSON and referenced by the main artifact. This precheck SHALL
not modify `ops/arc_solve_registry.yaml` and SHALL use
`inference_substrate=live_path_precheck_no_solve_claim`.

The result artifact SHALL include `registry_precheck_performed`,
`reproduced_total_levels_before`, `duplicate_solve_rejected`,
`off_path_solve_rejected`, `null_coordinate_exploit_valid`,
`perception_feature_receipts_path`, `target_shortlist`,
`recent_no_bank_targets_avoided_or_justified`,
`arc_metric_integrity_ready`, `inference_substrate`, and
`honest_verdict`. The artifact SHALL set
`arc_metric_integrity_ready=true` only when the registry precheck ran,
duplicate and off-path probe claims were rejected, no null-coordinate exploit
is valid for the reproduced-level metric, the perception receipts file exists,
and at least one shortlist target remains after recent no-bank avoidance.

Required field principles:

- `registry_precheck_performed`: principle "bare bool proving registry/public-game precheck ran before Exp5465 target selection."
- `reproduced_total_levels_before`: principle "authoritative registry total before this no-solve precheck."
- `duplicate_solve_rejected`: principle "bare bool: duplicate-depth solve claims fail closed and do not increment metrics."
- `off_path_solve_rejected`: principle "bare bool: source-derived, replay-only, or outer-loop provenance cannot receive live solve credit."
- `null_coordinate_exploit_valid`: principle "bare bool: true only if a banked reproduced level is validly explained by the null-coordinate exploit; metric-ready requires false."
- `perception_feature_receipts_path`: principle "path to JSON receipts for connected components, color blobs, changed pixels, salience tiers, and action-effect observations."
- `target_shortlist`: principle "Exp5465 candidates selected after avoiding already reached levels and recent duplicate no-bank lanes."
- `recent_no_bank_targets_avoided_or_justified`: principle "auditable list of recent no-bank targets avoided or any explicit justification for including one."
- `arc_metric_integrity_ready`: principle "true only when duplicate, provenance, null-coordinate, perception, and target-rotation gates are all clean."
- `inference_substrate`: principle "must equal live_path_precheck_no_solve_claim."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked: with no level-solve claim."

### REQ-ARC-FCP-5465: Gated Connected-Component Salience Level-Up Attempt

Experiment 5465 SHALL write
`results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json`
only after the Exp5464 metric-integrity precheck reports
`arc_metric_integrity_ready=true`. The workflow SHALL re-run the registry
precheck, select one target from Exp5464's `target_shortlist`, exercise
live-agent reachable connected-component, color-blob, changed-pixel,
salience-tier, and action-effect features, and run a bounded live/offline
agent attempt using the submitted `E3AgentPolicy`/`StepwiseExplorer` salience
path. It SHALL NOT read hidden game source, run offline ground-truth BFS, or
credit a hand-built per-game adapter as the solve path.

If the live attempt reaches a candidate new level, the workflow SHALL reproduce
that candidate through `arc_solver_kit.reproduce` or the official
`scripts/arc_loop_solve.py` reproduction gate before claiming progress. Success
SHALL require `offline_reproduced=true` and `reproduced_levels` strictly
greater than the selected target's registry-precheck depth. Otherwise the
artifact SHALL report an honest null while preserving the live-attempt feature
receipts and prohibited-input flags.

The result artifact SHALL include `solve_provenance`,
`registry_precheck_performed`, `target_game`, `target_level_before`,
`target_level_attempted`, `live_attempt_count`, `perception_features_used`,
`source_reading_used`, `offline_bfs_used`, `hand_adapter_credited`,
`offline_reproduced`, `reproduced_levels`, `new_level_banked`,
`arc_registry_update_required`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `solve_provenance`: principle "live_agent_self_discovery; credited path is the live agent's own attempts plus runtime reverse engineering."
- `registry_precheck_performed`: principle "bare bool proving the registry was re-read before selecting a target from Exp5464's shortlist."
- `target_game`: principle "game selected from Exp5464 target_shortlist after the rerun precheck."
- `target_level_before`: principle "registry-precheck reproduced depth for the selected target before this attempt."
- `target_level_attempted`: principle "one deeper level attempted by the live/offline agent."
- `live_attempt_count`: principle "bounded count of live-agent attempts actually executed."
- `perception_features_used`: principle "auditable list containing connected_component, color_blob, changed_pixel, salience_tier, and action_effect when exercised."
- `source_reading_used`: principle "must be false; hidden/public game source is not credited in this live self-discovery path."
- `offline_bfs_used`: principle "must be false; exhaustive offline ground-truth BFS is not the credited solve path."
- `hand_adapter_credited`: principle "must be false; a hand GameAdapter may not receive live-agent self-discovery credit."
- `offline_reproduced`: principle "true only when the live-agent candidate reproduces through the official reproduction gate beyond the precheck depth."
- `reproduced_levels`: principle "absolute reproduced level count after the gate; success requires this to exceed target_level_before."
- `new_level_banked`: principle "true only when offline_reproduced=true and reproduced_levels > target_level_before."
- `arc_registry_update_required`: principle "true only when a newly banked level should update ops/arc_solve_registry.yaml."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5479: Target Rotation Live-Path Precheck Without Solve Claim

Experiment 5479 SHALL write
`results/experiment_5479_arc_target_rotation_precheck_v497.json` as a
no-solve ARC target-rotation precheck before any level-up attempt. The workflow
SHALL read `ops/arc_solve_registry.yaml`, the Exp5464 precheck shortlist, the
Exp5465 no-bank artifact, and `ops/known-issues.md`; reject target levels that
the live mechanism already reproduces; avoid recent no-bank targets `bp35:L3`,
`ka59:L2`, and `cn04:L4` unless a new mechanism and target level are recorded;
and prefer a rotated Exp5464 shortlist target such as `sb26:L3`, `g50t:L3`,
`dc22:L3`, or `sp80:L3`.

The workflow SHALL verify live-path eligibility through frame-only
connected-component/color-blob salience diagnostics without reading hidden game
source, running offline ground-truth BFS, or crediting a hand per-game adapter.
The bounded dry check SHALL report connected components, color blobs, changed
cells, target-region candidates, and known blockers. The artifact SHALL make no
solve claim and SHALL use
`inference_substrate=arc_live_path_precheck_no_solve`.

The result artifact SHALL include `selected_game`,
`selected_target_level`, `registry_reproducible_total_levels_before`,
`duplicate_target_rejected`, `recent_no_bank_targets_avoided`,
`live_path_reachable`, `hidden_source_reading`, `offline_bfs_used`,
`hand_adapter_used`, `salience_feature_summary`,
`arc_target_rotation_ready`, `solve_claimed`, `inference_substrate`,
`random_seed`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "rotated non-duplicate Exp5464 shortlist game selected before any level-up attempt."
- `selected_target_level`: principle "selected next level, strictly greater than the registry reproduced depth."
- `registry_reproducible_total_levels_before`: principle "authoritative registry total before this no-solve precheck."
- `duplicate_target_rejected`: principle "bare bool proving already reproduced target levels are rejected before selection."
- `recent_no_bank_targets_avoided`: principle "auditable list containing bp35:L3, ka59:L2, and cn04:L4 unless a new mechanism justifies inclusion."
- `live_path_reachable`: principle "true only when submitted live salience/perception code emits the dry-check features."
- `hidden_source_reading`: principle "must be false; hidden/public source is not read in this precheck."
- `offline_bfs_used`: principle "must be false; no offline ground-truth BFS is used."
- `hand_adapter_used`: principle "must be false; target eligibility is not credited to a hand per-game adapter."
- `salience_feature_summary`: principle "dict summarizing connected components, color blobs, changed cells, target-region candidates, and known blockers."
- `arc_target_rotation_ready`: principle "true only when non-duplicate target rotation and live-path salience eligibility both pass."
- `solve_claimed`: principle "must be false; this artifact is a precheck, not a level solve."
- `inference_substrate`: principle "must equal arc_live_path_precheck_no_solve."
- `random_seed`: principle "deterministic seed for reproducible target ordering and dry-check fixtures."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked: with no solve claim."

### REQ-ARC-FCP-5480: Rotated Live Salience Level-Up Attempt

Experiment 5480 SHALL write
`results/experiment_5480_arc_live_salience_levelup_v497.json` only after
loading `selected_game` and `selected_target_level` from
`results/experiment_5479_arc_target_rotation_precheck_v497.json`. The workflow
SHALL abort with a blocked artifact when the Exp5479 target is missing or when
`ops/arc_solve_registry.yaml` already records the selected target level as
reproduced. Otherwise it SHALL run exactly one bounded live-agent
self-discovery attempt on that rotated target, using the submitted live
connected-component/color-blob/change-cell salience path as the prioritization
substrate.

The credited path SHALL be the agent's own observed frames, action effects, and
runtime reverse engineering only. It SHALL NOT read hidden game source, run an
offline ground-truth BFS, credit a hand per-game adapter, or rely on an
outer-loop reverse-engineered adapter. Any candidate new level SHALL be
reproduced through the registry-approved live reproduction mechanism before the
registry is updated. Success SHALL require `offline_reproduced=true` and
`reproduced_levels >= 1` beyond the selected target's precheck depth; otherwise
the artifact SHALL report an honest null and SHALL NOT modify
`ops/arc_solve_registry.yaml`.

The result artifact SHALL include `game`, `target_level`,
`solve_provenance`, `hidden_source_reading`, `offline_bfs_used`,
`hand_adapter_used`, `outer_loop_re_used`, `action_count`,
`explored_state_count`, `failed_hypotheses`, `offline_reproduced`,
`reproduced_levels`, `new_level_banked`, `reproduced_levels_before`,
`reproduced_levels_after`, `registry_updated`, `first_win_trace_path`,
`inference_substrate`, `random_seed`, and `honest_verdict`.

Required field principles:

- `game`: principle "selected Exp5479 game, or none when the target precondition blocks the attempt."
- `target_level`: principle "selected Exp5479 target level; success must reproduce at least this level."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `hidden_source_reading`: principle "must be false; hidden/public game source is not part of the credited path."
- `offline_bfs_used`: principle "must be false; exhaustive offline ground-truth BFS is not the credited path."
- `hand_adapter_used`: principle "must be false; a hand per-game adapter is not credited."
- `outer_loop_re_used`: principle "must be false; outer-loop reverse engineering is not credited."
- `action_count`: principle "bare integer count of bounded live-agent actions actually executed."
- `explored_state_count`: principle "bare integer count of live-agent states observed or tracked during the attempt."
- `failed_hypotheses`: principle "list of rejected salience/runtime hypotheses when no target level is banked."
- `offline_reproduced`: principle "true only when the live-agent candidate reproduces beyond the precheck depth."
- `reproduced_levels`: principle "new reproduced levels banked beyond the precheck depth; success requires >=1."
- `new_level_banked`: principle "true only when offline_reproduced=true and reproduced_levels>=1."
- `reproduced_levels_before`: principle "registry reproduced depth for the selected game before Exp5480."
- `reproduced_levels_after`: principle "registry reproduced depth after Exp5480; unchanged on honest null."
- `registry_updated`: principle "true only when ops/arc_solve_registry.yaml was updated for a newly reproduced level."
- `first_win_trace_path`: principle "relative path to the first reproduced winning trace, or empty string when none exists."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `random_seed`: principle "deterministic seed for the bounded live attempt."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5493: Registry-Only Trajectory Target Precheck

Experiment 5493 SHALL write
`results/experiment_5493_arc_trajectory_target_precheck_v498.json` as a
no-solve ARC registry and live-path precheck for the follow-on Exp5494
trajectory/option-induction attempt. The workflow SHALL read
`ops/arc_solve_registry.yaml`, `ops/exclusion_manifest.yaml`,
`ops/known-issues.md`, and the recent Exp5479/Exp5480 target artifacts before
selecting any target. It SHALL report each candidate game's current
`levels_reproduced`, reject any already reproduced target level, and avoid
recent no-bank targets `sb26:L3`, `bp35:L3`, `ka59:L2`, `cn04:L4`, and
`re86:L3` unless the registry proves a different target level and mechanism.

The workflow SHALL reject retired generic exploration-signal scopes from
Exp5154, including novelty-only, curiosity-only, energy-as-fitness
quality-diversity, and archive-granularity reruns. It SHALL prefer a target
whose registry row records live-path mechanism hooks and enough visible
runtime-observation structure for the live agent to form trajectory or option
hypotheses from its own observations. It SHALL NOT read game source, run
offline ground-truth BFS, or build or credit a per-game hand adapter. When no
eligible target exists, it SHALL emit a clean blocked no-target artifact rather
than selecting a stale duplicate.

The result artifact SHALL include `registry_path`,
`excluded_recent_no_bank_targets`, `duplicate_solve_avoided`,
`selected_game`, `selected_target_level`, `prior_levels_reproduced`,
`proposed_live_mechanism`, `trajectory_induction_preconditions`,
`offline_bfs_used`, `per_game_adapter_used`,
`arc_trajectory_precheck_ready`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `registry_path`: principle "must equal ops/arc_solve_registry.yaml."
- `excluded_recent_no_bank_targets`: principle "auditable list containing sb26:L3, bp35:L3, ka59:L2, cn04:L4, and re86:L3 unless the selected target proves a different level and mechanism."
- `duplicate_solve_avoided`: principle "bare bool proving the selected target is strictly deeper than the registry depth."
- `selected_game`: principle "selected game id, or empty string when no eligible target exists."
- `selected_target_level`: principle "selected next target level as a bare int, or 0 when blocked."
- `prior_levels_reproduced`: principle "authoritative registry depth for the selected game before Exp5493."
- `proposed_live_mechanism`: principle "one-line live-path mechanism to hand to Exp5494, not a retired exploration-signal rerun."
- `trajectory_induction_preconditions`: principle "list of live-observation prerequisites that must hold before Exp5494 attempts the target."
- `offline_bfs_used`: principle "must be false; this precheck is registry-only and no offline ground-truth BFS is run."
- `per_game_adapter_used`: principle "must be false; target eligibility is not credited to a hand adapter."
- `arc_trajectory_precheck_ready`: principle "true only when a non-duplicate, non-recent-no-bank, non-retired-scope target is selected."
- `inference_substrate`: principle "must equal registry_precheck_no_solve."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without a solve claim."

### REQ-ARC-FCP-5494: Exp5493-Selected Live Trajectory-Induction Level-Up Attempt

Experiment 5494 SHALL write
`results/experiment_5494_arc_live_trajectory_levelup_v498.json` after exactly
one bounded live ARC trajectory/option-induction attempt on the target selected
by Exp5493. The workflow SHALL read the Exp5493 precheck artifact and
`ops/arc_solve_registry.yaml` before spending live-agent budget; it SHALL block
without attempting when the target is already reproduced, appears in the recent
same-mechanism no-bank set, or lacks the live trajectory-induction
preconditions recorded by Exp5493. The credited path SHALL be
`live_agent_self_discovery`: the live agent's own attempted actions, observed
frame deltas, runtime hypotheses, verifier checks, and rejected option
sequences. It SHALL NOT read game source, run offline ground-truth BFS, or build
or credit a per-game hand adapter.

The workflow SHALL run through the live `E3AgentPolicy` path with
`LiveCoExLandmarkFrontierGenerator` as the trajectory/option-induction
generator when the preconditions pass. It SHALL record hypothesized action
sequences, observation deltas, verifier checks, and rejection reasons even when
no level is banked. If an LLM generator is invoked, the artifact's
`model_specs_if_llm_used` SHALL include
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; if those model specs are unavailable for the
live path, the workflow SHALL emit a blocked artifact instead of substituting a
legacy small model as a headline generator. The registry SHALL be updated only
when the standard live/offline reproduction gate reports
`offline_reproduced=true` and at least one new level beyond the Exp5493 prior
depth is reproduced.

The result artifact SHALL include `selected_game`, `target_level`,
`prior_levels_reproduced`, `post_levels_reproduced`, `solve_provenance`,
`offline_bfs_used`, `per_game_adapter_used`, `game_source_read`,
`trajectory_hypothesis_count`, `live_attempt_count`, `offline_reproduced`,
`reproduced_levels`, `new_level_banked`, `registry_updated`,
`model_specs_if_llm_used`, `failure_mode`, `inference_substrate`,
`random_seed`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5493-selected game id, or empty string only when the attempt blocks before target selection."
- `target_level`: principle "Exp5493-selected target level as a bare int."
- `prior_levels_reproduced`: principle "registry depth before Exp5494; success must be strictly deeper."
- `post_levels_reproduced`: principle "registry depth after Exp5494; unchanged on honest null."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `offline_bfs_used`: principle "must be false; no offline ground-truth BFS is credited."
- `per_game_adapter_used`: principle "must be false; no hand per-game adapter is credited."
- `game_source_read`: principle "must be false; source reading is outside the credited live path."
- `trajectory_hypothesis_count`: principle "bare int count of hypothesized action sequences induced from runtime observations."
- `live_attempt_count`: principle "bare int count of live actions actually executed."
- `offline_reproduced`: principle "true only after the live-discovered candidate passes the reproduction gate."
- `reproduced_levels`: principle "new reproduced levels beyond the prior depth; complete requires >=1."
- `new_level_banked`: principle "true only when offline_reproduced=true and reproduced_levels>=1."
- `registry_updated`: principle "true only when a newly reproduced level is written to ops/arc_solve_registry.yaml."
- `model_specs_if_llm_used`: principle "empty when no LLM was invoked; otherwise contains the three mandated headline GGUF model specs."
- `failure_mode`: principle "empty on success or concise blocked/no-bank reason."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `random_seed`: principle "deterministic seed for the bounded attempt."
- `honest_verdict`: principle "terminal status starts with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5507: Null-Coordinate And Perception-Grounding Target Precheck

Experiment 5507 SHALL write
`results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json`
as a no-solve aggregation artifact before Exp5508 spends any live-agent
budget. The workflow SHALL read `ops/arc_solve_registry.yaml`,
`ops/known-issues.md`, `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md`,
`results/experiment_5493_arc_trajectory_target_precheck_v498.json`, and
`results/experiment_5494_arc_live_trajectory_levelup_v498.json`. It MAY also
read prior ARC salience/precheck artifacts as upstream evidence, but it SHALL
NOT read game source, run offline ground-truth BFS, build a hand per-game
adapter, or claim a solve.

The workflow SHALL reject any requested target level that the registry already
records as reproduced. It SHALL audit recent no-bank targets and reject exact
same-target/same-mechanism reruns unless the selected mechanism is materially
changed. It SHALL audit prior null-coordinate or no-op behavior as either valid
game actions with zero frame effect or metric artifacts, and SHALL record
whether any banked level appears contaminated by a null-coordinate exploit. It
SHALL inspect upstream live observations for perception-grounding findings that
a classical connected-component/color-blob segmentation pass can expose.

The result artifact SHALL include `registry_path`,
`reproducible_total_levels_before`, `duplicate_targets_rejected`,
`recent_no_bank_targets_rejected`, `null_coordinate_audit`,
`perception_grounding_findings`, `selected_game`, `selected_level`,
`selected_mechanism`, `levelup_attempt_ready`, `solve_claimed`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `registry_path`: principle "must equal ops/arc_solve_registry.yaml."
- `reproducible_total_levels_before`: principle "authoritative registry total before this no-solve target precheck."
- `duplicate_targets_rejected`: principle "auditable list of requested target levels rejected because the registry already reproduces them."
- `recent_no_bank_targets_rejected`: principle "auditable list of recent same-target/same-mechanism no-bank reruns rejected before selection."
- `null_coordinate_audit`: principle "dict classifying null/missing/no-op coordinate evidence as valid game action, metric artifact, or contamination."
- `perception_grounding_findings`: principle "list of upstream live-observation findings exposed by connected components, color blobs, salience tiers, changed pixels, or action-effect asymmetry."
- `selected_game`: principle "selected game id, or empty string when blocked."
- `selected_level`: principle "selected level label such as L3, or empty string when blocked."
- `selected_mechanism`: principle "changed live-path perception-generation mechanism, not a same-mechanism rerun."
- `levelup_attempt_ready`: principle "bare bool true only when the selected level is non-duplicate, same-mechanism reruns are rejected, null-coordinate audit is clean, and perception findings are present."
- `solve_claimed`: principle "must be false; this artifact is a precheck and cannot bank a level."
- `inference_substrate`: principle "must equal aggregation_from_upstream_artifacts."
- `honest_verdict`: principle "terminal status starts with complete: or blocked: and makes no solve claim."

### REQ-ARC-FCP-5508: Live Perception-Generation Level-Up Attempt

Experiment 5508 SHALL write
`results/experiment_5508_arc_live_perception_generation_levelup_v499.json`
after re-reading `ops/arc_solve_registry.yaml` and the Exp5507 precheck
artifact immediately before any live-agent attempt. The workflow SHALL abort
with a blocked artifact when the Exp5507 target is absent, not ready, or already
reproducible in the re-read registry. Otherwise it SHALL run one bounded live
ARC attempt on the selected target using `E3AgentPolicy` with a reusable
classical perception-generation pass that extracts connected components, color
blobs, sprite overlays, salient motion, and action affordances from runtime
frames only.

The credited path SHALL be `live_agent_self_discovery`: the live agent's own
runtime observations, action effects, perception-generated candidates, and
standard reproduction gate. It SHALL NOT read game source, run offline
ground-truth BFS, or use a hand-built per-game adapter. Any candidate level-up
SHALL be reproduced through the standard live/offline reproduction gate before
the registry is updated. Success SHALL require `offline_reproduced=true` and
`reproduced_levels>=1` for a strictly new level beyond the selected game's
registry depth; otherwise the artifact SHALL report an honest null and SHALL
NOT modify `ops/arc_solve_registry.yaml`.

The workflow SHALL record per-step trajectory-taxonomy counts inspired by
Trajel for factual, referential, logical, procedural, and scope-based failures
when applicable. Honest-null artifacts SHALL still record enough live-attempt
duration, candidate-generation, prohibited-input, and methodology fields to
distinguish a bounded runtime null from a missing-methodology artifact.

The result artifact SHALL include `selected_game`, `selected_level`,
`registry_before_levels`, `registry_after_levels`, `arc_registry_delta`,
`offline_reproduced`, `reproduced_levels`, `solve_provenance`,
`live_agent_attempts`, `runtime_observation_steps`,
`perception_features_enabled`, `trajectory_taxonomy_counts`,
`offline_bfs_used`, `game_source_read`,
`hand_built_per_game_adapter_used`, `methodology_receipt`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5507-selected game id, or empty string only when the precheck blocks before a target can be attempted."
- `selected_level`: principle "Exp5507-selected level label such as L3; success must be strictly deeper than the re-read registry depth."
- `registry_before_levels`: principle "authoritative `ops/arc_solve_registry.yaml` total immediately before the live attempt."
- `registry_after_levels`: principle "authoritative registry total after the attempt; unchanged on honest null or blocked runs."
- `arc_registry_delta`: principle "bare int delta between after and before totals; success requires this to equal the newly reproduced levels."
- `offline_reproduced`: principle "true only when the live-discovered candidate passes the standard reproduction gate for a new level."
- `reproduced_levels`: principle "new reproduced levels banked beyond the selected game's pre-run depth; success requires >=1."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `live_agent_attempts`: principle "bare int count of runtime actions actually executed by the live agent."
- `runtime_observation_steps`: principle "bare int count of runtime frame/action observations available to perception generation."
- `perception_features_enabled`: principle "list containing connected_components, color_blobs, sprite_overlays, salient_motion, and action_affordances when the pass is active."
- `trajectory_taxonomy_counts`: principle "dict with factual, referential, logical, procedural, and scope_based failure counts."
- `offline_bfs_used`: principle "must be false; offline ground-truth BFS is not part of the credited path."
- `game_source_read`: principle "must be false; game source reading is outside live self-discovery credit."
- `hand_built_per_game_adapter_used`: principle "must be false; no hand per-game adapter is credited."
- `methodology_receipt`: principle "string receipt naming the bounded live runtime, candidate-generation mechanism, reproduction gate, and prohibited-input flags."
- `inference_substrate`: principle "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5520: Action-Diversity Target Precheck

Experiment 5520 SHALL write
`results/experiment_5520_arc_action_diversity_target_precheck.json` before the
next live ARC level-up attempt spends credit-bearing budget. The workflow SHALL
read `ops/arc_solve_registry.yaml` and the Exp5508 live perception-generation
artifact, reject any candidate whose target level is already reproduced by the
registry, and reject the Exp5508 repeated coordinate/action pattern as a target
or mechanism reuse.

The workflow SHALL configure a live-path candidate-generator change based on
connected-component/color-blob salience plus repeated-coordinate suppression,
target rotation, action entropy, or salience coverage. It SHALL measure that
changed path through a dry-run or limited no-credit probe only; the probe SHALL
NOT run offline ground-truth BFS, use a hand per-game adapter, or claim solve
credit. A target SHALL be selected only when the probe shows a meaningfully
different action-diversity path from Exp5508.

The result artifact SHALL include `registry_precheck_done`, `selected_game`,
`selected_level`, `already_reproduced`, `exp5508_pattern_reused`,
`candidate_generator_changes`, `action_entropy`,
`repeated_coordinate_rate`, `salience_coverage_rate`,
`no_credit_probe_attempts`, `arc_levelup_candidate_ready`,
`solve_provenance`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `registry_precheck_done`: principle "bare bool proving ops/arc_solve_registry.yaml was checked before target selection."
- `selected_game`: principle "one registry-safe game id selected for the next live level-up attempt, or empty string when blocked."
- `selected_level`: principle "next unreproduced target level as a string or bare int; it must be strictly deeper than the registry depth."
- `already_reproduced`: principle "must be false for any ready artifact."
- `exp5508_pattern_reused`: principle "must be false; Exp5508's repeated ACTION6 coordinate loop cannot be reused."
- `candidate_generator_changes`: principle "non-empty list naming live-path generation changes such as repeated-coordinate suppression, target rotation, action entropy gating, or salience coverage."
- `action_entropy`: principle "Shannon entropy over dry-run action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of dry-run consecutive coordinate choices that repeat a prior coordinate, as a bare float."
- `salience_coverage_rate`: principle "fraction of dry-run choices covering distinct salience candidates, as a bare float."
- `no_credit_probe_attempts`: principle "bare int count of no-credit dry-run choices measured before the live attempt."
- `arc_levelup_candidate_ready`: principle "bare bool true only when registry and Exp5508-pattern gates pass and the diversity metrics meet threshold."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `inference_substrate`: principle "must equal arc_live_precheck."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5521: Live Action-Diverse Level-Up Attempt

Experiment 5521 SHALL write
`results/experiment_5521_arc_live_action_diverse_levelup.json` after the
Exp5520 action-diversity precheck is ready. The attempt SHALL re-read
`ops/arc_solve_registry.yaml`, verify that Exp5520 selected a registry-safe
target with `arc_levelup_candidate_ready=true`, and run one bounded live-agent
self-discovery attempt against that target using the action-diverse
connected-component/color-blob generator. The live attempt MAY replay a
live-discovered trajectory through the standard offline reproduction gate, but
it SHALL NOT use offline ground-truth BFS, read game source, or credit a
hand-built per-game adapter.

The result artifact SHALL include `selected_game`, `selected_level`,
`offline_reproduced`, `reproduced_levels`, `banking_gate`, `registry_delta`,
`solve_provenance`, `live_attempts`, `action_entropy`,
`repeated_coordinate_rate`, `salience_coverage_rate`, `trajectory_log_path`,
`reproduction_command`, `arc_live_levelup_ready`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5520-selected registry-safe game id; empty only when the readiness gate blocks before live runtime."
- `selected_level`: principle "Exp5520-selected unreproduced level label or int; success must be strictly deeper than the registry depth."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the standard offline replay gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires >=1."
- `banking_gate`: principle "bare bool equal to offline_reproduced=true and reproduced_levels>=1 for solve_provenance=live_agent_self_discovery."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the banking gate is true."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `live_attempts`: principle "bare int count of runtime actions executed by the live agent."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed live coordinate choices that repeated a prior coordinate, as a bare float."
- `salience_coverage_rate`: principle "fraction of executed live coordinate choices covering proposed salience coordinates, as a bare float."
- `trajectory_log_path`: principle "path to the detailed trajectory log containing observations, proposed actions, verifier feedback, and diversity metrics."
- `reproduction_command`: principle "exact replay command when a live trajectory was reproduced, else null."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5520 and the registry reread allowed the live attempt."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5533: Strategy-Routing Precheck Before Live Level-Up Attempt

Experiment 5533 SHALL write
`results/experiment_5533_arc_strategy_routing_precheck.json` before the next
credit-bearing ARC live level-up attempt. The workflow SHALL read
`ops/arc_solve_registry.yaml`, Exp5520, and Exp5521; select one adjacent
frontier target whose target level is not already reproduced; and rotate away
from the stale Exp5521 target when Exp5521 records a no-bank live attempt with
repeated-coordinate collapse.

The workflow SHALL configure a bounded, deterministic strategy portfolio with
at least three live-path-compatible strategies, including salience-first,
action-effect memory, verifier-router candidate ranking, or conservative
reset/reinduction. The precheck SHALL verify that strategy routing is reachable
through the live candidate-router hook and that repeated-coordinate suppression
changes the candidate action selected before metrics are computed. It SHALL NOT
inspect hidden game source, run exhaustive offline ground-truth BFS, build a
hand per-game adapter, or claim a solve.

The result artifact SHALL include `selected_game`, `selected_level`,
`already_reproduced`, `registry_precheck_passed`, `strategy_portfolio`,
`strategy_routing_live_path_reachable`,
`repeated_coordinate_suppression_enabled`,
`repeated_coordinate_rate_precheck`, `action_entropy_precheck`,
`salience_coverage_rate_precheck`, `model_specs`,
`llm_strategy_proposer_used`, `solve_provenance`,
`arc_sge_candidate_ready`, `tests_added_or_reused`, `field_principles`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "registry-safe game id selected for the next strategy-guided live attempt, or empty string when blocked."
- `selected_level`: principle "adjacent unreproduced frontier level label; it must be deeper than the registry depth."
- `already_reproduced`: principle "must be false for any ready artifact because duplicate live levels cannot satisfy the standing floor."
- `registry_precheck_passed`: principle "bare bool proving the registry was read and the selected level is not already reproduced."
- `strategy_portfolio`: principle "list of at least three bounded live-path-compatible strategy descriptors used before the attempt."
- `strategy_routing_live_path_reachable`: principle "bare bool proving the router object reaches the live candidate-router hook used by E3AgentPolicy and graph exploration."
- `repeated_coordinate_suppression_enabled`: principle "bare bool true only when repeated-coordinate suppression changes candidate selection before metrics."
- `repeated_coordinate_rate_precheck`: principle "fraction of routed precheck coordinate choices repeating earlier coordinates after suppression."
- `action_entropy_precheck`: principle "Shannon entropy over routed precheck action/coordinate choices as a bare float."
- `salience_coverage_rate_precheck`: principle "fraction of salience candidate coordinates covered by routed precheck choices."
- `model_specs`: principle "allowed local-GGUF proposer specs recorded for audit; no model is invoked when llm_strategy_proposer_used=false."
- `llm_strategy_proposer_used`: principle "bare bool; false means deterministic strategy templates were used and no GGUF tokenizer/model path was loaded."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `arc_sge_candidate_ready`: principle "bare bool true only when target, strategy routing, suppression, and metric gates pass."
- `tests_added_or_reused`: principle "list of focused tests that cover the Exp5533 schema, target rotation, live routing hook, and suppression evidence."
- `inference_substrate`: principle "must equal arc_live_path_precheck_no_solve_claim."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5534: Strategy-Routed Live Level-Up Attempt

Experiment 5534 SHALL write
`results/experiment_5534_arc_strategy_routed_levelup.json` from the target
selected by Exp5533. The workflow SHALL load Exp5533's `selected_game` and
`selected_level`, reread `ops/arc_solve_registry.yaml`, and block as
`blocked_duplicate_target` without choosing a replacement if the registry now
already reproduces the selected level.

When not duplicate-blocked, the workflow SHALL run the live ARC agent with the
Exp5533 bounded strategy portfolio and repeated-coordinate suppression enabled.
It SHALL record action attempts, strategy choices, verifier route evidence,
suppression events, and level counter changes in a trajectory artifact. It
SHALL keep `solve_provenance=live_agent_self_discovery` and SHALL NOT inspect
hidden game source, run exhaustive offline ground-truth BFS, perform outer-loop
reverse engineering, or build a hand per-game adapter.

A new-level claim SHALL require `offline_reproduced=true` and
`reproduced_levels>=1` through the standard live-path reproduction gate. The
ARC solve registry SHALL be updated only when that gate passes. Otherwise the
artifact SHALL use an `honest_null:` verdict and preserve the trajectory
evidence.

The result artifact SHALL include `selected_game`, `selected_level`,
`solve_provenance`, `strategy_portfolio_used`, `strategy_switch_count`,
`attempts`, `action_entropy`, `repeated_coordinate_rate`,
`repeated_coordinate_suppression_events`, `salience_coverage_rate`,
`offline_reproduced`, `reproduced_levels`, `registry_delta`,
`trajectory_path`, `model_specs`, `llm_strategy_proposer_used`,
`arc_live_levelup_ready`, `tests_added_or_reused`, `field_principles`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5533-selected registry-safe game id; empty only when Exp5533 blocks before runtime."
- `selected_level`: principle "Exp5533-selected adjacent frontier level label; duplicate targets block rather than rotate."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `strategy_portfolio_used`: principle "bounded live-path-compatible strategy descriptors actually installed on the candidate router."
- `strategy_switch_count`: principle "integer count of changes between executed strategy labels after repeated-coordinate suppression."
- `attempts`: principle "bare int count of runtime actions executed by the live agent."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed live coordinate choices that repeated an earlier executed coordinate."
- `repeated_coordinate_suppression_events`: principle "bare int count of candidate-router repeated-coordinate suppressions recorded during live selection."
- `salience_coverage_rate`: principle "fraction of executed live coordinate choices covering proposed salience coordinates."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the standard offline replay gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires >=1."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the live reproduction gate passes."
- `trajectory_path`: principle "path to the detailed trajectory log containing attempts, strategies, verifier routes, suppression events, and level changes."
- `model_specs`: principle "allowed local-GGUF proposer specs recorded for audit; no model is invoked when llm_strategy_proposer_used=false."
- `llm_strategy_proposer_used`: principle "bare bool; false means deterministic strategy templates were used and no GGUF tokenizer/model path was loaded."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5533 and registry reread allowed live runtime."
- `tests_added_or_reused`: principle "list of focused tests that cover the Exp5534 schema, duplicate block, live routing trace, and registry gate."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5547: No-LLM ARC Substrate Clean Precheck

Experiment 5547 SHALL write
`results/experiment_5547_arc_no_llm_substrate_precheck.json` as a clean
no-credit ARC live-path precheck before any level-up attempt that follows the
flagged Exp5533/Exp5534 artifacts. The workflow SHALL read
`ops/arc_solve_registry.yaml`, run a duplicate registry precheck, and choose one
adjacent target level that is not already banked by the live mechanism. It SHALL
use only the live agent path, SHALL NOT invoke an LLM strategy proposer, and
SHALL NOT include `model_specs` or `target_model` because the declared
substrate is the offline arcade live-agent runtime with self-discovery and no
LLM.

The result artifact SHALL include `selected_game`, `selected_level`,
`registry_precheck_passed`, `already_reproduced`,
`llm_strategy_proposer_used`, `no_model_specs_required`, `random_seed`,
`reproducibility_checksum`, `strategy_routing_live_path_reachable`,
`repeated_coordinate_suppression_enabled`, `action_entropy_precheck`,
`solve_provenance`, `arc_clean_precheck_ready`, `tests_added_or_reused`,
`field_principles`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "registry-safe game id selected for the next clean no-LLM live-path attempt."
- `selected_level`: principle "adjacent unreproduced frontier level label selected after the duplicate registry precheck."
- `registry_precheck_passed`: principle "bare bool proving the registry was read and the selected level is not already reproduced."
- `already_reproduced`: principle "must remain false because duplicate live levels cannot satisfy the ARC standing progress floor."
- `llm_strategy_proposer_used`: principle "bare bool false proving this precheck did not load or invoke an LLM strategy proposer."
- `no_model_specs_required`: principle "bare bool true because the no-LLM substrate has no model invocation to name."
- `random_seed`: principle "deterministic seed required for third-party reruns of the target choice and checksum."
- `reproducibility_checksum`: principle "content-addressed hash over registry target, seed, substrate, and routing gates to catch silent drift."
- `strategy_routing_live_path_reachable`: principle "bare bool proving the bounded candidate router is reachable from the live candidate-router hook."
- `repeated_coordinate_suppression_enabled`: principle "bare bool proving repeated-coordinate suppression is active before action entropy is trusted."
- `action_entropy_precheck`: principle "bare float expectation for routed action/coordinate diversity before the live attempt."
- `solve_provenance`: principle "must equal live_agent_self_discovery even though this artifact claims no solve."
- `arc_clean_precheck_ready`: principle "bare bool true only when registry, no-LLM substrate, provenance, seed/checksum, live path, and suppression gates pass."
- `tests_added_or_reused`: principle "list of focused tests covering duplicate blocking, no-model metadata, checksum determinism, and schema gates."
- `field_principles`: principle "mapping of one-line principle annotations for every headline and gate field."
- `inference_substrate`: principle "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5548: Clean No-LLM Live Level-Up Attempt

Experiment 5548 SHALL write
`results/experiment_5548_arc_clean_live_levelup.json` after one gated live ARC
attempt using the clean no-LLM substrate precheck from Exp5547. The workflow
SHALL first confirm that Exp5547 reports `arc_clean_precheck_ready=true`, then
SHALL use Exp5547's `selected_game`, `selected_level`, `random_seed`, and
repeated-coordinate suppression setting unless a fresh registry reread shows
that target level is already banked. If the selected target is already banked,
the workflow SHALL rotate to a fresh adjacent frontier target and record the
rotation reason.

The live attempt SHALL use the live agent self-discovery path with a disabled
LLM strategy proposer, no model load, no `model_specs`, no hidden-game source
inspection, no exhaustive offline ground-truth BFS, and no hand-built per-game
adapter. It SHALL record the trajectory path, action entropy,
repeated-coordinate rate, attempt budget, terminal state, and exact replay or
banking result. A level-up claim SHALL be accepted only when
`offline_reproduced=true` and `reproduced_levels>=1` with
`solve_provenance=live_agent_self_discovery`; otherwise the artifact SHALL emit
an honest null and leave `registry_delta=0`.

The result artifact SHALL include `selected_game`, `selected_level`,
`solve_provenance`, `llm_strategy_proposer_used`, `no_model_specs_required`,
`random_seed`, `reproducibility_checksum`, `attempts`, `trajectory_path`,
`action_entropy`, `repeated_coordinate_rate`, `offline_reproduced`,
`reproduced_levels`, `registry_delta`, `arc_live_levelup_ready`,
`tests_added_or_reused`, `field_principles`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `selected_game`: principle "registry-safe game id used for the clean live attempt after Exp5547 and duplicate checks."
- `selected_level`: principle "adjacent unreproduced frontier level label attempted by the live agent."
- `solve_provenance`: principle "must equal live_agent_self_discovery so only the live runtime's own attempt can receive credit."
- `llm_strategy_proposer_used`: principle "bare bool false proving no LLM strategy proposer or model path was invoked."
- `no_model_specs_required`: principle "bare bool true because the declared no-LLM substrate has no model invocation to name."
- `random_seed`: principle "Exp5547-recorded deterministic seed reused for target rotation, trajectory gating, and checksum replay."
- `reproducibility_checksum`: principle "content-addressed hash over target, seed, trajectory metrics, and banking gate to catch silent drift."
- `attempts`: principle "bare int count of runtime actions executed during the live attempt."
- `trajectory_path`: principle "path to the detailed trajectory log containing actions, route evidence, suppression events, terminal state, and replay gate."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed coordinate actions that repeated an earlier executed coordinate."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the standard offline replay gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires at least one."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the accepted reproduction gate passes."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5547, registry reread, no-LLM metadata, and live harness preconditions allowed runtime."
- `tests_added_or_reused`: principle "list of focused tests covering clean schema, target rotation, trajectory metrics, checksum, and banking gate."
- `field_principles`: principle "mapping of one-line principle annotations for each headline and gate field."
- `inference_substrate`: principle "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5561: FSM Target Rotation Precheck

Experiment 5561 SHALL write
`results/experiment_5561_arc_fsm_target_rotation_precheck.json` before the
next credit-bearing ARC live attempt after Exp5548's honest no-bank null. The
workflow SHALL read `ops/arc_solve_registry.yaml` before selecting a target,
SHALL reject target levels already reproduced by the live mechanism, and SHALL
avoid the recent Exp5548 no-bank target unless a registry or ops-document reason
explicitly justifies retrying it.

The workflow SHALL build or reuse a simple finite-state action abstraction that
is reachable through the live candidate-router hook, SHALL enable
repeated-coordinate suppression before computing action entropy, and SHALL NOT
inspect hidden game source, run exhaustive offline ground-truth BFS, or build a
hand per-game calibration solver. The precheck is no-credit: it SHALL carry
`solve_provenance=live_agent_self_discovery`, SHALL invoke no LLM, and SHALL
claim readiness only when registry, target-rotation, FSM abstraction,
suppression, and entropy gates all pass.

The result artifact SHALL include `llm_invoked`,
`no_model_specs_required`, `selected_game`, `selected_level`,
`registry_precheck_passed`, `already_reproduced`,
`recent_no_bank_targets_avoided`, `fsm_action_abstraction_ready`,
`repeated_coordinate_suppression_enabled`, `action_entropy_precheck`,
`solve_provenance`, `arc_fsm_precheck_ready`, `tests_added_or_reused`,
`field_principles`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `llm_invoked`: principle "bare bool false proving the FSM target rotation precheck did not invoke any LLM."
- `no_model_specs_required`: principle "bare bool true because the declared no-LLM precheck substrate has no model invocation to name."
- `selected_game`: principle "registry-safe game id selected for the next FSM-guided live attempt, or empty string when blocked."
- `selected_level`: principle "adjacent unreproduced frontier level label selected after registry and recent no-bank target checks."
- `registry_precheck_passed`: principle "bare bool proving the registry was read and the selected level is not already reproduced."
- `already_reproduced`: principle "must remain false because duplicate live levels cannot satisfy the ARC standing progress floor."
- `recent_no_bank_targets_avoided`: principle "list of recent no-bank target markers rejected before selection unless an explicit retry reason exists."
- `fsm_action_abstraction_ready`: principle "bare bool proving the FSM action abstraction is live-router reachable and emits bounded action phases."
- `repeated_coordinate_suppression_enabled`: principle "bare bool proving repeated-coordinate suppression is active before action entropy is trusted."
- `action_entropy_precheck`: principle "bare float Shannon entropy over suppressed FSM action/coordinate choices before the live attempt."
- `solve_provenance`: principle "must equal live_agent_self_discovery even though this artifact claims no solve."
- `arc_fsm_precheck_ready`: principle "bare bool true only when registry, rotation, FSM abstraction, suppression, entropy, and no-LLM gates pass."
- `tests_added_or_reused`: principle "list of focused tests covering the Exp5561 schema, target rotation, FSM reachability, suppression, and artifact write."
- `field_principles`: principle "mapping of one-line principle annotations for each headline and gate field."
- `inference_substrate`: principle "must equal arc_live_path_precheck_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5562: Gated FSM Live Level-Up Attempt

Experiment 5562 SHALL write
`results/experiment_5562_arc_fsm_live_levelup.json` after the Exp5561-selected
FSM ARC target is rechecked against the registry. The workflow SHALL run a live
level-up attempt only when Exp5561 reports `arc_fsm_precheck_ready=true`, the
selected target is still adjacent and unreproduced, and the FSM action
abstraction remains live-router reachable. If the registry now records the
selected target level as already reproduced, the workflow SHALL emit a
duplicate-prevented artifact, SHALL NOT run the live attempt, and SHALL leave
`registry_delta=0`.

The credited attempt SHALL use only the live agent's own runtime attempts and
runtime-reachable FSM induction/routing. It SHALL NOT inspect hidden game
source, run an offline solver or exhaustive offline ground-truth BFS as solve
evidence, use an LLM strategy proposer, read hidden-source mechanics, or treat
LLM proposer text as the credited solver path. If the live agent reaches a new
candidate level, post-solve banking SHALL run only after that live path has
found the solution. A level-up claim SHALL be accepted only when
`solve_provenance=live_agent_self_discovery`, `offline_reproduced=true`, and
`reproduced_levels>=1`; otherwise the artifact SHALL emit an honest null and
leave `registry_delta=0`.

The result artifact SHALL include `llm_invoked`,
`no_model_specs_required`, `upstream_arc_precheck`, `solve_provenance`,
`llm_strategy_proposer_used`, `random_seed`, `reproducibility_checksum`,
`selected_game`, `selected_level`, `attempts`, `trajectory_path`,
`action_entropy`, `repeated_coordinate_rate`, `offline_reproduced`,
`reproduced_levels`, `registry_delta`, `arc_live_levelup_ready`,
`tests_added_or_reused`, `field_principles`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `llm_invoked`: principle "bare bool false proving the credited FSM live attempt did not invoke any LLM."
- `no_model_specs_required`: principle "bare bool true because this no-LLM live-agent substrate has no model invocation to name."
- `upstream_arc_precheck`: principle "path to the Exp5561 gate that selected the target and proved FSM live-path reachability before runtime."
- `solve_provenance`: principle "must equal live_agent_self_discovery so only the live runtime's own attempts and runtime reverse engineering receive solve credit."
- `llm_strategy_proposer_used`: principle "bare bool false proving no LLM strategy proposer text contributed to the credited solver path."
- `random_seed`: principle "deterministic seed used for target recheck, bounded live routing, trajectory logging, and checksum replay."
- `reproducibility_checksum`: principle "content-addressed hash over target, seed, trajectory metrics, duplicate gate, and banking result to catch silent drift."
- `selected_game`: principle "Exp5561-selected registry-safe game id rechecked immediately before the live attempt."
- `selected_level`: principle "Exp5561-selected adjacent unreproduced frontier level label, or the duplicate-prevented target label."
- `attempts`: principle "bare int count of live-agent runtime actions executed; zero means the duplicate or readiness gate prevented runtime."
- `trajectory_path`: principle "path to the detailed live trajectory or duplicate-prevented trajectory receipt."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed coordinate actions that repeated an earlier executed coordinate."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the post-solve offline reproduction gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires at least one."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the accepted reproduction gate passes after live discovery."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5561, registry reread, live-reachability, no-LLM metadata, and duplicate gates allowed runtime."
- `tests_added_or_reused`: principle "list of focused tests covering the Exp5562 schema, duplicate prevention, live trajectory metrics, checksum, and banking gate."
- `field_principles`: principle "mapping of one-line principle annotations for each headline and gate field."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, duplicate_prevented:, or blocked:."

### REQ-ARC-FCP-5575: SGE Anti-Stagnation Live-Path Precheck

Experiment 5575 SHALL write
`results/experiment_5575_sge_anti_stagnation_live_precheck.json` before any
follow-up SGE live inference spend. The workflow SHALL read the recorded SGE
trace at `results/outer_loop_sge_smoke_test.json`, define fixed collapse
thresholds before evaluation, and detect collapse using repeated normalized
strategy text, low pairwise strategy distance, repeated action proposals, and
consecutive null outcomes. It SHALL implement the anti-stagnation diversity
controller inside the existing `LLMStrategyProposer`/`SGECandidateRouter`
candidate-router path consumed by `E3AgentPolicy`, not as a standalone router.

When collapse is detected, the live-path router SHALL force a bounded portfolio
spanning observation, active coordinate probe, action-type probe, mechanic
falsification, and recovery/reset hypotheses. The controller SHALL apply an
outcome-conditioned taboo set against recently failed normalized strategies and
SHALL remain stable when the LLM proposer fails or returns malformed output. It
SHALL NOT use win-check, level-completion, hidden-source, scorecard, or oracle
signals for strategy generation, ranking, or readiness.

The precheck workflow SHALL read `ops/arc_solve_registry.yaml`, reject targets
already reproduced at the requested level, reject recently exhausted targets
unless the anti-stagnation controller is the new mechanism that justifies a
retry, select one unsolved adjacent frontier target and action budget, and emit
`live_path_ready=true` only when tests pass, the submitted E3 import path reaches
the controller, the controller activates on the recorded SGE trace, no oracle
leakage exists, and `target_unsolved=true`.

The result artifact SHALL include `field_principles`, `llm_invoked`,
`prior_trace_path`, `collapse_definition`,
`collapse_detected_on_prior_trace`, `diversity_metrics_before_after`,
`forced_portfolio`, `taboo_policy`, `e3_import_path`,
`live_path_reachable`, `verifier_is_oracle`,
`win_check_used_for_ranking`, `registry_precheck`, `target_game`,
`target_level`, `prior_levels_reproduced`, `action_budget`,
`tests_run`, `positive_control_passed`, `solve_provenance`,
`inference_substrate`, `target_unsolved`, `live_path_ready`, and
`honest_verdict`.

Required field principles:

- `llm_invoked`: principle "bare bool false proving Exp5575 is a deterministic precheck and spends no live LLM inference."
- `prior_trace_path`: principle "path to the recorded SGE trace used only for collapse diagnosis, not solve credit."
- `collapse_definition`: principle "fixed thresholds declared before evaluating the prior trace."
- `collapse_detected_on_prior_trace`: principle "bare bool true only when the fixed repeated-strategy, pairwise-distance, repeated-action, and null-outcome signals trip the collapse gate."
- `diversity_metrics_before_after`: principle "records strategy/action diversity before and after forced anti-stagnation routing."
- `forced_portfolio`: principle "bounded live-path-compatible strategy categories spanning observation, coordinate probe, action-type probe, mechanic falsification, and recovery/reset."
- `taboo_policy`: principle "outcome-conditioned taboo set derived only from recently failed normalized strategies."
- `e3_import_path`: principle "exact submitted path proving E3AgentPolicy reaches SGECandidateRouter through candidate_router.rank."
- `live_path_reachable`: principle "bare bool proving the controller is reachable through the scored E3 candidate-router hook."
- `verifier_is_oracle`: principle "must be false because precheck readiness cannot use oracle or hidden-source verifiers."
- `win_check_used_for_ranking`: principle "must be false because strategy ranking cannot inspect win, level-completion, scorecard, or source signals."
- `registry_precheck`: principle "structured evidence that registry depth was read and duplicate or exhausted targets were rejected before target selection."
- `target_game`: principle "registry-safe game id selected for the next SGE live attempt."
- `target_level`: principle "adjacent unreproduced frontier level selected after duplicate and exhaustion checks."
- `prior_levels_reproduced`: principle "registry depth before the attempted target; target_level must be greater."
- `action_budget`: principle "bounded action budget chosen before live inference."
- `tests_run`: principle "exact test commands proving fake-completer behavior and full Python suite status."
- `positive_control_passed`: principle "bare bool true only when fake-completer collapse activation and E3 reachability tests pass."
- `solve_provenance`: principle "must equal live_agent_self_discovery because the follow-up live attempt must be the scored runtime's own discovery path."
- `inference_substrate`: principle "must equal deterministic_live_path_precheck_no_llm."
- `target_unsolved`: principle "a target already reproduced at the requested level cannot receive duplicate solve credit."
- `live_path_ready`: principle "live inference requires tested E3 reachability, a new mechanism, and no oracle leakage."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

## Scenarios

### SCENARIO-ARC-FCP-5575: SGE Anti-Stagnation Controller Is E3-Reachable

Given the prior SGE trace repeatedly proposes passive waiting strategies with
null outcomes
When the fixed Exp5575 collapse thresholds are evaluated
Then repeated normalized strategy text, low pairwise strategy distance, repeated
action proposals, and consecutive null outcomes are reported, and
`collapse_detected_on_prior_trace=true`.

Given collapse has been detected inside `SGECandidateRouter`
When the live candidate-router hook ranks candidates for `E3AgentPolicy`
Then no live LLM completion is required for the collapsed step, the forced
portfolio emits observation, active coordinate probe, action-type probe,
mechanic falsification, and recovery/reset hypotheses, and diversity metrics
increase relative to the collapsed history.

Given recent failed strategies exist in router history
When the controller builds its taboo set
Then taboo entries are normalized from outcome-conditioned failed strategies
only, malformed or failed LLM completions degrade to deterministic fallback
ranking, and prompts/ranking diagnostics contain no win-check, level-completion,
hidden-source, scorecard, or oracle signal.

Given the ARC registry records the candidate target depth
When Exp5575 performs the registry precheck
Then it selects only a deeper unreproduced adjacent frontier target, records
`target_unsolved=true`, and sets `live_path_ready=true` only if tests pass, the
E3 import path reaches the controller, collapse controls activate on the prior
trace, and oracle leakage gates remain false.

### SCENARIO-ARC-FCP-5562: FSM Live Attempt Banks Only Reproduced Self-Discovery

Given Exp5561 reports `arc_fsm_precheck_ready=true` and selects `r11l:L3`
And the registry still records `r11l` below L3
When Exp5562 runs one bounded FSM live attempt with no LLM proposer
Then the artifact records `upstream_arc_precheck` pointing to Exp5561,
`solve_provenance=live_agent_self_discovery`,
`llm_invoked=false`, `llm_strategy_proposer_used=false`,
`no_model_specs_required=true`,
`inference_substrate=arc_live_agent_self_discovery_no_llm`, trajectory metrics,
and no `model_specs` or `target_model`.

Given the live runtime discovers a candidate level-up trajectory
When the post-solve offline reproduction gate verifies at least one new level
Then Exp5562 updates the registry according to repository practice and emits
`complete:`, `offline_reproduced=true`, `reproduced_levels>=1`, and matching
`registry_delta`.

Given the live runtime does not discover a reproducible new level
When the bounded live budget is exhausted
Then Exp5562 emits `honest_null:`, records the same trajectory and metric fields,
sets `offline_reproduced=false`, `reproduced_levels=0`, and leaves
`registry_delta=0`.

Given the registry now records the Exp5561-selected target level as already
reproduced
When Exp5562 performs the immediate duplicate recheck
Then it emits a duplicate-prevented artifact with `attempts=0`,
`arc_live_levelup_ready=false`, a reproducible trajectory receipt, and does not
rerun the solve.

### SCENARIO-ARC-FCP-5561: FSM Target Rotation Avoids Recent No-Bank Target

Given Exp5548 selected `g50t:L3`, banked no reproduced levels, and recorded
`registry_delta=0`
And the registry records multiple adjacent unreproduced frontier targets
When experiment 5561 runs the FSM target rotation precheck
Then it rejects `g50t:L3` as a recent no-bank target, selects a different
registry-safe adjacent frontier target, records
`recent_no_bank_targets_avoided=["g50t:L3"]`, and writes
`results/experiment_5561_arc_fsm_target_rotation_precheck.json` without a
solve claim.

Given the FSM action abstraction receives repeated click coordinates from
multiple finite-state phases
When experiment 5561 applies repeated-coordinate suppression through the live
candidate-router hook
Then selected FSM actions cover multiple coordinates, `action_entropy_precheck`
is a float above the readiness threshold,
`repeated_coordinate_suppression_enabled=true`, and
`arc_fsm_precheck_ready=true` only when no LLM was invoked and all registry and
provenance gates pass.

### SCENARIO-ARC-FCP-5548: Clean Live Attempt Banks Only Reproduced Self-Discovery

Given Exp5547 reports `arc_clean_precheck_ready=true` and the registry still
has the selected target as an adjacent unreproduced frontier level
When Exp5548 runs one bounded live attempt with the recorded seed and
repeated-coordinate suppression setting
Then the artifact uses `solve_provenance=live_agent_self_discovery`,
`llm_strategy_proposer_used=false`,
`no_model_specs_required=true`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
omits `model_specs` and `target_model`, and records trajectory metrics plus an
exact replay or no-bank result.

Given the registry has already banked Exp5547's selected target
When Exp5548 prepares the live attempt
Then it rotates to a fresh adjacent frontier target, records the rotation
reason, and still applies the same clean no-LLM metadata gates.

Given a live trajectory reaches a candidate level
When the standard offline replay gate does not reproduce at least one new level
Then Exp5548 emits `honest_null:`, `offline_reproduced=false`,
`reproduced_levels=0`, and `registry_delta=0`.

### SCENARIO-ARC-FCP-5547: Clean No-LLM Precheck Blocks Duplicates And Omits Model Specs

Given the ARC solve registry has at least one reproduced game with an
unbanked adjacent frontier level
When Exp5547 builds the clean substrate precheck
Then it selects a target whose requested level is deeper than the registry
depth, records `already_reproduced=false`,
`llm_strategy_proposer_used=false`, `no_model_specs_required=true`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`solve_provenance=live_agent_self_discovery`, a deterministic `random_seed`,
and a non-empty `reproducibility_checksum`
And the artifact contains no `model_specs` or `target_model` field.

Given the registry already banks a proposed target level
When Exp5547 runs the duplicate precheck
Then that target is rejected before readiness and the artifact only becomes
ready after selecting a non-duplicate adjacent frontier target.

### SCENARIO-ARC-FCP-5508: Classical Perception Generation Is Runtime-Grounded

Given the Exp5507 target precheck is ready and the registry re-read shows the
selected level is not already reproducible
When Exp5508 runs one bounded live `E3AgentPolicy` attempt with the classical
perception-generation pass
Then candidate actions are generated from connected components, color blobs,
sprite overlays, salient motion, and action affordances observed at runtime
And the artifact records trajectory-taxonomy counts, prohibited-input flags,
registry before/after totals, and an honest null unless the standard
reproduction gate confirms at least one new level.

### SCENARIO-ARC-FCP-4490: Positive-Control Candidate Ranking

Given a frame and candidate actions where only one click cell is known to
change the frame
When the behavior prior or scorer ranks the candidates
Then the changing click is ordered ahead of no-op candidates
And the legacy candidate order remains the tie-break for equal scores.

### SCENARIO-ARC-FCP-4726: Online Driver Arms Must Be Non-Degenerate Before Lift

Given the frozen, online-scratch, and online-warm goal-free driver arms are
available and the Qwen/arcade/CUDA preconditions pass
When experiment 4726 runs the valid-driver gate before aggregating held-out
first-win rates
Then it records distinct per-arm action distributions, positive-gradient
online Adam train steps, and online coordinate-head click proposals that differ
from the frozen prior before setting `arms_non_degenerate=true`.

Given any of those non-degeneracy witnesses is missing
When experiment 4726 writes its artifact
Then it reports
`complete: online_driver_arms_degenerate_confirmed_harness_bug`, keeps
`chosen_submitted_config` unchanged, and does not treat a flat first-win delta
as a capability null.

### SCENARIO-ARC-FCP-4629: Submitted E3 Uses Live-Reachable Action-Effect Ranking

Given local cached transition effects and a small trained CNN action-effect
checkpoint are available
When the submitted `E3AgentPolicy` builds its `StepwiseExplorer`
Then the explorer receives a live action-effect scorer by default
And `rich_action_candidates` applies that scorer as the final ordering pass
after any candidate-router tie-break
And experiment 4629 writes a terminal artifact with the matched bare control,
bootstrap action-delta CI, parity-test result, orphan-lint result,
`verifier_is_oracle=false`, and `solve_provenance=live_agent_self_discovery`.

### SCENARIO-ARC-FCP-4641: Submitted E3 Uses Live-Reachable Expansion Priority

Given the 4629 live action-effect scorer is available
When `graph_explore_solve_v2` or the submitted `E3AgentPolicy` has multiple
frontier branches with remaining untested actions
Then the branch whose remaining actions have higher predicted frame-change is
expanded before predicted no-op branches
And the ranker-only mode remains available as the matched control
And experiment 4641 writes a terminal artifact with live solve-rate,
depth-of-live-solve, median actions-to-win, first-win-rate deltas, bootstrap
CIs, parity-test result, orphan-lint result, `verifier_is_oracle=false`, and
`solve_provenance=live_agent_self_discovery`.

### SCENARIO-ARC-FCP-4491: Missing Corpus Does Not Fabricate Results

Given the ARC Public Demo replay corpus and replay-derived weights are absent
When experiment 4490 writes its terminal artifact
Then the artifact is valid JSON, starts `honest_verdict` with a terminal
prefix, records the missing corpus in `preconditions_checked`, leaves
`weights_bundled=false`, and reports no held-out efficiency win.

### SCENARIO-ARC-FCP-4492: Passing Structural Energy Gate Wires Ranking

Given the discriminative v3 verifier reports leave-one-game-out AUROC above
0.600 versus the 0.503 baseline
When experiment 4492 writes its terminal artifact
Then `honest_verdict` is terminal-prefixed, `loo_gate_passed=true`, the feature
classes that materially moved AUROC are recorded, and frame-change ranking can
rank candidates by `P(change) * (-delta_E)` with stable ties.

### SCENARIO-ARC-FCP-4495: Attributed Mirror Shards Load Without Cold Download

Given the official ARC Public Demo short link is not directly reachable from
the staging machine but an attributed mirror is reachable
When experiment 4495 writes replay-derived training shards under the gitignored
data directory
Then the loader reads rows containing `frame`, `action`, `frame_delta`, and
`level_progress`, the artifact records provenance and attribution, and
`weights_committed=false`.

### SCENARIO-ARC-FCP-4496: Scoreboard Separates Headline From Banked Levels

Given a cached submitted-default benchmark artifact and the current
variant-transfer signal
When experiment 4496 writes its scoreboard artifact
Then the row reports the exact submitted-default held-out generic solve-rate and
variant-transfer rate as headline metrics, preserves source provenance for both
measurements, keeps `reproducible_total_levels` in a context-only field, and
records that `test_arc_submitted_agent_parity.py` is the focused parity gate.

### SCENARIO-ARC-FCP-4505: Refreshed Scoreboard Pins Submitted Defaults

Given the refreshed submitted-default held-out benchmark, the current
variant-transfer signal, and the value-weight remeasurement verdict
When experiment 4505 writes its scoreboard artifact
Then the artifact reports the current submitted-default config with
`value_weight==0.0`, records the parity test as green, reports the generic
solve-rate and variant-transfer rate as headline metrics, and leaves
`reproducible_total_levels` in context-only metadata.

### SCENARIO-ARC-FCP-4524: Stop Policy Preserves CORE Level Depth

Given the fixed eight-game local submission gate has a measured
run-to-completion control
When experiment 4524 compares that control with a stop-at-scored-target run
Then the artifact records actions-to-reach-each-level versus total actions per
game, uses `action_field_used="actions"` for both arms, preserves every CORE
solve, preserves every game's before/after `best_level`, and only reports
success when the treatment's median CORE total actions is strictly lower than
the measured control.

### SCENARIO-ARC-FCP-4547: CNN Ranker Beats Only With Positive Control

Given the cached ARC Public Demo human replay shards are present and
`arc_solver_kit.offline_arcade()` imports successfully
When experiment 4547 trains the small CNN on frame-only action-effect rows and
measures CNN-ranked candidates against the matched blind-BFS order
Then the artifact records the required median action, solve-rate, AUROC,
positive-control, random-seed, checksum, and precondition fields, and a success
verdict is emitted only when the CNN median actions-to-first-level-up is
strictly lower than blind BFS while preserving solve-rate.

Given the CNN does not reduce held-out median actions
When the held-out transition-delta AUROC is above the trivial baseline
Then the artifact reports `complete: frame_change_cnn_no_action_reduction_honest_null`,
sets `false_negative_risk_checked=true`, and records hidden-field probing for
the `ka59`/`ar25` L2 stall as the secondary input.

### SCENARIO-ARC-FCP-4568: Pooled Clickability Predictor Emits Honest Efficiency Delta

Given the local human replay shards and local transition corpus are present
When experiment 4568 trains the small frame-only action-effect CNN and measures
predictor-ranked candidates against the matched blind order
Then the artifact records the required median actions, explicit action delta,
bootstrap CI, efficiency score, generic-transfer fields, non-oracle verifier
flag, positive control, null note when needed, checksum, and preconditions, and
emits a success verdict only when the action delta is positive with no
solve-rate drop.

Given the predictor does not reduce held-out median actions
When the positive control passes
Then the artifact reports
`complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened`,
sets `false_negative_risk_checked=true`, keeps `chosen_submitted_config` as
`unchanged`, and records the residual missing-verifier gaps.

### SCENARIO-ARC-FCP-4501: Staged Frame-Only Rerun Has Interpretable Null Guard

Given locally staged replay shards but no bundled 14,672-example
`action_effect_dict.npz`
When experiment 4501 trains or smoke-trains the frame-only predictor and writes
its terminal artifact
Then the artifact is valid JSON with terminal-prefixed `honest_verdict`,
records the staged corpus count and the missing NPZ target, emits a behavior
prior, keeps `solve_rate_dropped=false`, reports before/after held-out action
metrics from the same candidate-order harness, and marks
`false_negative_risk_guard=positive_control_passed_null_interpretable` when the
positive control detects a known ranking win.

### SCENARIO-ARC-FCP-4502: Energy-Augmented Ranking Is Measured Against Predictor-Only

Given the 4492 structural energy gate passed and the 4501 frame-change
predictor/prior can rank held-out cached candidates
When experiment 4502 scores each candidate with both `P(frame_change)` and the
v3 structural `delta_E`
Then the artifact compares predictor-only and energy-augmented held-out
solve-rate and action-efficiency on identical candidate groups, records
`inference_substrate=verifier_ensemble_against_cached_candidates`, and uses a
terminal-prefixed honest verdict that reports either an added-value win or an
honest null over predictor-only.

### SCENARIO-ARC-FCP-4511: Predicted No-Ops Are Pruned Before Expansion

Given a frame, legacy candidate actions, and a frame-change scorer that assigns
low probability to no-op candidates
When `rich_action_candidates` is called with a pruning threshold
Then candidates below the threshold are removed before ranking or expansion,
at least one candidate is retained, and the experiment 4511 artifact records
the baseline 7760 control, pruned median action measurement, solve-rate guard,
held-out no-op precision, positive control status, random seed, and
reproducibility checksum.

### SCENARIO-ARC-FCP-4512: Imitation Prior Orders and Prunes Expansion

Given a frame, a legacy candidate list, and an imitation prior that assigns
higher likelihood to the human-like action sequence for that frame class
When the explorer builds candidates with an action prior and bottom-quantile
prune setting
Then the high-likelihood action is expanded before low-likelihood candidates,
the bottom quantile is pruned while retaining at least one candidate, and the
experiment 4512 artifact records the 7760 baseline, prior-guided median action
measurement, solve-rate guard, prior source, positive control status, false
negative risk check, random seed, and reproducibility checksum.

### SCENARIO-ARC-FCP-4513: Easy Frames Commit One Candidate, Ambiguous Frames Expand

Given a frame, a legacy ranked candidate list, value-margin/no-op/novelty
signals, and an adaptive budget threshold
When the ambiguity score is below the threshold
Then the explorer retains only the top candidate for that frame and records a
budget-1 commit diagnostic.
When the ambiguity score is at or above the threshold
Then the explorer retains the normal candidate width, records an expanded
diagnostic, and the experiment 4513 artifact records the 7760 baseline,
adaptive median action measurement, threshold sweep, per-game solve-rate guard,
ambiguity signal components, positive control status, false negative risk check,
random seed, and reproducibility checksum.

### SCENARIO-ARC-FCP-4514: Lazy Value Scores Reorder Without Filtering

Given a live `StepwiseExplorer` frontier with more expandable nodes than the
lazy top-K value-scoring budget
When the explorer chooses a best-first frontier node with `value_weight>0`
Then only the top-K cheap-priority nodes pay the value-head cost, repeated frame
hashes reuse cached scores, unscored tail nodes remain in the frontier, and the
experiment 4514 artifact reports every swept weight, the explicit
`value_weight=0.0` control, the chosen submitted value weight, lazy-eval speedup
confirmation, false-negative-risk check, random seed, and reproducibility
checksum.

### SCENARIO-ARC-FCP-4516: Integrated Submission Gate Reports Null Or Win Honestly

Given the A1-A4 artifacts, the submitted agent config, and the fixed 8-game
local submission gate
When experiment 4516 selects integration levers
Then flagged artifacts are skipped, non-flagged levers are accepted only when
they preserve the core solve set and reduce core median actions versus the
no-lever control, and the submitted config keeps `value_weight=0.0` when no
positive value weight wins.

Given a live `StepwiseExplorer` with known forward edges from the current node
to an ancestor of a selected frontier node but no exact path to the frontier
When the explorer serves navigation for that frontier
Then it walks forward to the deepest reachable ancestor, replays only the
suffix before probing the frontier action, records navigation diagnostics, and
falls back to RESET replay only when no forward ancestor is reachable.

Given the integrated submitted config is measured on the fixed 8-game gate
When the artifact is written
Then it records the 7760 baseline, integrated median actions, integrated
solve-rate, held-out solve-rate, nav-loop finding, false-negative-risk check,
random seed, reproducibility checksum, and the exact levers integrated.

### SCENARIO-ARC-FCP-4518: Canonical Gate Guards The Fixed Metric

Given the verified 7760 baseline and the fixed eight-game local submission
gate
When the gate builds a dashboard row for a named lever
Then the row uses total `actions` for both baseline and treatment, preserves
the CORE set-containment verdict, reports bonus solves separately, and fails a
treatment that reports a different action field than the baseline.

Given the gate's regression fixtures
When the focused gate tests run
Then the A1 and A2 CORE-loss fixtures fail, the positive-control and neutral
fixtures pass, bonus solves are reported, the legacy baseline fallback remains
covered, and the fixed game set plus 7760 baseline guard cannot move silently.

Given candidate budgets `{8000,12000,16000,24000}`
When experiment 4518 measures baseline headroom
Then it selects the smallest `B*` whose solved set matches the solved set at
`1.5B`, keeps `8000` only if the measurement is unavailable, and writes the
terminal artifact with the required field principles.

### SCENARIO-ARC-FCP-4523: Batch And Navigation Tie-Break Are Swept Against CORE

Given a live `StepwiseExplorer` with multiple equal-depth frontier nodes
When navigation-cost tie-break is enabled
Then the selected frontier still comes from the shallowest eligible depth, but
equal-depth ties prefer exact forward walks from the current node before
shorter RESET replay paths.

Given a selected frontier node with several untested salient actions
When the explorer navigates to that node with frontier batch size greater than
one
Then up to `k` actions from that node are queued before global frontier
selection moves elsewhere, while `k=1` preserves the prior single-probe
control behavior.

Given experiment 4523 measures the fixed eight-game gate
When it compares the sweep against the `k=1`/no-tie-break control
Then every row uses the same total `actions` field, reports CORE preservation
and navigation diagnostics, passes the positive-control reduction guard, and
wires `SUBMITTED_AGENT_CONFIG` only for a strict `IMPROVED` CORE median action
reduction.

### SCENARIO-ARC-FCP-4527: Nav Metrics Are CI-Guarded But Secondary

Given the local submission gate measures the fixed eight-game set with the
authoritative per-level scorecard
When a measured config preserves CORE solves and per-level efficiency but has
more `reset_replay_steps` at equal total actions
Then the verdict remains non-inferior, the dashboard emits a navigation
regression warning, and the per-game rows expose `deepest_level_reached`,
`per_level_efficiency`, `reset_replay_steps`, and `forward_walk_hit_rate`.

Given the gate is invoked with `--update-baseline`
When the measured candidate baseline is missing CORE efficiency or drops `lp85`
below the canonical per-level efficiency floor
Then the baseline is not persisted and the command reports the canonical
baseline guard failure.

### SCENARIO-ARC-FCP-4528: .417 B-Track Audit Records No Blind Budget Raise

Given the upstream `.417 B2` canonical metric harness artifact
When experiment 4528 audits the artifact
Then it records the CORE containment gate, fixture CI guard, and measured
headroom table in `b_track_status`, cites the upstream artifact fields it
imports, and reports a complete audit if the headroom table has no stable `B*`
candidate instead of raising the local submission-gate default blindly.

### SCENARIO-ARC-FCP-4575: Learned-CNN Duration Floor

Given a fixture artifact with a CNN/torch marker, `duration_s` around five
seconds, no LLM/GGUF marker, and
`inference_substrate=verifier_ensemble_against_cached_candidates`
When adversarial verification and the summary reader process the artifact
Then the applied floor is the one-second verifier-scoring floor and no
`DURATION_TOO_SHORT` flag is emitted.

Given a fixture artifact with `inference_substrate=live_llm_inference`, a GGUF
marker, and `duration_s < 60`
When adversarial verification processes the artifact
Then it emits a critical `DURATION_TOO_SHORT` flag.

### SCENARIO-ARC-FCP-5360: Color-Blob Salience Is Live-Reachable

Given a rendered ARC frame with single-color connected components that include
a button-like salient-color blob, a larger dull blob, and a status-bar-colored
blob
When the color-blob salience prior scores live `rich_action_candidates`
Then the button-like blob is ranked ahead of the dull and status-bar blobs.

### SCENARIO-ARC-FCP-5373: Salience Repair Is Validated Before Probing

Given a rendered ARC frame with a button-like blob, a large flat distractor, and
a status-bar-like strip
When the live salience repair scores `rich_action_candidates`
Then the button-like blob is ranked ahead of both flat/status distractors and
the artifact records the measured rank delta.

Given a frame-diff scorer assigns high confidence before any observed transition
has confirmed that high scores predict real frame change
When the submitted `E3AgentPolicy` asks for live candidate ordering
Then the frame-diff score is gated out until an observed transition validates it
against the actual before/after frame pixels.

Given the submitted `E3AgentPolicy` constructor builds its `StepwiseExplorer`
without an explicit action prior
When the live policy is instantiated
Then the explorer receives the classical color-blob salience prior and the
submitted config declares the mechanism live-reachable.

Given the registry records `re86`, `sb26`, `bp35`, and `lf52` at shallow depth
2
When experiment 5360 performs its registry precheck
Then it selects the next unbanked target level rather than duplicating an
already reproduced L1/L2 solve, runs the perception audit, and writes an
artifact whose credited progress fields only pass when offline reproduction
banks at least one new live-path level.

### SCENARIO-ARC-FCP-5385: Geometric Salience Stays On The Live Path

Given a rendered ARC frame with two same-tier button-like blobs and a later
agent-observed transition changing cells near only one of those blobs
When the geometric salience prior ranks live `rich_action_candidates`
Then the geodesic transition anchor moves the nearer blob ahead of the equal
base-salience distractor and the prior reports hyperbolic/geodesic ranking as
enabled.

Given the submitted `StepwiseExplorer` has an action prior that can observe
transitions
When the live policy ingests its own before/action/after observation
Then the action prior receives that observation through the same live path that
feeds the frame-change scorer.

Given `ops/arc_solve_registry.yaml` records `re86` below L3
When Experiment 5385 performs its registry precheck
Then it selects `re86` L3, marks duplicate credit disallowed, and only sets
`new_level_banked=true` if the live-agent-discovered action labels reproduce
beyond the prior level.

### SCENARIO-ARC-FCP-5397: Blob Tiers Shape Live Candidate Generation

Given a rendered ARC frame containing a status strip, a large flat colored
region, and a compact salient button-like component
When the live `StepwiseExplorer` asks `rich_action_candidates` for click
candidates with the connected-component salience prior enabled
Then the compact salient button-like component is generated before the click cap
can drop it, status-bar components are pushed to the lowest tier, and the live
explorer records emitted salience tiers.

Given `ops/arc_solve_registry.yaml` records `re86` below L3
When Experiment 5397 performs its registry precheck
Then it selects `re86` L3, marks duplicate solve avoidance complete, runs a
bounded live-agent attempt, and only emits `complete:` when a
live-agent-discovered new level is reproduced offline.

### SCENARIO-ARC-FCP-5410: Trajectory Prefixes Are Live-Observed And Uncertainty-Gated

Given the live agent observes repeated frame-changing button-like blob clicks
from its own attempts
When the trajectory frontier generator is asked for a multi-action prefix
Then it emits only prefixes supported by the observed transition stream,
records the blob/color salience route, and increments frontier expansion
evidence.

Given a single low-support or conflicting inferred dynamic
When the same generator is asked to promote that dynamic into a prefix
Then the uncertainty gate rejects it, records an uncertainty rejection, and
keeps `offline_reproduced=false` unless a live self-discovered solution later
passes the reproduction gate.

### SCENARIO-ARC-FCP-5423: CoEx Landmarks Persist Live Runtime Frontier Evidence

Given the live agent observes repeated frame-changing actions from its own
attempts
When the CoEx landmark frontier is asked for a multi-action prefix
Then it promotes only live-observed action clusters, persists the prefix across
resets, records a frontier transition, emits a measurement-access receipt, and
reports the discovered landmark count.

Given `lf52` L3 is not already banked in the registry
When experiment 5423 performs its registry precheck
Then it selects `target_game=lf52`, `target_level=L3`, and
`duplicate_solve_avoided=true` before running any bounded live attempt.

Given the bounded live attempt does not reproduce a new level
When the experiment writes its artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`arc_new_level_banked=false`, and `honest_verdict` starts with
`honest_null:` while preserving attempt counts, reset counts, frontier
transitions, landmarks, runtime observations, action-sequence receipts,
`no_offline_bfs=true`, and `no_per_game_adapter=true`.

### SCENARIO-ARC-FCP-5437: Registry-Guided Reinduction Attempts New Frontier

Given `cn04` has three reproduced levels and no reproduced L4 in the ARC solve
registry
When experiment 5437 performs its registry precheck
Then it selects `target_game=cn04`, `target_level=L4`, records the target as an
eligible unbanked frontier, and sets `duplicate_solve_avoided=true` before any
bounded live attempt runs.

Given the live agent gathers frame-changing runtime observations from its own
attempts
When experiment 5437 summarizes the attempt
Then it records runtime predicates, frontier transitions,
measurement-access action sequence receipts, reset count, and action count
without crediting any level unless the live-discovered sequence reproduces
beyond the registry depth.

Given the bounded reinduction attempt does not reproduce a new level
When the artifact is validated
Then `offline_reproduced=false`, `reproduced_levels=0`,
`arc_new_level_banked=false`, and `honest_verdict` starts with `honest_null:`
while preserving `registry_precheck=true`, `no_offline_bfs=true`,
`no_per_game_adapter=true`, and `inference_substrate=live_arc_agent_runtime`.

### SCENARIO-ARC-FCP-5450: Measurement-Access Rotation Avoids Stale Frontiers

Given the registry records `cn04` L3 and `re86` L2, and recent artifacts record
no-bank `cn04` L4 and repeated `re86` L3 salience attempts
When experiment 5450 performs its target precheck
Then it selects a different eligible next-level frontier, records
`registry_precheck_total_levels`, `selected_game`, `selected_target_level`, and
`target_rotation_reason`, and runs the bounded attempt only after duplicate
solve avoidance.

Given the live attempt gathers frame measurements, action-effect observations,
state-change summaries, or verifier-routed predicates from its own transitions
When the artifact is built
Then `runtime_predicates_induced` records those predicates and
`live_attempt_count` records the bounded live effort without reading hidden
source, running offline ground-truth BFS, or crediting a per-game adapter.

Given the bounded measurement-access attempt does not reproduce a level deeper
than the registry precheck
When the artifact is validated
Then `offline_reproduced=false`, `new_level_reproduced=false`,
`new_levels_banked=0`, `arc_new_level_banked=false`, and `honest_verdict`
starts with `honest_null:` while preserving the target rotation reason,
frontier evidence, attempted predicates, and residual wall.

### SCENARIO-ARC-FCP-5464: Metric Precheck Rejects Duplicate And Off-Path Credit

Given the ARC registry records a reproduced depth for a public game
When experiment 5464 audits a claimed solve whose target level is less than or
equal to that depth
Then the claim is rejected as duplicate, `duplicate_solve_rejected=true`, and
the reproduced total is unchanged.

Given a claimed solve uses `outer_loop_re`, source reading, offline
ground-truth BFS, a replay-only artifact, or a hand per-game adapter as the
credited path
When experiment 5464 audits provenance
Then the claim is rejected before solve credit, `off_path_solve_rejected=true`,
and the artifact remains a no-solve precheck.

Given reproduced loop artifacts and registry rows are available
When experiment 5464 audits null-coordinate exploit validity
Then one-step ACTION6 solves with null or missing click coordinates are treated
as metric contamination, while normal multi-step or coordinate-bearing replays
leave `null_coordinate_exploit_valid=false`.

### SCENARIO-ARC-FCP-5464: Perception Receipts And Target Shortlist Are Live-Path Ready

Given the live `E3AgentPolicy` can reach the connected-component salience prior
When experiment 5464 runs perception diagnostics
Then the receipts JSON contains connected component rows, color-blob rows,
changed-pixel rows, salience-tier rows, and action-effect observation rows
from live-path reachable code.

Given recent no-bank targets include `re86` L3, `lf52` L3, `cn04` L4, and
`ka59` L2
When experiment 5464 builds the Exp5465 shortlist
Then shortlisted targets avoid already reached levels and those recent no-bank
lanes unless a row records an explicit justification.

### SCENARIO-ARC-FCP-5465: Gated Salience Attempt Uses Live Features And Reproduction Credit

Given Exp5464 reports `arc_metric_integrity_ready=true` and a non-empty
`target_shortlist`
When experiment 5465 re-runs the registry precheck
Then it selects a target from that shortlist, records the precheck depth as
`target_level_before`, and attempts `target_level_before + 1`.

Given the live `E3AgentPolicy` salience path is available
When experiment 5465 builds its live attempt receipts
Then `perception_features_used` contains connected-component, color-blob,
changed-pixel, salience-tier, and action-effect features from live-path
reachable code.

Given a bounded live/offline attempt does not reproduce deeper than the selected
precheck depth
When experiment 5465 validates its artifact
Then `offline_reproduced=false`, `reproduced_levels` remains at
`target_level_before`, `new_level_banked=false`,
`arc_registry_update_required=false`, `source_reading_used=false`,
`offline_bfs_used=false`, `hand_adapter_credited=false`, and
`honest_verdict` starts with `honest_null:`.

Given a candidate reaches a deeper level through the live-agent route
When the official reproduction gate confirms the candidate beyond the selected
precheck depth
Then and only then `offline_reproduced=true`, `new_level_banked=true`,
`arc_registry_update_required=true`, and `honest_verdict` starts with
`complete:`.

### SCENARIO-ARC-FCP-5479: Rotated Target Precheck Avoids Duplicate And Recent No-Bank Lanes

Given Exp5464 shortlisted `bp35:L3`, `sb26:L3`, `g50t:L3`, `dc22:L3`, and
`sp80:L3`, Exp5465 no-banked `bp35:L3`, and the registry records those games at
L2
When experiment 5479 selects a target before any level-up attempt
Then it rejects already reproduced duplicate target probes, avoids
`bp35:L3`, `ka59:L2`, and `cn04:L4`, selects the first eligible rotated target
from `sb26:L3`, `g50t:L3`, `dc22:L3`, or `sp80:L3`, and records the registry
total before the attempt.

Given the submitted live salience path is importable
When experiment 5479 runs the bounded dry check
Then the salience summary reports connected components, color blobs, changed
cells, target-region candidates, and known blockers without hidden source
reading, offline BFS, or hand-adapter credit.

Given the artifact is validated
When experiment 5479 writes
`results/experiment_5479_arc_target_rotation_precheck_v497.json`
Then `arc_target_rotation_ready=true`, `solve_claimed=false`,
`inference_substrate=arc_live_path_precheck_no_solve`, and `honest_verdict`
contains no level-solve claim.

### SCENARIO-ARC-FCP-5480: Rotated Salience Attempt Banks Only Reproduced New Levels

Given Exp5479 selected `sb26:L3` and the registry records `sb26` at L2
When experiment 5480 loads the target before attempting a level-up
Then it records `game=sb26`, `target_level=3`,
`reproduced_levels_before=2`, and proceeds only because the target is not
already reproduced.

Given the registry already records the selected target level or the Exp5479
target fields are missing
When experiment 5480 runs
Then it emits a `blocked:` artifact with `action_count=0`,
`offline_reproduced=false`, `new_level_banked=false`, and
`registry_updated=false`.

Given the bounded live salience attempt does not reproduce a level beyond the
precheck depth
When experiment 5480 validates its artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`reproduced_levels_after=reproduced_levels_before`,
`new_level_banked=false`, `registry_updated=false`,
`hidden_source_reading=false`, `offline_bfs_used=false`,
`hand_adapter_used=false`, `outer_loop_re_used=false`, and
`honest_verdict` starts with `honest_null:`.

Given a candidate reaches the selected target level through the live-agent
self-discovery path
When the registry-approved reproduction gate confirms it beyond the precheck
depth
Then and only then `offline_reproduced=true`, `reproduced_levels>=1`,
`new_level_banked=true`, `registry_updated=true`, `first_win_trace_path` is
non-empty, and `honest_verdict` starts with `complete:`.

### SCENARIO-ARC-FCP-5493: Trajectory Target Precheck Avoids Stale And Retired Lanes

Given the registry records `sb26`, `bp35`, and `re86` at L2, `ka59` at L1,
`cn04` at L3, and `dc22` at L2
When experiment 5493 evaluates trajectory/option-induction candidates
Then it excludes `sb26:L3`, `bp35:L3`, `ka59:L2`, `cn04:L4`, and `re86:L3`,
rejects any candidate target whose level is already reproduced, rejects
novelty-only, curiosity-only, energy-as-fitness quality-diversity, and
archive-granularity reruns from the Exp5154 retired scope, and selects the
first remaining target with live-observation trajectory hooks.

Given `dc22:L3` survives those filters
When experiment 5493 writes
`results/experiment_5493_arc_trajectory_target_precheck_v498.json`
Then `registry_path=ops/arc_solve_registry.yaml`,
`duplicate_solve_avoided=true`, `selected_game=dc22`,
`selected_target_level=3`, `prior_levels_reproduced=2`,
`offline_bfs_used=false`, `per_game_adapter_used=false`,
`arc_trajectory_precheck_ready=true`,
`inference_substrate=registry_precheck_no_solve`, and `honest_verdict`
starts with `complete:`.

Given no candidate survives duplicate, recent no-bank, and retired-scope
filters
When experiment 5493 writes its artifact
Then `selected_game` is an empty string, `selected_target_level=0`,
`arc_trajectory_precheck_ready=false`, `offline_bfs_used=false`,
`per_game_adapter_used=false`, and `honest_verdict` starts with `blocked:`.

### SCENARIO-ARC-FCP-5494: Exp5493 Target Attempts Through Live Trajectory Induction

Given Exp5493 selected `dc22:L3`, the registry records `dc22` at L2, and the
Exp5493 trajectory preconditions include runtime action-effect observations,
visible toggle/navigation state changes, level-counter deltas, and frontier
prefix grouping
When experiment 5494 performs its pre-attempt gate
Then it selects `selected_game=dc22`, `target_level=3`,
`prior_levels_reproduced=2`, `solve_provenance=live_agent_self_discovery`,
`offline_bfs_used=false`, `per_game_adapter_used=false`,
`game_source_read=false`, and proceeds only because the target is not already
reproduced and is not a recent same-mechanism no-bank duplicate.

Given the live agent executes a bounded `E3AgentPolicy` attempt with
`LiveCoExLandmarkFrontierGenerator`
When the attempt is summarized
Then the artifact records `trajectory_hypothesis_count`,
`live_attempt_count`, hypothesized action sequences, observation deltas,
verifier checks, rejection reasons, and `model_specs_if_llm_used=[]` when no
LLM generator was invoked.

Given the bounded live trajectory-induction attempt does not reproduce a new
level beyond the prior registry depth
When experiment 5494 validates the artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`new_level_banked=false`, `registry_updated=false`,
`post_levels_reproduced=prior_levels_reproduced`, and `honest_verdict` starts
with `honest_null:`.

Given a live self-discovered trajectory candidate reaches the target level
When the standard reproduction gate confirms it beyond the prior depth
Then and only then `offline_reproduced=true`, `reproduced_levels>=1`,
`new_level_banked=true`, `registry_updated=true`,
`post_levels_reproduced>prior_levels_reproduced`, and `honest_verdict` starts
with `complete:`.

### SCENARIO-ARC-FCP-5507: Null-Coordinate Perception Precheck Selects Changed Mechanism

Given Exp5494 records a `dc22:L3` no-bank trajectory attempt, the registry
records `dc22` at L2, and prior no-bank artifacts record `bp35:L3` and
`sb26:L3` salience-only nulls
When experiment 5507 aggregates upstream evidence
Then it rejects the already reproduced duplicate probes, rejects recent
same-target/same-mechanism no-bank reruns, audits Exp5494's zero-change click
receipts as valid recorded ACTION6 coordinates rather than missing/null
coordinates, records perception-grounding findings from connected components,
color blobs, salience tiers, changed pixels, and action-effect asymmetry, and
selects `dc22` `L3` only with a materially changed perception-generation
mechanism.

Given experiment 5507 writes
`results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json`
When the artifact is validated
Then `registry_path=ops/arc_solve_registry.yaml`,
`reproducible_total_levels_before` is a bare int,
`duplicate_targets_rejected` and `recent_no_bank_targets_rejected` are lists,
`null_coordinate_audit` is a dict, `perception_grounding_findings` is a list,
`selected_game` and `selected_level` are strings,
`levelup_attempt_ready=true`, `solve_claimed=false`,
`inference_substrate=aggregation_from_upstream_artifacts`, and
`honest_verdict` starts with `complete:`.

Given every candidate is either already reproduced, a same-mechanism no-bank
rerun, or lacks perception evidence
When experiment 5507 builds its artifact
Then it leaves `selected_game`, `selected_level`, and `selected_mechanism`
empty, sets `levelup_attempt_ready=false`, keeps `solve_claimed=false`, and
uses a `blocked:` honest verdict with exact reasons.

### SCENARIO-ARC-FCP-5520: Action-Diversity Precheck Selects Changed Target

Given Exp5508 repeatedly chose ACTION6 at the same small coordinate set and did
not bank a new level
And the registry records the Exp5508 target depth but has a different candidate
game with an unreproduced next level
When experiment 5520 runs the no-credit action-diversity precheck
Then it rejects the Exp5508 target/pattern, rotates to one registry-safe target,
measures action entropy, repeated-coordinate rate, and salience coverage from
the changed live-path candidate generator, and writes
`results/experiment_5520_arc_action_diversity_target_precheck.json` without a
solve claim.

Given the registry already reproduces the candidate level or the dry-run probe
collapses to the Exp5508 repeated-coordinate pattern
When experiment 5520 validates the artifact
Then `arc_levelup_candidate_ready=false`, `already_reproduced` or
`exp5508_pattern_reused` records the blocker, and `honest_verdict` starts with
`blocked:`.

### SCENARIO-ARC-FCP-5521: Live Action-Diverse Attempt Is Reproduction-Gated

Given Exp5520 selected `sb26:L3` with a changed action-diverse generator and
the registry still records only `sb26` depth 2
When experiment 5521 runs the bounded live self-discovery attempt
Then the attempt records observations, proposed actions, verifier feedback,
action entropy, repeated-coordinate rate, and salience coverage in a trajectory
log
And the result writes
`results/experiment_5521_arc_live_action_diverse_levelup.json` with
`solve_provenance=live_agent_self_discovery` and
`inference_substrate=arc_live_agent_self_discovery`.

Given the live attempt does not reproduce a new level through the standard
offline replay gate
When experiment 5521 validates the artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`banking_gate=false`, `registry_delta=0`, `reproduction_command=null`, and
`honest_verdict` starts with `honest_null:` while preserving enough trajectory
detail to distinguish the attempt from Exp5508.

Given the live attempt reaches a new level and the replay gate reproduces it
When experiment 5521 banks the result
Then `offline_reproduced=true`, `reproduced_levels>=1`,
`banking_gate=true`, `registry_delta` equals `reproduced_levels`, and the
artifact records the exact reproduction command used for the live-discovered
trajectory.

### SCENARIO-ARC-FCP-5533: Strategy-Routing Precheck Selects A Non-Stale Live Target

Given Exp5521 selected `sb26:L3` but banked no level and recorded repeated
coordinate collapse
And the registry records multiple adjacent unreproduced frontier targets
When experiment 5533 runs the strategy-routing precheck
Then it rejects the stale Exp5521 target, selects one non-duplicate adjacent
frontier target, records a bounded strategy portfolio with at least three
strategies, proves the strategy router is reachable through the live candidate
router hook, and writes
`results/experiment_5533_arc_strategy_routing_precheck.json` without a solve
claim.

Given repeated candidate coordinates would otherwise be selected by multiple
strategies
When experiment 5533 applies repeated-coordinate suppression
Then the selected action sequence changes before diversity metrics are
computed, `repeated_coordinate_suppression_enabled=true`,
`repeated_coordinate_rate_precheck` is lower than the unsuppressed route, and
`arc_sge_candidate_ready=true` only when target, routing, suppression, and
metric gates pass.

### SCENARIO-ARC-FCP-5534: Strategy-Routed Live Attempt Is Reproduction-Gated

Given Exp5533 selected `g50t:L3` and `arc_sge_candidate_ready=true`
And the registry still records only `g50t` depth 2
When experiment 5534 runs the bounded live self-discovery attempt
Then the attempt installs the Exp5533 strategy portfolio on the live candidate
router, enables repeated-coordinate suppression, records strategy choices,
verifier routes, suppression events, and level counter changes, and writes
`results/experiment_5534_arc_strategy_routed_levelup.json` with
`solve_provenance=live_agent_self_discovery` and
`inference_substrate=arc_live_agent_self_discovery`.

Given the registry already reproduces the Exp5533 target level
When experiment 5534 starts
Then it writes a `blocked_duplicate_target` artifact, keeps
`registry_delta=0`, chooses no replacement target, and does not run the live
agent.

Given the live attempt does not reproduce a new level through the standard
offline replay gate
When experiment 5534 validates the artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`registry_delta=0`, and `honest_verdict` starts with `honest_null:` while
preserving the trajectory evidence.

Given the live attempt reaches a new level and the replay gate reproduces it
When experiment 5534 banks the result
Then `offline_reproduced=true`, `reproduced_levels>=1`, `registry_delta`
equals `reproduced_levels`, and the registry update records the Exp5534
live-path reproduction evidence.

### REQ-ARC-FCP-5591: Translation-Invariant Object Identity + Containment/Adjacency Topology

`ops/known-issues.md`'s 2026-07-11 task 10 entry (folded into task 2, not a
separate experiment) identified two sub-components a real top-3 ARC-AGI-3
competitor's open-sourced classical connected-component segmentation adds
beyond `ColorBlobSaliencePrior`'s existing size/color salience tiers: a
translation-invariant object-identity signature (so the same shape+color
object hashes identically across frames regardless of position -- attacking
the GAP-4891 / `project_arc_live_agent_learning_gaps` binding constraint that
frame-only, position-only features sit at LOO=chance), and a containment tree
+ adjacency graph over the blob list (which objects sit inside which, and
which touch).

`python/carnot/agentic/arc_color_blob_salience.py` SHALL expose
`object_hash(blob: ColorBlob) -> str`, a sha1 signature of the blob's color
plus its cell-shape pattern normalized to its own bounding box's top-left
corner as origin, and `blob_topology(frame) -> dict`, computing the FULL
(unfiltered, `min_pixels=1, max_component_fraction=1.0`) connected-component
partition of a frame via the existing `connected_color_blobs` (unmodified)
and returning `blobs` (list, index is blob id), `object_hashes` (id ->
`object_hash` value), `children` (id -> sorted list of directly-enclosed
blob ids, computed via complement flood-fill from the frame border per blob),
and `adjacency_list` (sorted `[i, j]` id pairs for 4-connected-edge-sharing
blobs). Both additions SHALL be pure functions over existing primitives with
no change to `ColorBlob`'s fields or `connected_color_blobs`'/
`ColorBlobSaliencePrior`'s existing signatures or outputs -- this is an
additive perception-data extension, not a scoring or ranking change.

Required field principles:

- `object_hash`: principle "two blobs with the same shape and color hash identically regardless of position, giving the agent a position-invariant object-identity feature the existing bbox/centroid/tier fields do not provide."
- `blob_topology.children`: principle "innermost-encloser parent assignment yields a clean nesting tree (an object's parent is not every ancestor, just the tightest enclosing blob)."
- `blob_topology.adjacency_list`: principle "any two blobs sharing a 4-connected edge, including parent/child pairs since they physically touch -- lets a consumer reason about spatial relationships beyond raw bbox overlap."

#### SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY

Given a frame containing two same-color, same-shape blobs at different
positions (e.g. two 3x3 squares of the same color, one elsewhere on the
grid)
When `object_hash` is computed for each blob
Then both blobs receive the identical hash value

Given a frame containing two blobs of the same color but a DIFFERENT shape,
or the same shape but a DIFFERENT color
When `object_hash` is computed for each
Then the two blobs receive different hash values

#### SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY

Given a frame with a ring-shaped blob of one color fully enclosing a smaller
solid blob of a different color
When `blob_topology` runs
Then the inner blob's id appears in `children[ring_id]`, and the pair
appears in `adjacency_list` (the ring and its contents share a 4-connected
edge)

Given a frame with two same-shape, same-color blobs where one is enclosed by
a third blob and the other sits freely on the background
When `blob_topology` runs
Then only the enclosed blob's id appears under its true parent's `children`
entry, the freely-sitting blob has no parent, and `object_hashes` reports
the SAME hash for both despite their different topological position

### REQ-ARC-FCP-5591-2: Object-Hash Change-History Bonus (Task 10's Deferred Live-Wiring Step)

REQ-ARC-FCP-5591's DONE note deferred its own suggested live-consuming
mechanism -- "preferring an object whose hash was seen to change in a prior
frame" -- as "a distinct, separately-scoped design + empirical-validation
step per the Phase Prototype + Empirical Validation discipline." This
requirement closes that gap. `ObjectHistorySaliencePrior`
(`python/carnot/agentic/arc_object_history_salience.py`) SHALL wrap
`ColorBlobSaliencePrior` (mirroring the existing
`arc_geometric_salience.GeometricSaliencePrior` precedent for composing a
mutable history-tracking layer on top of the frozen base prior) with a
per-`object_hash` `(obs, changed)` tally, updated via `observe_transition`
from real observed click transitions and consumed via `score` as an additive
bonus proportional to the observed change rate, gated behind a
`min_observations` evidence floor (mirroring `InertClickSigPruner`'s
trust+specificity discipline, inverted polarity: boost instead of prune,
identity-hash-keyed instead of structural-signature-keyed). An
under-observed hash SHALL receive zero bonus, never a penalty.

`coerce_object_history_salience_prior` SHALL provide the standard
`None`/`False` -> unchanged base, instance -> passthrough, `True` -> wrap
coercion contract, threaded through `E3AgentPolicy.__init__` as
`object_history_salience`, gated OFF by default
(`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG`). Because `action_prior` is already a generic,
externally-composable slot that `_ingest`'s existing `hasattr`-gated
`observe_transition`/`reset` hooks and `_candidates`' existing
`action_prior.score` consumption dispatch to generically, wiring this in
requires NO new hook call sites in `arc_competition_agent.py` -- unlike
REQ-ARC-FCP-5595's `InertClickSigPruner`, which needed a brand-new
`rank_candidates` call site.

### SCENARIO-ARC-FCP-5591-2-CHANGE-RATE-BONUS: Reliable Change History Outscores Reliably-Inert History

Given the SAME object (identical `base_prior` score), observed `>=
min_observations` times either always changing the frame or never changing
it when clicked
When `score` is called for a click candidate on that object
Then the reliably-changing history's score is strictly higher than the
reliably-inert history's score, and the reliably-inert history's score
equals the unmodified base score (zero bonus, not a penalty)

### SCENARIO-ARC-FCP-5591-2-EVIDENCE-FLOOR: Under-Observed Hashes Get Zero Bonus

Given an object observed fewer than `min_observations` times, regardless of
how many of those observations changed the frame
When `score` is called for a click candidate on that object
Then the score equals the unmodified base score -- no premature boost from
sparse evidence

### SCENARIO-ARC-FCP-5591-2-NOT-DEGENERATE: Adversarial Check Against Base-Tier Redundancy

Given two click candidates whose `base_prior` scores are IDENTICAL (same
color, size, and shape, hence the same tier and button-likelihood features)
before any observed history
When one of the shared-hash objects accumulates a reliable change-history
tally and `score` is called for both candidates
Then both candidates' final scores are boosted identically above the shared
base score (the mechanism is hash-identity-based, not position-based) --
confirming the bonus is genuinely new information, not a re-derivation of
the base tier features under a different name

### SCENARIO-ARC-FCP-5591-2-DEFAULT-OFF-PARITY: Unwrapped By Default

Given the SUBMITTED default configuration
When a `StepwiseExplorer` or `E3AgentPolicy` is constructed with no explicit
`object_history_salience` argument
Then `action_prior` is a plain, unwrapped `ColorBlobSaliencePrior`
(`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG["object_history_salience_enabled"]`) -- byte-identical
to the agent's behavior before this requirement, until a matched-budget
offline A/B validates flipping it on, per the `solve_rate_dropped` guardrail

### SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL: Real-Game Empirical Validation

Given real click transitions collected from a real `E3AgentPolicy`
exploration run on a click-heavy game (`m0r0`, per REQ-ARC-FCP-5595's own
confirmed roster)
When those transitions are fed through `ObjectHistorySaliencePrior.observe_
transition`
Then the artifact reports, honestly, whether any `object_hash` cleared both
`min_observations` and shows a nonzero change rate (a genuine, non-degenerate
signal for the mechanism to act on) -- zero is an honest, valid outcome when
evidence is sparse within the measured budget, not an error; and an
adversarial degeneracy check reports whether any two real click candidates
sharing an identical `base_prior` score were differentiated by history alone

### REQ-ARC-FCP-5591-3: Matched-Budget A/B -- The Flip-Decision Measurement

The `solve_rate_dropped` guardrail names a matched-budget A/B as the
precondition for ever flipping `SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED` to
`True`. Unlike REQ-ARC-FCP-5595-2's `InertClickSigPruner` A/B,
`ObjectHistorySaliencePrior` cannot be measured through `OfflineSolver`
(`action_prior` is not a `move_pruner`; `OfflineSolver` has no `action_prior=`
concept), and no `states_expanded`-equivalent counter exists on the live
`StepwiseExplorer`/`E3AgentPolicy` path. This requirement therefore measures
TRAJECTORY DIVERGENCE -- whether enabling the mechanism actually changes
which candidates a real `E3AgentPolicy` exploration run selects, compared
against an identical-budget baseline with the mechanism off -- as the
honestly-available substitute, since the tested game/policy does not reach a
level-up within the tested budget (no `actions_to_first_levelup` available).
The experiment SHALL run three arms on the SAME game/budget: baseline
(disabled), the real default `change_bonus_weight`, and a diagnostic arm with
`change_bonus_weight` rescaled to match `ColorBlobSaliencePrior`'s own
tier-score magnitude -- isolating whether the mechanism CAN influence
behavior at an appropriate scale, separate from whether the current default
is well-tuned. Any hypothesis formed from an informal check without a real
baseline comparison SHALL be re-verified against the formal three-arm result
before being reported, per this project's own "cross-check surprising
results" discipline.

### SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP: Identical Trajectories At Default Weight Is An Honest Null

Given the baseline (`object_history_salience=False`) and default-weight
treatment (`object_history_salience=True`, `change_bonus_weight=10.0`) arms
run on the same game and matched budget
When their action sequences are compared
Then an identical trajectory is reported as an honest, valid null (the
mechanism tracks real evidence but the bonus magnitude never changes
candidate ranking at this weight), not a failure requiring escalation

### SCENARIO-ARC-FCP-5591-3-RESCALED-WEIGHT-STILL-NO-OP: A Rescaled Bonus Weight Isolates Scale From Structure

Given a diagnostic arm with `change_bonus_weight` rescaled to match
`ColorBlobSaliencePrior`'s real tier-score magnitude, run on the same game
When compared against the baseline trajectory
Then the artifact reports honestly whether the rescaled weight changes
behavior (confirming the mechanism CAN matter at an appropriate scale) or
remains identical to baseline (an open question for follow-up, not assumed
to be an over-exploitation risk without a real baseline-comparison check --
a repeated-coordinate pattern that appears IDENTICALLY in the baseline
disproves that it is bonus-induced)

### REQ-ARC-FCP-5590: Dict-Candidate CNN Fix Matched-Budget A/B

`docs/research-notes/arc-perception-grounding-audit-2026-07-13.md` found that
`FrameChangeScorer.candidate_score`'s `getattr(candidate, "action_id")`
raised `AttributeError` on dict-shaped candidates, silently zeroing the CNN
term of the live default frame-change scorer on every
`ActionEffectExpansionPrior.frontier_priority` call (the always-on default
frontier-priority computation,
`SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED = True`). The fix
(`arc_frame_change_predictor._as_action_like`, normalizing a dict candidate
to an attribute-bearing shim before the CNN call) is low-risk by
construction (the pre-fix behavior was "silently contributes zero," so a
correct fix can only add signal, never regress the unaffected
`PersistentAEM` memory term), but per CLAUDE.md's Phase Prototype +
Empirical Validation + Adversarial Check Discipline a fix that stops a
silent bug is not automatically a capability win and must be measured before
being trusted as one.

`python/carnot/experiment_5590_frame_change_cnn_dict_candidate_fix_ab.py`
SHALL run a matched-budget A/B across the full `CLAIMED` 11-game roster,
CONTROL reproducing the PRE-FIX behavior via a scoped monkeypatch of
`arc_frame_change_predictor._as_action_like` to the identity function (the
same shipped `E3AgentPolicy` construction otherwise, restored after each
game so concurrent TREATMENT runs are unaffected) and TREATMENT using the
real, unmodified fixed default. Tier-3 LLM induction SHALL be disabled
(`CARNOT_ARC_DISABLE_INDUCTION=1`) so the measurement isolates the
search/frontier-priority effect from induction noise and wall-clock.

The artifact SHALL report `levels_gained_control_total`,
`levels_gained_treatment_total`, `per_game_levels_delta`,
`states_expanded_control_total`, `states_expanded_treatment_total` (each
arm's total distinct explored-state count, `len(policy.explorer.graph)`, a
search-behavior proxy independent of whether any level was actually
reached), and `levels_gained_headroom_present` (CLAUDE.md
FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at
least one arm reached a nonzero level somewhere on the roster).

**RESOLUTION (2026-07-13).** Ran cleanly on the full 11-game roster,
budget=200/game. `levels_gained_headroom_present: true` (1 level reached in
both arms, `lp85`) -- the null is interpretable, not a degenerate
zero-headroom test. `per_game_levels_delta`: zero on every single game.
`states_expanded_control_total` and `states_expanded_treatment_total` were
IDENTICAL, and per-game `states_expanded` matched EXACTLY (not just in
aggregate) for all 11 games -- a much stronger and more informative null
than "levels didn't change": the fix produced literally zero measurable
difference in which states the explorer visited, in either direction.
Consistent with the audit's own honest bound (the CNN term's blend weight is
already small, `cnn_weight=0.05` vs `memory_weight=1.0`) and with the
project's prior LOO=chance finding on frame-only features
(`project_arc_live_agent_learning_gaps`): the underlying signal the bug was
blocking was itself not load-bearing at this weight and on this roster, not
that the bug was somehow inert. The fix remains correct code hygiene (a
scorer that silently discards part of its own blended signal on a
type-shape mismatch is a latent hazard regardless of today's measured
weight), but does not currently move live-agent capability.

Required field principles:

- `levels_gained_headroom_present`: principle "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster, else the null may just mean neither arm had headroom."
- `states_expanded_control_total` / `states_expanded_treatment_total`: principle "a search-behavior proxy independent of whether any level was reached -- a per-game exact match is a stronger null signal than an aggregate levels-only comparison."
- `control_results`: principle "PRE-FIX behavior reproduced via a scoped monkeypatch of _as_action_like to identity -- the SAME shipped construction with one function swapped for the duration of the run, not a different code path."

#### SCENARIO-ARC-FCP-5590-MATCHED-BUDGET-DELTA

Given the real `E3AgentPolicy` cascade run on the same game under CONTROL
(pre-fix `_as_action_like` identity monkeypatch) and TREATMENT (the real
fixed default) with the same budget and induction disabled
When both arms complete
Then `per_game_levels_delta` and any `states_expanded` difference for that
game reflect ONLY the dict-candidate CNN-term fix, with no other
construction difference between the two arms

### REQ-ARC-FCP-5592: Candidate-Scoring Stack Vs Bare-Control Matched-Budget A/B

`docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-2026-07-11.md`
(O2) found the 3rd-place ARC-AGI-3 team ("forge") had an architecturally
equivalent candidate-selection "arbiter" slot to our own
`candidate_router` + DAgger value head + goal-energy candidate guidance +
navigation-cost-tiebreak stack, but disabled their arbiter (a second LLM
call) in their winning config for cost, keeping only a hand-tuned static
fallback. `SUBMITTED_AGENT_CONFIG["bare_control_config"]` already documents
the exact on/off toggle matching forge's own ablation, but no experiment had
ever run it against the real full-stack default and reported the delta --
this is the missing measurement, not a new scorer build.

`python/carnot/experiment_5592_candidate_scoring_stack_bare_control_ab.py`
SHALL run a matched-budget A/B across the full `CLAIMED` 11-game roster,
FULL STACK being the real, unmodified live `E3AgentPolicy` default, and
BARE CONTROL being `E3AgentPolicy` constructed with
`SUBMITTED_AGENT_CONFIG["bare_control_config"]`'s knobs mapped to their real
constructor kwargs (`target_levels=1`, `value_weight=0.0`,
`candidate_router=None`, `navigation_cost_tiebreak=False`,
`action_effect_expansion_prior=False`, `goal_bias=None`,
`goal_candidate_guidance=False`). Tier-3 LLM induction SHALL be disabled
(`CARNOT_ARC_DISABLE_INDUCTION=1`) so the measurement isolates the
candidate-SELECTION axis forge's ablation targeted.

The artifact SHALL report `levels_gained_full_stack_total`,
`levels_gained_bare_control_total`, `per_game_levels_delta`,
`efficiency_full_stack_total`/`efficiency_bare_control_total` (summed
per-game `arc_agi.scorecard.EnvironmentScoreCalculator` efficiency, the
action-efficiency half of forge's own reported metric), and
`levels_gained_headroom_present` (CLAUDE.md FALSE_NEGATIVE_RISK discipline).

**RESOLUTION (2026-07-13).** Ran cleanly on the full 11-game roster,
budget=200/game. `levels_gained_headroom_present: true` (`lp85` reached L1
in both arms) -- interpretable, not a degenerate zero-headroom test.
`per_game_levels_delta`: zero on every single game;
`efficiency_full_stack_total == efficiency_bare_control_total` (2.7778,
matching to 4 decimal places). Verified this is a real result, not a
construction bug: the two arms' `lp85` rows show genuinely DIFFERENT search
behavior (`actions_to_first_levelup`: 7 for full stack vs 5 for bare
control; `actions` total 198 vs 5, since `bare_control_config`'s
`target_levels=1` correctly stops bare control immediately after L1 while
the full stack continues toward its default target of 3) -- the ablation
demonstrably took effect. The efficiency SCORE nonetheless matched exactly
because `arc_agi.scorecard`'s per-level score is `min((human/agent)^2*100,
115)`: both 5 and 7 actions are already well under `lp85` L1's human
baseline, so both arms saturate at the same capped per-level score
regardless of the 2-action difference between them -- a real property of
the (capped) metric, not evidence the ablation was a no-op.

**Honest conclusion: on this roster/budget, the richer candidate-scoring
stack (candidate_router + DAgger value head + goal-energy candidate
guidance + navigation-cost tiebreak) produces NO measured level-up or
action-efficiency advantage over the bare control forge's own ablation
methodology would compare it against.** This does not establish the stack
is worthless (it may earn its keep at a different budget, on a different
roster, or via a mechanism `EnvironmentScoreCalculator`'s capped-per-level
formula does not reward, e.g. reducing variance or avoiding worse-case
failures elsewhere), but it means the claim "our scoring stack is the
arbiter forge wanted but couldn't afford" is NOT YET empirically supported
on this measurement and should not be cited as a moat without a
follow-up at a different budget/roster or metric that is genuinely
sensitive to the ablation.

Required field principles:

- `levels_gained_headroom_present`: principle "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster."
- `bare_control_kwargs`: principle "the real E3AgentPolicy constructor kwargs mapped from SUBMITTED_AGENT_CONFIG['bare_control_config'] -- documents exactly what was ablated, matching forge's own on/off toggle."
- `efficiency_full_stack_total` / `efficiency_bare_control_total`: principle "sum of the leaderboard harness's own per-game efficiency score -- the action-efficiency half of forge's own reported metric, not just level count; a capped-per-level formula can saturate identically for two genuinely different search behaviors on a shallow level, which is a real property of the metric, not evidence of a construction bug."

#### SCENARIO-ARC-FCP-5592-STACK-VS-BARE-DELTA

Given the real `E3AgentPolicy` cascade run on the same game under FULL
STACK (the real unmodified default) and BARE CONTROL
(`SUBMITTED_AGENT_CONFIG["bare_control_config"]`'s knobs applied) with the
same budget and induction disabled
When both arms complete
Then `per_game_levels_delta` and any `efficiency` difference for that game
reflect ONLY the candidate-selection-stack ablation, with no other
construction difference between the two arms, and a per-game row inspection
(actions taken, actions to first level-up) can distinguish a genuine null
from a construction bug that silently made both arms identical

### REQ-ARC-WMTE-5593: Goal-Predicate Consistency Against Real Observed Level-Progress

`ops/known-issues.md`'s 2026-07-11 task 11 entry (hallucination-consistency
checks) found two independent top-3 ARC-AGI-3 teams each carry an
unexploited self-report-vs-ground-truth gap: Reki's `board_change_assessment`
(what it thinks changed) is never cross-checked against `changed_pixels`
(the real pixel diff); Duck's free-text Goal/Action hypothesis is
regenerated each turn but never checked against observed level-up/no-change
transitions. Investigating this project's own architecture found no direct
analog to Reki's exact natural-language self-report, but found the DYNAMICS
half of the same gap-class was already closed
(`WorldModelVerifier.score(engine)` checks the induced `engine()`'s
predicted next-grid against the real observed next-grid) while the GOAL
half was genuinely open: nothing validated `is_level_complete` (the induced
code's formalized goal hypothesis, installed by
`execute_bounded_llm_reinduction` as a search termination condition) against
real observed level-progress ground truth.

`python/carnot/agentic/arc_executable_world_model.py` SHALL expose
`score_goal_predicate_consistency(is_level_complete, transitions) ->
GoalPredicateConsistency`, the goal-hypothesis sibling of
`WorldModelVerifier`/`VerifyResult`: for each transition, the real ground
truth is `level_after > level_before` (a genuine level-up occurred); the
claim under test is `is_level_complete(next_grid)`. Agreement is a cheap,
deterministic sign check -- no second LLM call, matching forge's own
competitive-pressure finding (docs/research-notes/arc-agi3-milestone1-
winners-sota-ingestion-2026-07-11.md, O3) that an expensive LLM judge was
not worth the cost while a deterministic filter was kept. The function
SHALL treat a raising predicate as a `claimed=False` miss rather than
propagating the exception, and SHALL return a well-formed zero result (no
`ZeroDivisionError`) for an empty transition list. The CALLER CONTRACT
(documented in the function's own docstring) is to pass transitions from a
SINGLE level boundary, since `is_level_complete` is a per-boundary predicate
in the real pipeline (freshly re-induced after every level-up).

This is a NEW, purely additive verification primitive -- it does not modify
`WorldModelVerifier`, `VerifyResult`, or any existing induction/planning
behavior, and is NOT yet wired into any live decision (e.g. vetoing a goal
predicate before planning); that is a distinct, separately-scoped design +
empirical-validation step, consistent with how the color-blob salience
topology extension (REQ-ARC-FCP-5591) was left additive-only pending its
own validation.

**RESOLUTION (2026-07-13).** The function itself is validated by 5 direct
unit tests on realistic (non-toy) synthetic data
(`tests/python/test_arc_goal_predicate_consistency.py`): a perfect
predictor scores 1.0; a predictor that never claims a win is CAUGHT missing
a real level-up (not silently trusted); a predictor that claims every state
is a win is CAUGHT false-positiving a no-op (both miscalibration
directions detected, not just one); a raising predicate is handled
gracefully; an empty transition list returns a well-formed zero result.

The offline-sim prototype (`experiment_5593_goal_predicate_consistency_
offline_sim_prototype.py`) attempted a REAL end-to-end test against `lp85`
(the only game with any measured headroom across the full 11-game roster in
this session's exp5590/exp5592 A/Bs) using the real default Qwen3.5-9B-MTP
proposer, and found a genuine, precisely-diagnosed limitation of the
EXISTING (pre-dating this task) `induce_prompt`/`LocalGGUFProposer`
machinery: `lp85`'s logical grid is 64x64 (much larger than typical), and
`induce_prompt`'s fixed "render the full initial grid" overhead alone
produces a prompt whose token count is already close to the induction
pipeline's available budget (`n_ctx=16384` minus `max_tokens=2560` reserved
for the completion = 13,824 tokens) -- confirmed by direct debugging
(bypassing `LocalGGUFProposer.induce()`'s own truncated error-repr handling
to read the llama-server's actual JSON error body): an 8-transition window
measured 18,355 prompt tokens (`exceed_context_size_error`), and even a
MINIMAL single-transition window (the level-up transition alone) measured
~13,400+ tokens -- already at the edge of the budget regardless of how few
transitions are used. This is a real, useful finding about the shared
induction pipeline's scalability on large-grid games, not a flaw in
`score_goal_predicate_consistency` itself (which never got the chance to
run against a real induced predicate on `lp85`, since induction itself
never produced one). Fixing `induce_prompt`'s large-grid scalability is
out of scope for this task; the checked-in artifact honestly records
`goal_predicate_accuracy: null` and
`induction_failure_detail` (the truncated exception repr
`LocalGGUFProposer.generate()` currently captures -- the fuller JSON error
body is not preserved by the existing error handling, a minor, separately
fixable gap not addressed here).

Required field principles:

- `real_levelup_present_in_sample`: principle "the check is only interpretable if the collected transitions include at least one genuine level-up -- otherwise the accuracy figure reflects only no-op agreement, which any always-False predictor would also score perfectly (CLAUDE.md FALSE_NEGATIVE_RISK discipline, applied to this new consistency check)."
- `goal_predicate_mismatches`: principle "the specific transitions where the induced is_level_complete disagreed with real observed level-progress, for honest post-hoc inspection."

#### SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR

Given a goal predicate whose sign correctly matches every real observed
level-up and no-op transition in a single level boundary's transition
window
When `score_goal_predicate_consistency` scores it
Then `accuracy` is 1.0 and `mismatches` is empty

#### SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT

Given a goal predicate that never claims a win (or, in the opposite
direction, claims every state is a win)
When `score_goal_predicate_consistency` scores it against a transition
window containing at least one real level-up and one real no-op
Then the predicate's disagreement with the real observed transitions is
recorded in `mismatches`, not silently scored as a correct prediction

### REQ-ARC-WMTE-5593-2: `induce_prompt` Large-Grid Scalability Fix -- Real Positive-Control Demo

REQ-ARC-WMTE-5593's own RESOLUTION left the `induce_prompt` large-grid-scalability limitation
explicitly out of scope, and `ops/known-issues.md` task 11's DONE note named it as the natural
prerequisite for a real positive-control demonstration of `score_goal_predicate_consistency` on
`lp85` specifically. This requirement closes that gap.

**Root cause, precisely measured.** `_transitions_block`'s two full-grid renders (the INITIAL
grid and, when a level-up occurred in the window, the WIN STATE grid) used `to_ascii` --
one raw character per cell. On `lp85`'s 64x64 grid this is exactly correct at the character
level (~4,160 chars/grid including newlines) but catastrophically token-inefficient: a
single-transition window (up to 2 full grids) already measured ~13,400+ tokens against the
13,824-token available budget, and an 8-transition window measured 18,355 tokens
(`exceed_context_size_error`, 32.8% over budget).

**Fix, two parts, both real-data-measured (not estimated):**

1. `_rle_grid(g) -> str` -- a NEW function, run-length-encodes a FULL grid one line per row,
   `r<row>:<v0>x<n0>,<v1>x<n1>,...`, with the starting column of each run left IMPLICIT (the
   running sum of prior counts in that row) since every cell in a row is covered with no gaps.
   An earlier attempt that spelled out an explicit column per run (matching `_rle_delta`'s own
   style) measured WORSE than `to_ascii` on `lp85`'s real grid for its WIN-state render (5,164
   vs 4,159 chars) -- the per-run column overhead dominated for `lp85`'s actual run-length
   distribution (avg run length ~11.6 cells, 352 runs on the measured INITIAL grid). Dropping the
   column entirely removed that overhead: measured on the SAME real `lp85` grid, INITIAL
   4,159->1,857 chars (2.24x), WIN 4,159->2,368 chars (1.76x), both verified lossless
   round-trip.
2. `_rle_delta_compact(g0, g1) -> str` -- a NEW function (kept separate from the existing
   `_rle_delta`, which has its own round-trip tests and another caller and is NOT modified here).
   After fix (1) above, the per-transition DELTAS became the new dominant cost on `lp85`'s real
   transitions (measured: 8 deltas via `_rle_delta` = 9,308 tokens, still over budget even after
   the full-grid fix). `_rle_delta_compact` sub-compresses each changed run's NEW values as
   `<value>x<count>` pairs instead of listing one value per changed cell -- large changes are
   often a single-color object moving or appearing, which the raw comma-per-cell format cannot
   exploit. Measured on the SAME 8 real deltas: 9,308 -> 5,992 tokens (a 3,316-token additional
   saving), verified lossless round-trip against 200 random synthetic diffs.

`induce_prompt`'s own explanatory text (the part of the prompt teaching the model how to decode
the compact forms) SHALL be updated to describe both new formats precisely enough for the model
to reconstruct a grid/delta from them -- this is a prompt-CONTRACT change, not just an internal
encoding change, since the model must correctly parse the new notation to reason about the game.

**RESOLUTION (2026-07-14).** Real, tokenizer-measured result on `lp85`'s actual 8-transition
window (via `llama_cpp.Llama(vocab_only=True)` against the real `Qwen3.5-9B-MTP` GGUF, the
exact model `induce_prompt` targets): the SAME window that measured 18,355 tokens before this fix
now measures 11,167 tokens against the 13,824-token budget -- comfortably under budget with
~2,657 tokens of real headroom (a ~39% total token reduction). Re-running
`experiment_5593_goal_predicate_consistency_offline_sim_prototype.py` end-to-end (real GPU
inference against the real port-8920 `Qwen3.5-9B-MTP` server, `duration_s=33.452`,
`inference_substrate=live_llm_inference`) now produces the real positive-control demo REQ-ARC-
WMTE-5593 could not: `induction_ok=true` (no context overflow), `induce_transition_count=8` (the
exact target window size), `real_levelup_present_in_sample=true` (interpretable per
FALSE_NEGATIVE_RISK), and `score_goal_predicate_consistency` scores the REAL induced
`is_level_complete` against the 8 real transitions: `goal_predicate_accuracy=0.75` (6/8 correct,
2 false-negative mismatches at transition indices 6 and 7 -- both real level-ups the induced
predicate missed). `honest_verdict:
"complete: goal_predicate_consistency_prototype_induced_predicate_miscalibrated"` -- the
CHECK works correctly on real data (that is this requirement's job); the induced predicate
itself being imperfect is a separate, honest finding about induction QUALITY on `lp85`
specifically, not a defect in the check or the scalability fix.

Existing tests unaffected (`test_rle_delta_lossless.py`'s 3 tests, `test_arc_goal_predicate_
consistency.py`'s 5 tests, `test_experiment_5593_...py`'s 7 tests all still pass unmodified --
`_rle_delta` itself was never touched). New tests:
`tests/python/test_arc_induce_prompt_large_grid_scalability.py`.

Required field principles: none new (this requirement modifies internal encoding functions and
`induce_prompt`'s prompt text; no new artifact schema fields).

#### SCENARIO-ARC-WMTE-5593-2-LOSSLESS-ROUND-TRIP

Given a full grid or a changed-cell delta encoded via `_rle_grid` / `_rle_delta_compact`
When the encoding is decoded back into a grid using the exact reconstruction rule stated in
`induce_prompt`'s own explanatory text (implicit column for full grids; explicit run-start
column + implicit sub-run column for deltas)
Then the reconstructed grid is byte-identical to the original for both real `lp85` grids and
randomized synthetic grids/diffs across multiple colors (0-15) and shapes

#### SCENARIO-ARC-WMTE-5593-2-REAL-BUDGET-FIT

Given `lp85`'s real 64x64 grid and a real 8-transition induction window selected the same way
`experiment_5593_...py` selects it (through the first real level-up plus one)
When `induce_prompt` renders the window and the result is tokenized with the real
`Qwen3.5-9B-MTP` tokenizer
Then the token count is below the 13,824-token available budget (`n_ctx=16384` minus
`max_tokens=2560`), where it previously measured 18,355 tokens and overflowed with
`exceed_context_size_error`

### REQ-ARC-WMTE-5594: `/think` vs `/no_think` Induction Quality A/B on the Frozen Live Generator

`ops/known-issues.md`'s 2026-07-11 task 7 entry (cheap, dev-side-only) was
filed after ARC Prize's public GPT-5.6 results showed reasoning-effort
scaling ARC-AGI-3 solve rate ~26x (Low->Max) versus only ~1.3x on
ARC-AGI-1 for the same model -- a domain-specific signal that induction
tasks (the thing GPT-5.6 gains the most from extra reasoning on) benefit
disproportionately more from think-mode than the field's typical single-
grid puzzle tasks do. The frozen live-submission generator
(`project_arc_live_generator` memory: Qwen3.5-9B-MTP, MTP + q8 KV +
`n_predict>=2048` + `/no_think`) was decided under June sprint time
pressure and never re-measured against this specific finding.

**Precondition (a) mechanism finding.** Before any A/B could run, this
task discovered that `LocalGGUFProposer`'s `no_think_prefix` instance
attribute (set to `"/no_think\n"` in the live default config) has **no
effect on real induction calls today**: `CARNOT_ARC_CODEONLY_INDUCE`
defaults ON (2026-06-25 operator directive), and codeonly mode's own
`_L2_CODEONLY_DIRECTIVE` module-level constant in
`arc_executable_world_model.py` hardcodes a literal `"/no_think\n"` as its
own first line, which wins in `generate()`'s
`if _codeonly: ... elif self.no_think_prefix: ...` branching (codeonly
always applies when `codeonly_eligible=True`, which every `induce()` call
passes). Testing `/think` fairly therefore requires a scoped module-level
monkeypatch of `_L2_CODEONLY_DIRECTIVE` (swap the leading `/no_think\n`
for `/think\n`), restored in a `finally` block after each induction call
-- not the dead `no_think_prefix` attribute. This is a real, previously
undocumented fact about the induction pipeline, independent of this task's
headline A/B result.

**Precondition (a) compatibility-check bug, found and fixed in-session.**
The first automated compatibility probe (`check_think_mode_compatibility`)
falsely reported `think_mode_compatible_with_mtp: False`, directly
contradicting two separate manual verifications (direct `/completion`
calls against the live port-8920 server) that `/think` mode produces
genuinely different, clearly-reasoning-shaped output. Root cause: the
check's tag test was a bare `"<think>" in think_content` substring check,
but the model's actual opening tag varies across calls -- one probe
emitted `<thinking>` (observed verbatim:
`'<thinking>\nThe user wants me to write a Python world-model
engine...'`), and the literal characters `<think>` are not a substring of
`<thinking>` (the closing `>` does not align). The check's length-ratio
fallback (`len(think) > 1.5 * len(no_think)`) was also too strict for a
short `n_predict=120` probe budget: one real probe measured 549 vs 403
chars (a genuine 36% delta), below the 1.5x (604.5-char) threshold. Fixed
by checking a tuple of known reasoning-tag prefixes
(`("<think>", "<thinking>", "<reasoning>")`) via `str.startswith` and
lowering the length-ratio fallback to 1.15x. Re-verified standalone after
the fix: `compatible=True`, `"think content starts with a reasoning tag
(527 vs 460 chars) -- compatible"` -- consistent with the manual
verification. This is a lesson for any future probe of LLM-emitted
reasoning-tag content: match a set of known tag spellings, not one
literal string, and do not rely solely on a length-ratio heuristic at a
short completion budget.

**RESOLUTION (2026-07-13).** With the compatibility check fixed, the real
4-attempt measurement (2 roster games -- `m0r0`, `sk48` -- x 2 arms) ran
against the live default Qwen3.5-9B-MTP proposer (161.6s total,
`inference_substrate: live_llm_inference`). Both arms induced successfully
on both games (4/4 induction_ok). Per-game `heldout_accuracy` (via the
existing `WorldModelVerifier`, reused unmodified from REQ-ARC-WMTE-4494):
`m0r0` no_think=0.5 vs think=0.0 (no_think better); `sk48` no_think=1.0 vs
think=1.0 (tie). Neither roster game's 10-transition window contained a
real level-up (`real_levelup_present_in_sample: false` on all four
attempts), so `score_goal_predicate_consistency` (REQ-ARC-WMTE-5593) was
never triggered and goal-predicate accuracy is not part of this result --
an honest scope limitation, not a gap papered over. `honest_verdict:
"complete: think_mode_ab_equal_success_no_think_higher_accuracy"`: equal
induction-success count (2 vs 2), but no_think's mean heldout accuracy
(0.75) exceeds think's (0.5) on this 2-game roster.

**What this does and does not show.** This is a real, informative,
NEGATIVE result for switching the frozen live stack to `/think` on
induction quality specifically -- on this small roster, `/think` mode
never wins outright and loses once. It does NOT settle the broader
GPT-5.6-style reasoning-effort-scaling question for the ARC-AGI-3 domain
in general: (1) the roster is 2 games, well below the CLAUDE.md sample-
size floor for any percentage-point claim; (2) `heldout_accuracy` scores
world-model INDUCTION quality only, not the fuller "actions-to-first-win
across a real solve" metric GPT-5.6's own comparison used, which this
task's scope explicitly excluded (reusing existing verifiers, not
extending `lb.run_game` into a full solve loop); (3) neither game's
sample included a real level-up, so the goal-predicate half of induction
quality is entirely unmeasured here. Per the task's own explicit
instruction and the ARC-AGI-3 Submission Sprint / November-Floor
disciplines' "the live stack is frozen" rule, this result does NOT change
the frozen live stack's `/no_think` setting -- it is reported as an
offline dev measurement requiring an explicit operator decision, and on
these numbers the honest recommendation is "no evidence yet to justify
unfreezing the stack for `/think`."

Required field principles:

- `think_mode_compatible_with_mtp`: principle "task 7 precondition (a),
  checked here rather than assumed -- if False, this experiment stops per
  the task's `blocked_think_mode_incompatible_with_mtp` instruction rather
  than proceeding on an untested assumption."
- `think_max_tokens`: principle "materially larger than
  `no_think_max_tokens` by design -- think mode needs completion budget
  for reasoning tokens before code; comparing truncated-mid-thought
  output to quick code would not be a fair test, and this asymmetry is
  disclosed, not hidden."

#### SCENARIO-ARC-WMTE-5594-INCOMPATIBLE-BLOCKS-CLEANLY

Given the live-server compatibility probe finds no reasoning-tag prefix
and no material length delta between `/think` and `/no_think` output
When `build_artifact` runs
Then `honest_verdict` is
`"complete: blocked_think_mode_incompatible_with_mtp"` and no induction
attempt is made on any roster game

#### SCENARIO-ARC-WMTE-5594-TAG-VARIANT-RECOGNIZED

Given the live-server compatibility probe's `/think`-arm response begins
with `<thinking>` rather than the literal `<think>`
When `check_think_mode_compatibility` evaluates the probe responses
Then it recognizes the `<thinking>` prefix as a reasoning-tag match and
reports `compatible=True`, not a false `no observable difference`
negative

### REQ-ARC-FCP-5595: InertClickSigPruner -- Dead-Signature Click Pruning

`ops/known-issues.md`'s 2026-07-11 task 9 entry ("New 2026-07-11, cheap,
reuses an existing code shape") asked for Reki's dead-signature click-pruning
mechanism -- track a clicked component's structural signature `(color, size,
is_rect, twin_count)`; if a click on that signature never changes the frame,
suppress future clicks on components sharing it -- built by extending the
`arc_hazard_pruner.HazardMovePruner` trust+specificity gating pattern rather
than Reki's own greedy K=2 threshold, which the audit that surfaced this
(`docs/research-notes/arc-perception-grounding-audit-2026-07-13.md`) flagged
as over-aggressive: two observations is a thin evidence floor, and a strict
"first effective observation ever = permanently sacred, otherwise K=2 kills
it" rule has no tolerance for a signature that is mostly inert but
occasionally does something.

**The implementation (`python/carnot/agentic/arc_inert_click_pruner.py`,
`InertClickSigPruner`).** Per structural signature, the pruner accumulates
observed `(obs, inert, leveled)` counts from the search's own clicks -- no
offline ground truth, transferring to any game the same way both sibling
pruners (`HazardMovePruner`, `RelationalMaskMovePruner`) do. It differs from
Reki's rule in two ways: the evidence floor is raised from 2 to
`min_observations` (default 4, matching `RelationalMaskMovePruner`'s own
explicitly-not-K=2 default), and a `min_specificity` threshold (default 0.9)
replaces literal-zero-tolerance -- a signature is pruned once its OBSERVED
inert fraction clears the bar, not the instant a single effective click is
seen. A signature that has ever produced a real level-up is permanently
sacred (never pruned, regardless of specificity), mirroring both sibling
pruners' hard binary veto for level-ups specifically.

The clicked component's signature is computed via `click_signature(blob,
blobs)`, a new free function built on `connected_color_blobs`/`blob_at_click`
(`arc_color_blob_salience`, REQ-ARC-FCP-5591). `blob_at_click` (REQ-ARC-FCP-
5595) is the free-function promotion of `ColorBlobSaliencePrior`'s existing
private `_blob_for_click` lookup, added purely additively so callers outside
`ColorBlobSaliencePrior` can reuse the exact same click-to-blob resolution.
`twin_count` is the number of OTHER blobs in the same frame sharing a blob's
`(color, pixel_count, is_rect)` triple, so evidence about one component
transfers to its structural twins even when they sit at different frame
positions -- confirmed directly by `test_click_signature_twin_count_matches_
shared_shape_only` and by the offline-sim prototype's cross-twin evidence
transfer in `test_prunes_signature_confidently_inert_after_evidence_floor`.

The pruner implements the SAME `should_prune(frame, label) -> bool` /
`observe(frame_before, label, frame_after, leveled_up)` protocol as both
sibling pruners, so it composes through the existing `arc_relational_mask_
pruner.CompositeMovePruner` and is consumable by `OfflineSolver` via the
identical `move_pruner=` constructor parameter both siblings already use --
this makes it live-path-reachable per the ARC Live-Path Reachability
Discipline without any new wiring code (`OfflineSolver` is one of the two
recognized live entrypoints). A separate `rank_candidates(frame, rows) ->
rows` method implements the identical gating logic in the filter-protocol
shape `StepwiseExplorer._candidates` (`arc_competition_agent.py`) already
composes with (matching `program_synthesis_filter`/`goal_candidate_
guidance`'s contract) -- at the time this task was originally scoped, tested
and ready but deliberately NOT wired into that live composition chain,
consistent with how the color-blob salience front-end (REQ-ARC-FCP-5591) was
left additive-only pending its own live-wiring decision (`ops/known-issues.md`
task #97, still open).

**WIRING FOLLOW-ON (2026-07-13, same day).** The gap above is closed.
`coerce_inert_click_pruner` (new, `arc_inert_click_pruner.py`, matching
`coerce_program_synthesis_filter`'s `None`/`False` -> no pruner, instance ->
passthrough, `True` -> construct-default shape) plugs an `InertClickSigPruner`
into both `StepwiseExplorer` (a new `inert_click_pruner` constructor param,
threaded through a `rank_candidates` call inside `_candidates` alongside
`program_synthesis_filter`/`goal_candidate_guidance`) and `E3AgentPolicy`
(same param name, passed through). The pruner also gets a real `observe()`
call from `_ingest`'s existing per-transition OBSERVE hook -- the same site
that already feeds `dense_curiosity`/`controllable_novelty_policy`/
`object_centric_proposal_policy`/`action_prior` -- so it accumulates evidence
from the search's OWN live clicks, matching `HazardMovePruner`'s own online
discipline; without this half, `rank_candidates` would be wired but
permanently a no-op (every signature stays "unproven" forever). Gated OFF by
default (`SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG["inert_click_pruner_enabled"]`), matching every other
freshly-wired-but-unvalidated component in that file -- per the
`solve_rate_dropped` guardrail, flipping the default for the SCORED agent
needs its own matched-budget offline A/B (states/actions-expanded reduction,
zero regression in reproduced levels) first, not assumed safe just because
it is reachable. Verified with 8 new focused tests
(`tests/python/test_arc_inert_click_pruner_live_wiring.py`): the coercion
function's full branch set, `_candidates` calling and propagating
`rank_candidates`'s filtered result, a no-pruner no-op, a raising-pruner
non-fatal fallback (matching every sibling hook's try/except discipline),
`_ingest` feeding `observe()` the realized transition, a no-pruner no-op for
that hook too, default-off parity against `SUBMITTED_AGENT_CONFIG`, and
`E3AgentPolicy` opt-in threading. Ruff and mypy clean; 46 pre-existing
`arc_competition_agent.py`-adjacent tests (submitted-parity, program-synthesis
filter, E3 fidelity/named-tail gates, HUD mask, competition-agent adapter)
still pass unchanged.

**RESOLUTION (2026-07-13).** The pruner itself is validated by 7 direct unit
tests on realistic synthetic grids
(`tests/python/test_arc_inert_click_pruner.py`): confidently-inert signatures
are pruned only after clearing both the evidence floor and specificity bar;
evidence transfers between structural twins; below-floor and below-specificity
cases are conservatively NOT pruned; a signature that ever leveled up is
permanently sacred even after many subsequent inert observations; keyboard
actions and undecodable labels are safely ignored; `rank_candidates` drops
only the confidently-inert click rows from a mixed candidate list.

The offline-sim prototype (`experiment_5595_inert_click_sig_pruner_offline_
sim_prototype.py`) ran the pruner against REAL transitions collected from a
REAL `E3AgentPolicy`/`lb.run_game` exploration of `m0r0` (confirmed
click-heavy by direct probe before selection: 21 of 22 transitions were
action=6). GAME-SELECTION NOTE (found investigating, not assumed): a first
attempt to probe click-action prevalence via a bare `E3AgentPolicy(game,
explore_budget=6)` with no explicit `proposer=` stalled twice with near-zero
CPU growth over many minutes -- some default-constructed component the
exploration loop depends on appears to block. The script therefore always
constructs `E3AgentPolicy` with an explicit `LocalGGUFProposer`, matching
exp5594's proven-reliable pattern.

The real run (37 transitions collected, 32 clicks, 19.3s and 32.8s on two
runs) produced an honest, informative NULL: 12 distinct signatures were
tracked but ZERO cleared BOTH the `min_observations=4` floor and the
`min_specificity=0.9` bar (`honest_verdict:
"complete: inert_click_sig_pruner_prototype_ran_but_no_signature_cleared_
evidence_floor"`). This is not a flaw in the mechanism -- with 32 clicks
spread across 12 distinct signatures (an average of under 3 observations per
signature), most signatures simply did not accumulate enough repeated
evidence within this budget to clear the conservative gate. The gate is
DESIGNED to fail closed under sparse evidence (per the trust+specificity
discipline this task explicitly asked for, replacing Reki's more aggressive
K=2), so an honest null here is the expected behavior at this budget, not a
bug. A larger `total_budget` (more actions per game) or a roster with more
same-signature repeat clicks (e.g. a game with many decorative identical
sprites) is the natural follow-on if a positive pruning demonstration is
wanted; this task's "cheap" framing scoped the prototype to confirming the
mechanism runs correctly end-to-end on real data, which it does.

`inference_substrate` was corrected mid-task from an initial conservative
`live_llm_inference` guess to `offline_arcade_live_agent_runtime_self_
discovery_no_llm`: the real measured duration (19.3s) was far under the 60s
`live_llm_inference` floor, and `adversarial_verify.py` correctly flagged the
mismatch (`DURATION_TOO_SHORT`) before this artifact was accepted. A real
`LocalGGUFProposer` IS constructed and wired into `E3AgentPolicy`, but this
script never calls `induce()`/`generate()` on it, and the measured duration
confirms the exploration loop itself never invokes the LLM either -- the
`model_specs` GGUF entry is vestigial, matching the documented substrate's own
disclosed pattern for vestigial model strings.

Required field principles:

- `total_signatures_confidently_inert`: principle "count of distinct
  signatures that cleared BOTH the evidence floor AND the specificity bar
  with zero level-ups -- the load-bearing claim behind building the pruner
  at all, measured against real click data rather than synthetic grids."
- `inference_substrate`: principle "offline_arcade_live_agent_runtime_
  self_discovery_no_llm -- confirmed empirically (measured duration far
  under the live_llm_inference floor), not assumed."

#### SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA

Given real (frame_before, label, frame_after, leveled_up) transitions
collected from a real offline-arcade exploration run on a click-heavy game
When those transitions are fed through `InertClickSigPruner.observe`
Then the pruner's `stats()` reports a well-formed count of tracked
signatures and confidently-inert signatures (zero is an honest, valid
outcome when no signature clears the evidence floor within the budget, not
an error)

#### SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK

Given the same real transitions replayed as a candidate-row list against a
real collected frame
When `InertClickSigPruner.rank_candidates` filters that list
Then it runs without error and `rows_kept + rows_dropped == rows_in`, with
only rows whose click signature is confidently inert removed

#### SCENARIO-ARC-FCP-5595-LIVE-WIRING-CANDIDATES

Given a `StepwiseExplorer` with an `inert_click_pruner` configured
When `_candidates` builds the candidate-row list for a frame
Then it calls `inert_click_pruner.rank_candidates(frame, rows)` and returns
its (filtered) result, exactly like the `program_synthesis_filter`/
`goal_candidate_guidance` hooks it composes alongside; a raising pruner is
caught and never breaks candidate generation; no pruner configured is a
clean no-op

#### SCENARIO-ARC-FCP-5595-LIVE-WIRING-OBSERVE

Given a `StepwiseExplorer` with an `inert_click_pruner` configured and a
pending action recorded in `self.awaiting`
When `_ingest` processes the resulting frame
Then it calls `inert_click_pruner.observe(frame_before, label, frame_after,
leveled_up)` with the realized transition, from the same per-transition
OBSERVE hook that feeds `dense_curiosity`/`controllable_novelty_policy`/
`object_centric_proposal_policy`/`action_prior` -- without this, the pruner's
tally never accumulates evidence from live play and `rank_candidates` would
be wired but permanently inert

#### SCENARIO-ARC-FCP-5595-DEFAULT-OFF-PARITY

Given the SUBMITTED default configuration
When a `StepwiseExplorer` or `E3AgentPolicy` is constructed with no explicit
`inert_click_pruner` argument
Then `explorer.inert_click_pruner` is `None` (tracking
`SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG["inert_click_pruner_enabled"]`) -- the SCORED agent's
behavior is unchanged until a matched-budget offline A/B validates flipping
it on, per the `solve_rate_dropped` guardrail

### REQ-ARC-FCP-5595-2: Matched-Budget A/B -- The Flip-Decision Measurement

The `solve_rate_dropped` guardrail (REQ-ARC-FCP-5595-LIVE-WIRING-*'s own
default-off framing) names a matched-budget offline A/B (states/actions-
expanded reduction, zero regression in reproduced levels) as the precondition
for ever flipping `SUBMITTED_INERT_CLICK_PRUNER_ENABLED` to `True`. This
requirement runs that measurement, mirroring `HazardMovePruner`'s own tu93 A/B
precedent exactly: `scripts/arc_loop_solve.solve_adaptered` (extended with a
new `inert_click_prune: bool = False` parameter, composing
`InertClickSigPruner` into the same `move_pruner` construction `hazard_prune`/
`mask_prune` already use) SHALL be run twice at the SAME `--target-level` with
`hazard_prune`/`mask_prune` held fixed at `False` in both arms, varying ONLY
`inert_click_prune`, using the pruner's real, already-validated default
parameters. `OfflineSolver.last_states_expanded` SHALL be the efficiency
metric; `arc_solver_kit.reproduce` (the offline reproduction gate) SHALL be
the correctness backstop for both arms -- a states_expanded reduction only
counts as a genuine win if both arms reach the same target level and both
pass `offline_reproduced=True`.

Because `OfflineSolver`'s directed, verifier-guided search may never exercise
repeated inert clicks the way broad live exploration does, this requirement
ALSO runs an independent live-wired supplementary check: a real
`E3AgentPolicy` construction with `inert_click_pruner=True` (matching
exp5595's own real-game construction), reporting the pruner's own `stats()`
after a real exploration run -- confirming (or refuting) engagement
independent of the `OfflineSolver` harness.

### SCENARIO-ARC-FCP-5595-2-MATCHED-BUDGET-AB: Honest Verdict Taxonomy

Given the baseline (`inert_click_prune=False`) and treatment
(`inert_click_prune=True`) `OfflineSolver` runs plus the live-wired
supplementary check
When both arms pass the reproduction gate and reach the same target level
Then the verdict reports either a genuine `states_expanded` reduction (if
either the offline treatment arm or the live-wired check actually pruned
anything) or an honest no-op (if neither did) -- a zero-reduction result is
valid and informative at the tested budget, not a failure requiring
escalation; a failed reproduction gate or a lower reached level on the
treatment arm is classified as a regression regardless of any states_expanded
number, overriding the efficiency signal with the correctness backstop

### REQ-ARC-FCP-5699: SGE Anti-Stagnation Controller -- Genuine Live Collapse-Escape Re-Test

`ops/known-issues.md` task 6's own "NEXT STEP" named the fix: detect repeated
null-outcome LLM strategies and force diversity instead of converging on a
passive "wait" strategy. The conductor built `AntiStagnationDiversityController`
(now `SGECandidateRouter`'s default `anti_stagnation_controller`) and ran a
DETERMINISTIC precheck (exp5575) replaying the original recorded collapse
trace -- real evidence the controller LOGIC is correct, but not yet evidence
a genuine LIVE run (fresh strategies from the model, not a replayed trace)
actually escapes a collapse it would otherwise fall into. exp5575's own
follow-on live attempt (exp5576) never ran -- it GATE_BLOCKed on unrelated
project-wide gates (a pre-existing 16-failure full-suite pytest run and a
1262-test spec-coverage backlog), not on anything about the SGE mechanism
itself.

This requirement runs the genuine live re-test. The ORIGINAL null target
(g50t L3) is no longer usable as a registry-frontier attempt -- it was
independently fully cleared (`levels_reproduced=7, full_game_clear=true`)
one day after exp5575's precheck by an unrelated hand-derived mechanism. Per
"(or the original null target)", this requirement structurally replicates the
original scenario instead: a fresh-episode g50t session
(`prior_levels=0, target_level=1`, no registry credit claimed) reproduces the
early-exploration regime the original collapse occurred in, run TWICE (n=2
independent episodes) per CLAUDE.md's cross-check-surprising-results
discipline. A secondary, bonus real attempt at the current shallowest
not-fully-cleared registry frontier (read live, not hardcoded) tests whether
the fix also helps bank new territory.

### SCENARIO-ARC-FCP-5699-LIVE-COLLAPSE-ESCAPE: Real Live Collapse, Real Live Escape

Given a fresh `E3AgentPolicy` session on g50t with `SGECandidateRouter`'s
default anti-stagnation controller active and real GPU-backed local-GGUF LLM
inference (not a fake-completer, not a replayed trace)
When the live strategy proposer genuinely converges on a repeated
"observe/wait" strategy across several steps -- the exact failure mode task 6
was filed against
Then the controller detects the live collapse (`collapse_detected_live=True`,
with a real `collapse_trigger_step`), switches ranking away from the LLM
proposer to the deterministic forced diverse portfolio
(`forced_portfolio_activated_live=True`, `post_collapse_strategy_diversity>1`),
and this behavior reproduces independently across both replication episodes;
the escape is reported honestly including its own limitation (the forced
portfolio may itself settle into a smaller, static repetition on a frozen
game state) rather than being overclaimed as full resolution; a genuine
no-collapse observation on the secondary frontier pass (e.g. because the
perception layer yields empty candidate lists at a deep, already-explored
frontier) is reported as an honest, separately-flagged null, not silently
folded into the headline collapse-escape claim

### REQ-ARC-WMTE-5596: Generator-Size A/B -- Qwen3.6-27B-MTP vs the Frozen Live Generator

`ops/known-issues.md` task 13 (2026-07-12, HIGH PRIORITY) queued a re-verification of the Kaggle
VRAM budget and an offline A/B of a larger generator against the frozen live Qwen3.5-9B-MTP,
after the operator asked: "are we still using qwen-3.5-9B when the leaders are using the larger
and newer qwen-3.6-27B and Gemma-4-31b models?"

**13(a) -- Kaggle backend hardware re-verification.** A fresh, first-party check (fetched
2026-07-13, not the stale May-2026 staff-post evidence the original investigation relied on)
found `docs.arcprize.org/arc-prize-2026`'s starter kit explicitly names an `rtx6000` accelerator
option mapped to `Nvidia RTX 6000 (g4-standard-48)`, labelled "Heavy ML; ARC-AGI-3 exclusive."
This session's own clone of the ARC-AGI-3 Milestone-1 winners' code
(`external/arc-m1-3rd-forge/kernel-metadata.json`, 3rd place, LB 0.86) confirms this is not
theoretical: forge's real, scored submission requests `"machine_shape":
"NvidiaRtxPro6000"` directly and ran `gemma-4-31b-it`. This corroborates (does not replace) the
project's existing accelerator investigation
(`docs/research-notes/arc-kaggle-accelerator-upgrade-2026-06-21.md`, updated 2026-07-13 with
this finding). Also clarified: `results/kaggle_env_probe.json`'s P100 finding is from an
unrelated auxiliary dev/build-verify kernel with no `machine_shape` field (a known, previously
flagged gap, not evidence the SCORED submission kernel's own `NvidiaL4` request -- a deliberate
2026-06-21 quota-cost tradeoff, unchanged here -- is broken).

**13(c) -- MTP compatibility at the larger size (checked, not assumed).** Direct GGUF metadata
inspection (`gguf_dump.py --no-tensors`) found the FIRST candidate considered,
`unsloth/gemma-4-31B-it-GGUF`, declares NO `nextn_predict_layers` key at all -- no native MTP
support. The operator surfaced two better-fitting official candidates mid-task:
`unsloth/Qwen3.6-27B-MTP-GGUF` and `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, both genuinely MTP-capable
(`qwen35.nextn_predict_layers = 1`, confirmed by the same metadata inspection) at the exact
"Qwen3.6-27B-class" size the operator's original question named. Downloaded and verified the
dense 27B variant (16.3GB Q4_K_M, full download, `unsloth/Qwen3.6-27B-MTP-GGUF`).

**A second, deeper compatibility gap found via a real launch attempt.** Even with genuine
architectural MTP support, a manual CUDA launch with `--spec-type draft-mtp --model-draft <same
path>` OOM'd on a single 24GB RTX 3090 (verbatim from the real launch log: `ggml_backend_cuda_
buffer_type_alloc_buffer: allocating 15621.78 MiB on device 0: cudaMalloc failed: out of
memory`): llama.cpp's SELF-draft MTP loads the SAME GGUF file TWICE (main + draft model), so the
real VRAM footprint is roughly 2x the file size (~32.6GB for this 16.3GB file) plus KV
cache/CUDA overhead -- exceeding a single 3090's capacity entirely.
This is exactly the kind of gap task 13(c) asked to rule out before assuming a speedup carries
over, and it generalizes past "does the metadata say MTP" to "does self-draft actually fit."
Fixed by adding `_candidate_mtp_self_draft_fits_vram()` (a real free-VRAM-vs-2x-filesize
feasibility check, +2GB margin) alongside the existing metadata check
(`_candidate_declares_mtp_metadata()`); the experiment's actual runtime `mtp` flag
(`candidate_mtp_used`) is the AND of both, so the candidate arm cleanly falls back to non-MTP on
this hardware instead of crash-looping (the first real run, before this fix, silently fell back
to a much slower AMD iGPU each time the OOM'd process died -- see the second bug below).

**A separate, structural GPU-pinning bug found and fixed.** The experiment's first working draft
never called `LocalGGUFProposer.stop()` between arms. Because both arms are pinned to the SAME
GPU (GPU 1, the outer loop's own allocated 3090), the "current" arm's still-resident ~11GB server
left only ~13GB free on the 24GB card -- right at
`_generator_server_and_env()`'s `_GENERATOR_CUDA_MIN_FREE_MB=13000` guard floor -- so the
candidate's launch attempt tripped the guard and silently fell back to the slow iGPU (`build-hip`
binary), defeating the "both arms on the same hardware tier" design goal with no error or
warning. Fixed by wrapping each arm's body in `try/finally: proposer.stop();
_wait_for_port_down(proposer.port)`, a new helper that polls the stopped server's `/health`
endpoint (bounded, 30s timeout) so the NEXT arm's free-VRAM guard sees genuinely reclaimed
memory, not a still-warm process.

**RESOLUTION (2026-07-13).** With both bugs fixed, the real 4-attempt measurement (m0r0 + sk48,
both arms, GPU 1, both arms confirmed via wall-clock plausibility to have run on the CUDA build
not the iGPU) produced a real, informative result: both arms 2/2 induction success;
`heldout_accuracy` -- m0r0 current=0.0 vs candidate=0.5; sk48 current=0.2 vs candidate=1.0. The
candidate (Qwen3.6-27B, running WITHOUT MTP per the VRAM-infeasibility finding) shows materially
HIGHER induction quality on both tested games. `honest_verdict:
"complete: generator_size_ab_equal_success_candidate_higher_accuracy"`.

**What this does and does not show.** This is a real, positive signal for the candidate's
induction QUALITY specifically, on a 2-game roster (well below the CLAUDE.md sample-size floor
for any percentage-point claim), measuring `WorldModelVerifier.heldout_accuracy` (not a full
actions-to-first-win live-solve sweep). It does NOT show the candidate would be faster or more
efficient on the actual Kaggle hardware: MTP is confirmed infeasible for this model on a single
24GB RTX 3090 (this dev box), but the ACTUAL scored hardware may be the 96GB RTX Pro 6000 per
13(a)'s finding, where self-draft MTP (2x ~16.3GB = ~32.6GB) would fit comfortably alongside KV
cache -- a materially different VRAM regime this experiment did not test. Per the task's own
guardrail and CLAUDE.md's frozen-live-stack discipline (mirroring REQ-ARC-WMTE-5594's task-7
precedent), this result does NOT change the frozen live generator -- it is reported as an offline
dev measurement requiring an explicit operator decision, and the honest recommendation is: this
result is promising enough to justify a deeper look (a larger roster, the 35B MoE variant, and/or
a real VRAM-matched test if 96GB hardware becomes available), not an automatic swap.

Required field principles:

- `candidate_mtp_self_draft_fits_vram`: principle "self-draft MTP loads the SAME GGUF file twice
  (main + draft); found mid-task that this can OOM even when the metadata declares support, so
  this is checked separately before attempting a real launch, not assumed from metadata alone."
- `candidate_mtp_used`: principle "the actual runtime decision (metadata support AND VRAM
  feasibility); this is what determines whether any wall-clock delta between arms is confounded
  by an MTP-vs-no-MTP asymmetry, not the metadata declaration alone."

#### SCENARIO-ARC-WMTE-5596-MTP-SUPPORT-VERIFIED

Given a candidate GGUF whose metadata declares an MTP self-draft head
When `_candidate_mtp_self_draft_fits_vram` estimates the self-draft footprint (2x file size plus
a margin) against real free VRAM on the target GPU
Then a candidate whose self-draft footprint exceeds available VRAM is correctly identified as
MTP-infeasible on this hardware, and the experiment runs that arm without MTP instead of
attempting a doomed launch

#### SCENARIO-ARC-WMTE-5596-INDUCTION-QUALITY-DELTA

Given real induction attempts from both the current frozen generator and the candidate generator,
run sequentially on the SAME GPU with each arm's server fully stopped before the next starts
When `build_artifact` compares `heldout_accuracy` across arms
Then the comparison reflects a genuine same-hardware-tier measurement (not confounded by one arm
silently falling back to a slower device), and the verdict honestly reports which arm has higher
accuracy without flipping the frozen live-submission generator

### REQ-ARC-WMTE-5597: Generator-Size A/B -- Qwen3.6-35B-A3B-MTP (MoE) vs the Frozen Live Generator

Follow-on to REQ-ARC-WMTE-5596. exp5596's dense `Qwen3.6-27B-MTP` candidate showed materially
higher `heldout_accuracy` than the current generator on both tested games, and its own spec entry
flagged the 35B MoE variant (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) as "a natural follow-on if this
result is promising." This experiment is that follow-on, reusing exp5596's exact two-step MTP
feasibility check (`_candidate_declares_mtp_metadata` + `_candidate_mtp_self_draft_fits_vram`,
duplicated per-experiment per this session's established one-file-per-experiment convention) and
its GPU-pinning/stop-and-wait fix unmodified.

**MoE VRAM confirmation.** The candidate's Q4_K_M quant is 21.6GB (larger than the dense 27B
candidate's 16.3GB, despite the 35B MoE architecture activating only ~3B params per token --
self-draft MTP stores the FULL weight file regardless of expert-routing sparsity, so MoE does not
relax the self-draft VRAM problem). GGUF metadata confirms genuine MTP support
(`qwen35moe.nextn_predict_layers = 1`), but the self-draft feasibility check correctly found it
infeasible on a single 24GB RTX 3090 (self-draft estimate 43227MB vs 24120MB free), so the
candidate ran without MTP -- as expected, the same outcome as exp5596. A direct manual single-load
sanity check (no MTP) confirmed the model DOES fit non-MTP, but only barely: 21.9GB used of 24GB,
leaving 2.2GB free -- confirmed BEFORE the real run to avoid another crash-loop.

**RESOLUTION (2026-07-13).** The real 4-attempt measurement (m0r0 + sk48, both arms, GPU 1, both
arms confirmed via the same stop-and-wait GPU-pinning fix) produced a DIFFERENT result from
exp5596's dense-27B finding: both arms 2/2 induction success; `heldout_accuracy` -- m0r0
current=0.5 vs candidate=0.3 (current wins); sk48 current=1.0 vs candidate=1.0 (tie). Mean:
current=0.75, candidate=0.65. `honest_verdict:
"complete: generator_size_ab_equal_success_current_higher_accuracy"` -- the MoE candidate
performed WORSE than the current frozen 9B generator on this roster, in direct contrast to the
dense 27B candidate's positive result on the SAME two games. `induce_duration_s` for the
candidate was also notably higher (67.3s and 14.4s vs the current generator's 1.5s and 0.7s) --
the larger MoE model is substantially slower even without the MTP self-draft doubling, since its
weights are ~4x the current generator's file size.

**Note on cross-run comparability.** exp5596's OWN `current` arm scores on this same roster
(m0r0=0.0, sk48=0.2) differ from THIS run's `current` arm scores (m0r0=0.5, sk48=1.0) despite
identical model, game, and budget configuration -- real sampling variance in LLM-driven
exploration + induction (temperature > 0, no fixed decoding seed control at this layer), not a
bug. This means neither run's `current`-arm baseline should be treated as a fixed reference point;
the WITHIN-RUN candidate-vs-current comparison is the only sound reading, and even that is a
single-draw comparison on a 2-game roster -- well below the CLAUDE.md sample-size floor for any
percentage-point claim. Combined with exp5596's contradictory-direction result, the honest
reading across both experiments is: **neither the dense-27B nor the MoE-35B candidate has
demonstrated a reliable induction-quality edge over the current generator on this hardware and
roster** -- exp5596's positive signal and exp5597's negative signal could both be sampling noise
at n=2 games per arm. A larger roster (5+ games) and/or multiple seeds per (game, arm) pair would
be needed before either direction is trustworthy enough to inform an operator decision.

Required field principles: identical to REQ-ARC-WMTE-5596 (same field set, same rationale).

#### SCENARIO-ARC-WMTE-5597-MOE-MTP-FEASIBILITY-CHECKED

Given a MoE candidate GGUF whose metadata declares an MTP self-draft head and whose file is
LARGER than a previously-tested dense candidate's file
When `_candidate_mtp_self_draft_fits_vram` estimates the self-draft footprint
Then MoE sparsity does NOT relax the self-draft VRAM requirement (the full file is still loaded
twice), and a candidate whose self-draft footprint exceeds available VRAM is correctly identified
as MTP-infeasible regardless of active-parameter count

#### SCENARIO-ARC-WMTE-5597-INDUCTION-QUALITY-DELTA

Given real induction attempts from both the current frozen generator and the MoE candidate
generator, run sequentially on the SAME GPU with each arm's server fully stopped before the next
starts
When `build_artifact` compares `heldout_accuracy` across arms
Then the comparison honestly reports whichever arm has higher accuracy -- including a result
where the LARGER candidate performs WORSE than the current generator, without any pressure to
report a positive-sounding outcome

### REQ-ARC-WMTE-5598: Properly-Powered Multiseed Generator-Size A/B -- Resolving the exp5596/5597 Contradiction

exp5596 (dense Qwen3.6-27B-MTP, 2 games, n=1 draw/arm) found the candidate BEAT the current
generator. exp5597 (MoE Qwen3.6-35B-A3B-MTP, same 2 games, n=1) found the candidate LOST. Both
spec entries flagged this as likely sampling noise at n=1 -- this experiment resolves that by
testing all THREE arms (current, both candidates) together on a widened 4-game roster (m0r0,
sk48, cd82, sp80) with 3 independent repeated draws per (arm, game) cell (n=12 draws/arm total),
batching the loop by arm (one server per arm, reused across all its draws, stopped once before
the next arm starts) to avoid the per-attempt server-restart overhead exp5596/5597's interleaved
loop would have incurred at this scale.

**A genuine hardware fault interrupted the first attempt.** Mid-run, during the candidate_35b_moe
arm, GPU 1 (the outer loop's dedicated RTX 3090, hosted in an external eGPU enclosure) dropped
off the PCI bus entirely -- `nvidia-smi -q -i 1` returned "No devices were found," and even the
purpose-built `nvidia-smi --gpu-reset -i 1` recovery tool could not reach it. This is a hardware/
driver-level fault, not a script bug: the existing `_generator_server_and_env()` GPU-pinning
guard (by design, per its own docstring, "never fight the conductor for the 3090s") silently fell
back to the slow AMD iGPU when the free-VRAM check on GPU 1 started returning -1 (unreadable),
which would have contaminated the arm's later draws with a different, inconsistent hardware tier
partway through -- exactly the kind of confound this experiment was designed to eliminate. The
operator physically power-cycled the machine to recover the eGPU; both RTX 3090s came back
healthy afterward (confirmed via `nvidia-smi`).

**Hardening added before the retry.** (1) A per-arm `n_ctx` override (`candidate_35b_moe`
reduced from 16384 to 10240) intended to relax the arm's tight VRAM margin -- a direct manual
comparison found this saved only ~80MB (21.9GB used either way; the footprint is dominated by
model WEIGHTS, not KV cache, so this change alone would not have prevented the fault, and is kept
as a modest, honest, non-load-bearing precaution rather than oversold as a fix). (2) A genuine
fix: `_gpu1_free_mb() < 0` is now checked BEFORE every draw (not just once per arm); if GPU 1
becomes unreachable mid-run, the experiment immediately stops collecting further draws and
returns a distinct `generator_size_multiseed_ab_blocked_gpu1_lost_mid_run_partial_ranked_*`
verdict rather than silently continuing on degraded hardware -- fail closed, not silently degrade.
This guard was built defensively; it did NOT trigger on the successful retry (confirmed: no
`gpu1_unreachable_mid_run_aborting_remaining_draws` row in the checked-in artifact).

**RESOLUTION (2026-07-13).** The retried run completed cleanly on GPU 1 for all three arms
(3061.6s total, 35 of 36 draws succeeded -- one candidate_35b_moe draw on sp80 hit a normal,
honest induction failure: "syntax error line 159: expected an indented block," a real generated-
code defect after 3 retries, unrelated to the earlier hardware fault). Mean `heldout_accuracy`:
current=0.100 (std 0.289), candidate_27b=0.525 (std 0.352), candidate_35b_moe=0.391 (std 0.437).
Paired win/loss/tie against current (per (game, repeat) cell): candidate_27b **10 wins / 0
losses / 2 ties**; candidate_35b_moe **5 wins / 1 loss / 5 ties**. `honest_verdict:
"complete: generator_size_multiseed_ab_ranked_candidate_27b_gt_candidate_35b_moe_gt_current"`.

**This resolves the exp5596-vs-exp5597 contradiction: at real statistical power, BOTH candidates
beat the current generator, with the dense 27B model the clearer, more decisive winner.**
candidate_27b's 10-0-2 paired record is a near-unanimous signal (a naive sign test on the 10
decisive draws: P(10/10 or better under a fair-coin null) = 0.5^10 ~ 0.001) -- exp5596's original
positive finding replicates and strengthens with more data. candidate_35b_moe's 5-1-5 record is
net positive but much weaker and noisier (sign test on 6 decisive draws: P(>=5/6) ~ 0.11, not
strong evidence on its own) -- consistent with exp5597's negative n=1 draw having been an
unlucky single sample from a real but modest, noisy positive distribution, not a genuine
candidate-loses-to-current effect. Both candidates ran WITHOUT MTP (self-draft infeasible on a
single 24GB card for both, reusing exp5596/5597's feasibility check unmodified) -- this remains
an open axis: performance AND speed on the actual Kaggle hardware (a 96GB RTX Pro 6000, per
REQ-ARC-WMTE-5596's task-13(a) finding) where self-draft would fit is untested.

**What this does and does not show.** Still a 4-game, offline dev-quality-only measurement (not a
full actions-to-first-win live-solve sweep, and n=12/arm is still below the CLAUDE.md N>=30 floor
for a firm percentage-point claim on the absolute accuracy VALUES) -- but the PAIRED, WITHIN-RUN,
same-hardware-tier, multi-draw comparison is now genuinely more trustworthy than either single-
draw prior result, and candidate_27b's near-unanimous win record is a real, replicated, promising
signal specifically for the dense 27B candidate. Per the task's own guardrail and CLAUDE.md's
frozen-live-stack discipline, this result does NOT change the frozen live generator -- it is
reported as an offline dev measurement requiring an explicit operator decision. The honest
recommendation: candidate_27b's induction-quality edge is now well-supported enough to justify a
genuine cost/benefit evaluation for switching (weighed against Kaggle quota, actual scored-
hardware VRAM/MTP behavior, and a fuller live-solve test) -- it is no longer just a single-draw
curiosity.

Required field principles: identical to REQ-ARC-WMTE-5596/5597's `candidate_declares_mtp_
metadata`/`candidate_mtp_self_draft_fits_vram`/`candidate_mtp_used` (per-arm here), plus:

- `per_draw_results`: principle "every individual (arm, game, repeat) draw is recorded, not just
  aggregates -- exp5596/5597's contradiction came from single draws, so preserving the full draw
  list is what lets a reader assess variance directly rather than trusting a summary alone."
- `paired_vs_current`: principle "per-(game, repeat) win/loss/tie counts against the current arm,
  the higher-power paired comparison (controls for per-game/per-draw difficulty) vs comparing
  unpaired means across arms."

#### SCENARIO-ARC-WMTE-5598-MULTISEED-PAIRED-COMPARISON

Given multiple independent draws per (arm, game) cell across a widened roster
When `build_artifact` computes `per_arm_summary` and `paired_vs_current`
Then the paired win/loss/tie counts against the current arm are reported per candidate, giving a
higher-power signal than any single-draw comparison, without asserting formal statistical
significance beyond what the sample size supports

#### SCENARIO-ARC-WMTE-5598-ARM-BATCHED-SERVER-LIFECYCLE

Given three arms sharing one GPU, each needing multiple draws across multiple games
When `build_artifact` runs the game/repeat loops
Then exactly one proposer/server is constructed per arm (reused across all its draws) and
explicitly stopped before the next arm starts, and a mid-run GPU-health check aborts the run
honestly (rather than silently falling back to different hardware) if the GPU becomes unreachable
between draws

### REQ-ARC-WMTE-5599: Real Reinduction-Path A/B -- candidate_27b Reverses on the Actual Scored Code Path

Follow-on to REQ-ARC-WMTE-5598, prompted by the operator's cost/benefit question. Investigating
`E3AgentPolicy._induce_and_plan()` (the method the SCORED live agent actually calls) found its
LLM tier (`execute_bounded_llm_reinduction`, `arc_llm_reinduction.py`) is ONLY invoked when the
induction reason is `"level_up_reinduction"` -- a genuine level-up just happened. For the
initial `"stall"` reason (exploration exhausted without ever winning), the agent tries a
zero-LLM TTT-prior model, then classical DSL/active-probe tiers -- the LLM is never invoked.
Confirmed empirically: a direct `lb.run_game` measurement on `m0r0` (never-leveled, like every
roster game in exp5596/5597/5598) completed `_induce_and_plan()` in 17.6s -- far too fast for
real LLM inference -- and `level_induction_events` stayed empty. **This means exp5596/5597/5598's
induction-quality measurements, while real and valid as a measure of induction quality via the
`LocalGGUFProposer.induce()` wrapper, could not have exercised the real live-agent reinduction
code path on their own roster** -- that roster structurally cannot trigger it.

This experiment instead calls `execute_bounded_llm_reinduction` directly -- the exact function
the scored agent invokes -- on real, reproducible post-level-up transitions from `lp85` (the one
game with a session-confirmed level-up, `first_levelup_index` around 6). A widened
`n_ctx=22000` (up from the class default 16384) was required and verified first via a manual
pre-check: lp85's 64x64 grid previously overflowed the induction prompt at 16384 (exp5593's real
HTTP 400 `exceed_context_size_error`); 22000 resolved it (a real 101.6s call completed cleanly).
3 independent repeats per arm (lp85's exploration is stochastic; fresh transitions collected
every draw), current vs `candidate_27b` (exp5598's clear induction-quality winner), same
GPU-1/stop-between-arms/GPU-health-guard discipline as exp5598.

**DISCLOSED METHODOLOGY GAP (found reviewing the real result, not hidden).** `_induce_and_plan()`
calls `execute_bounded_llm_reinduction` with `min_heldout_accuracy=1.0` plus several other
policy-configured kwargs (`enable_subgoal_search`, `subgoal_budget`, `value_head`,
`enable_factored_planner`, `factored_trust_threshold`, `structural_goal_provider`). This
experiment's call used the function's own bare defaults for all of these (`min_heldout_accuracy
=0.0` etc.) rather than replicating the exact policy-configured values -- a real fidelity gap
against the true scored call site, not a bit-identical replication. This is flagged explicitly
because `current`'s one "planned" draw (repeat 2, below) had `heldout_accuracy=0.0`, which would
almost certainly have been REJECTED under the real agent's strict 1.0 threshold -- so even
`current`'s single apparent success may not survive under the real, stricter gating. This
experiment answers "does the generator work through the reinduction machinery under standard/
default gating," a real and useful question, but not a bit-identical replication of the scored
call site; a follow-on with exact parameter matching is the natural next step if this result is
acted on.

**RESOLUTION (2026-07-13) -- REVERSES exp5598's induction-quality-only finding.** All 6 draws
reached the real level-up transition set (37 total transitions, level-up at action 7, 8-transition
induce window, consistent across every draw). `current`: plan_rate_given_levelup = **1/3 (33%)**,
`reinduce_duration_s` = 67.0s / 78.1s / 20.7s (mean ~55s), mean `heldout_accuracy` = 0.0.
`candidate_27b`: plan_rate_given_levelup = **0/3 (0%)** -- WORSE, not better -- and
`reinduce_duration_s` = 294.9s / 476.4s / 431.2s (mean ~401s, i.e. **~7x slower** than current),
mean `heldout_accuracy` = 0.222 (nominally higher than current's 0.0, but never actually
producing a usable plan in any of the 3 draws). `honest_verdict:
"complete: reinduction_ab_current_plans_more_reliably"`.

**What this changes for the cost/benefit recommendation.** exp5598's induction-quality signal
(candidate_27b wins 10-0-2 on a held-out-accuracy proxy, n=12/arm) was real and well-powered, but
this experiment shows that signal does NOT carry over to the real live-agent reinduction code
path: candidate_27b never once produced a usable plan here, while current did once (with the
above-noted caveat about whether that one success would survive stricter gating), AND
candidate_27b is dramatically slower per attempt (~7x). **This reverses the tentative
recommendation from the cost/benefit discussion following exp5598** -- candidate_27b's induction-
quality edge does not translate into better live-agent planning reliability on this test, and its
severe speed cost (already flagged as a theoretical risk in that discussion) is now empirically
confirmed as large. The honest conclusion: **do not switch the frozen live-submission generator
based on the evidence gathered so far** -- exp5598's quality-only signal was measuring the wrong
thing for this decision, and the real-reinduction-path test (despite its own disclosed parameter-
fidelity gap) points the opposite direction, on top of a severe throughput cost.

**Scope limits, honestly stated.** n=3 repeats per arm on a SINGLE game (lp85) is a small sample;
`plan_rate 1/3 vs 0/3` is not independently statistically decisive on its own (a single flip on
either side would change the ranking). The speed finding (~7x slower) is the more robust,
sample-size-independent part of this result. The disclosed parameter-fidelity gap (bare function
defaults vs the real policy's stricter configured values) means this is not the final word on
the planning-reliability question either. Taken together with exp5598's contradicted signal, the
overall picture is: no candidate has yet shown a reliable, real advantage on the metric that
matters (live planning success), and the cost (speed) is real and severe.

Required field principles: identical structure to REQ-ARC-WMTE-5596/5597/5598.

#### SCENARIO-ARC-WMTE-5599-REAL-REINDUCTION-PATH

Given real post-level-up transitions and a candidate generator with a strong induction-quality
signal from a prior, narrower experiment
When `execute_bounded_llm_reinduction` (the actual function the scored live agent calls) is
invoked directly with that candidate as the proposer
Then the plan-success rate and wall-clock cost are measured on the real code path, which may
disagree with the narrower induction-quality proxy -- and the experiment reports whichever
direction the real data shows, without assuming the narrower result carries over

#### SCENARIO-ARC-WMTE-5599-CONTEXT-BUDGET-FIX-VERIFIED

Given a large-grid game whose induction prompt previously overflowed the default context window
When the reinduction call is made with a widened `n_ctx`
Then the call completes without a context-size error, isolating any subsequent
planning-quality result from the context-budget confound

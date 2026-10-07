# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_8240_v712_qualified_delayed_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Causal persistent corrections and exact crash recovery succeed on historically exposed development data, while establishing no generalized learning or utility trajectory benefit.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. Resumed state hashes differed from baseline execution after an injected crash (`resumed_sha256` != `baseline_sha256` in `restart_state_hashes`).
2. Any required child process failed, timed out, or produced an unexpected exit code (`actual_exit` != `expected_exit` or `passed` is `false` in `child_exit_rows`).
3. Actions were altered before an item's admission slot rather than strictly after, violating causal ordering.
4. Retention labels were opened prematurely (`retention_labels_opened` is `true`), or in-sample fitting was reported as generalized learning.

## WAS THAT CHECKED
Yes. Active crash recovery was tested via injected crashes at slots 90 and 170 (`crash90`, `crash170`) and verified via exact state hash identity (`resumed_sha256` matches `baseline_sha256`) in `restart_state_hashes`. All required execution processes in `child_exit_rows` completed with expected exit codes. Decision updates were verified to take effect only after admission slots in `later_decision_changes`. Retention labels were verified unopened (`retention_labels_opened: false`), and generalization/learning benefit scores were explicitly logged as 0 with H2 unmeasured (`measured_here: false`), correctly recording an honest null outcome.

## EVIDENCE
- `claim_scope`: `Causal persistent corrections and exact recovery on exposed development only; no learning benefit established.`
- `honest_verdict`: `complete_null_utility_trajectory_benefit_reserved_for_8241`
- `acceptance_gates`: `benefit`: `H2 and retention verdicts reserved for Exp8241`, `causal_execution`: `true`, `owned_validation`: `true`
- `measured_here`: `false`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `retention_labels_opened`: `false`
- `required_checks_passed`: `true`
- `restart_state_hashes`: `baseline_sha256`: `sha256:0f689ff235c9e2582695315956c96fe7c2e732a244c1444c1dd5f648959e4214`, `resumed_sha256`: `sha256:0f689ff235c9e2582695315956c96fe7c2e732a244c1444c1dd5f648959e4214`, `passed`: `true`
- `child_exit_rows`: `name`: `crash90`, `actual_exit`: `73`, `expected_exit`: `73`, `passed`: `true`
- `child_exit_rows`: `name`: `resume90`, `actual_exit`: `0`, `expected_exit`: `0`, `passed`: `true`
- `child_exit_rows`: `name`: `crash170`, `actual_exit`: `73`, `expected_exit`: `73`, `passed`: `true`
- `child_exit_rows`: `name`: `resume170`, `actual_exit`: `0`, `expected_exit`: `0`, `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_8241_v712_delayed_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The delayed utility learning patches provide no delayed decision benefit for `global_plus_group` over `global_only`, resulting in a complete null verdict on hypothesis H2.

## WHAT WOULD REFUTE IT
A statistically significant positive cost gain for `global_plus_group` over `global_only`, such as a bootstrap `lower_bound` exceeding `0.02` (`lower_cost_gain_gt`), at least 5 improved sources (`improved_sources_min`), or `H2.passed` evaluating to `true`.

## WAS THAT CHECKED
Yes; checked across 158 completed sources evaluated over 10,000 block bootstrap draws (at block lengths 8, 16, and 32) and paired comparisons against `global_only` and baseline controls in `H2`, `block_bootstrap_diagnostics`, `excess_brier_rows`, `per_source_deltas`, and `retention_rows`.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_delayed_decision_benefit"`
- `"verdict_class"`: `"null"`
- `"passed"`: `false`
- `"failed_conditions"`: `["lower_gain", "improved_sources"]`
- `"improved_sources"`: `0`
- `"lower_bound"`: `0.0`
- `"mean_gain"`: `0.0`
- `"operands"`: `{"complete_sources": true, "improved_sources": false, "lower_gain": false, "nonoverlapping_blocks": true, "per_class": true, "per_seed_safety": true, "valid_draws": true}`
- `"completed_count"`: `158`
- `"comparison"`: `"global_plus_group versus global_only"`
- `"gain"`: `0.0`
- `"cost_worsening"`: `0.0`
- `"brier_worsening"`: `0.0`
- `"generalized_learning_benefit_score"`: `0`
- `"h2_development_signal_score"`: `0`
- `"independent_generalization_score"`: `0`

## RECOMMENDATION
KEEP

## experiment_8242_v712_independent_concurrent_service.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8243_v712_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8244_v712_kv260_decision_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8245_v712_polarfire_state_dispatch.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A refutation is not applicable because the artifact does not assert a comparative or substantive performance claim. The artifact explicitly restricts its scope to execution mechanics and exact hash validation, registering zero independent benefit and a disqualified verdict. If an assertion of comparative superiority, FPGA speedup, or learning advantage were made, it would be falsified by the absence of comparative baselines, zero model calls, and unvalidated board execution.

## WAS THAT CHECKED
No; comparative evaluation against baselines or rival methods was not tested or checked, as the run was designed strictly as a dispatch and receipt scaffolding harness rather than a comparative experiment.

## EVIDENCE
- `claim_scope`: `"execution mechanics and exact hashes only; independent benefit remains zero"`
- `honest_verdict`: `"complete_disqualified_polarfire_state_dispatch"`
- `verdict_class`: `"disqualified"`
- `methodology`: `"Full selected state and every query are hashed. The original host evaluator defines parity. Current board execution is one bounded Linux CPU dispatch. No generator loads, state fitting, FPGA acceleration, speed or learning gain is inferred. Historical Qwen is provenance only."`
- `exposure_scope`: `"reused exposed development probabilities; no independent evaluation"`
- `independent_generalization_score`: `0`
- `generalized_learning_benefit_score`: `0`
- `current_model_calls`: `0`
- `current_device_execution_count`: `0`
- `polarfire_workload_validated`: `false`
- `required_checks_passed`: `false`
- `acceptance_gates`: `"board_parity": false`, `"scientific_benefit": false`, `"owned_checks": false`

## RECOMMENDATION
KEEP

## experiment_8246_v712_gatemate_change_ledger.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8247_v712_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The evaluated energy margin and group adaptation treatment methods fail to demonstrate significant decision cost gains over baselines, yielding completed null results across both H1 and H2 while upstream hardware evidence remains blocked or null.

## WHAT WOULD REFUTE IT
Observing statistically significant positive cost gains exceeding the protocol thresholds in the paired bootstrap evaluations: specifically, for H1, observing `lower_gain` > 0.02 and `improved_sources` >= 5 without violating the control constraints for Brier increase (<= 0.01) or extra false accepts (<= 0); and for H2, observing `lower_gain` > 0.02 and `improved_sources` >= 5 in the moving block bootstrap across sensitivity block lengths.

## WAS THAT CHECKED
Yes. Refutation was given a real chance to occur and was thoroughly checked under `H1.statistics` and `H2.statistics`. For H1, 97 independent completed sources were evaluated across 10,000 bootstrap draws against primary and mandatory controls; `lower_gain` observed -0.04296875 and only 2 improved sources, directly failing protocol conditions. For H2, 158 completed sources were evaluated across 10,000 block bootstrap draws across block lengths 8, 16, and 32; observed gain was 0.0 with 0 improved sources, failing the protocol.

## EVIDENCE
- `completed_null`
- `complete_blocked_upstream_evidence`
- `failed_conditions`
- `lower_gain`
- `-0.04296875`
- `improved_sources`
- `2`
- `original_frozen_v707_radial_cost_increase`
- `0.0234375`
- `energy_margin versus calibration-selected primary`
- `global_plus_group versus global_only`
- `passed`
- `false`
- `retention_passed`
- `0`
- `10000`
- `97`
- `158`
- `complete_null_kv260_decision_boundary`
- `complete_disqualified_polarfire_state_dispatch`
- `complete_blocked_gatemate_physical_change`
- `independent_science`

## RECOMMENDATION
KEEP

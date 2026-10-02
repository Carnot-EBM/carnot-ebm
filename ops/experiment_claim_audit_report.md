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
| CLAIM_SUPPORTED | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8023_v695_likelihood_calibration.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8024_likelihood_decision_test.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8025_v695_causal_online_updates.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8026_v695_learning_retention_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8027_v695_native_update_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The native update implementation achieves float64 numerical parity and matched CPU costs on exposed cached development, but is disqualified from deployment readiness (`complete_disqualified_native_update_cost`) because durable transaction throughput fails NFR-01 tenfold targets and owned validation checks fail.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if:
1. Numerical parity failed between Python and Rust implementations (e.g., probability or coefficient differences exceeding tolerance `1e-10`, or `action_equal` evaluating to false).
2. End-to-end durable transaction speedup actually achieved the 10x target over Python (`transaction_speedup` >= 10.0, satisfying `nfr01_met: true`), contradicting the disqualification.
3. Owned validation checks passed cleanly (`owned_checks: true` with zero exit codes and no crashes), which would contradict the disqualification and yield a readiness score of 1.

## WAS THAT CHECKED
Yes.
1. Numerical parity was checked across 4,480 completed updates and 256 stress cases in `parity_rows` and `serialized_restart_rows`, confirming errors below `1e-16` and setting `parity_passed` to `true`.
2. Durable transaction latency was checked across batch sizes 1, 16, and 64 across 30 repetitions against the Python baseline arm in `timing_distributions` and `complete_service_estimates`, showing transaction speedups of only ~1.0006x to 1.005x because feature construction and persistence dominate runtime, properly recording `nfr01_met` as `false`.
3. Owned validation was checked in `validation_receipts`, capturing a segmentation fault (`exit_code` `-11` on `unit_consumers_e2e015_019`), resulting in `owned_checks: false`, `native_update_ready_score: 0`, and the honest disqualified verdict.

## EVIDENCE
- `honest_verdict`: `"complete_disqualified_native_update_cost"`
- `verdict_class`: `"disqualified"`
- `claim_scope`: `"Opt-in float64 numerical parity and matched CPU costs on exposed cached development; no deployment promotion or hardware speed claim."`
- `native_update_ready_score`: `0`
- `parity_passed`: `true`
- `tier1_arithmetic_met`: `true`
- `nfr01_met`: `false`
- `acceptance_gate_results`: `{"measurement": true, "owned_checks": false, "parity": true}`
- `transaction_speedup`: `{"1": 1.0053599322342952, "16": 1.0006318120280278, "64": 1.0048504343406452}`
- `kernel_speedup`: `{"1": 23.200951248513675, "16": 28.00542028018679, "64": 27.55844913903419}`
- `exit_code`: `-11`
- `passed`: `false`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_8028_v695_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8029_v695_hardware_workload_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent hardware custody is verified under read-only replay, with no current device execution, no compatible kernels, and no measured device acceleration benefit.

## WHAT WOULD REFUTE IT
Any board or operation recording active hardware execution (`current_device_execution_count` > 0 or `current_hardware_execution` as true), an executable device kernel (`executable_device_kernel` as true), positive device acceleration benefit (`device_benefit` as true), or failed custody verification (`custody` as false or `custody_valid` as false).

## WAS THAT CHECKED
Yes. Custody verification and device benefit were evaluated in `acceptance_gate_results` (`custody`, `device_benefit`, `numeric_pass`), `board_rows` across audited boards (`custody_valid`, `current_hardware_execution`, `terminal_criterion_met`, `compatible_sparse_kernel`, `compatible_update_kernel`), and `workload_placement_rows` across all pipeline operations (`device_execution_count`, `executable_device_kernel`).

## EVIDENCE
- `honest_verdict`: `"complete_null_independent_hardware_custody"`
- `verdict_class`: `"null"`
- `claim_scope`: `"Current read-only custody and CPU quantized replay of exposed development. No current device execution, model load, natural deployment benefit or vendor speedup."`
- `purchase_recommendation`: `"No purchase: no qualified useful measured compatible device bottleneck and executable kernel"`
- `acceptance_gate_results`:
  - `custody`: `true`
  - `device_benefit`: `false`
  - `numeric_pass`: `false`
- `current_device_execution_count`: `0`
- `hardware_custody_ready_score`: `1`
- `quantized_update_ready_score`: `0`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8030_v695_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

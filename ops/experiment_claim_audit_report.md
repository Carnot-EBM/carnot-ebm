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

## experiment_8046_v697_branch_protocols.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or empirical claim is asserted to refute. If construed as an assertion of protocol readiness or custody validity, any corrupted hash, missing upstream asset, or failed gate check among the sealed preconditions would refute it.

## WAS THAT CHECKED
Yes, custody and precondition gate checks were verified across the listed upstream inputs and code configs (`acceptance_gate_results` and `preconditions_checked`), but hypothesis testing was not checked because the invocation explicitly did not execute comparative evaluation or model inference.

## EVIDENCE
- `claim_scope`: `This invocation seals original historically exposed public source roles and feedback acceptance. It measures custody and owned validation only; no live scoring, new training trajectory, unseen evaluation or deployment benefit.`
- `honest_verdict`: `complete_null_branch_protocols`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `inference_substrate_class`: `no_model_load`
- `disposition`: `unavailable in protocol-only invocation`
- `measured`: `false`
- `scope`: `protocol only`
- `arm`: `protocol`
- `condition`: `public_custody`
- `learning_protocol_ready_score`: `1`
- `field_principles`: `Require qualified protocol custody and all owned checks; readiness needs no scientific win and certifies no deployment correctness.`

## RECOMMENDATION
KEEP

## experiment_8047_fit_score_capture.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8051_v697_feedback_constrained_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Empirical guard checks on persistent small-head updates establish no future safety, retention, or independent learning benefit over unconstrained updates.

## WHAT WOULD REFUTE IT
The claim would be refuted by observing a measurable, statistically meaningful performance advantage for the feedback-constrained arm over the unconstrained or frozen baselines—such as a lower Brier score, lower typed cost, superior retention accuracy, or a positive generalized learning benefit score on evaluated streams.

## WAS THAT CHECKED
Yes. In `rows`, `attempted_gradient_counts`, and `committed_update_counts`, both `unconstrained` and `feedback_constrained` arms were evaluated across multiple seeds on identical released candidates, measuring Brier scores, typed costs, false accepts, and guard rejections, while `positive_control_results` confirmed the guard mechanism could detect beneficial versus destructive updates.

## EVIDENCE
- `"honest_verdict": "complete_null_feedback_constrained_learning"`
- `"verdict_class": "null"`
- `"claim_scope": "This invocation measures persistent CPU small-head updates on one historically exposed development stream. Empirical guard checks establish no future safety, retention or independent learning benefit."`
- `"generalized_learning_benefit_score": 0`
- `"arm": "unconstrained"`
- `"arm": "feedback_constrained"`
- `"arm": "frozen_no_write"`
- `"brier": 0.11997091123721107`
- `"brier": 0.1217728610957661`
- `"typed_cost": 0.5170940170940171`
- `"typed_cost": 0.5341880341880342`
- `"positive_control_results"`
- `"passed": true`

## RECOMMENDATION
KEEP

## experiment_8052_v697_learning_benefit_audit.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8053_v697_guarded_transaction_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Native execution of guarded numerical transactions achieves numerical parity and readiness with Python but yields a complete null speedup.

## WHAT WOULD REFUTE IT
Observation of a statistically significant performance advantage where the lower bound of the 95% bootstrap confidence interval strictly exceeds 1.0 (refuting the null speedup claim), or observation of numerical discrepancies where `parity_passed` is false or restart actions and probabilities diverge beyond numerical tolerance (refuting parity and readiness).

## WAS THAT CHECKED
Yes. Paired timing measurements against Python across 30 non-warmup repetitions were conducted for all transaction classes (`accepted`, `rejected`, `reset`) under both `feedback_constrained` and `unconstrained` conditions in `complete_numerical_transaction_speedup`, and numerical equivalence was evaluated across transactions in `parity_rows` and `acceptance_gate_results`.

## EVIDENCE
- `"honest_verdict": "complete_null_guarded_transaction_cost"`
- `"claim_scope": "This invocation measures exposed-development guarded numerical transactions only. No full verification-service, generalized learning, deployment safety or hardware speed claim."`
- `"parity_passed": true`
- `"native_transaction_ready_score": 1`
- `"nfr01_met": false`
- `"default_enabled": false`
- `"speedup": 0.9968559812593174`
- `"lower_95": 0.9929087490172696`
- `"upper_95": 0.9998244589543033`
- `"speedup": 0.9937533980426855`
- `"lower_95": 0.9848318546893519`
- `"upper_95": 1.0000066418105247`
- `"speedup": 0.9875195597377358`
- `"lower_95": 0.9580954974124196`
- `"upper_95": 1.0042421273635151`
- `"speedup": 0.9949596995660783`
- `"lower_95": 0.9856279911470285`
- `"upper_95": 1.0011461633110714`
- `"completed_count": 240`

## RECOMMENDATION
KEEP

## experiment_8054_v697_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8055_v697_hardware_guard_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware acceleration provides no measurable whole-service benefit or genuine headroom over CPU execution for the guarded boundary, supporting a complete null verdict and deferring device acquisition.

## WHAT WOULD REFUTE IT
An observation of substantial speedup bounds (significantly greater than 1.0x) showing that compatible arithmetic accounts for a dominant share of total transaction latency, an authenticated device execution demonstrating whole-service speedup, or active FPGA fabric kernels compatible with the guarded workload.

## WAS THAT CHECKED
Yes; checked in `acceleration_bounds` (evaluating hypothetical 100x arithmetic acceleration against measured CPU transaction profiles across `native` and `python` arms), `board_rows` (evaluating device custody, execution status, and kernel compatibility across KV260, PolarFire, and GateMate targets), and `acceptance_gate_results`.

## EVIDENCE
- `honest_verdict`: `"complete_null_guarded_hardware_boundary"`
- `claim_scope`: `"This invocation authenticates historical custody and CPU guard emulation on exposed development. No device performance, future safety or service acceleration."`
- `measured_device_benefit`: `false`
- `genuine_headroom`: `false`
- `speedup_bound`: `1.0000684639280863`
- `purchase_recommendation`: `"defer: no useful measured device bottleneck"`
- `acquisition_relevance`: `"defer: no measured board whole-service benefit"`
- `current_device_execution_count`: `0`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8056_v697_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

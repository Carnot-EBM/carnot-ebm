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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 4 |
| CANNOT_DETERMINE | 3 |

## experiment_8289_v715_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any comparative empirical or generalization claim being asserted from this artifact, as neither hypothesis was executed and no outcome rows were generated.

## WAS THAT CHECKED
No; both registered hypothesis protocols were halted prior to measurement due to missing prerequisite operands, leaving empirical verification unperformed.

## EVIDENCE
`"status"`
`"blocked_unmeasured"`
`"failed_operand"`
`"eligible_independent_audit_and_primitives"`
`"completed_count"`
`null`
`"independent_science"`
`false`
`"supported_transferable_evidence"`

## RECOMMENDATION
KEEP

## experiment_8290_v716_runtime_localization.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A finding of successful CUDA device initialization, allocation, and native execution (e.g., `cuInit` returning `0`, `cudaGetDeviceCount` > `0`, or `native_compatible` being `true`) would refute the diagnostic finding of an unavailable CUDA runtime. Furthermore, if a comparative scientific or learning claim were asserted, observing `cuda_context` as `false` or zero improvement over a baseline would refute it. However, the artifact makes no comparative or scientific claim to refute.

## WAS THAT CHECKED
No comparative hypothesis was checked because `H1` and `H2` were explicitly not measured (`measured_here` is `false`) and `scientific_benefit` was gated `false`. Hardware diagnostics and environment localization were executed (in `cuda_probe_rows` and `gate_check_summary`), confirming that the runtime is blocked across driver, runtime, and native layers.

## EVIDENCE
`"H1"`: `{"measured_here": false}`
`"H2"`: `{"measured_here": false}`
`"scientific_benefit"`: `false`
`"cuda_context"`: `false`
`"generalized_learning_benefit_score"`: `0`
`"phase"`: `"terminal_blocked"`
`"honest_verdict"`: `"complete_blocked_cuda_runtime"`
`"inference_substrate"`: `"deterministic_runtime_receipt_validation_no_llm"`
`"cuInit"`: `101`
`"cudaGetDeviceCount"`: `101`
`"native_compatible"`: `false`

## RECOMMENDATION
KEEP

## experiment_8291_v716_dependency_scoped_admission.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Dependency-scoped transaction admission preserves soundness and provides an efficiency advantage over full constraint scans.

## WHAT WOULD REFUTE IT
- Disagreement or constraint violations identified by an independent external oracle comparing post-state validity against true semantics rather than the verifier checking itself.
- End-to-end transaction latency of dependency-scoped admission exceeding that of full constraint scans across general graph topologies (`paired_median_time_ratio > 1.0`).

## WAS THAT CHECKED
No. Soundness refutation was not given a chance to happen because the verifier acts as its own oracle (`verifier_is_oracle = true`), making the soundness claim circular by construction. For efficiency, refutation was observed on chain and cyclic topologies where scoped admission was slower than full evaluation (`paired_median_time_ratio` of `1.1067801379041788` and `1.0396705054985425`), but the efficiency gate masked this by scoping the criterion strictly to the sparse DAG fixture.

## EVIDENCE
- `honest_verdict`: `"complete_circular_positive_dependency_scoped_admission"`
- `verifier_is_oracle`: `true`
- `efficiency_signal`: `true`
- `dependency_efficiency_signal_score`: `1`
- `dependency_efficiency_signal_score`: `"Sparse graph-unit median constraint fraction <=0.5 and paired median whole-transaction ratio <=1; repetitions add no independent sources."`
- `paired_median_time_ratio`: `1.1067801379041788`
- `paired_median_time_ratio`: `1.0396705054985425`
- `paired_median_time_ratio`: `0.9507309161694101`
- `scientific_benefit_measured`: `false`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8292_evidence_view_canary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8300_v716_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8301_v716_kv260_evidence_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8302_v716_gatemate_physical_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact is an audit receipt and evidence ledger making no comparative or empirical performance claim, there is no headline claim to refute. If the ledger's recording of blocked status were taken as a factual assertion, it would be refuted by an authenticated operator physical-change receipt, modified bringup documentation in the change ledger, or a live JTAG probe returning a valid GM1Ax IDCODE (`0x20000001`) instead of `0xffffffff`.

## WAS THAT CHECKED
Yes, for document and ledger continuity: `change_ledger` checked document hashes against upstream experiment 8288 (all unchanged), and `physical_change_evidence` verified that no new physical change receipts exist. No comparative modeling or device execution was attempted.

## EVIDENCE
`claim_scope`
`"Evidence ledger; future physical preflight remains unexecuted"`
`arm`
`"documentation_audit"`
`honest_verdict`
`"complete_blocked_gatemate_physical_change"`
`verdict_class`
`"blocked"`
`scientific_benefit_score`
`0`
`model_invoked`
`false`
`inference_substrate`
`"aggregation_from_upstream_artifacts"`

## RECOMMENDATION
KEEP

## experiment_8303_v716_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A concrete observation demonstrating statistically significant independent scientific gain or advantage over baseline controls on the target tasks would refute the artifact's recorded blocked and null state; however, because this artifact is a receipt and status aggregation that advances no positive comparative claim, there is no performance claim to refute.

## WAS THAT CHECKED
No; the artifact explicitly records H1 and H2 as unmeasured due to upstream prerequisite failures, ARC evaluations as disqualified with zero games executed, and hardware dispatch as execution mechanics only with zero benefit score, omitting unblocked comparative evaluations.

## EVIDENCE
`status`
`blocked_unmeasured`
`failed_operand`
`eligible_independent_audit_and_primitives`
`energy_specific_advantage`
`false`
`natural_benefit`
`circular_positive_fixture`
`independent_science`
`claim_scope`
`execution mechanics and exact hashes only; independent benefit remains zero`
`benefit_score`
`0`
`scientific_benefit`
`verdict_class`
`disqualified`
`honest_verdict`
`complete_disqualified_owned_checks`
`complete_circular_positive_polarfire_state_dispatch`

## RECOMMENDATION
KEEP

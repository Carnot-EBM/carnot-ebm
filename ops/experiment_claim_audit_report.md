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
| NO_CLAIM | 6 |

## experiment_8620_v740_triviaplus_extraction.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative efficacy or generalization claim is made. The descriptive extraction audit would be contradicted by missing census rows, incorrect majority-label aggregation, or abstentions counted as usable scores.

## WAS THAT CHECKED
Yes, audit checks are reported: all 190 intended rows completed, with 135 usable rows and 55 explicit budget abstentions. Shown aggregations match the release. The target head was not evaluated, and the lexical scores remain exploratory without winner selection. The artifact does not turn audit readiness into a detector-benefit claim.

## EVIDENCE

- `honest_verdict`: `complete_null_v740_triviaplus_extraction`; `verdict_class`: `null`
- `intended_count`: `190`; `completed_count`: `190`
- `usable_rows`: `135`; `source_windows_over_budget`: `55`
- `matches_release`: `true`; `invalid_count`: `0`
- `status`: `descriptive_exploratory`; `selected_winner`: `false`
- `evaluated_this_invocation`: `false`
- `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`
- `Numeric1 iff original inputs and owned checks qualify the complete honest census, including zero usable rows.`

## RECOMMENDATION
KEEP

## experiment_8621_v740_lean_resident_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports execution qualification without claiming comparative speed, learning, or generalization benefit.

## WAS THAT CHECKED
No comparative benefit test is reported; benchmarking is explicitly skipped. The displayed acceptance gates concern execution mechanics.

## EVIDENCE
- `accepted_benchmark_claim`: `false`
- `benchmark_skipped`: `true`
- `speed_inferred`: `false`
- `generalized_learning_benefit_score`: `0`
- `Invocation-bound mechanics evidence; no independent learning or speed benefit.`

## RECOMMENDATION
KEEP

## experiment_8622_v740_lean_resident_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On the exposed answers and single host, lean_native improves throughput over legacy_native at length 16, batch 1, preserving decisions with p95 latency within 5% of baseline.

## WHAT WOULD REFUTE IT
A primary throughput confidence interval with lower bound ≤1, p95 latency ratio >1.05, or any decision mismatch would fail the stated benefit criterion. A tie against lean_python would undermine a separate claim of added benefit from native arithmetic.

## WAS THAT CHECKED
Yes. The primary comparison reports throughput bounds above 1, latency within tolerance, and full parity across ten paired repeats. These timing outcomes could have failed independently of the correctness oracle.

The cheapest direct arithmetic control, lean_python, is present and loses at length 16. The length-1 arithmetic interval includes 1, but that comparison is explicitly descriptive. Generalization scores are zero; the headline concerns execution cost on exposed inputs. The reported roster is complete, with no failures, censoring, or exclusions.

## EVIDENCE
- `lean_service_benefit_score`: `Primary lean_native/legacy_native throughput lower95>1, p95<=1.05 and exact decisions.`
- `primary_contrast`: `baseline`: `legacy_native`; `treatment`: `lean_native`; `length`: `16`; `batch`: `1`; `throughput_ratio_ci95`: `2.6021614396438366`, `2.7140499917159295`; `p95_request_latency_ratio`: `1.026794316978981`; `full_parity`: `true`.
- `arithmetic_contrast`: `baseline`: `lean_python`; `throughput_ratio_ci95`: `1.179907577230295`, `1.2584953582202063`.
- `exposure_scope`: `original exposed64 answers, repeated on one host`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`; `generalized_learning_benefit_score`: `0`.
- `completed_count`: `240`; `intended_count`: `240`; `failed_count`: `0`; `censored_count`: `0`; `excluded_count`: `0`.

## RECOMMENDATION
KEEP

## experiment_8624_v740_arc_outcome_observer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted. The narrower instrumentation assertion would fail if enabling the observer changed actions, RNG state, receipts or termination, or classified a reset-shortened window as a completed no-progress horizon.

## WAS THAT CHECKED
Yes, for instrumentation: the displayed parity rows check actions, effects, RNG, receipts, termination and horizon status across full-horizon, early-levelup and reset fixtures. These checks could fail. Historical short windows remain censored rather than supplying negative outcomes or arm credit. This establishes a mechanical receipt; it does not test comparative benefit.

## EVIDENCE

- `methodology_note`: `Horizon accounting qualifies instrumentation only. No policy, learner, deadline behavior, game solver, arm ranking or solve count changes.`
- `honest_verdict`: `complete_null_arc_outcome_observer`
- `arc_observer_ready_score`: `Numeric0/1: reachable action-preserving instrumentation with all owned checks passed.`
- `promotion_ready_score`: `Numeric0: no policy or solve promotion is measured.`
- `full_horizon`, `early_levelup`, `reset`
- `equality`: `actions`, `effects`, `horizon_status`, `numpy_rng`, `receipt`, `rng`, `termination`; `true`
- `horizon_status`: `censored`

## RECOMMENDATION
KEEP

## experiment_8625_v740_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to falsify. The narrower accounting statement that compatible physical execution remains unauthenticated would be contradicted by an authenticated compatible execution receipt in its own records.

## WAS THAT CHECKED
Yes, as an evidence-availability check: `gate_check_summary` records the physical-execution prerequisite as failed. No accelerator performance comparison was conducted. The positive boundary score explicitly concerns accounting readiness, while acceleration estimates remain conditional and physical readiness remains zero.

## EVIDENCE

- `exposure_scope`: `exposed_engineering_accounting_only`
- `rows`: `Each unit is an operation-to-board mapping, not an independent question or device measurement.`
- `hardware_boundary_ready_score`: `One requires authenticated mapping, qualified cost, current authority and passing owned checks. It makes no acceleration claim.`
- `physical_hardware_ready_score`: `0`
- `measured_acceleration`: `null`
- `conditional`: `true`

## RECOMMENDATION
KEEP

## experiment_8626_v740_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V740 remains blocked on qualified learning while retaining a qualified lean-service throughput benefit over the native legacy baseline.

## WHAT WOULD REFUTE IT
A matched native legacy comparator tying or beating the lean service on complete-session throughput, or an advantage obtained through unequal outputs or omitted costs, would defeat the benefit claim. Qualified current learning satisfying the failed prerequisites would contradict the learning-blocked assessment.

## WAS THAT CHECKED
Yes, within the reported scope. The primary cost comparison uses a serious native baseline, ten paired repeats, declared full parity, and a throughput-ratio interval entirely above one. The slower prototype measurements are explicitly unqualified; the positive score is defined for inclusive session costs.

Failed execution prerequisites and absent current learning evidence retain the learning block. Historical unqualified losses receive no learning or generalization credit. Administrative completion counts terminal dispositions, not successful experiments. Oracle-based evidence replay is not presented as proof of the verifier’s added scientific value.

## EVIDENCE
- `baseline`: `legacy_native`; `treatment`: `lean_native`
- `paired_repeats`: `10`; `full_parity`: `true`
- `throughput_ratio_ci95`: `2.6021614396438366`, `2.7140499917159295`
- `complete_cost_benefit_score`: `1`
- `prototype_operation_subtotal_not_complete_session_cost`; `qualified_cost`: `false`
- `qualified_learning`: `false`; `learning_execution_ready_score`: `0`
- `conditional_exposed_learning_benefit_score`: `0`; `generalized_learning_benefit_score`: `0`
- `All twelve scheduling dispositions have terminal accounting; this does not count successful experiments.`

## RECOMMENDATION
KEEP

## experiment_8618_causal_brier_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No learning outcome or comparative value is asserted. This artifact records a prerequisite failure that blocked execution.

## WAS THAT CHECKED
The prerequisites were checked: execution readiness was 0 against required 1; methods readiness passed. There is no experimental result to assess for refutation.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `learning_execution_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8623_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental success or comparative value claim is made. This artifact only records a failed prerequisite gate.

## WAS THAT CHECKED
Yes, the prerequisite gates were evaluated: runtime readiness was 0 against required 1; canary input readiness passed. The artifact reports the experiment as blocked, without claiming experimental success.

## EVIDENCE
`blocked_gate_check_v1`, `blocked`, `blocked_gate_check_failed`, `runtime_repair_ready_score`, `actual=0 == expected=1`, `canary_inputs_ready_score`, `actual=1 == expected=1`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

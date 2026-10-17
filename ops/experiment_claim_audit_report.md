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
| NO_CLAIM | 5 |

## experiment_8593_v738_resident_provenance_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. The operational qualification would fail if an invalid descriptor were accepted, corrected provenance still prevented publication, or required resident tests failed.

## WAS THAT CHECKED
Yes, for operational qualification. The classifier reproduction rows show original publication refused and corrected publication accepted. All three descriptor mutations exited with code 1, and the exact test union reports success. These are execution receipts; the artifact assigns no speed, learning, or generalization credit.

## EVIDENCE

- `complete_null_v738_resident_provenance_qualification`
- `original exposed head; provenance qualification only`
- `No language model is loaded. The original small numerical energy head is evaluated by unchanged resident tests. Qualification earns no speed or learning credit.`
- `absent_descriptor`, `wrong_head`, `inconsistent_scaler`
- `historical_union_preserved`

## RECOMMENDATION
KEEP

## experiment_8594_causal_brier_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt asserting no experimental outcome or comparative benefit.

## WAS THAT CHECKED
No learning outcome was tested. The prerequisite checks in gates_evaluated show that learning readiness failed: actual 0 versus required 1. Execution was blocked before measurement.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `learning_execution_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8596_v738_resident_complete_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On one host at length 16/batch 1, resident native execution improves throughput over resident Python within the latency and parity gates, with a separate benefit from Python residency.

## WHAT WOULD REFUTE IT
Resident Python tying or beating native execution on complete-session throughput, a native throughput confidence bound failing to exceed 1, latency exceeding the 1.05 ratio gate, or any decision mismatch would defeat the registered native-benefit criterion. The lifecycle claim faces the corresponding comparison against cold Python.

## WAS THAT CHECKED
Yes. The primary contrast uses resident Python, a serious baseline that controls for residency. Its throughput interval exceeds 1, latency passes, and parity holds. The separate lifecycle contrast also passes. These timing outcomes could have failed; oracle-defined correctness does not guarantee faster execution or establish a separate verifier-quality benefit.

The artifact discloses exposed inputs and assigns zero generalization scores. It reports all 240 cells completed, with zero failures, censoring or exclusions. The displayed rows show no false validity flag or demonstrated wrong-key error.

## EVIDENCE
- `primary_contrast`: `baseline` = `resident_python`; `treatment` = `resident_native`; `throughput_ratio_ci95` endpoints = `1.0740525244779107`, `1.105777420052135`; `p95_request_latency_ratio` = `0.836081817018632`; `full_parity` = `true`.
- `lifecycle_contrast`: `baseline` = `cold_python`; `treatment` = `resident_python`; `throughput_ratio_ci95` endpoints = `11.462653466221406`, `11.702661328052224`; `p95_request_latency_ratio` = `0.903149598723823`.
- `methodology_note`: `Ten paired repeats on one host are timing evidence, not independent machines. Lifecycle benefit is separate from native arithmetic. No warmup or answer/feature cache.`
- `generalized_learning_benefit_score` = `0`; `independent_generalization_score` = `0`; `nfr01_score` = `0`.
- `completed_count` = `240`; `failed_count` = `0`; `censored_count` = `0`; `excluded_count` = `0`.

## RECOMMENDATION
KEEP

## experiment_8597_v738_medexpert_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no method-value or generalization claim to falsify. Authenticated sentence rows acquired during this invocation would contradict the narrower report that acquisition remained blocked.

## WAS THAT CHECKED
Yes, for acquisition status: gate_check_summary records assisted access returning 401 and both sentence-byte checks returning null. Both inventory rows report zero acquired rows. This is a blocked evidence audit; it asserts no comparative evaluation result.

## EVIDENCE

- `honest_verdict`: `complete_blocked_medexpert_release_unavailable`
- `release_rows_acquired`: `false`
- `completed_count`: `0`
- `No detector or head is evaluated.`
- `One means completed bounded audit, including explicit release blockage. It does not permit evaluation.`

## RECOMMENDATION
KEEP

## experiment_8598_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental value claim to falsify. The blocking receipt would be contradicted if its own gate rows showed every prerequisite passed.

## WAS THAT CHECKED
Yes, for the blocking status. In `gates_evaluated`, the runtime readiness gate requires 1, observes 0, and records failure. This supports the blocking receipt; it makes no claim about the method’s effectiveness.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `runtime_repair_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8599_v738_arc_outcome_identifiability.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The imported panel provides insufficient resolved, cross-game intervention evidence to support an arm ranking or priority update.

## WHAT WOULD REFUTE IT
Resolved outcome rows meeting the stated support requirements—three overlapping training games and five resolved observations per shared mode/arm/game cell—and supporting an eligible arm ranking would contradict the reported support deficit.

## WAS THAT CHECKED
Yes, through intervention counts, censoring records, and leave-one-game-out eligibility checks. The displayed support rows have zero resolved outcomes; displayed holdouts have no overlapping training games or ranking. All eight paired comparisons are censored and show zero peak-level differences. These checks support the limited insufficiency conclusion; the artifact claims no demonstrated effectiveness or generalization gain.

## EVIDENCE

- `honest_verdict`: `complete_null_arc_outcome_identifiability`
- `no_supported_update`: `true`
- `resolved_count`: `0`
- `minimum_overlapping_games`: `3`
- `minimum_resolved_per_shared_cell`: `5`
- `overlapping_training_game_count`: `0`
- `eligible`: `false`; `fitted`: `false`; `ranking`: `null`
- `right_censored`: `true`
- `methodology_note`: `Observed states and policy changes are associations. Censored windows provide no failure rate or arm ranking. No new game, solve, hidden-game score or generalization gain.`

## RECOMMENDATION
KEEP

## experiment_8600_v738_resident_hardware_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Resident Python improves the measured cold-start lifecycle, while compatible hardware acceleration remains unqualified and acquisition is not recommended.

## WHAT WOULD REFUTE IT
Resident execution failing parity or showing no throughput improvement would undermine the lifecycle benefit. Authenticated compatible device execution with parity and complete costs beating resident Python would contradict the hardware qualification boundary.

## WAS THAT CHECKED
Yes, within the stated scope. The lifecycle comparison reports full parity and a throughput confidence interval above one. Per-board qualification checks record missing execution evidence as null and fail the corresponding gates. Arithmetic ceilings remain conditional.

The artifact checks whether acceleration is established; it does not claim to have experimentally ruled out hardware benefit. Resident Python is the relevant serious comparator for any future device claim.

## EVIDENCE

- `baseline`: `cold_python`; `treatment`: `resident_python`; `contrast`: `lifecycle`.
- `full_parity`: `true`; `throughput_ratio_ci95`: `[11.462653466221406, 11.702661328052224]`.
- `authenticated_compatible_resident_execution`; `expected`: `true`; `observed`: `null`; `passed`: `false`.
- `compatible_physical_execution`: `false`.
- `conditional`: `true`; `measured_acceleration`: `null`; `independent_count`: `0`.
- `purchase_authorized`: `false`; `recommended`: `false`.

## RECOMMENDATION
KEEP

## experiment_8601_v738_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No method-value claim is asserted to falsify. Missing or misclassified scheduling slots could contradict administrative accounting, but that is separate from experimental success.

## WAS THAT CHECKED
No comparative efficacy test is asserted. The blocked-input rows retain failed prerequisites and unexecuted producers; administrative completion is explicitly separated from successful studies. The oracle designation therefore does not expose a circular value claim.

## EVIDENCE
- `administrative_complete`: `true`
- `successful_studies_not_implied`: `true`
- `qualified_learning_benefit`: `false`
- `qualified_arc_panel`: `false`
- `producer_executed`: `false`
- `Stored ARC replay checks execution mechanics and grants no hidden-game capability benefit.`

## RECOMMENDATION
KEEP

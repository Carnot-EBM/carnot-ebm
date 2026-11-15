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
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 7 |

## experiment_9083_v775_prospective_custody.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative effectiveness claim to falsify. The limited custody qualification would be contradicted by a successful worker bypass; the reported collection block would be contradicted by proposals or collection proceeding despite unmet prerequisites.

## WAS THAT CHECKED
Yes, within the displayed fixture: bypass controls report passing, enforcement records denied access, and proposals and collection attempts remain zero. The artifact reports blocked readiness, with fixture qualification recorded separately.

The oracle supplies no claimed added value, and no model or generalization success is asserted. A rival baseline is therefore unnecessary. The false acceptance gates are reflected in blocked readiness; the displayed rows establish no validity-filter or wrong-key contradiction.

Review was limited to supplied text; elided rows and referenced receipts remain uninspected.

## EVIDENCE

- `honest_verdict`: `complete_blocked_historical_attempt_topology_coverage`
- `claim_scope`: `Finite supervised custody qualification only. Exact graph truth is an oracle; any graph success would be circular_positive.`
- `custody_ready_score`: `0`; `fixture_custody_ready_score`: `1`
- `paths_denied`: `true`; `symlink_denied`: `true`; `fd_denied`: `true`
- `proposal_count`: `0`; `construction_started`: `false`
- `model_invoked`: `false`; `independent_generalization_score`: `0`
- `exposure_scope`: `Private scripted fixture and historical metadata only. Human LPHB remains exposed development; no independent/generalized claim.`

## RECOMMENDATION
KEEP

## experiment_9084_v775_human_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. If benefit were claimed, valid results showing the method tying or losing to text-length logistic or regularized linear controls would refute added value.

## WAS THAT CHECKED
No comparison was performed. The artifact explicitly reports a blocked, protocol-only result, with extraction and training unperformed. It does not present inventory completion as detector performance or generalization.

## EVIDENCE
- `claim_scope`: `Original exposed LPHB inventory and frozen embedding comparison protocol only; no detector benefit, independent source qualification or MDHJudgments repair.`
- `honest_verdict`: `complete_blocked_historical_producer_hash`
- `head_comparison_status`: `protocol_only_not_measured`
- `extraction_performed`: `false`
- `head_training_performed`: `false`
- `per_arm_metrics`: `null`

## RECOMMENDATION
KEEP

No rerun or external inspection; this assessment is limited to the supplied artifact.

## experiment_9085_collect_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental success or comparative value claim is asserted. This artifact records a blocked prerequisite check.

## WAS THAT CHECKED
Yes—the blocking condition was checked in both `gates_evaluated` entries: each required 1, observed 0, and failed. Experimental performance was not assessed; this receipt supports only the gate-block report.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `custody_ready_score` and `roster_ready_score`; both entries have `expected`: `1`, `actual`: `0`, and `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_9086_collect_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records blocked prerequisites without claiming successful collection or comparative value.

## WAS THAT CHECKED
Yes, the blocking condition was checked in `gates_evaluated`: both prerequisites required 1, observed 0, and failed. No experimental outcome was tested.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- Both `custody_ready_score` and `roster_ready_score`: `expected`: `1`, `actual`: `0`, `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9088_human_embeddings.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental claim is asserted. This artifact records a prerequisite failure, not successful extraction or comparative value.

## WAS THAT CHECKED
No experimental claim was tested. The prerequisite gate was checked in `gates_evaluated` and failed: observed readiness was 0 against required readiness of 1.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `human_methods_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

Upstream evidence was not inspected; this receipt establishes no experimental outcome.

## experiment_9093_v775_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an accounting inventory, with no comparative claim about service cost, verifier value, speedup, or generalization.

## WAS THAT CHECKED
No comparative hypothesis was tested. The scientific-service completeness gate fails, costs per request remain unavailable, and service ratios are withheld. The oracle and exposure disclosures therefore do not undermine an asserted benefit.

## EVIDENCE

- `claim_scope`: `Authenticated accounting inventory only; no scientific service ratio or hardware speedup.`
- `honest_verdict`: `complete_null_inventory_unavailable_service`
- `complete_service_ratios`: `[]`
- `scientific_service_complete`: `passed`: `false`
- `Neither complete scientific costs nor request denominators are authenticated.`

## RECOMMENDATION
KEEP

Not checked: elided rows or external receipts. This inventory supports no scientific performance inference.

## experiment_9094_v775_susceptibility.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The CPU equation diagnostic reproduces suppression of hidden-layer susceptibility at the cancellation setting despite equal binary energies.

## WHAT WOULD REFUTE IT
A stable equilibrium at alpha 0.5 showing nonzero first-order hidden response to output forcing, or independently computed derivatives and forced-equilibrium displacements disagreeing with the analytic susceptibility prediction.

## WAS THAT CHECKED
Yes, within the numerical diagnostic’s scope. Paired rows show zero midpoint response and nonzero endpoint responses. Derivative rows compare analytic derivatives with finite differences; finite-perturbation rows compare solved equilibria with predicted linear displacement as forcing shrinks. Those numerical comparisons could fail, even though binary-energy equality alone cannot establish the mechanism.

The oracle flag concerns exact graph BFS; the headline claims no verifier advantage. No model generalization or computational advantage is claimed, so a competitive baseline is unnecessary. The displayed flags do not invalidate these numerical rows, and their cited metric fields exist.

No experiment was rerun. Elided rows limit direct inspection, particularly of midpoint finite-perturbation results.

## EVIDENCE

- `claim_scope`: `CPU numerical equation diagnostic only; no generator, inference-speed, joule or hardware benefit.`
- `arm`: `alpha_0_0.5_1`; `hidden_response_norms`: `0.24959953591282544`, `0.0`, `0.24959953591282544`.
- `mixed_curvature_error`: `7.331371304486467e-08`; `derivative_atol`: `1e-06`.
- `error_per_beta`: `6.474635825458494e-10`, `6.474635018019086e-08`, `6.474809844277076e-06`.
- `floating_point_floor`: `false`; `independent_generalization_score`: `0`; `model_invoked`: `false`.

## RECOMMENDATION
KEEP

## experiment_9095_v775_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact aggregates administrative outcomes without asserting verifier benefit or independent generalization. Oracle-defined wins would not establish added value, but this artifact explicitly labels them circular.

## WAS THAT CHECKED
Yes, for the administrative boundary: the displayed custody gate observes zero against a required one and fails; the overall verdict remains blocked. Exposed human data is explicitly identified as development data. No comparative scientific success is asserted.

## EVIDENCE
- `honest_verdict`: `complete_blocked_declared_gate`
- `claim_scope`: `Fourteen administrative outcomes; graph labels are an executable oracle and graph wins are circular_positive. No new detector, online or general verifier benefit.`
- `exposure_scope`: `LPHB is exposed development; no independent human evaluation or generalization. Historical graph access and source authority remain blocked.`
- `artifact_field`: `custody_ready_score`; `expected`: `1`; `observed`: `0`; `passed`: `false`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

Not checked: upstream artifacts or elided rows. This assessment covers the supplied text only.

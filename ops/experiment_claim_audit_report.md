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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 5 |

## experiment_7672_v669_bound_relations.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7673_v669_fresh_relation_cohort.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7674_relation_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is a pre-execution gate check receipt recording that upstream gate criteria were unsatisfied, making no empirical, comparative, or performance claim.

## WAS THAT CHECKED
No; execution was halted at the pre-gate layer prior to running any experiment or comparative evaluation.

## EVIDENCE
`schema`
`"blocked_gate_check_v1"`
`status`
`"blocked"`
`honest_verdict`
`"blocked_gate_check_failed"`
`duration_s`
`0.0`
`blocked_at_layer`
`"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7676_v669_qwen_quote_relations.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7679_v669_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Production promotion is blocked because required independent static and online evidence is absent or unconfirmed, establishing no independent quality or efficiency benefit.

## WHAT WOULD REFUTE IT
The observation that required static and online evidence artifacts (`results/experiment_7675_v669_static_decision.json`, `results/experiment_7677_v669_online_learning.json`, and `results/experiment_7678_v669_continuous_learning.json`) were present with valid producer contracts, passing all eight acceptance gates with positive confirmed quality and efficiency scores.

## WAS THAT CHECKED
Yes. The audit verified producer existence across all slots under `preconditions_checked.producer_classes`, evaluated file custody in `source_artifact_hashes.missing_evidence`, audited upstream contracts in `gate_check_summary`, and evaluated all eight gates in `acceptance_gate_results`. Refutation was given a real chance to occur but did not; missing artifacts, failed contract checks, and failed acceptance gates confirmed the blocked verdict.

## EVIDENCE
`honest_verdict`
`complete_blocked_required_static_online_evidence`
`verdict_class`
`blocked`
`production_promotion`
`false`
`activation`
`claim_limit`
`No independent static or online benefit confirmed.`
`independent_audit_complete_score`
`0`
`independent_efficiency_confirmed_score`
`independent_quality_confirmed_score`
`independent_quality_confirmed_gates`
`failed_contract_checks`
`11`
`missing_evidence`
`results/experiment_7675_v669_static_decision.json`
`results/experiment_7677_v669_online_learning.json`
`results/experiment_7678_v669_continuous_learning.json`
`passed`

## RECOMMENDATION
KEEP

## experiment_7680_v669_arc_probe_protocol.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The ARC probe protocol is ready and achieves a positive verdict on an ARC generalization task.

## WHAT WOULD REFUTE IT
The claim of probe protocol utility and generalization readiness would be refuted by observing:
1. The guided probe arm failing to discriminate better than a trivial heuristic comparator arm (e.g., tying or underperforming a simple novelty baseline on admitted and rejected probes).
2. Evaluation on unseen, independent ARC game instances exposing false goal confirmations or invalid probe decisions.
3. An independent ground-truth environment or external oracle refuting the verifier's terminal state evaluations.

## WAS THAT CHECKED
No. Refutation was not given a real chance to happen:
1. The protocol was tested solely on a scripted CPU test fixture without loading or invoking any model (`model_invoked` is false), with zero independent hidden games tested (`independent_hidden_games` is 0, `new_hidden_game_wins` is 0).
2. The verifier was designated as its own oracle (`verifier_is_oracle` is true), guaranteeing zero false terminal confirmations by construction across all scripted ambiguous goals.
3. When evaluated against the rival arm, the `guided` arm exactly tied the `novelty` arm on admitted and rejected probes, demonstrating no added value over the simple baseline.
4. All operational acceptance gates assessing real-world generalization (`decision_utility`, `freshness`, `efficiency`, `probability`, `retention`) explicitly failed.

## EVIDENCE
- `honest_verdict`: `"complete_circular_positive_probe_fixture"`
- `verdict_class`: `"circular_positive"`
- `verifier_is_oracle`: `true`
- `arc_generalization_task`: `true`
- `arc_probe_protocol_ready_score`: `1`
- `inference_substrate`: `"cpu_scripted_scored_wrapper_no_model_load"`
- `inference_substrate_class`: `"no_model_load"`
- `model_invoked`: `false`
- `MODEL_SPECS`: `[]`
- `prior_exposure`: `"scripted development proxy; no hidden-game inference"`
- `solve_provenance`: `"development_proxy"`
- `historical_model_provenance`: `"V668 fixture is inherited; no current model calls"`
- `decision_utility`: `{"admitted_probes": 192, "new_hidden_game_wins": 0, "passed": false}`
- `freshness`: `{"live_producer_available": false, "passed": false}`
- `probability`: `{"independent_hidden_games": 0, "passed": false}`
- `efficiency`: `{"live_action_cost_measured": false, "passed": false}`
- `retention`: `{"live_retest_groups": 0, "passed": false}`
- `arm`: `"novelty"` -> `raw_metrics`: `{"admitted": 2, "false_confirmation": 0, "rejected": 2, "sdk_progress": 0}`
- `arm`: `"guided"` -> `raw_metrics`: `{"admitted": 2, "false_confirmation": 0, "rejected": 2, "sdk_progress": 0}`
- `ambiguous_goals`: `448`
- `false_terminal_confirmations`: `0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7681_v669_arc_live_probes.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

## experiment_7684_v669_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6-sol
provider: openai
approval: never
sandbox: workspace-write [

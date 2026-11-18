# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

**PARTIAL RUN** -- wall-clock budget 750s exhausted after 4 of 8 artifact(s).

| verdict | count |
|---|---|
| NO_CLAIM | 4 |

## experiment_9194_v783_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an administrative task-binding and receipt-replay scaffolding artifact that makes no comparative method or generalization claim and does not even claim administrative readiness (its pre-pass verdict is complete_disqualified_v783_contract). Its non-claim and disqualified posture would be contradicted if its own fields asserted a positive outcome, contract readiness, activation or publication authorization, or scientific generalization credit (such as contract_ready_score equal to 1, activation_authorized equal to true, current_science_paper_ready equal to true, external_publication_authorized equal to true, or independent_generalization_score greater than 0) while executing zero model calls on exposed development data, having lagged archive state, or failing validation obligations. The artifact excerpt is also truncated mid-way through historical_model_provenance, so top-level rows, required_checks_passed, validation_obligations, and honest_verdict are omitted from the JSON excerpt itself, and no live model execution or rival baseline comparison is performed in this infrastructure task.

## WAS THAT CHECKED
Yes, in the top-level readiness, authorization, scope, gate, and invocation fields plus the mechanical pre-pass notes: contract readiness is 0, activation and publication booleans are false, independent generalization and effective independent group counts are 0, LLM invocations and generated tokens are 0, verifier_is_oracle is false, and the pre-pass honest_verdict is complete_disqualified_v783_contract. Per-unit rows and validation_obligations appear alphabetically after historical_model_provenance where the artifact text truncates, so row-level disqualification causes are not inspectable in this excerpt, though no positive or comparative claim depends on them.

## EVIDENCE
- `track`: `infrastructure`
- `inference_substrate_class`: `no_model_load`
- `claim_scope` / `scope`: `Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.`
- `allowed_claims`: `administrative task binding`
- `scientific_benefit`: `unmeasured; fixtures and historical receipts grant no independent benefit`
- `exposure_scope`: `exposed_development`
- `contract_ready_score`: `0`
- `independent_generalization_score`: `0`
- `effective_independent_groups`: `0`
- `current_llm_invocations`: `0`
- `generated_tokens`: `0`
- `activation_authorized`: `false`
- `current_science_paper_ready`: `false`
- `external_publication_authorized`: `false`

## RECOMMENDATION
KEEP

## experiment_9195_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or experimental claim is made because execution stopped before the titled byte-prefilter experiment ran; as a pre-gate block receipt, its own record would be contradicted only if the evaluated gate showed the upstream requirement satisfied (observed score equal to expected score with passed true) while still declaring a gate failure, or if it asserted prefilter qualification despite lacking corpus-walk execution, losslessness verification, and baseline comparisons against an unfiltered corpus walk.

## WAS THAT CHECKED
No experimental claim was checked because execution halted at the conductor pre-gate with zero runtime after evaluating the upstream contract readiness gate and observing 0 instead of 1; the artifact therefore lacks any corpus-walk rows, prefilter recall/losslessness measurements, or baseline arms, while the upstream gate failure itself was checked and recorded consistently across gates_evaluated, blocked_diagnostic_contract, and the top-level failure fields.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `duration_s`: `0.0`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_at_layer`: `conductor_pre_gate`
- `failed_upstream`: `exp9194-contract`
- `failed_field`: `contract_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `gate_check_summary`: `gate-unsat(final): 1 of 1 gate(s) failed; first failure: exp9194-contract.contract_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_9197_label_ambiguity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or empirical claim is made to refute because this artifact is a pre-gate block receipt and lacks corpus rows, candidate outputs, and comparator baselines. As a diagnostic receipt, its own record would be contradicted only if the evaluated gate entry recorded an observed value satisfying the equality check (actual 1 equal to expected 1 with passed set to true) while still reporting a failed gate check, or if it asserted ambiguity-envelope findings despite halting before execution.

## WAS THAT CHECKED
No — no empirical hypothesis, model output, or baseline comparison was checked because execution halted prior to running the experiment and the artifact lacks evaluation rows and comparator arms. Only the prerequisite readiness gate was checked in gates_evaluated at conductor_pre_gate, where upstream exp9194-contract field source_ready_score was observed as int 0 against expected int 1 (passed false), matching the top-level blocked fields.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_at_layer`: `conductor_pre_gate`
- `duration_s`: `0.0`
- `title`: `Exp9197: Freeze exact human-label ambiguity envelopes without model work`
- `gate_check_summary`: `gate-unsat(final): 1 of 1 gate(s) failed; first failure: exp9194-contract.source_ready_score (actual=0 == expected=1)`
- `gates_evaluated`: `upstream` is `exp9194-contract`, `artifact_field` is `source_ready_score`, `expected` is `1`, `actual` is `0`, `passed` is `false`

## RECOMMENDATION
KEEP

## experiment_9207_v783_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This artifact is an administrative milestone-accounting and gate-receipt aggregation making no comparative or live scientific claim. That non-claim posture would be contradicted if its own fields inferred scientific success, authorized activation or external publication, populated confirmed branch results, asserted live or retracted scientific claims, or credited historical disqualified, blocked, unmeasured, or unstarted branches despite failed upstream gates. Any actual comparative claim in downstream work still lacks a passing upstream contract gate, measured lossless corpus-prefilter parity, executed forward-validation obligations, and live model or human evaluations against serious baselines.

## WAS THAT CHECKED
Yes, in declared_gate_rows, G3, claim_scope, field_principles, current_milestone_accounting, confirmed_branch_results, collection_partition, arc_summary, ces_summary, corpus_scan_summary, forward_qualification_summary, and cost_summary. Those sections check and record that 12 of 14 milestone tasks were skipped after exp9194-contract observed a zero gate score, no retracted phrasings appear as live claims, historical ARC deterministic controls remain disqualified with no live path or added value, zero LLM invocations or scientific answers were attempted, and all readiness, activation, and publication flags remain false. Because execution stopped at the declared gates, live comparative tests (such as CES against mean/max entropy and probability-length comparators, full-corpus prefilter parity, and hardware latency probes) were not executed.

## EVIDENCE
- `honest_verdict`: `complete_blocked_declared_gate`
- `allowed_claims`: `fourteen terminal administrative outcomes`, `authenticated historical failures`, `unmeasured human science, secondary CES and two label sensitivity branches`
- `field_principles`: `Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim.`, `Exact retained operands and executed receipts support accounting; absent science stays blocked`
- `detail`: `no retracted phrasings asserted as live claims`
- `confirmed_branch_results`: `[]`
- `scientific_success_inferred`: `false`
- `activation_authorized`: `false`
- `current_science_paper_ready`: `false`
- `external_publication_authorized`: `false`
- `independent_generalization_score`: `0`
- `current_llm_invocations`: `0`
- `attempted_scientific_answers`: `0`
- `contract_ready_score`: `0`
- `value_added`: `false`
- `verdict_class`: `disqualified`, `blocked`
- `status`: `blocked`, `unmeasured`, `unstarted`

## RECOMMENDATION
KEEP

## experiment_9208_v784_contract.json

**NOT_REVIEWED_BUDGET**

## experiment_9209_corpus_prefilter.json

**NOT_REVIEWED_BUDGET**

## experiment_9211_v784_label_ambiguity.json

**NOT_REVIEWED_BUDGET**

## experiment_9221_v784_capstone.json

**NOT_REVIEWED_BUDGET**

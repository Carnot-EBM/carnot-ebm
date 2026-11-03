"""REQ-REPORT-8768: qualification describes a frozen constructed test, not natural transfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import evidence_set_methods_8768 as m
from carnot.reporting import evidence_set_analysis_8768 as analysis
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS
from carnot.reporting.v746_retained_evidence import load, reference

Json = dict[str, Any]


def build(work: Json, validation: Json, output: Path) -> Json:
    """Execution checks gate readiness; constructed comparative gains remain circular."""
    inputs, labels, decisions = [
        load(work[k])["rows"]
        for k in ["corpus_reference", "oracle_reference", "decisions_reference"]
    ]
    rows = analysis.score(inputs, decisions, labels)
    summary = analysis.summarize(rows)
    checks = bool(validation["receipts"]) and all(r["passed"] for r in validation["receipts"])
    checks = checks and validation["union"]["passed"] and validation["coverage"]["complete"]
    gates = dict(
        preconditions=work["preconditions_checked"]["passed"],
        complete_rows=len(rows) == 6480,
        independent_row_audit=analysis.audit(inputs, decisions),
        disjoint_splits=work["disjoint_splits"],
        truth_controls=summary["truth_controls_passed"],
        scoped_validation=bool(checks),
    )
    qualified = all(gates.values()) and not work["private_control"]
    verdict = (
        "null"
        if work["private_control"]
        else "blocked"
        if not gates["preconditions"]
        else (
            "disqualified"
            if not qualified
            else "circular_positive"
            if summary["scientific_promotion"]
            else "null"
        )
    )
    failures = [
        dict(
            upstream=m.TASK,
            **work["method_reference"],
            check=k,
            field=k,
            operator="==",
            expected=True,
            observed=False,
            passed=False,
        )
        for k, passed in gates.items()
        if not passed and not work["private_control"]
    ]
    value: Json = dict(
        experiment_id=8768,
        task_id=m.TASK,
        milestone="2026.11.751",
        run_date="20261103",
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v751_evidence_set_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=rows,
        intended_count=6480,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=120,
        independent_count_scope="120 evaluation worlds; development 60; 2160 views are not independent cases",
        sample_size_budget=dict(
            development_worlds=60,
            evaluation_worlds=120,
            units=2160,
            decision_rows=6480,
            evaluation_rows=4320,
            repeats_multiply_evidence=False,
            natural_questions=0,
            v750_regression_rows_in_N=0,
        ),
        verifier_is_oracle=False,
        oracle_dependent_benchmark=True,
        exposure_scope="Implementation agent inspected generators and development controls; prepublication tests also instantiate draft and final evaluation cases. Final evaluation labels sealed separately before comparison; no claimed blindness. All records are constructed.",
        independent_generalization_score=0,
        evidence_set_methods_ready_score=int(qualified),
        scientific_promotion_score=int(qualified and summary["scientific_promotion"]),
        acceptance_gates=gates,
        required_checks_passed=bool(checks),
        flagged_adversarial=verdict == "disqualified",
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        retained_child_custody=validation["union"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=m.SEED,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            work[k]
            for k in [
                "method_reference",
                "corpus_reference",
                "oracle_reference",
                "source_records_reference",
                "decisions_reference",
                "validation_reference",
                "nodes_reference",
            ]
        ]
        + validation.get("raw_references", []),
        cited_upstream_artifacts=work["references"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            owned=m.OWNED,
            tests=m.TESTS,
            consumers=m.CONSUMERS,
            named_e2es=["E2E-018 private CLI", "E2E-021 private CLI"],
            full_suite_pass_claimed=False,
        ),
        method_reference=work["method_reference"],
        corpus_reference=work["corpus_reference"],
        oracle_reference=work["oracle_reference"],
        decisions_reference=work["decisions_reference"],
        split_rows=[
            dict(
                split=s,
                worlds=n,
                units=n * 12,
                decision_rows=n * 36,
                role="implementation_only" if s == "development" else "comparative_evaluation",
            )
            for s, n in [("development", 60), ("evaluation", 120)]
        ],
        leakage_controls=dict(
            strict_feature_allowlist=["facts", "query"],
            truth_labels_in_verifier_inputs=False,
            separate_label_seal=True,
            evaluator_process_reads_oracle=False,
            label_access_os_sandbox=False,
            decision_rules_frozen_before_comparison=True,
            exposure_honestly_disclosed=True,
            all_units_executed=True,
        ),
        arm_specs=[
            dict(
                arm=a,
                method="signed relational join with conflict escalation"
                if a == "evidence_set"
                else "unchanged V749 truth per premise; unchanged V750 local reduction"
                if a == "local_record"
                else "unchanged V749 joint symbolic truth",
                maximum_invocations=5,
            )
            for a in m.ARMS
        ],
        statistical_protocol=summary["statistical_protocol"],
        oracle_access_rows=work["oracle_access_rows"],
        historical_regression_reference=work["historical_regression_reference"],
        publication_output=str(output.absolute()),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        validation_reference=work["validation_reference"],
        methodology_note="Constructed signed relational records only. Independent finite Boolean-world enumeration supplies three-way labels. Missing evidence has both satisfiable alternatives; contradictory sets have no models and escalate. The existing joint comparator and local-record baseline are unchanged. Exact outcomes on deterministic designed controls are not population accuracy estimates. Bonferroni paired world bootstrap uses 10000 draws and 120 evaluation clusters; views and development worlds do not enlarge N. Any comparative gain is oracle-dependent circular_positive, not natural extraction, learned verification, independent semantics or hidden-game transfer. No LLM or numerical training executes.",
        **{
            k: v
            for k, v in summary.items()
            if k not in ["independent_count", "statistical_protocol"]
        },
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind actual invocation evidence and constructed scope; qualification does not establish natural usefulness.",
    )
    value["field_principles"].update(
        evidence_set_methods_ready_score="One only after frozen corpus, independent truth controls and every owned check qualify.",
        verifier_is_oracle="The scorer never reads scoring labels; benchmark conclusions still depend on constructed oracle semantics.",
        independent_count="Only the 120 evaluation world clusters define the primary estimand.",
        model_invocation_counts="No model is loaded or called by this task.",
        repository_health_reference="A bounded diagnostic is neither scoped acceptance nor a full-suite pass.",
        gate_check_summary="Preserve exact comparison operands when readiness is suppressed.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value

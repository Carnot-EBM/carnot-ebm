"""REQ-REPORT-8555: terminal accounting preserves every validator finding.

A failure artifact may be published to explain what happened. That publication
receipt does not qualify the online adapter or clear a historical quarantine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import canonical_update_8555 as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

Json = dict[str, Any]
REQUIRED = {
    "owned_tests",
    "coverage_report",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "cold",
    "missing",
    "owned_failure",
    "rehashed_tamper",
}


def build(work: Json, output: Path) -> Json:
    """Reduce exact primitive bytes and require all owned gates before granting readiness."""
    raw = Path(work["raw"])
    events = json.loads(Path(work["events"]["path"]).read_bytes())
    records, metrics, visible, relations = e.canonical(events)
    controls = e.controls()
    control_records, _, control_visibility, control_relations = e.canonical(controls)
    receipts = work["receipts"]
    gates = dict(
        historical=work["historical"].get("historical_learning_validation_ready_score") == 1,
        original_stream_consumer=work["upstream_consumer"]["passed"],
        numerical=bool(work["numerical"]) and all(r["passed"] for r in work["numerical"]),
        restart=work["restart_passed"],
        original_terminal_union=work["union"]["passed"],
        projection=visible["passed"] and control_visibility["passed"],
        negative_validator_control=work["negative_control"]["passed"],
        required_commands=REQUIRED <= {r["name"] for r in receipts if r["passed"]},
        owned_children=all(r["passed"] for r in receipts),
        unchanged_validators_clean=not work["findings"],
        production=not work["private_control"],
    )
    required = all(gates.values()) and not work["failures"]
    owned_failed = bool(work["findings"]) or any(not r["passed"] for r in receipts)
    owned_failed = owned_failed or any(r.get("owned_required_check") for r in work["failures"])
    owned_failed = owned_failed or any(
        not gates[k]
        for k in gates
        if k not in {"historical", "original_stream_consumer", "production"}
    )
    kind = (
        "disqualified"
        if owned_failed or work["private_control"]
        else "blocked"
        if work["failures"] or not gates["historical"] or not gates["original_stream_consumer"]
        else "circular_positive"
    )
    failures = list(work["failures"])
    failures += [
        e.gate(raw / "measurement.json", name, True, observed)
        for name, observed in gates.items()
        if not observed
    ]
    rows = [
        dict(
            unit_id=r["namespace"] + ":" + r["event_id"],
            family="exposed_historical_transition",
            arm=r.get("arm", "constructed"),
            seed=r.get("seed"),
            status="completed",
            absolute_metric=e.reduce([r])["changed_coordinate_count"],
            missing_reason=None,
            censored=False,
            excluded=False,
        )
        for r in events
    ]
    log_refs = [
        dict(path=r[s + "_path"], sha256=r[s + "_sha256"])
        for r in receipts
        for s in ("stdout", "stderr")
        if r.get(s + "_sha256")
    ]
    value: Json = dict(
        experiment_id=8555,
        task_id=e.TASK,
        milestone="2026.10.735",
        run_date="20261015",
        honest_verdict="complete_" + kind + "_v735_canonical_update_accounting",
        verdict_class=kind,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=work["historical"].get("historical_model_provenance", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="original transition",
            intended=len(rows),
            independent_units=0,
            repeated_events_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_history_and_constructed_controls_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=required,
        flagged_adversarial=bool(work["findings"]),
        acceptance_gates=gates,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work["findings"],
        preconditions_checked=work["preconditions"],
        duration_s=(work["end_ns"] - work["start_ns"]) / 1e9,
        phase_spans=work["phases"],
        random_seed=7358555,
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code"],
        raw_shard_hashes=[e.reference(raw / "measurement.json"), work["events"], *log_refs],
        cited_upstream_artifacts=work["sources"],
        online_adapter_ready_score=int(required),
        canonical_counter_rows=records,
        metrics=metrics,
        counter_visibility_receipt=visible,
        alias_map=e.ALIAS_MAP,
        metric_relation_rows=relations,
        event_rows_reference=work["events"],
        numerical_update_controls=work["numerical"],
        restart_receipts=[r for r in receipts if r["name"] in {"baseline", "restart", "resume"}],
        historical_validation_reference=work["historical_validation_reference"],
        historical_union_receipt=work["union"],
        upstream_consumer_receipt=work["upstream_consumer"],
        recovery_reference=work.get("recovery_reference"),
        accounting_controls=dict(
            rows=controls,
            canonical_counter_rows=control_records,
            metric_relation_rows=control_relations,
            visibility=control_visibility,
        ),
        negative_validator_control=work["negative_control"],
        primitive_reference=e.reference(raw / "measurement.json"),
        output_path=str(output),
        preserved_quarantines=[8530, 8533, 8537, 8541, 8545, 8548],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8556, 8565)
        ],
        methodology_note="No model loads. Recompute original sealed transition operands and assignment costs through shipped objective mechanics. Exact numeric counters remain visible to unchanged validators. Finite differences and real exit73/resume are constructed controls, not learning or generalization evidence. Failed qualification remains disqualified; publication of a failure is not scientific acceptance.",
    )
    value["field_principles"] = {
        k: "Bind this invocation and its immutable primitive operands; qualification never establishes independent benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(e.DEFINITIONS)
    value["reproducibility_checksum"] = e.checksum(value)
    return value

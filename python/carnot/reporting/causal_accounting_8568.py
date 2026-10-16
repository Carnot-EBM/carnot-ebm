"""REQ-REPORT-8568: bind fresh accounting to original transitions and current checks.

The original numerical methods remain authoritative. This adapter separates
current qualification from historical diagnostics and from beneficial learning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import canonical_update_8555 as old
from carnot.reporting import count_semantics_8566 as sealed
from carnot.reporting import count_validator_8567 as validator
from carnot.reporting.count_semantics_evidence_8566 import authenticate as failed_original, read
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8568_v736_causal_accounting_qualification"
TASK = "exp8568-causal-accounting-qualification"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
TESTS = [
    "tests/python/test_causal_accounting_8568.py",
    "tests/python/test_causal_accounting_runner_8568.py",
]
OWNED = [
    f"python/carnot/reporting/causal_accounting{s}_8568.py" for s in ("", "_runner", "_validation")
] + [CLI]
VALIDATOR_PIN = "sha256:ca6b5a53c8f4e0de8eb2f3dfd01ec1d08c462d50eac19dde904275c9c8b17a86"


def authority(root: Path) -> tuple[Json, list[Json], list[Json]]:
    """Require original qualification hashes; current code cannot silently reseal a revision."""
    historical, refs, failures = old.authenticate(root)
    path = root / "results" / (validator.NAME + ".json")
    try:
        revision = old.read(dict(path=str(path), sha256=VALIDATOR_PIN))
        refs.append(old.reference(path))
        terminal = old.read(old.reference(Path(revision["terminal_validation_sidecar_path"])))
        bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if revision["count_validator_ready_score"] != 1 or not bound["report"]["passed"]:
            raise ValueError("validator_qualification")
        for ref in [
            *revision["code_config_hashes"],
            *revision["source_artifact_hashes"],
            *revision["raw_shard_hashes"],
        ]:
            current = old.reference(Path(ref["path"]))
            refs.append(current)
            if current != ref:
                failures.append(
                    old.gate(
                        Path(ref["path"]),
                        "validator_revision_hash",
                        ref["sha256"],
                        current["sha256"],
                        upstream=validator.TASK,
                    )
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            old.gate(
                path,
                "validator_revision_authentication",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=validator.TASK,
            )
        )
    return historical, refs, failures


def diagnostics(root: Path) -> list[Json]:
    """Read all saved failures before choosing the narrow current orchestration repair."""
    value = failed_original(root)
    rows = []
    for failure in (r for r in value["gate_check_summary"] if r.get("owned_required_check")):
        path = root / failure["path"]
        ref = old.reference(path)
        if ref["sha256"] != failure["sha256"]:
            raise ValueError("controller_log_hash")
        text = path.read_text()
        regression = (
            "live_parent_save" if "no such table: context" in text else "implicit_configuration"
        )
        if regression == "implicit_configuration" and "AssertionError: []" not in text:
            raise ValueError("unknown_controller_failure")
        rows.append(
            dict(
                source=ref,
                historical_exit=failure["observed"],
                regression=regression,
                current_required_success=False,
                repair="explicit coverage configuration; retain child shards; never save live parent",
            )
        )
    return rows


def proof(events: list[Json]) -> tuple[Json, Json]:
    """Keep distinct counters visible and supply only data operands to the shipped proof checker."""
    metrics = old.reduce(events)
    totals = sealed.reduce(events)
    if any(totals[k] != n for k, n in metrics.items()):
        raise ValueError("independent_count_reduction")
    table = dict(
        rows=[
            {
                k: row[k]
                for k in (
                    "namespace",
                    "event_id",
                    "before",
                    "after",
                    "assignment_indices",
                    "admitted",
                    "label_released",
                    "applied",
                )
            }
            for row in events
        ]
    )
    operations = {key: op for op, key in sealed.KEYS.items()}
    operations["intended_count"] = "rows"
    claims = {
        key: dict(
            value=n,
            operation=operations[key],
            unit=sealed.CONTRACT["operations"][operations[key]],
            table="original",
            table_sha256=canonical_hash(table),
        )
        for key, n in totals.items()
    }
    return metrics, dict(
        version=sealed.CONTRACT["version"], tables=dict(original=table), claims=claims
    )


def build(work: Json, output: Path) -> Json:
    """Read immutable operands and require every current owned check before readiness."""
    raw = Path(work["raw"])
    events = read(work["events"])
    records, metrics, visible, relations = old.canonical(events)
    _, count_proof = proof(events)
    gates = dict(
        preconditions=work["preconditions"]["passed"],
        historical=work["historical"].get("historical_learning_validation_ready_score") == 1,
        validator_revision=not work["failures"],
        original_stream=work["upstream_consumer"]["passed"],
        numeric=bool(work["numerical"]) and all(r["passed"] for r in work["numerical"]),
        restart=work["restart_passed"],
        projection=visible["passed"],
        historical_union=work["historical_union"]["passed"],
        current_owned_validation=work["owned_ready"],
        owned_children=all(r["passed"] for r in work["receipts"]),
        count_validator=not any(r["severity"] == "critical" for r in work.get("findings", [])),
    )
    owned = all(
        gates[k]
        for k in (
            "numeric",
            "restart",
            "projection",
            "current_owned_validation",
            "owned_children",
            "count_validator",
        )
    )
    ready = owned and all(gates.values())
    kind = "null" if ready else "blocked" if owned else "disqualified"
    failures = [
        *work["failures"],
        *[old.gate(raw / "final.json", k, True, v) for k, v in gates.items() if not v],
    ]
    rows = [
        dict(
            unit_id=r["namespace"] + ":" + r["event_id"],
            status="completed",
            absolute_metric=old.reduce([r])["changed_coordinate_count"],
            missing_reason=None,
            censored=False,
            excluded=False,
        )
        for r in events
    ]
    value: Json = dict(
        experiment_id=8568,
        task_id=TASK,
        milestone="2026.10.736",
        run_date="20261016",
        honest_verdict="complete_" + kind + "_v736_causal_accounting_qualification",
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
            unit="sealed original transition",
            intended=len(rows),
            independent_units=0,
            repeated_events_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_history_and_constructed_controls_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=any(r["severity"] == "critical" for r in work.get("findings", [])),
        acceptance_gates=gates,
        validation_receipts=work["receipts"],
        validation_node_union=work.get("node_union"),
        owned_coverage_reference=work.get("coverage_reference"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phases"],
        random_seed=7368568,
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code"],
        raw_shard_hashes=[
            old.reference(raw / "final.json"),
            work["events"],
            work["preliminary_reference"],
        ],
        cited_upstream_artifacts=work["sources"],
        online_adapter_ready_score=int(ready),
        historical_validation_reference=work["historical_reference"],
        validator_revision_reference=work["validator_reference"],
        canonical_counter_rows=records,
        metrics=metrics,
        alias_map=old.ALIAS_MAP,
        counter_visibility_receipt=visible,
        metric_relation_rows=relations,
        discrete_count_proof=count_proof,
        event_rows_reference=work["events"],
        numerical_update_controls=work["numerical"],
        restart_receipts=work["restart_receipts"],
        controller_failure_regressions=[
            dict(
                r,
                current_path_passed=work["owned_ready"],
                targeted_test="tests/python/test_causal_accounting_runner_8568.py::test_explicit_coverage_and_no_live_parent_save",
            )
            for r in work["diagnostics"]
        ],
        historical_union_receipt=work["historical_union"],
        accounting_controls=work["controls"],
        primitive_reference=old.reference(raw / "final.json"),
        output_path=str(output),
        preliminary_primitive_reference=work["preliminary_reference"],
        upstream_consumer_receipt=work["upstream_consumer"],
        preserved_quarantines=[8530, 8533, 8537, 8541, 8545, 8548, 8555],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8569, 8579)
        ],
        methodology_note="Fresh CPU accounting qualification only. Original numerical mechanics, tolerances and decision thresholds stay fixed. Count proofs cannot establish beneficial learning, independent generalization, source information or capability panel benefit.",
    )
    value["field_principles"] = {
        k: "Bind exact current invocation evidence; qualification establishes no beneficial learning."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(old.DEFINITIONS)
    value["reproducibility_checksum"] = old.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct sealed original events so repaired checksums cannot bless fabricated totals."""
    from carnot.reporting.causal_accounting_validation_8568 import ready

    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != old.checksum(value):
            return False
        work = old.read(value["primitive_reference"])
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if old.reference(Path(ref["path"])) != ref:
                return False
        for row in work["receipts"]:
            saved = json.loads(Path(row["stdout_path"]).with_suffix(".receipt.json").read_bytes())
            if row != saved:
                return False
            for stream in ("stdout", "stderr"):
                if old.reference(Path(row[stream + "_path"]))["sha256"] != row[stream + "_sha256"]:
                    return False
        historical, _, failures = authority(ROOT)
        if not work["preconditions"]["passed"]:
            failures.append(
                old.gate(
                    Path(work["preconditions"]["scratch_path"]),
                    "disk_memory_tools",
                    True,
                    work["preconditions"],
                )
            )
        original = old.controls() if work["private"] else old.history()[0]
        return bool(
            read(work["events"]) == original
            and work["controls"] == old.controls()
            and work["historical"] == historical
            and work["failures"] == failures
            and work["diagnostics"] == diagnostics(ROOT)
            and work["owned_ready"] == ready(work)
            and value == build(work, Path(value["output_path"]))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False

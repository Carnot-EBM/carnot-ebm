"""REQ-REPORT-8934: retain paired depth evidence without claiming detector benefit."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting import pilot_label_audit_runner_8932 as audit
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.experiment_template import ExperimentTemplate as ExperimentTemplate

Json = dict[str, Any]
ROOT = pilot.ROOT
NAME = "experiment_8934_v764_depth_control"
TASK = "exp8934-depth-control"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/depth_control_8934.py",
    "python/carnot/inference/depth_runtime_8934.py",
    "python/carnot/reporting/depth_control_runner_8934.py",
    CLI,
]
TEST = "tests/python/test_depth_control_8934.py"
INPUTS = [
    (
        "results/experiment_8931_v764_graph_source_methods.json",
        "sha256:810b976a4d7883a3a600c9501982eb882d1feb658b0a54cb86afea00320f5b57",
        "source_methods_ready_score",
        "exp8931-graph-source-methods",
    ),
    (
        "results/experiment_8932_v764_pilot_label_audit.json",
        "sha256:f98f862d034531080553eb8b6757e305b5dd81e3f21a30684fd6277f63cf1c5e",
        "pilot_labels_ready_score",
        "exp8932-pilot-label-audit",
    ),
]


def authenticate(root: Path) -> Json:
    """Reject missing original bytes or failed source gates before spending GPU time."""
    checked = methods.authenticate(root)
    failures, refs, protocol = list(checked["failures"]), [], {}
    for index, (name, pin, field, task) in enumerate(INPUTS):
        path = root / name
        ref = methods.reference(path)
        refs.append(ref)
        reason = None
        try:
            value = json.loads(path.read_text())
            actual = value.get(field)
            if field not in value:
                reason = "missing_field"
            elif type(actual) is int and actual == 0:
                reason = "observed_zero"
            elif (
                actual != 1
                or type(actual) is not int
                or ref["sha256"] != pin
                or value.get("task_id") != task
                or value.get("required_checks_passed") is not True
            ) or not (methods.replay(path) if index == 0 else audit.replay(path))["passed"]:
                reason = "rejected_producer"
            elif index == 0:
                protocol = {
                    k: value[k]
                    for k in ("graph_manifest", "feature_schema", "generator_settings", "bounds")
                }
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            reason = "missing_file" if not path.is_file() else "rejected_producer"
        if reason:
            failures.append(
                dict(
                    check="upstream_identity_and_gate",
                    upstream=task,
                    artifact_field=field,
                    expected=1,
                    observed=reason,
                    path=ref["path"],
                    hash=ref["sha256"],
                )
            )
    return dict(failures=failures, references=refs, protocol=protocol)


def ordered_units(protocol: Json) -> list[Json]:
    """Freeze equal condition order without using answers, labels or feature values."""
    units = source.scheduled(protocol, "control")
    orders = [False] * 25 + [True] * 25
    random.Random(764034).shuffle(orders)
    return [
        u
        for index in range(50)
        for u in sorted(
            units[index * 2 : index * 2 + 2], key=lambda u: u["k"], reverse=orders[index]
        )
    ]


def reduce(units: list[Json], rows: list[Json]) -> Json:
    """Retain fifty clusters and compute discordance from observed paired error labels."""
    if [r["unit_id"] for r in rows] != [u["unit_id"] for u in units]:
        raise ValueError("row_roster")
    control = source.paired_control(units, rows)
    independent_labels = all(
        sorted(node for node, distance in audit.a.distances(u).items() if distance == u["k"])
        == u["expected"]
        for u in units
    )
    control["labels_verified"] = control["labels_verified"] and independent_labels
    control["accepted"] = control["accepted"] and independent_labels
    pairs, counts = (
        [],
        dict(k2_correct_k3_wrong=0, k2_wrong_k3_correct=0, both_correct=0, both_wrong=0),
    )
    for index in range(0, 100, 2):
        pair = {units[i]["k"]: rows[i] for i in (index, index + 1)}
        complete = all(r["status"] == "completed" for r in pair.values())
        a, b = pair[2]["value"], pair[3]["value"]
        if complete and control["labels_verified"]:
            counts[
                "both_wrong"
                if a == b == 1
                else "both_correct"
                if a == b == 0
                else "k2_correct_k3_wrong"
                if a == 0
                else "k2_wrong_k3_correct"
            ] += 1
        pairs.append(
            dict(
                cluster_id=units[index]["cluster_id"],
                seed=units[index]["seed"],
                error_k2=a,
                error_k3=b,
                difference=b - a if complete else None,
                status="completed" if complete else "censored",
            )
        )
    completed = sum(r["status"] == "completed" for r in rows)
    return dict(
        rows=rows,
        paired_depth_rows=pairs,
        discordant_pairs=counts,
        error_rate_k2=control["error_k2"],
        error_rate_k3=control["error_k3"],
        paired_interval=control["paired_interval"],
        depth_control=control,
        intended_count=100,
        completed_count=completed,
        failed_count=0,
        censored_count=100 - completed,
        excluded_count=0,
        independent_count=sum(p["status"] == "completed" for p in pairs),
        null_inspection=dict(
            labels_verified=control["labels_verified"],
            ceiling_k2=control["error_k2"] is not None and control["error_k2"] >= 0.9,
            ceiling_k3=control["error_k3"] is not None and control["error_k3"] >= 0.9,
            categories={
                str(k): {
                    category: sum(r["k"] == k and r["parser_category"] == category for r in rows)
                    for category in ("correct", "valid_wrong_distance", "out_of_graph", "malformed")
                }
                for k in (2, 3)
            },
            action="Stop collection unless the operational paired source gate passes; no extra graphs or greater depth.",
        ),
    )


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Completion is separate from depth success, and an oracle never proves a moat."""
    runtime = work["runtime"]
    result = reduce(work["units"], runtime["rows"])
    owned = (
        qualified(validation)
        and not runtime["owned_failure"]
        and not validation.get("terminal_rejected", False)
        and (result["completed_count"] != 100 or result["depth_control"]["labels_verified"])
    )
    failures = work["authentication"]["failures"] + runtime["checks"]
    cls = (
        "disqualified"
        if not owned and not fixture
        else "blocked"
        if failures
        else "circular_positive"
        if result["depth_control"]["accepted"] and not fixture
        else "null"
    )
    counts = runtime["counts"]
    ready = int(cls == "circular_positive" and runtime["live"])
    signal = int(
        owned and runtime["live"] and result["depth_control"]["usable_signals"] and not fixture
    )
    value = dict(work["template"])
    value.update(
        result,
        experiment_id=8934,
        task_id=TASK,
        milestone="2026.11.764",
        run_date="20261108",
        honest_verdict="complete_" + cls + "_paired_depth_control",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        inference_substrate="live_llm_inference"
        if counts["generation_calls_attempted"]
        else "model_loading"
        if counts["model_loads_attempted"]
        else "blocked_no_run",
        inference_substrate_class="model_bounded_generation"
        if counts["generation_calls_attempted"]
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[]
        if fixture
        else [dict(hf_id=source.SETTINGS["model"], quantization="Q4_K_M", gpu=0)],
        model_invocation_counts=counts,
        numerical_training_calls=0,
        model_invoked=bool(counts["model_loads_attempted"]),
        verifier_is_oracle=True,
        exposure_scope="constructed_graph_final_answers_only",
        independent_generalization_score=0,
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"] + runtime["phase_spans"],
        preconditions_checked=dict(
            authentication=work["authentication"],
            runtime=runtime.get("preconditions_checked", {}),
            arc_suspended=True,
            generator_weights_frozen=True,
            gpu=0,
        ),
        random_seed=764034,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            methods.reference(Path(work["raw"]) / "work.json"),
            methods.reference(Path(work["raw"]) / "validation.json"),
            *work["raw_references"],
        ],
        cited_upstream_artifacts=[
            dict(ref, scope="opaque_original_producer_no_current_compute")
            for ref in work["authentication"]["references"]
        ],
        gate_check_summary=failures,
        acceptance_gates=dict(
            owned_checks=owned,
            authenticated_inputs=not work["authentication"]["failures"],
            **result["depth_control"],
        ),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        depth_source_ready_score=ready,
        signal_capture_ready_score=signal,
        model_receipt=runtime["model_receipt"],
        resource_lease=runtime["resource_lease"],
        generator_settings=runtime["settings"],
        feature_schema_hash=canonical_hash(source.FEATURE_SCHEMA),
        condition_order=[u["unit_id"] for u in work["units"]],
        fixture_only=fixture,
        flagged_adversarial=bool(validation.get("terminal_rejected")),
        repository_health=work["repository_health"],
        response_identity_controls=dict(
            stateless=True,
            raw_before_labels=True,
            actual_usage=True,
            first_response_only=True,
            retries=0,
        ),
        collection_authorized=bool(ready),
    )
    value["field_principles"] = {
        key: "Current paired graph source evidence only; no independent generalization or learned detector benefit."
        for key in value
    }
    value["field_principles"].update(
        depth_source_ready_score="Operational source admission, not statistical proof of a depth effect.",
        signal_capture_ready_score="Finite server log probabilities and actual token usage; never inferred from truth.",
        paired_interval="Frozen graph-paired normal 95-percent interval; fifty independent graphs, one hundred answers.",
        null_inspection="Label integrity and observed ceiling diagnostics; a null stops collection.",
        model_invocation_counts="Owned current ledger only; upstream compute remains opaque.",
    )
    value["model_specs"] = value["MODEL_SPECS"]
    value["schema"] = sorted(set(value) | {"reproducibility_checksum"})
    value["reproducibility_checksum"] = pilot.checksum(value)
    return value


def qualified(validation: Json) -> bool:
    """Bind measured coverage to this producer rather than the upstream methods files."""
    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == methods.CHECKS
        and all(
            r["passed"] and r.get("exit_code") == 0 and not r.get("timed_out") for r in receipts
        )
        and verify_measured(validation["coverage"], OWNED)
    )

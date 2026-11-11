"""REQ-REPORT-8948: retain paired depth evidence without claiming detector benefit."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting import matched_source_methods_8946 as matched
from carnot.reporting import depth_control_8934 as prior
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.experiment_template import ExperimentTemplate as ExperimentTemplate

Json = dict[str, Any]
ROOT = pilot.ROOT
NAME = "experiment_8948_v765_matched_depth_control"
TASK = "exp8948-matched-depth-control"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/matched_depth_control_8948.py",
    "python/carnot/inference/matched_depth_runtime_8948.py",
    "python/carnot/reporting/matched_depth_runner_8948.py",
    CLI,
]
TEST = "tests/python/test_matched_depth_control_8948.py"
INPUTS = [
    (
        "results/experiment_8946_v765_matched_source_methods.json",
        "sha256:7cd1769932adf184a01e75351d83a997b2f7d15797095b581ae54b6886777b19",
        "matched_source_ready_score",
        "exp8946-matched-source-methods",
    ),
]


def verify_references(value: Json) -> None:
    """Reject changed retained bytes without rerunning retired scientific tasks."""
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if methods.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise matched.audit.Rejected(
                dict(
                    check="retained_bytes_drift",
                    upstream=INPUTS[0][3],
                    artifact_field="sha256",
                    expected=ref["sha256"],
                    observed=methods.reference(Path(ref["path"]))["sha256"],
                    path=ref["path"],
                    hash=methods.reference(Path(ref["path"]))["sha256"],
                    failure_kind="missing_file"
                    if not Path(ref["path"]).is_file()
                    else "rejected_producer",
                )
            )


def authenticate(root: Path) -> Json:
    """Authenticate the prospective producer, retained inputs and matching branch."""
    checked = matched.authenticate(root)
    failures, refs, protocol = list(checked["failures"]), list(checked.get("references", [])), {}
    for name, pin, field, task in INPUTS:
        path = root / name
        ref = methods.reference(path)
        refs.append(ref)
        try:
            value = json.loads(path.read_text())
            for key, expected in (
                (field, 1),
                ("sha256", pin),
                ("task_id", task),
                ("required_checks_passed", True),
            ):
                actual = ref["sha256"] if key == "sha256" else value.get(key)
                matched.audit.require(actual, expected, path, key, task)
            verify_references(value)
            for key, expected in (
                ("future_generator_settings", source.SETTINGS),
                ("feature_schema", source.FEATURE_SCHEMA),
            ):
                matched.audit.require(value.get(key), expected, path, key, task)
            protocol = dict(
                graph_manifest=value["graph_manifest"],
                feature_schema=value["feature_schema"],
                generator_settings=value["future_generator_settings"],
            )
        except matched.audit.Rejected as error:
            failures.append(error.receipt)
        except (OSError, ValueError, AttributeError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    check="producer_structure",
                    upstream=task,
                    artifact_field="producer_json",
                    expected="original complete object",
                    observed=str(error),
                    failure_kind="missing_file" if not path.is_file() else "rejected_producer",
                    path=ref["path"],
                    hash=ref["sha256"],
                )
            )
    return dict(failures=failures, references=refs, protocol=protocol)


def ordered_units(protocol: Json) -> list[Json]:
    """Preserve the producer's balanced, stateless order without another shuffle."""
    units: list[Json] = source.scheduled(protocol, "control")
    return units


def reduce(units: list[Json], rows: list[Json]) -> Json:
    """Retain fifty clusters and compute discordance from observed paired error labels."""
    result: Json = prior.reduce(units, rows)
    result["rows"] = [
        dict(
            r,
            condition=f"k{r['k']}",
            censor_reason=r["missing_reason"],
            valid_set_size=len(u["expected"]),
            chance_accuracy=len(u["expected"]) / len(u["nodes"]),
        )
        for u, r in zip(units, rows, strict=True)
    ]
    equal = [
        dict(
            cluster_id=units[i]["cluster_id"],
            k2_count=len(next(u["expected"] for u in units[i : i + 2] if u["k"] == 2)),
            k3_count=len(next(u["expected"] for u in units[i : i + 2] if u["k"] == 3)),
        )
        for i in range(0, 100, 2)
    ]
    result["equal_cardinality_checks"] = [
        dict(r, passed=r["k2_count"] == r["k3_count"] and r["k2_count"] > 0) for r in equal
    ]
    control = result["depth_control"]
    interval = None
    if result["completed_count"] == 100 and control["labels_verified"]:
        differences = np.array(control["paired_differences"])
        draws = np.random.default_rng(765048).choice(differences, size=(10000, 50)).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        interval = dict(
            method="whole-graph paired percentile bootstrap 95%",
            random_seed=765048,
            draws=10000,
            denominator=50,
            mean_difference=float(differences.mean()),
            low=float(low),
            high=float(high),
            significance_claim=False,
        )
    result["paired_interval"] = control["paired_interval"] = interval
    control.update(
        equal_cardinality=all(r["passed"] for r in result["equal_cardinality_checks"]),
        minimum_k2_error=control["error_k2"] is not None and control["error_k2"] >= 0.20,
    )
    control["accepted"] = bool(
        control["accepted"] and control["minimum_k2_error"] and control["equal_cardinality"]
    )
    return result


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
        experiment_id=8948,
        task_id=TASK,
        milestone="2026.11.765",
        run_date="20261109",
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
        inference_mode="live_gpu" if runtime["live"] else "no_generation",
        started_at=work["started_at"],
        finished_at=work["finished_at"],
        global_test_debt=work["repository_health"],
        methodology_note="Prospective matched-cardinality graph source only; oracle truth cannot establish detector advantage. Operational rise is not a significance claim.",
        reproduction_commands=[
            f"CUDA_VISIBLE_DEVICES=0 CARNOT_FORCE_LIVE=1 PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary.json>",
        ],
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
        random_seed=765048,
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
        paired_interval="Seeded whole-graph paired bootstrap 95-percent interval; fifty independent graphs, one hundred answers.",
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

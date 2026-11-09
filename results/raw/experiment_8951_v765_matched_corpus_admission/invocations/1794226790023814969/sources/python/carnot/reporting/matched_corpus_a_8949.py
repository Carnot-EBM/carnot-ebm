"""REQ-REPORT-8949: qualify shard custody without fitting or claiming detector benefit."""

from __future__ import annotations

from collections import Counter
import json
import math
import sys
from types import ModuleType
from pathlib import Path
from typing import Any

from carnot.reporting import matched_depth_control_8948 as depth
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import pilot_label_audit_8932 as audit
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.experiment_template import ExperimentTemplate as ExperimentTemplate

Json = dict[str, Any]
ROOT = depth.ROOT
NAME = "experiment_8949_v765_matched_corpus_a"
TASK = "exp8949-matched-corpus-a"
EXPERIMENT_ID = 8949
SHARD = "A"
SEED = 765049
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/matched_corpus_a_8949.py",
    "python/carnot/inference/matched_corpus_runtime_8949.py",
    "python/carnot/reporting/matched_corpus_runner_8949.py",
    CLI,
]
TEST = "tests/python/test_matched_corpus_a_8949.py"
TESTS = [TEST]
INPUTS = [
    *depth.INPUTS,
    (
        "results/experiment_8948_v765_matched_depth_control.json",
        "sha256:da6efe2bb0adcc3e48f6b04b2b387d4e653a0cb9bed9df92a66dbc7f9e1fd9dd",
        "depth_source_ready_score",
        depth.TASK,
    ),
]


def authenticate(root: Path) -> Json:
    """Keep historical custody checks while admitting only the current matched control."""
    evidence: Json = depth.authenticate(root)
    name, pin, field, task = INPUTS[-1]
    path = root / name
    ref = methods.reference(path)
    evidence["references"].append(ref)
    try:
        value = json.loads(path.read_text())
        for key, expected in (
            (field, 1),
            ("task_id", task),
            ("sha256", pin),
            ("required_checks_passed", True),
            ("collection_authorized", True),
        ):
            depth.matched.audit.require(
                ref["sha256"] if key == "sha256" else value.get(key), expected, path, key, task
            )
        depth.verify_references(value)
    except depth.matched.audit.Rejected as error:
        evidence["failures"].append(error.receipt)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        evidence["failures"].append(
            dict(
                upstream=task,
                path=str(path),
                hash=ref["sha256"],
                check="producer_structure",
                artifact_field="producer_json",
                expected="original complete matched control",
                observed=str(error),
                failure_kind="missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    return evidence


def check_units(units: list[Json], shard: str = "A") -> None:
    """A censored slot keeps its original graph and role rather than drawing a replacement."""
    if (
        len(units) != 160
        or len({u["graph_hash"] for u in units}) != 160
        or len({u["seed"] for u in units}) != 160
        or Counter(u["role"] for u in units) != dict(train=80, calibration=40, test=40)
        or any(u["shard"] != shard or u["k"] != 2 for u in units)
    ):
        raise ValueError("shard_roster")


def ordered_units(protocol: Json, shard: str = "A") -> list[Json]:
    """Use the frozen label-blind order and never depend on the other scheduled shard."""
    units: list[Json] = source.scheduled(protocol, shard)
    check_units(units, shard)
    return units


def reduce(units: list[Json], rows: list[Json], shard: str = "A") -> Json:
    """Recompute labels by edge relaxation while features remain response-only values."""
    check_units(units, shard)
    if [r["unit_id"] for r in rows] != [u["unit_id"] for u in units]:
        raise ValueError("row_roster")
    labels, identities, typed = [], [], []
    verified = True
    for unit, row in zip(units, rows, strict=True):
        lengths = audit.distances(unit)
        expected = sorted(n for n, distance in lengths.items() if distance == 2)
        third = sorted(n for n, distance in lengths.items() if distance == 3)
        valid = (
            expected == unit["expected"]
            and len(expected) == len(third) > 0
            and source.graph_hash(unit) == unit["graph_hash"]
        )
        if row["status"] == "completed":
            category = source.category(row["answer"], unit["nodes"], expected)
            valid = (
                valid
                and row["parser_category"] == category
                and row["value"] == row["numerator"] == int(category != "correct")
            )
        verified = verified and valid
        identities.append(
            dict(
                cluster_id=unit["cluster_id"],
                graph_hash=unit["graph_hash"],
                node_edge_hash=source.graph_hash(unit),
                role=unit["role"],
                shard_id=shard,
                prompt_hash=unit["prompt_hash"],
                seed=unit["seed"],
            )
        )
        labels.append(
            dict(
                unit_id=unit["unit_id"],
                cluster_id=unit["cluster_id"],
                role=unit["role"],
                expected=expected,
                distances=lengths,
                parser_category=row["parser_category"],
                label=row["value"],
                independent_reference_agrees=valid,
            )
        )
        typed.append(dict(row, condition="k2", censor_reason=row["missing_reason"]))
    complete = sum(r["status"] == "completed" for r in rows)
    usable = all(
        r["signal_status"] == "usable"
        and isinstance(r["features"], dict)
        and set(r["features"]) == set(source.FEATURE_SCHEMA["learned_features"])
        and all(type(v) in (int, float) and math.isfinite(v) for v in r["features"].values())
        for r in rows
    )
    return dict(
        rows=typed,
        response_rows=typed,
        completed_count=complete,
        independent_count=complete,
        intended_count=160,
        failed_count=0,
        censored_count=160 - complete,
        excluded_count=0,
        labels_verified=verified,
        usable_signals=usable,
        split_counts=dict(Counter(u["role"] for u in units)),
        graph_identity_rows=identities,
        label_sidecar=labels,
        detector_view=[r["features"] for r in rows],
    )


def raw_complete(work: Json) -> bool:
    """A headline count cannot replace durable first responses and backend receipts."""
    raw = Path(work["raw"]) / "runtime" / "collection"
    try:
        for unit, row in zip(work["units"], work["runtime"]["rows"], strict=True):
            folder = raw / unit["unit_id"]
            envelope = json.loads((folder / "response.json").read_text())
            response = envelope["raw_response"]
            if (
                json.loads((folder / "wire_response.json").read_text()) != response
                or response["model"] != source.SETTINGS["model"]
                or not response["id"]
                or source.signals(response)["features"] != row["features"]
                or not all(
                    type(response["usage"][key]) is int
                    for key in ("prompt_tokens", "completion_tokens")
                )
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def build(
    work: Json, validation: Json, fixture: bool, *, producer: ModuleType | None = None
) -> Json:
    """Readiness certifies one shard's custody, never pooled support or learned benefit."""
    config = producer or sys.modules[__name__]
    runtime = work["runtime"]
    result = config.reduce(work["units"], runtime["rows"])
    owned = (
        config.qualified(validation)
        and not runtime["owned_failure"]
        and not validation.get("terminal_rejected", False)
        and result["labels_verified"]
        and (result["completed_count"] != 160 or config.raw_complete(work))
    )
    failures = work["authentication"]["failures"] + runtime["checks"]
    ready = int(
        owned
        and not failures
        and runtime["live"]
        and result["completed_count"] == 160
        and result["usable_signals"]
        and not fixture
    )
    cls = (
        "disqualified"
        if not owned and not fixture
        else "blocked"
        if failures
        else "circular_positive"
        if ready
        else "null"
    )
    counts = runtime["counts"]
    raw = Path(work["raw"])
    value = dict(work["template"])
    value.update(
        {k: v for k, v in result.items() if k not in {"detector_view", "label_sidecar"}},
        experiment_id=config.EXPERIMENT_ID,
        task_id=config.TASK,
        milestone="2026.11.765",
        run_date="20261109",
        honest_verdict="complete_" + cls + "_matched_corpus_" + config.SHARD.lower(),
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
        methodology_note=f"Frozen matched-population shard {config.SHARD} only. Independent graph labels are evaluation-only; detector features use backend confidence alone. No fitting, balancing, retries or pooled admission claim.",
        reproduction_commands=[
            f"CUDA_VISIBLE_DEVICES=0 CARNOT_FORCE_LIVE=1 PYTHONPATH=python .venv/bin/python -u {config.CLI} --date 20261109",
            f".venv/bin/python -u {config.CLI} --cold-replay <primary.json>",
        ],
        MODEL_SPECS=[dict(hf_id=source.SETTINGS["model"], quantization="Q4_K_M", gpu=0)],
        model_invocation_counts=counts,
        numerical_training_calls=0,
        model_invoked=bool(counts["model_loads_attempted"]),
        verifier_is_oracle=True,
        exposure_scope="matched_graph_final_answers_evaluation_only",
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
        random_seed=config.SEED,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            methods.reference(raw / "work.json"),
            methods.reference(raw / "validation.json"),
            *work["raw_references"],
        ],
        gate_check_summary=failures,
        acceptance_gates=dict(
            owned_checks=owned,
            authenticated_inputs=not work["authentication"]["failures"],
            all_responses=result["completed_count"] == 160,
            independent_labels=result["labels_verified"],
            finite_features=result["usable_signals"],
            raw_receipts=config.raw_complete(work),
            no_detector_fitting=True,
        ),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        global_test_debt=work["repository_health"],
        shard_ready_score=ready,
        shard_id=config.SHARD,
        model_receipt=runtime["model_receipt"],
        resource_lease=runtime["resource_lease"],
        generator_settings=runtime["settings"],
        feature_schema_hash=canonical_hash(source.FEATURE_SCHEMA),
        label_sidecar_hash=methods.reference(raw / "evaluation_labels.json"),
        detector_view_hash=methods.reference(raw / "detector_view.json"),
        fixture_only=fixture,
        flagged_adversarial=bool(validation.get("terminal_rejected")),
    )
    value["response_rows"] = [
        dict(
            r,
            response_hash=methods.reference(
                raw / "runtime" / "collection" / r["unit_id"] / "response.json"
            )["sha256"],
            prompt_hash=u["prompt_hash"],
            node_edge_hash=u["graph_hash"],
            split=u["role"],
        )
        for u, r in zip(work["units"], result["response_rows"], strict=True)
    ]
    value["field_principles"] = {
        key: "Current first-response custody only; oracle truth does not establish detector benefit."
        for key in value
    }
    value["field_principles"].update(
        shard_ready_score="All 160 authentic responses and owned checks; pooled class support belongs to admission.",
        detector_view_hash="Only response confidence features in frozen order; no graph, truth, label or role access.",
        model_invocation_counts="Owned invocation ledger only; retained upstream work is never current compute.",
    )
    value["model_specs"] = value["MODEL_SPECS"]
    value["schema"] = sorted(set(value) | {"reproducibility_checksum"})
    value["reproducibility_checksum"] = pilot.checksum(value)
    return value


def qualified(validation: Json, owned: list[str] = OWNED) -> bool:
    """Require the complete check roster and measured statements from this producer."""
    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == methods.CHECKS
        and all(
            r["passed"] and r.get("exit_code") == 0 and not r.get("timed_out") for r in receipts
        )
        and verify_measured(validation["coverage"], owned)
    )

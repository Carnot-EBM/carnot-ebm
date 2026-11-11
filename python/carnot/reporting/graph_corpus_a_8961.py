"""REQ-REPORT-8961: qualify shard custody without fitting or claiming detector benefit."""

from __future__ import annotations

from collections import Counter
import json
import math
import sys
from types import ModuleType
from pathlib import Path
from typing import Any

from carnot.reporting import matched_depth_control_8948 as depth
from carnot.inference import graph_corpus_runtime_8961 as capture
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import pilot_label_audit_8932 as audit
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.invocation_provenance_9000 import (
    serialize_authentication,
    serialize_invocation,
)
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.experiment_template import ExperimentTemplate as ExperimentTemplate

Json = dict[str, Any]
ROOT = depth.ROOT
NAME = "experiment_8961_v766_graph_corpus_a"
TASK = "exp8961-graph-corpus-a"
EXPERIMENT_ID = 8961
SHARD = "A"
SEED = 766061
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/graph_corpus_a_8961.py",
    "python/carnot/inference/graph_corpus_runtime_8961.py",
    "python/carnot/reporting/graph_corpus_runner_8961.py",
    CLI,
]
TEST = "tests/python/test_graph_corpus_a_8961.py"
TESTS = [TEST]
INPUTS = [
    (
        "results/experiment_8959_v766_relational_methods.json",
        "sha256:eaa01cdf08424373c0ea12b703a09099bf489cd015593111a5142c1c2f25c966",
        "relational_methods_ready_score",
        "exp8959-relational-methods",
    )
]
verify_references = depth.verify_references


def authenticate(root: Path) -> Json:
    """Authenticate retained methods and manifests without executing retired producers."""
    evidence: Json = dict(failures=[], references=[], protocol={})
    name, pin, field, task = INPUTS[0]
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
        ):
            depth.matched.audit.require(
                ref["sha256"] if key == "sha256" else value.get(key), expected, path, key, task
            )
        verify_references(value)
        if value["frozen_protocol"]["generator_settings"] != source.SETTINGS:
            raise ValueError("generator_settings_drift")
        roles = [
            dict(
                unit_id=q["unit_id"],
                cluster_id=g["graph_id"],
                graph_hash=g["graph_hash"],
                role=g["role"],
                shard=g["shard"],
                stream_id=g["stream_id"],
                stream_position=g["stream_position"],
            )
            for g in value["graph_manifest"]
            for q in g["questions"]
        ]
        if roles != value["role_manifest"]:
            raise ValueError("role_manifest_drift")
        evidence["protocol"] = {
            k: value[k] for k in ("graph_manifest", "role_manifest", "frozen_protocol")
        }
        evidence["references"].extend(
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        )
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
                expected="original frozen methods, weights and roles",
                observed=str(error),
                failure_kind="missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    return evidence


def check_units(units: list[Json], shard: str = "A") -> None:
    """A censored slot keeps its original graph and role rather than drawing a replacement."""
    if (
        len(units) != 512
        or len({u["graph_hash"] for u in units}) != 512
        or len({u["seed"] for u in units}) != 512
        or Counter(u["role"] for u in units) != dict(calibration=64, test=64, online=384)
        or any(u["shard"] != shard or u["k"] != 2 for u in units)
    ):
        raise ValueError("shard_roster")


def ordered_units(protocol: Json, shard: str = "A") -> list[Json]:
    """Use the frozen label-blind order and never depend on the other scheduled shard."""
    units: list[Json] = source.scheduled(protocol, shard)
    for unit in units:
        unit["expected"] = source.distance_nodes(unit["nodes"], unit["edges"], unit["start"], 2)
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
        typed.append(
            dict(
                row,
                condition="k2",
                censor_reason=row["missing_reason"],
                feature_mask={
                    k: int(row["signal_status"] == "usable")
                    for k in source.FEATURE_SCHEMA["learned_features"]
                },
                answer_present=row["parser_category"] in {"correct", "valid_wrong_distance"},
            )
        )
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
        intended_count=512,
        failed_count=0,
        censored_count=512 - complete,
        excluded_count=0,
        labels_verified=verified,
        usable_signals=usable,
        split_counts=dict(Counter(u["role"] for u in units)),
        role_counts=dict(Counter(u["role"] for u in units)),
        graph_identity_rows=identities,
        label_sidecar=labels,
        detector_view=[r["features"] for r in rows],
    )


def raw_complete(work: Json) -> bool:
    """A headline count cannot replace durable first responses and backend receipts."""
    raw = Path(work["raw"]) / "runtime" / "collection"
    try:
        ledger = {
            r["call_id"]: r
            for r in json.loads((raw.parent / "invocation_ledger.json").read_text())["rows"]
        }
        for index, (unit, row) in enumerate(
            zip(work["units"], work["runtime"]["rows"], strict=True)
        ):
            folder = capture.folder(raw, index, unit)
            envelope = json.loads((folder / "response.json").read_text())
            response = envelope["raw_response"]
            request = json.loads((folder / "intent.json").read_text())["request"]
            call = ledger[unit["unit_id"]]
            if (
                request
                != dict(source.SETTINGS, messages=[dict(role="user", content=unit["prompt"])])
                or call["request_sha256"] != canonical_hash(request)
                or call["response_sha256"] != canonical_hash(response)
                or call["status"] != "completed"
                or json.loads((folder / "wire_response.json").read_text()) != response
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
        and (result["completed_count"] != 512 or config.raw_complete(work))
    )
    failures = work["authentication"]["failures"] + runtime["checks"]
    ready = int(
        owned
        and not failures
        and runtime["live"]
        and result["completed_count"] == 512
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
    invocation = serialize_invocation(counts, runtime.get("ledger", []), raw)
    value = dict(work["template"])
    value.update(
        {k: v for k, v in result.items() if k not in {"detector_view", "label_sidecar"}},
        experiment_id=config.EXPERIMENT_ID,
        task_id=config.TASK,
        milestone="2026.11.766",
        run_date="20261110",
        honest_verdict="complete_" + cls + "_graph_corpus_" + config.SHARD.lower(),
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        inference_substrate="live_llm_inference"
        if counts["generation_calls_attempted"]
        else "model_loading"
        if counts["model_loads_attempted"]
        else "blocked_no_run",
        inference_substrate_class=invocation["inference_substrate_class"],
        execution_venue="host",
        inference_mode="live_gpu" if runtime["live"] else "no_generation",
        minimum_duration_s=10
        if counts["generation_calls_attempted"]
        else 2
        if counts["model_loads_attempted"]
        else 0,
        started_at=work["started_at"],
        finished_at=work["finished_at"],
        methodology_note=f"Frozen matched-population shard {config.SHARD} only. Independent graph labels are evaluation-only; detector features use backend confidence alone. No fitting, balancing, retries or pooled admission claim.",
        reproduction_commands=[
            f"CUDA_VISIBLE_DEVICES=0 CARNOT_FORCE_LIVE=1 PYTHONPATH=python .venv/bin/python -u {config.CLI} --date 20261110",
            f".venv/bin/python -u {config.CLI} --cold-replay <primary.json>",
        ],
        MODEL_SPECS=[dict(hf_id=source.SETTINGS["model"], quantization="Q4_K_M", gpu=0)],
        model_invocation_counts=invocation["model_invocation_counts"],
        numerical_training_calls=0,
        model_invoked=bool(counts["model_loads_attempted"]),
        verifier_is_oracle=True,
        exposure_scope="matched_graph_final_answers_evaluation_only",
        independent_generalization_score=0,
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"] + runtime["phase_spans"],
        preconditions_checked=dict(
            authentication=serialize_authentication(work["authentication"], raw),
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
            all_responses=result["completed_count"] == 512,
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
                capture.folder(raw / "runtime" / "collection", index, u) / "response.json"
            )["sha256"],
            prompt_hash=u["prompt_hash"],
            node_edge_hash=u["graph_hash"],
            split=u["role"],
        )
        for index, (u, r) in enumerate(zip(work["units"], result["response_rows"], strict=True))
    ]
    prior = work.get("prior_terminal")
    if (
        prior
        and cls in {"blocked", "null"}
        and prior["verdict_class"] == cls
        and prior["gate_check_summary"] == failures
    ):
        value["honest_verdict"] = prior["honest_verdict"]
    value["field_principles"] = {
        key: "Current first-response custody only; oracle truth does not establish detector benefit."
        for key in value
    }
    value["field_principles"].update(
        shard_ready_score="All 512 authentic responses and owned checks; pooled class support belongs to admission.",
        detector_view_hash="Only response confidence features in frozen order; no graph, truth, label or role access.",
        model_invocation_counts="Owned invocation ledger only; retained upstream work is never current compute.",
    )
    value["acceptance_gates"] = {
        k: dict(
            passed=v,
            principle="Readiness needs complete authentic evidence; failed owned checks disqualify.",
        )
        for k, v in value["acceptance_gates"].items()
    }
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

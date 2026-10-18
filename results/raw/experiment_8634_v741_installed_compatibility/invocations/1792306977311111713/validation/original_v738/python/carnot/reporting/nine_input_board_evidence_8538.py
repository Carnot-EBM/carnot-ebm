"""REQ-REPORT-8538: fixed historical seals keep CPU costs separate from fabric support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import board_cost_boundary_8524 as shipped
from carnot.reporting.board_operation_boundary_8454 import resources as resources
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT, base = shipped.ROOT, shipped.base
reference = shipped.reference
COST = "results/experiment_8534_v733_nine_input_inference_cost.json"
OPERATIONS = [
    "text_features",
    "scaling",
    "raw_logit_offset",
    "cubic_residual",
    "normalization",
    "threshold_fallback",
    "response_serialization",
]
COST_OPERATIONS = ["startup", *OPERATIONS, "host_device_transfer"]
SOURCES = [
    (
        "results/experiment_8524_v732_board_cost_boundary.json",
        "sha256:84d2393d711e1b8040de2e7265af40efcea18515b1a91e6e37a0794169fa7ee1",
        "20261014",
    ),
    (
        "results/experiment_8440_v726_board_reader_qualification.json",
        "sha256:f1fe951405765fc39318a440f90c3baa2cb34d713f75945de638aa3b74a887bb",
        "20261012",
    ),
]
PRESERVED = [
    "kv260_supported_operations",
    "polarfire_graduation",
    "polarfire_workload_validated",
    "kv260_synthesis_succeeded",
    "gatemate_obligation_rows",
    "missing_source_hashes",
    "last_authenticated_idcode",
    "reopening_condition",
]


def checksum(value: Json) -> str:
    """Independent source pins, rather than this summary checksum, authorize evidence."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def failure(path: Path, field: str, expected: Any, observed: Any, upstream: str) -> Json:
    """Missing values stay null so absence cannot be mistaken for measured zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def history(root: Path) -> Json:
    """Each original reader has its own hash, date, primitive hashes and terminal binding."""
    result: Json = dict(qualified=False, checks=[], refs=[], fields={}, readers=[])
    for name, digest, date in SOURCES:
        path = root / name
        expected: Any
        observed: Any
        field, expected, observed = "original_sha256", digest, None
        try:
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(field)
            value = json.loads(path.read_bytes())
            for field, expected in dict(
                run_date=date,
                required_checks_passed=True,
                flagged_adversarial=False,
                board_reader_ready_score=1,
            ).items():
                observed = value.get(field)
                if type(observed) is not type(expected) or observed != expected:
                    raise ValueError(field)
            field, expected, observed = "original_primitive_hashes", True, False
            refs = [
                reference(path),
                value["work_reference"],
                *value["source_artifact_hashes"],
                *value["raw_shard_hashes"],
            ]
            for ref in refs:
                base.checked(ref)
            terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
            report = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
            if report["report"]["passed"] is not True:
                raise ValueError("original_terminal_binding")
            field, expected = "supported_operation_rows", ["quadratic_Ising_energy", 5, True]
            observed = [
                value["kv260_supported_operations"][0]["operation"],
                value["kv260_supported_operations"][0]["k_max"],
                value["polarfire_workload_validated"],
            ]
            if observed != expected:
                raise ValueError(field)
            result["refs"] += refs
            result["readers"].append(
                dict(
                    path=str(path),
                    sha256=digest,
                    run_date=date,
                    supported_operation_rows=value["kv260_supported_operations"],
                )
            )
            if name == SOURCES[0][0]:
                result["fields"] = {k: value[k] for k in PRESERVED}
                result["models"] = value["historical_model_provenance"]
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            result["checks"].append(failure(path, field, expected, observed, Path(name).stem))
    result["qualified"] = len(result["readers"]) == len(SOURCES)
    return result


def ceiling(spans: list[Json]) -> Json:
    """Remove only measured arithmetic; startup, text, responses and transfers remain charged."""
    clocks = sorted(spans, key=lambda r: r["start_ns"])
    if (
        set(r["operation"] for r in clocks) != set(COST_OPERATIONS)
        or len(clocks) != len(COST_OPERATIONS)
        or any(type(r[k]) is not int or r[k] < 0 for r in clocks for k in ["start_ns", "end_ns"])
        or any(r["end_ns"] < r["start_ns"] for r in clocks)
        or any(a["end_ns"] != b["start_ns"] for a, b in zip(clocks, clocks[1:]))
    ):
        raise ValueError("disjoint_complete_spans")
    total = clocks[-1]["end_ns"] - clocks[0]["start_ns"]
    arithmetic = sum(
        r["end_ns"] - r["start_ns"]
        for r in clocks
        if r["operation"] in {"scaling", "raw_logit_offset", "cubic_residual"}
    )
    if not total or arithmetic == total:
        raise ValueError("disjoint_complete_spans")
    return dict(
        complete_clock_ns=total,
        eligible_clock_ns=arithmetic,
        ceiling=total / (total - arithmetic),
        installed_operations_ceiling=1.0,
        counterfactual=True,
        deployed_speed=False,
        input_count=9,
        transport_and_startup_included=True,
    )


def costs(root: Path) -> Json:
    """An optional future producer must pass its exact fields, terminal binding and span hashes."""
    path = root / COST
    result: Json = dict(
        path=str(path), qualified=False, checks=[], refs=[], spans=[], conditional_ceiling=None
    )
    expected: Any
    observed: Any
    field, expected, observed = "nine_input_inference_cost_ready_score", 1, None
    try:
        value = json.loads(path.read_bytes())
        result["refs"].append(reference(path))
        for field, expected in dict(
            nine_input_inference_cost_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            experiment_id=8534,
            task_id="exp8534-nine-input-inference-cost",
            run_date="20261015",
        ).items():
            observed = value.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(field)
        field, expected, observed = "bound_terminal_bytes", True, False
        validate_primary(value, path)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        if (
            read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))["report"][
                "passed"
            ]
            is not True
        ):
            raise ValueError(field)
        field, expected, observed = (
            "disjoint_complete_spans_reference",
            "hash_verified_nine_input_complete_spans",
            None,
        )
        ref = value[field]
        spans = json.loads(base.checked(ref).read_bytes())["spans"]
        result.update(qualified=True, spans=spans, conditional_ceiling=ceiling(spans))
        result["refs"].append(ref)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        result["checks"].append(
            failure(path, field, expected, observed, "exp8534-nine-input-inference-cost")
        )
    return result

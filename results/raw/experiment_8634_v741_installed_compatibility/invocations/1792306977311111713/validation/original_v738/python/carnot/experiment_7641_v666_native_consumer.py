"""Expose the qualified native recalibration service as an opt-in package API.

This experiment measures integration readiness. It carries the dated Exp7627
ratios as historical evidence and makes no new speed or learned-benefit claim.

Spec: REQ-REPORT-7641 and SCENARIO-REPORT-7641-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_7626_v665_native_service as exp7626
from carnot.pipeline.native_calibrated_decision_service import (
    NativeServiceClient,
    load_native_extension,
)
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260925"
MILESTONE = "2026.09.666"
EXPERIMENT_ID = "exp7641-v666-native-consumer"
SCHEMA = "carnot.exp7641.v666.native_consumer.v1"
RESULT_PATH = Path("results/experiment_7641_v666_native_consumer.json")
RAW_DIR = Path("results/raw/experiment_7641_v666_native_consumer")
EXP7626_PATH = Path("results/experiment_7626_v665_native_service.json")
EXP7627_PATH = Path("results/experiment_7627_v665_native_cost.json")
MODULE_PATH = Path("python/carnot/experiment_7641_v666_native_consumer.py")
CLIENT_PATH = Path("python/carnot/pipeline/native_calibrated_decision_service.py")
OLD_EXPERIMENT_PATH = Path("python/carnot/experiment_7626_v665_native_service.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7641_v666_native_consumer.py")
TEST_PATH = Path("tests/python/test_experiment_7641_v666_native_consumer.py")
EXP7626_TEST_PATH = Path("tests/python/test_experiment_7626_v665_native_service.py")
EXAMPLE_PATH = Path("examples/native_calibrated_decision.py")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PIPELINE_SPEC_PATH = Path("openspec/capabilities/pipeline/spec.md")
CORE_PATH = Path("crates/carnot-core/src/portable_recalibration.rs")
BINDING_PATH = Path("crates/carnot-python/src/portable_recalibration.rs")
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATIONS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
UNIT_GROUPS = {
    "real_native": (
        "real_predict",
        "real_feedback",
        "state_summary",
        "cold_reload",
    ),
    "unavailable": (
        "invalid_probability",
        "missing_extension",
        "corrupt_state",
        "closed_client",
    ),
    "durability": (
        "duplicate_feedback",
        "unknown_feedback",
        "invalid_label",
        "interrupted_write",
    ),
}


def _row(unit_id: str, group: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Build one independently checkable integration unit."""

    return {
        "unit_id": unit_id,
        "arm": "package_direct_pyo3",
        "group": group,
        "input_or_error_condition": unit_id,
        "expected_typed_value": deepcopy(expected),
        "observed_typed_value": deepcopy(observed),
        "passed": passed,
        "absolute_metric": 1 if passed else 0,
        "numerator": 1 if passed else 0,
        "denominator": 1,
        "seed": None,
        "direction": "exact_contract",
        "censored": False,
        "raw_provenance": "current_exp7641_package_integration",
    }


def synthetic_integration_rows() -> list[JsonDict]:
    """Return the exact twelve-unit fixture used by reducer mutation tests."""

    rows: list[JsonDict] = []
    for group, unit_ids in UNIT_GROUPS.items():
        for unit_id in unit_ids:
            rows.append(_row(unit_id, group, True, True, True))
    return rows


def reduce_integration_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require each registered package, failure, and durability unit once."""

    expected = {unit for units in UNIT_GROUPS.values() for unit in units}
    identifiers = [str(row.get("unit_id")) for row in rows]
    if (
        len(rows) != len(expected)
        or set(identifiers) != expected
        or len(set(identifiers)) != len(rows)
    ):
        raise ValueError("integration_units_missing")
    if any(row.get("passed") is not True for row in rows):
        raise ValueError("integration_unit_failed")
    counts = {group: sum(row.get("group") == group for row in rows) for group in UNIT_GROUPS}
    if any(counts[group] != len(UNIT_GROUPS[group]) for group in UNIT_GROUPS):
        raise ValueError("integration_group_invalid")
    return {
        "complete": True,
        "independent_units": len(rows),
        "passed_units": len(rows),
        "failed_units": 0,
        "real_native_units": counts["real_native"],
        "unavailable_units": counts["unavailable"],
        "durability_units": counts["durability"],
    }


def hardware_dispositions(root: Path) -> list[JsonDict]:
    """Carry qualified device scope forward without probing any device."""

    source = "results/experiment_7627_v665_native_cost.json#hardware_dispositions"
    return [
        {
            "hardware": "KV260",
            "disposition": "graduated_fpga_fabric_scope_preserved",
            "claim_scope": "fpga_fabric_k_max_at_most_5",
            "k_max": 5,
            "current_execution": False,
            "source_receipt": source,
            "reopen_condition": "separately scoped FPGA-fabric task",
        },
        {
            "hardware": "PolarFire",
            "disposition": "graduated_linux_cpu_dispatch_scope_preserved",
            "claim_scope": "linux_cpu_dispatch_only",
            "fpga_sampling_measured": False,
            "current_execution": False,
            "source_receipt": source,
            "reopen_condition": "separate measured FPGA-sampling task",
        },
        {
            "hardware": "GateMate",
            "disposition": "blocked_unchanged_physical_prerequisite",
            "claim_scope": "physical_jtag_unexecuted",
            "last_observed": "0xffffffff",
            "current_execution": False,
            "source_receipt": source,
            "reopen_condition": (
                "dated operator cable, port, power, board, JTAG, or DirtyJTAG change after Exp6559"
            ),
        },
        {
            "hardware": "CUDA",
            "disposition": "historical_two_rtx3090_scope_not_current_execution",
            "claim_scope": "historical_input_only",
            "current_execution": False,
            "source_receipt": "research-hardware-wishlist.md",
            "reopen_condition": "separately authorized GPU task",
        },
        {
            "hardware": "NPU",
            "disposition": "unqualified_runtime_unavailable",
            "qualified": False,
            "current_execution": False,
            "source_receipt": "research-hardware-wishlist.md#AMD-XDNA-NPU-Status",
            "reopen_condition": "installed runtime plus measured local benchmark",
        },
        {
            "hardware": "TSU",
            "disposition": "unqualified_no_authenticated_access",
            "qualified": False,
            "current_execution": False,
            "source_receipt": "research-hardware-wishlist.md#Extropic-TSU",
            "reopen_condition": "authenticated hardware or SDK access",
        },
    ]


def _check(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    passed = observed == expected if operator == "eq" else bool(observed)
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
    }


def _load_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"object_required:{path}")
    return value


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate prior receipts, source ownership, and the real extension."""

    exp7626_path = root / EXP7626_PATH
    exp7627_path = root / EXP7627_PATH
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    required = (
        EXP7626_PATH,
        EXP7627_PATH,
        CORE_PATH,
        BINDING_PATH,
        OLD_EXPERIMENT_PATH,
        REPORT_SPEC_PATH,
        PIPELINE_SPEC_PATH,
        Path("research-hardware-wishlist.md"),
    )
    for relative in required:
        path = root / relative
        exists = path.is_file()
        rows.append(
            _check("named_input", "task prompt", relative.as_posix(), "exists", "eq", True, exists)
        )
        sources[relative.as_posix()] = {
            "path": relative.as_posix(),
            "role": "pre_gate_receipt" if relative in {EXP7626_PATH, EXP7627_PATH} else "producer",
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
            "bytes": path.stat().st_size if exists else None,
        }
    if not exp7626_path.is_file() or not exp7627_path.is_file():
        failed = next(row for row in rows if not row["passed"])
        return {
            "rows": rows,
            "blocker": {key: failed[key] for key in failed if key != "passed"},
            "source_artifact_hashes": sources,
        }

    service = _load_object(exp7626_path)
    cost = _load_object(exp7627_path)
    for check, upstream, path, field, expected, observed in (
        (
            "native_service_ready",
            "Exp7626",
            EXP7626_PATH.as_posix(),
            "native_service_ready_score",
            1,
            service.get("native_service_ready_score"),
        ),
        (
            "native_parity",
            "Exp7626",
            EXP7626_PATH.as_posix(),
            "parity_reduction.complete",
            True,
            service.get("parity_reduction", {}).get("complete"),
        ),
        (
            "native_durability",
            "Exp7626",
            EXP7626_PATH.as_posix(),
            "durability_reduction.complete",
            True,
            service.get("durability_reduction", {}).get("complete"),
        ),
        (
            "native_reader",
            "Exp7626",
            EXP7626_PATH.as_posix(),
            "flagged_adversarial",
            False,
            service.get("flagged_adversarial"),
        ),
        (
            "historical_speed_valid",
            "Exp7627",
            EXP7627_PATH.as_posix(),
            "native_speed_benefit_score",
            1,
            cost.get("native_speed_benefit_score"),
        ),
        (
            "historical_reader",
            "Exp7627",
            EXP7627_PATH.as_posix(),
            "flagged_adversarial",
            False,
            cost.get("flagged_adversarial"),
        ),
    ):
        rows.append(_check(check, upstream, path, field, "eq", expected, observed))
    extension_receipt = service.get("source_artifact_hashes", {}).get("native_extension", {})
    extension = Path(str(extension_receipt.get("path", "")))
    observed_hash = sha256_file(extension) if extension.is_file() else None
    rows.append(
        _check(
            "native_extension_exists",
            "Exp7626",
            str(extension),
            "exists",
            "eq",
            True,
            extension.is_file(),
        )
    )
    rows.append(
        _check(
            "native_extension_hash",
            "Exp7626",
            str(extension),
            "sha256",
            "eq",
            extension_receipt.get("sha256"),
            observed_hash,
        )
    )
    sources["native_extension"] = {
        "path": str(extension),
        "role": "authenticated_exp7626_native_module",
        "exists": extension.is_file(),
        "sha256": observed_hash,
        "bytes": extension.stat().st_size if extension.is_file() else None,
    }
    sources[RESULT_PATH.as_posix()] = {
        "path": RESULT_PATH.as_posix(),
        "role": "planned_output_not_input",
        "exists": False,
        "sha256": None,
        "bytes": None,
    }
    failed = next((row for row in rows if row["passed"] is not True), None)
    blocker = {key: failed[key] for key in failed if key != "passed"} if failed else None
    return {
        "rows": rows,
        "blocker": blocker,
        "source_artifact_hashes": sources,
        "native_extension": extension,
        "exp7626": service,
        "exp7627": cost,
    }


def _gate(category: str, check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {
        "category": category,
        "check": check,
        "condition": f"{check} must preserve its declared claim boundary",
        "upstream": "current Exp7641 evidence",
        "path": "integration_rows",
        "field": check,
        "operator": "eq",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": "Validity, readiness, benefit, retention, and freshness are separate claims.",
    }


def acceptance_gates(ready: int) -> list[JsonDict]:
    """Keep package readiness separate from every benefit class."""

    return [
        _gate("validity", "integration_rows_complete", True, ready == 1, ready == 1),
        _gate("readiness", "native_consumer_ready_score", 1, ready, ready == 1),
        _gate("probability_benefit", "new_probability_benefit_measured", True, False, False),
        _gate("utility", "new_total_cost_benefit_measured", True, False, False),
        _gate("retention", "production_defaults_changed", False, False, True),
        _gate("freshness", "current_model_or_board_execution", False, False, True),
    ]


def historical_speed_evidence(cost: Mapping[str, Any] | None = None) -> JsonDict:
    """Carry only the dated Exp7627 workload ratios without remeasurement."""

    if cost is None:
        python_ratio = 7.827002746229495
        rust_ratio = 2.6492877005903535
        nfr = False
    else:
        reduction = cost["timing_reduction"]
        values = reduction["equal_stratum_geometric_mean"]
        python_ratio = values["python_over_direct_native"]["estimate"]
        rust_ratio = values["rust_over_direct_native"]["estimate"]
        nfr = reduction["nfr_10x_met"]
    return {
        "source": EXP7627_PATH.as_posix(),
        "measurement_date": "2026-09-24",
        "workload": "Exp7627 120 paired durable consumer blocks across four strata",
        "python_over_native": python_ratio,
        "jsonl_over_native": rust_ratio,
        "nfr_10x_met": nfr,
        "current_measurement": False,
    }


def field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Place a short governing rule beside every top-level field."""

    specific = {
        "honest_verdict": "Completion and benefit are different outcomes.",
        "verdict_class": "Package readiness alone uses null; external absence uses blocked.",
        "flagged_adversarial": "A flagged result opens no downstream gate.",
        "gate_check_summary": "Every block retains all exact comparison operands.",
        "rows": "Each independent integration condition counts once.",
        "sample_size_budget": "Views and replays do not multiply independent units.",
        "preconditions_checked": "Unavailable evidence cannot become fabricated execution.",
        "inference_substrate": "This work loads native code but no model.",
        "duration_s": "Duration is current monotonic wall time without padding.",
        "source_artifact_hashes": "Inputs, receipts, and planned outputs keep separate roles.",
        "native_consumer_ready_score": "One requires real PyO3, restart, and failure behavior.",
        "new_speed_claim": "Historical ratios do not become a new benchmark.",
        "production_defaults_changed": "The JSON-lines production default remains unchanged.",
    }
    return {
        key: specific.get(key, f"Retain {key} so its terminal scope stays auditable.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash immutable inputs, rows, configuration, and reduction code identity."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "run_date": value.get("run_date"),
            "rows": value.get("integration_rows"),
            "sample_size_budget": value.get("sample_size_budget"),
            "historical_speed_evidence": value.get("historical_speed_evidence"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "MODEL_SPECS": value.get("MODEL_SPECS"),
            "random_seed": value.get("random_seed"),
        }
    )


def build_artifact(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    preconditions: Mapping[str, Any] | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the complete readiness artifact from independent integration rows."""

    reduction = reduce_integration_rows(rows)
    context = dict(preconditions or {})
    sources = deepcopy(context.get("source_artifact_hashes", {}))
    if not sources:
        for relative in (EXP7626_PATH, EXP7627_PATH, MODULE_PATH, CLIENT_PATH):
            path = root / relative
            sources[relative.as_posix()] = {
                "path": relative.as_posix(),
                "role": "test_fixture_input",
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        sources[RESULT_PATH.as_posix()] = {
            "path": RESULT_PATH.as_posix(),
            "role": "planned_output_not_input",
            "exists": False,
            "sha256": None,
            "bytes": None,
        }
    cost = context.get("exp7627")
    artifact: JsonDict = {
        "experiment": 7641,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "schema": SCHEMA,
        "honest_verdict": "complete_null_native_consumer_ready",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": None,
        "acceptance_gate_results": acceptance_gates(1),
        "rows": deepcopy(list(rows)),
        "integration_rows": deepcopy(list(rows)),
        "integration_reduction": reduction,
        "sample_size_budget": {
            "intended": 12,
            "observed": 12,
            "excluded": 0,
            "censored": 0,
            "views_per_unit": "not_counted_as_independent_units",
        },
        "preconditions_checked": deepcopy(context.get("rows", [])),
        "inference_substrate": "host_cpu_direct_pyo3_package_integration_no_model",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "historical_models": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATIONS),
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "device_uuid": None,
            "owned_pid": os.getpid(),
            "native_device": "host_cpu",
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "duration_s": float(duration_s),
        "random_seed": {"integration": None, "reason": "deterministic exact contract"},
        "source_artifact_hashes": sources,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "verifier_is_oracle": True,
        "native_consumer_ready_score": 1,
        "historical_speed_evidence": historical_speed_evidence(cost),
        "new_speed_claim": False,
        "hardware_dispositions": hardware_dispositions(root),
        "production_defaults_changed": False,
        "jsonl_default_preserved": True,
        "safetensors_scope_changed": False,
        "state_format": "JSON sufficient statistics; separate from unchanged safetensors",
    }
    artifact["field_principles"] = field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(root: Path, scratch: Path) -> JsonDict:
    """Build a complete exact fixture without claiming a current native run."""

    del scratch
    return build_artifact(root, synthetic_integration_rows())


def build_blocked_artifact(root: Path, blocker: Mapping[str, Any], scratch: Path) -> JsonDict:
    """Close one unchanged external absence without fabricating integration work."""

    del scratch
    artifact = build_artifact(root, synthetic_integration_rows())
    artifact.update(
        {
            "honest_verdict": f"complete_blocked_{blocker['check']}",
            "verdict_class": "blocked",
            "gate_check_summary": deepcopy(dict(blocker)),
            "integration_rows": [],
            "rows": [],
            "integration_reduction": {
                "complete": False,
                "independent_units": 0,
                "passed_units": 0,
                "failed_units": 0,
                "real_native_units": 0,
                "unavailable_units": 0,
                "durability_units": 0,
            },
            "native_consumer_ready_score": 0,
            "acceptance_gate_results": acceptance_gates(0),
            "sample_size_budget": {
                "intended": 12,
                "observed": 0,
                "excluded": 12,
                "censored": 0,
                "views_per_unit": "not_counted_as_independent_units",
            },
        }
    )
    artifact["field_principles"] = field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any], *, check_files: bool = True) -> list[str]:
    """Independently reject readiness, provenance, or claim-boundary drift."""

    errors: list[str] = []
    verdict = str(value.get("honest_verdict", ""))
    verdict_class = value.get("verdict_class")
    if not verdict.startswith("complete_"):
        errors.append("honest_verdict")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if value.get("MODEL_SPECS") != [] or value.get("planned_MODEL_SPECS") != []:
        errors.append("model_specs")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATIONS
    ):
        errors.append("model_invocation")
    if value.get("new_speed_claim") is not False:
        errors.append("new_speed_claim")
    if value.get("production_defaults_changed") is not False:
        errors.append("production_defaults")
    if value.get("verifier_is_oracle") is not True:
        errors.append("oracle_declaration")
    if verdict_class == "blocked":
        summary = value.get("gate_check_summary")
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not isinstance(summary, Mapping) or not required.issubset(summary):
            errors.append("blocked_gate_summary")
    else:
        try:
            reduction = reduce_integration_rows(value.get("integration_rows", []))
        except (TypeError, ValueError):
            errors.append("integration_rows")
            reduction = None
        if reduction != value.get("integration_reduction"):
            errors.append("integration_reduction")
        if value.get("native_consumer_ready_score") != 1:
            errors.append("ready_score")
    speed = value.get("historical_speed_evidence")
    if not isinstance(speed, Mapping):
        errors.append("historical_speed")
    else:
        if speed.get("python_over_native") != 7.827002746229495:
            errors.append("historical_python_ratio")
        if speed.get("jsonl_over_native") != 2.6492877005903535:
            errors.append("historical_jsonl_ratio")
        if speed.get("nfr_10x_met") is not False or speed.get("current_measurement") is not False:
            errors.append("historical_nfr")
    categories = {row.get("category") for row in value.get("acceptance_gate_results", [])}
    if categories != {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }:
        errors.append("acceptance_gate_categories")
    hardware = {row.get("hardware"): row for row in value.get("hardware_dispositions", [])}
    if set(hardware) != {"KV260", "PolarFire", "GateMate", "CUDA", "NPU", "TSU"}:
        errors.append("hardware_dispositions")
    elif (
        hardware["KV260"].get("k_max") != 5
        or hardware["PolarFire"].get("claim_scope") != "linux_cpu_dispatch_only"
        or hardware["GateMate"].get("last_observed") != "0xffffffff"
    ):
        errors.append("hardware_scope")
    principles = value.get("field_principles")
    expected_principles = {*value, "field_principles", "reproducibility_checksum"}
    if not isinstance(principles, Mapping) or set(principles) != expected_principles:
        errors.append("field_principles")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum")
    if check_files:
        for receipt in value.get("source_artifact_hashes", {}).values():
            if receipt.get("role") == "planned_output_not_input":
                continue
            path = Path(str(receipt.get("path")))
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            if receipt.get("exists") and (
                not path.is_file() or sha256_file(path) != receipt.get("sha256")
            ):
                errors.append(f"source_hash:{receipt.get('path')}")
    return sorted(set(errors))


def cold_replay(path: Path) -> JsonDict:
    """Read exact bytes in a fresh process and rerun the terminal validator."""

    value = _load_object(path)
    errors = validate_artifact(value)
    return {"valid": not errors, "errors": errors, "artifact_sha256": sha256_file(path)}


def independent_replay(path: Path) -> JsonDict:
    """Recompute integration and historical comparisons from terminal rows."""

    value = _load_object(path)
    try:
        reduction = reduce_integration_rows(value["integration_rows"])
        valid = reduction == value.get("integration_reduction")
    except (KeyError, TypeError, ValueError) as error:
        return {"valid": False, "error": str(error), "artifact_sha256": sha256_file(path)}
    speed = value.get("historical_speed_evidence", {})
    valid = valid and speed.get("python_over_native") == 7.827002746229495
    valid = valid and speed.get("jsonl_over_native") == 2.6492877005903535
    return {
        "valid": valid,
        "integration_reduction": reduction,
        "historical_comparison_recomputed": valid,
        "artifact_sha256": sha256_file(path),
    }


def exercise_native_consumer(
    root: Path, extension: Path, scratch: Path
) -> list[JsonDict]:  # pragma: no cover - exercised by task E2E.
    """Exercise real native, explicit failure, and durability boundaries."""

    binding = load_native_extension(extension)
    state = scratch / "consumer.json"
    client = NativeServiceClient(binding, state)
    decision = client.predict("event-one", 0.31)
    rows = [
        _row(
            "real_predict",
            "real_native",
            {"available": True, "verified": False},
            {"available": decision.available, "verified": decision.verified},
            decision.available and not decision.verified,
        )
    ]
    acknowledgment = client.release_feedback("event-one", 1)
    rows.append(
        _row(
            "real_feedback",
            "real_native",
            {"acknowledged": True, "durable": True},
            {"acknowledged": acknowledgment.acknowledged, "durable": acknowledgment.durable},
            acknowledgment.acknowledged and acknowledgment.durable,
        )
    )
    summary = client.state_summary()
    rows.append(
        _row(
            "state_summary",
            "real_native",
            {"sample_count": 1, "processed": ["event-one"]},
            {
                "sample_count": summary["sample_count"],
                "processed": summary["processed_event_ids"],
            },
            summary["sample_count"] == 1 and summary["processed_event_ids"] == ["event-one"],
        )
    )
    reopened = NativeServiceClient(binding, state)
    reopened_summary = reopened.state_summary()
    rows.append(
        _row(
            "cold_reload",
            "real_native",
            1,
            reopened_summary["sample_count"],
            reopened_summary["sample_count"] == 1,
        )
    )

    invalid = reopened.predict("invalid", float("nan"))
    rows.append(
        _row(
            "invalid_probability",
            "unavailable",
            "finite_probability_required",
            invalid.error,
            not invalid.available and invalid.error == "finite_probability_required",
        )
    )
    missing = NativeServiceClient.from_extension(
        state_path=scratch / "missing.json", extension_path=scratch / "missing.so"
    )
    missing_result = missing.predict("missing", 0.4)
    rows.append(
        _row(
            "missing_extension",
            "unavailable",
            "native_extension_unavailable",
            missing_result.error,
            not missing_result.available
            and str(missing_result.error).startswith("native_extension_unavailable:"),
        )
    )
    corrupt_path = scratch / "corrupt.json"
    corrupt_path.write_text("not-json", encoding="utf-8")
    corrupt = NativeServiceClient(binding, corrupt_path).predict("corrupt", 0.2)
    rows.append(
        _row(
            "corrupt_state",
            "unavailable",
            "native_service_unavailable",
            corrupt.error,
            not corrupt.available and str(corrupt.error).startswith("native_service_unavailable:"),
        )
    )
    client.close()
    closed = client.predict("closed", 0.2)
    rows.append(
        _row(
            "closed_client",
            "unavailable",
            "native_client_closed",
            closed.error,
            closed.error == "native_client_closed",
        )
    )

    duplicate = reopened.release_feedback("event-one", 1)
    rows.append(
        _row(
            "duplicate_feedback",
            "durability",
            "duplicate_feedback:event-one",
            duplicate.error,
            duplicate.error == "duplicate_feedback:event-one" and not duplicate.durable,
        )
    )
    unknown = reopened.release_feedback("unknown", 1)
    rows.append(
        _row(
            "unknown_feedback",
            "durability",
            "unknown_prediction:unknown",
            unknown.error,
            unknown.error == "unknown_prediction:unknown" and not unknown.acknowledged,
        )
    )
    pending = reopened.predict("pending", 0.25)
    invalid_label = reopened.release_feedback("pending", 2)
    rows.append(
        _row(
            "invalid_label",
            "durability",
            "binary_label_required",
            invalid_label.error,
            pending.available
            and invalid_label.error == "binary_label_required"
            and not invalid_label.durable,
        )
    )
    interruption = exp7626.run_interrupted_write_probe(root, extension, state)
    rows.append(
        _row(
            "interrupted_write",
            "durability",
            {"acknowledged": False, "prior_state_survived": True},
            {
                "acknowledged": interruption["acknowledged"],
                "prior_state_survived": interruption["prior_state_survived"],
            },
            interruption["acknowledged"] is False and interruption["prior_state_survived"] is True,
        )
    )
    return rows


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit one flushed boundary with truthful monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7641] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(name: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": name,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "pending_operations": 0,
        "checkpoint_position": units,
    }


def build_validation_commands(
    root: Path, private: Path, extension: Path
) -> list[validation.CommandSpec]:
    """Freeze focused tests, changed-code coverage, and package E2E checks."""

    commands = validation.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix(), CLIENT_PATH.as_posix()],
        static_paths=[
            OLD_EXPERIMENT_PATH.as_posix(),
            EXP7626_TEST_PATH.as_posix(),
            WRAPPER_PATH.as_posix(),
            EXAMPLE_PATH.as_posix(),
        ],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage-exp7641",
    )
    python = str(root / ".venv/bin/python")
    installed = private / "installed"
    installed_program = """
from pathlib import Path
import sys
target, extension, state = sys.argv[1:]
sys.path.insert(0, target)
from carnot.pipeline.native_calibrated_decision_service import NativeServiceClient
client = NativeServiceClient.from_extension(state_path=Path(state), extension_path=Path(extension))
decision = client.predict('installed-event', 0.33)
ack = client.release_feedback('installed-event', 1)
raise SystemExit(not (decision.available and ack.durable))
"""
    commands.extend(
        [
            validation.CommandSpec(
                "private_package_install",
                (
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--target",
                    str(installed),
                    ".",
                ),
                "private installed-style package tree",
                300.0,
            ),
            validation.CommandSpec(
                "installed_style_native_import",
                (
                    python,
                    "-I",
                    "-u",
                    "-c",
                    installed_program,
                    str(installed),
                    str(extension.resolve()),
                    str(private / "installed-state.json"),
                ),
                "private installed package plus exact PyO3 extension",
                120.0,
            ),
            validation.CommandSpec(
                "e2e_003_real_pyo3",
                (
                    str(root / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'basetemp/e2e003'}",
                    f"{TEST_PATH}::test_scenario_pipeline_7641_native_lifecycle_is_durable",
                    "-q",
                ),
                "E2E-003 package-to-PyO3 round trip",
                300.0,
            ),
            validation.CommandSpec(
                "e2e_004_json_interrupt_reload",
                (
                    str(root / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'basetemp/e2e004'}",
                    f"{TEST_PATH}::test_scenario_report_7641_interrupted_write_preserves_state",
                    "-q",
                ),
                "E2E-004 JSON state interruption and reload",
                300.0,
            ),
        ]
    )
    return commands


def terminal_commands(candidate: Path, root: Path) -> list[validation.CommandSpec]:
    """Build fresh reducers and strict readers for one exact candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact terminal candidate",
            60.0,
        ),
        validation.CommandSpec(
            "independent_raw_reduction",
            (python, "-u", wrapper, "--independent-replay", str(candidate)),
            "exact terminal candidate",
            60.0,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact terminal candidate",
            120.0,
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact terminal candidate",
            120.0,
        ),
    ]


def _all_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(row.get("passed") is True for row in receipts)


def run_experiment(root: Path, run_date: str, output: Path) -> JsonDict:  # pragma: no cover
    """Authenticate, exercise, validate, and atomically publish the consumer."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(started, "preconditions", "start", root=root)
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    context = collect_preconditions(root)
    destination = output if output.is_absolute() else root / output
    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    if context["blocker"] is not None:
        blocked = build_blocked_artifact(root, context["blocker"], raw)
        blocked["preconditions_checked"] = context["rows"]
        blocked["source_artifact_hashes"] = context["source_artifact_hashes"]
        blocked["duration_s"] = time.monotonic() - started
        blocked["field_principles"] = field_principles(
            [*blocked, "field_principles", "reproducibility_checksum"]
        )
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        atomic_json(destination, blocked)
        progress(started, "preconditions", "blocked", check=context["blocker"]["check"])
        return blocked
    progress(started, "preconditions", "complete", checks=len(context["rows"]))

    private = Path(tempfile.mkdtemp(prefix="carnot-exp7641-private-", dir="/tmp"))
    spans: list[JsonDict] = []
    phase_started = time.monotonic()
    progress(started, "integration", "before_native_calls", planned=12)
    rows = exercise_native_consumer(root, context["native_extension"], private / "integration")
    spans.append(_span("integration", phase_started, started, len(rows)))
    progress(started, "integration", "after_native_calls", completed=len(rows))
    reduce_integration_rows(rows)
    rows_path = raw / "integration_rows.json"
    atomic_json(rows_path, {"rows": rows})

    (private / "validation/basetemp").mkdir(parents=True, exist_ok=True)
    commands = build_validation_commands(root, private / "validation", context["native_extension"])
    affected = (
        MODULE_PATH,
        CLIENT_PATH,
        OLD_EXPERIMENT_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        EXP7626_TEST_PATH,
        EXAMPLE_PATH,
        REPORT_SPEC_PATH,
        PIPELINE_SPEC_PATH,
    )
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    manifest = {
        "schema": "carnot.exp7641.affected_manifest.v1",
        "frozen_before_validation": True,
        "worktree": str(root),
        "worktree_status_sha256": canonical_hash(status.stdout),
        "affected_files": [
            {"path": path.as_posix(), "sha256": sha256_file(root / path)} for path in affected
        ],
        "commands": [
            {
                "name": command.name,
                "argv": list(command.argv),
                "scope": command.scope,
                "timeout_s": command.timeout_s,
            }
            for command in commands
        ],
        "private_basetemp": str((private / "validation/basetemp").resolve()),
        "coverage_file": str((private / "validation/.coverage-exp7641").resolve()),
        "PYTHONPATH": f"{root / 'python'}:{root}",
        "extension": str(context["native_extension"].resolve()),
    }
    manifest_path = raw / "affected_validation_manifest.json"
    atomic_json(manifest_path, manifest)
    for path, role in (
        (rows_path, "current_raw_rows"),
        (manifest_path, "affected_validation_manifest"),
        (root / MODULE_PATH, "current_producer"),
        (root / CLIENT_PATH, "package_producer"),
        (root / WRAPPER_PATH, "current_entrypoint"),
        (root / TEST_PATH, "current_tests"),
        (root / EXAMPLE_PATH, "package_usage_example"),
    ):
        label = path.relative_to(root).as_posix()
        context["source_artifact_hashes"][label] = {
            "path": label,
            "role": role,
            "exists": True,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    progress(started, "validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected_receipts = validation.run_commands(
        root,
        commands,
        log_dir=raw / "validation/scoped",
        extra_env={
            "CARNOT_EXP7626_EXTENSION": str(context["native_extension"].resolve()),
            "COVERAGE_FILE": str((private / "validation/.coverage-exp7641").resolve()),
        },
        heartbeat_s=60.0,
    )
    spans.append(_span("validation", phase_started, started, len(affected_receipts)))
    progress(
        started,
        "validation",
        "after_subprocesses",
        passed=_all_passed(affected_receipts),
    )
    if not _all_passed(affected_receipts):
        failed = [row["name"] for row in affected_receipts if row.get("passed") is not True]
        raise RuntimeError("affected_validation_failed:" + ",".join(failed))

    provisional = build_artifact(
        root,
        rows,
        preconditions=context,
        validation_receipts=affected_receipts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate = private / "terminal-candidate.json"
    atomic_json(candidate, provisional)
    progress(started, "terminal_readers", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = validation.run_commands(
        root,
        terminal_commands(candidate, root),
        log_dir=raw / "validation/terminal_candidate",
        extra_env={"CARNOT_EXP7626_EXTENSION": str(context["native_extension"].resolve())},
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_readers", phase_started, started, len(terminal)))
    progress(started, "terminal_readers", "after_subprocesses", passed=_all_passed(terminal))
    if not _all_passed(terminal):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        root,
        rows,
        preconditions=context,
        validation_receipts=[*affected_receipts, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    final["started_at_utc"] = started_at
    final["completed_at_utc"] = datetime.now(UTC).isoformat()
    final["affected_validation_manifest_path"] = manifest_path.relative_to(root).as_posix()
    final["affected_validation_manifest_sha256"] = sha256_file(manifest_path)
    final["flagged_adversarial"] = not next(
        row for row in terminal if row["name"] == "adversarial_verify"
    )["passed"]
    final["field_principles"] = field_principles(
        [*final, "field_principles", "reproducibility_checksum"]
    )
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    exact = private / "exact-terminal-candidate.json"
    atomic_json(exact, final)

    progress(started, "exact_terminal_readers", "before_subprocesses", planned=4)
    exact_receipts = validation.run_commands(
        root,
        terminal_commands(exact, root),
        log_dir=raw / "validation/exact_terminal",
        extra_env={"CARNOT_EXP7626_EXTENSION": str(context["native_extension"].resolve())},
        heartbeat_s=60.0,
    )
    progress(
        started,
        "exact_terminal_readers",
        "after_subprocesses",
        passed=_all_passed(exact_receipts),
    )
    if not _all_passed(exact_receipts):
        raise RuntimeError("exact_terminal_validation_failed")
    atomic_json(raw / "exact_terminal_reader_receipts.json", {"receipts": exact_receipts})
    progress(started, "publication", "before_atomic_write", output=destination)
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(exact):
        raise RuntimeError("published_bytes_differ")
    progress(started, "publication", "after_atomic_write", bytes=destination.stat().st_size)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer and read-only replay modes for the thin wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode while keeping the repository wrapper thin."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        outcome = cold_replay(args.cold_replay)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(not outcome["valid"])
    if args.independent_replay is not None:
        outcome = independent_replay(args.independent_replay)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(not outcome["valid"])
    run_experiment(args.root.resolve(), args.date, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

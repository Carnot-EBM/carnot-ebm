#!/usr/bin/env python3
"""Build the REQ-REPORT-7147 V627 evidence matrix.

This module only reads checked-in JSON.  It never imports an upstream
experiment, reruns inference, or sends a hardware command.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = "openspec/capabilities/research-reporting/spec.md"
OUTPUT_PATH = "results/experiment_7147_v627_capstone.json"
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts: independent V627 evidence matrix"
)
LEGAL_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
DISPOSITIONS = {
    "promote",
    "continue",
    "retire",
    "repair",
    "blocked_pending_external_state",
}

UPSTREAM_SPECS: dict[int, dict[str, str]] = {
    7136: {
        "task": "contract_preflight",
        "path": "results/experiment_7136_v627_contract_preflight.json",
        "completion_field": "v627_task_contract_conforms_score",
    },
    7137: {
        "task": "source_delta",
        "path": "results/experiment_7137_v627_source_delta.json",
        "completion_field": "v627_source_delta_complete_score",
    },
    7138: {
        "task": "relational_fixture",
        "path": "results/experiment_7138_v627_relational_fixture.json",
        "completion_field": "source_grounding_fixture_ready_score",
    },
    7139: {
        "task": "symbolic_grounding_ab",
        "path": "results/experiment_7139_v627_symbolic_grounding_ab.json",
        "completion_field": "symbolic_grounding_complete_score",
    },
    7140: {
        "task": "symbolic_intervention_audit",
        "path": "results/experiment_7140_v627_symbolic_intervention_audit.json",
        "completion_field": "symbolic_intervention_audit_complete_score",
    },
    7141: {
        "task": "csl_event_stream",
        "path": "results/experiment_7141_v627_csl_event_stream.json",
        "completion_field": "csl_event_stream_ready_score",
    },
    7142: {
        "task": "flowbalance_memory_csl",
        "path": "results/experiment_7142_v627_flowbalance_memory_csl.json",
        "completion_field": "flowbalance_memory_csl_complete_score",
    },
    7143: {
        "task": "flowbalance_memory_cold_audit",
        "path": "results/experiment_7143_v627_flowbalance_memory_cold_audit.json",
        "completion_field": "flowbalance_memory_cold_audit_complete_score",
    },
    7144: {
        "task": "rebudgeted_arc_loo",
        "path": "results/experiment_7144_v627_rebudgeted_arc_loo.json",
        "completion_field": "arc_loo_cell_complete_score",
    },
    7145: {
        "task": "rust_multiscale_sampler",
        "path": "results/experiment_7145_v627_rust_multiscale_sampler.json",
        "completion_field": "rust_multiscale_parity_score",
    },
    7146: {
        "task": "gatemate_changed_state",
        "path": "results/experiment_7146_v627_gatemate_changed_state.json",
        "completion_field": "gatemate_terminal_receipt_score",
    },
}
UPSTREAM_ORDER = tuple(UPSTREAM_SPECS)

GATE_SPECS = (
    (7139, 7138, "source_grounding_fixture_ready_score", 1),
    (7140, 7139, "symbolic_grounding_complete_score", 1),
    (7142, 7141, "csl_event_stream_ready_score", 1),
    (7143, 7142, "flowbalance_memory_csl_complete_score", 1),
)

EXPLICIT_DEFERRALS = (
    "kan",
    "pwa_kan",
    "lora",
    "external_text_scorer",
    "grammar_induction",
    "fsnet",
    "within_chain_activation_steering",
    "arc_per_game_adapter",
    "arc_offline_ground_truth_bfs",
    "arc_public_registry_resolve",
    "arc_hidden_game_solve_claim",
    "learned_verifier_replacing_exact_authorities",
    "gemma_to_qwen_portability_matrix",
    "fpga_redesign",
    "extropic_runtime_claim",
    "kona_baseline_claim",
    "scaling_claim_from_host_only_benchmark",
)

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        "artifact_inventory_rows",
        "artifact_hash_rows",
        "gate_recompute_rows",
        "completion_rows",
        "verdict_class_rows",
        "row_count_rows",
        "source_grounding_rows",
        "symbolic_intervention_rows",
        "csl_rows",
        "cold_retention_rows",
        "arc_rows",
        "rust_sampler_rows",
        "gatemate_rows",
        "prd_gap_rows",
        "deferral_rows",
        "branch_disposition_rows",
        "inference_rerun_count",
        "hardware_command_count",
        "v627_capstone_complete_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)

ROW_FIELDS = (
    "rows",
    "artifact_inventory_rows",
    "artifact_hash_rows",
    "gate_recompute_rows",
    "completion_rows",
    "verdict_class_rows",
    "row_count_rows",
    "source_grounding_rows",
    "symbolic_intervention_rows",
    "csl_rows",
    "cold_retention_rows",
    "arc_rows",
    "rust_sampler_rows",
    "gatemate_rows",
    "prd_gap_rows",
    "deferral_rows",
    "branch_disposition_rows",
)


def _validate_date(run_date: str) -> None:
    """Reject labels that are not real YYYYMMDD calendar dates."""

    try:
        parsed = datetime.strptime(run_date, "%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise ValueError("run date must be a valid YYYYMMDD value") from exc
    if parsed.strftime("%Y%m%d") != run_date:  # pragma: no cover - defensive parser check
        raise ValueError("run date must be a valid YYYYMMDD value")


def _field_principles() -> dict[str, str]:
    """Explain why every required top-level field exists."""

    principles = {
        field: "This field preserves independently recomputed V627 capstone evidence."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "rows": "Inventory rows are the capstone's primary evidence units.",
            "artifact_inventory_rows": "One ordered row preserves each exact upstream path.",
            "v627_capstone_complete_score": (
                "One means the evidence matrix is structurally complete, not scientifically promoted."
            ),
            "inference_rerun_count": "The capstone must never rerun model inference.",
            "hardware_command_count": "The capstone must never issue a hardware command.",
            "verifier_is_oracle": "The capstone is an independent aggregator, not a scoring oracle.",
        }
    )
    return principles


def initialize_artifact(run_date: str) -> dict[str, Any]:
    """Return the complete fail-closed schema used before any file read."""

    _validate_date(run_date)
    artifact: dict[str, Any] = {
        "schema": "carnot.v627.capstone.v1",
        "experiment_id": 7147,
        "status": "blocked",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": {
            UPSTREAM_SPECS[number]["path"]: None for number in UPSTREAM_ORDER
        },
        "inference_rerun_count": 0,
        "hardware_command_count": 0,
        "v627_capstone_complete_score": 0,
        "random_seed": 714720260908,
        "reproducibility_checksum": None,
        "gate_check_summary": {
            "passed": False,
            "failed_check": "capstone_initialized",
            "expected_value": "upstream_checks_complete",
            "observed_value": "not_started",
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_capstone_initialized_before_upstream_checks",
    }
    artifact.update({field: [] for field in ROW_FIELDS})
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: dict[str, Any]) -> str:
    """Hash stable evidence while excluding wall time and the checksum itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def write_atomic(path: Path, artifact: dict[str, Any]) -> None:
    """Replace one JSON artifact only after its full bytes are ready."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_upstreams(root: Path) -> dict[int, dict[str, Any]]:
    """Read and hash the 11 exact files without importing producer code."""

    records: dict[int, dict[str, Any]] = {}
    for number in UPSTREAM_ORDER:
        spec = UPSTREAM_SPECS[number]
        path = root / spec["path"]
        record: dict[str, Any] = {
            "experiment": number,
            "task": spec["task"],
            "path": spec["path"],
            "present": path.is_file(),
            "size_bytes": None,
            "sha256": None,
            "payload": None,
            "read_error": None,
        }
        if not record["present"]:
            record["read_error"] = "artifact_not_found"
            records[number] = record
            continue
        try:
            raw = path.read_bytes()
            record["size_bytes"] = len(raw)
            record["sha256"] = f"sha256:{hashlib.sha256(raw).hexdigest()}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                record["read_error"] = "json_root_not_object"
            else:
                record["payload"] = payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            record["read_error"] = f"{type(exc).__name__}:{exc}"
        records[number] = record
    return records


def recompute_gates(records: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate dependency contracts from bare producer fields only."""

    rows = []
    for consumer, producer, field, expected in GATE_SPECS:
        producer_payload = records.get(producer, {}).get("payload")
        observed = producer_payload.get(field) if isinstance(producer_payload, dict) else None
        consumer_payload = records.get(consumer, {}).get("payload")
        rows.append(
            {
                "consumer_experiment": consumer,
                "producer_experiment": producer,
                "producer_path": UPSTREAM_SPECS[producer]["path"],
                "field": field,
                "expected_value": expected,
                "observed_value": observed,
                "producer_present": isinstance(producer_payload, dict),
                "consumer_present": isinstance(consumer_payload, dict),
                "passed": observed == expected,
            }
        )
    return rows


def recompute_verdict(
    payload: dict[str, Any] | None, gate_passed: bool | None
) -> str:
    """Lower a declared class whenever rows, gate, or oracle cannot support it."""

    if payload is None:
        return "blocked"
    declared = str(payload.get("verdict_class", "blocked")).strip().lower()
    honest = str(payload.get("honest_verdict", "")).strip().lower()
    if honest.startswith("blocked") or declared == "blocked":
        return "blocked"
    if honest.startswith("disqualified") or declared == "disqualified":
        return "disqualified"
    if honest.startswith("null") or declared == "null":
        return "null"
    if honest.startswith("partial") or declared == "partial":
        return "partial"
    if declared not in LEGAL_VERDICTS:
        return "disqualified"
    if gate_passed is False:
        return "blocked"
    if declared in {"positive", "circular_positive"}:
        if payload.get("verifier_is_oracle") is True:
            return "circular_positive"
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            return "disqualified"
    return declared


def _upstream_validation(
    record: dict[str, Any], gate_passed: bool | None
) -> tuple[str, list[str]]:
    """Check the evidence fields needed by this matrix."""

    if not record["present"]:
        return "missing", ["artifact_not_found"]
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return "unreadable", [str(record.get("read_error"))]
    failures = []
    for field in (
        "run_date",
        "inference_substrate",
        "execution_venue",
        "verdict_class",
        "honest_verdict",
    ):
        if field not in payload:
            failures.append(f"missing_{field}")
    if payload.get("verdict_class") not in LEGAL_VERDICTS:
        failures.append("illegal_verdict_class")
    declared = payload.get("verdict_class")
    recomputed = recompute_verdict(payload, gate_passed)
    if declared != recomputed:
        failures.append(f"verdict_ceiling:{declared}->{recomputed}")
    if recomputed == "blocked":
        gate = payload.get("gate_check_summary")
        if not isinstance(gate, dict):
            failures.append("blocked_gate_summary_missing")
        elif any(gate.get(key) is None for key in ("failed_check", "expected_value", "observed_value")):
            failures.append("blocked_gate_summary_incomplete")
    return ("valid" if not failures else "inconsistent"), failures


def build_inventory_rows(
    records: dict[int, dict[str, Any]], gates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build one auditable row for each fixed upstream slot."""

    consumer_gates = {row["consumer_experiment"]: row for row in gates}
    rows = []
    for number in UPSTREAM_ORDER:
        record = records[number]
        payload = record.get("payload")
        gate = consumer_gates.get(number)
        gate_passed = gate["passed"] if gate else None
        validation_status, validation_findings = _upstream_validation(
            record, gate_passed
        )
        declared_rows = payload.get("rows") if isinstance(payload, dict) else None
        completion_field = UPSTREAM_SPECS[number]["completion_field"]
        row = {
            "experiment": number,
            "task": UPSTREAM_SPECS[number]["task"],
            "path": record["path"],
            "present": record["present"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
            "run_date": payload.get("run_date") if isinstance(payload, dict) else None,
            "inference_substrate": payload.get("inference_substrate") if isinstance(payload, dict) else None,
            "inference_substrate_class": payload.get("inference_substrate_class") if isinstance(payload, dict) else None,
            "execution_venue": payload.get("execution_venue") if isinstance(payload, dict) else None,
            "declared_verdict_class": payload.get("verdict_class") if isinstance(payload, dict) else None,
            "recomputed_verdict_class": recompute_verdict(payload, gate_passed),
            "honest_verdict": payload.get("honest_verdict") if isinstance(payload, dict) else "blocked_artifact_not_found",
            "verifier_is_oracle": payload.get("verifier_is_oracle") if isinstance(payload, dict) else None,
            "completion_field": completion_field,
            "completion_value": payload.get(completion_field) if isinstance(payload, dict) else None,
            "row_count": len(declared_rows) if isinstance(declared_rows, list) else None,
            "validation_status": validation_status,
            "validation_findings": validation_findings,
            "structured_gate_value": gate,
        }
        row["task_complete"] = row["completion_value"] == 1
        row["scientific_promotion_eligible"] = (
            row["recomputed_verdict_class"] == "positive"
            and row["verifier_is_oracle"] is False
            and gate_passed is not False
            and validation_status == "valid"
        )
        rows.append(row)
    return rows


def recompute_source_grounding(
    fixture: dict[str, Any] | None, comparison: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Count source-grounding coverage from explicit fixture rows."""

    rows = []
    if fixture is not None:
        loader_rows = fixture.get("independent_loader_rows", [])
        rows.append(
            {
                "experiment": 7138,
                "fixture_row_count": len(fixture.get("fixture_rows", [])),
                "model_view_row_count": len(fixture.get("model_view_rows", [])),
                "sealed_scorer_row_count": len(fixture.get("sealed_scorer_rows", [])),
                "independent_loader_row_count": len(loader_rows),
                "independent_loader_all_passed": bool(loader_rows)
                and all(row.get("passed") is True for row in loader_rows),
                "label_exposure_count": fixture.get("label_exposure_count"),
                "measurement_available": bool(fixture.get("fixture_rows", [])),
            }
        )
    if comparison is not None:
        arm_rows = comparison.get("arm_rows", comparison.get("rows", []))
        rows.append(
            {
                "experiment": 7139,
                "comparison_row_count": len(arm_rows),
                "measurement_available": bool(arm_rows),
            }
        )
    return rows


def recompute_symbolic_interventions(
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Count each intervention class without treating absent rows as zero."""

    if payload is None:
        return []
    return [
        {
            "experiment": 7140,
            "useful_count": len(payload.get("useful_intervention_rows", [])),
            "harmful_count": len(payload.get("harmful_intervention_rows", [])),
            "null_count": len(payload.get("null_intervention_rows", [])),
            "measurement_available": any(
                isinstance(payload.get(field), list)
                for field in (
                    "useful_intervention_rows",
                    "harmful_intervention_rows",
                    "null_intervention_rows",
                )
            ),
        }
    ]


def _rates_by_arm(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    """Average a binary per-unit field within each named arm."""

    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm")
        value = row.get(field)
        if isinstance(arm, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
            values[arm].append(float(value))
    return {arm: sum(items) / len(items) for arm, items in sorted(values.items())}


def recompute_csl(
    stream: dict[str, Any] | None, learning: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Recompute stream and learning outcomes from their event rows."""

    rows = []
    if stream is not None:
        events = stream.get("event_rows", [])
        rows.append(
            {
                "experiment": 7141,
                "event_count": len(events),
                "split_counts": dict(sorted(Counter(row.get("split") for row in events).items())),
                "measurement_available": bool(events),
            }
        )
    if learning is not None:
        future = learning.get("future_success_rows", [])
        retention = learning.get("protected_retention_rows", [])
        rows.append(
            {
                "experiment": 7142,
                "learning_event_count": len(learning.get("event_rows", [])),
                "future_success_row_count": len(future),
                "future_success_rates": _rates_by_arm(future, "exact_success"),
                "protected_retention_row_count": len(retention),
                "protected_retention_rates": _rates_by_arm(retention, "exact_success"),
                "measurement_available": bool(future) and bool(retention),
            }
        )
    return rows


def recompute_cold_retention(
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Recompute the cold retention audit only when its rows exist."""

    if payload is None:
        return []
    retention = payload.get("protected_retention_rows", [])
    negative = payload.get("negative_transfer_rows", [])
    numeric = [
        float(row["negative_transfer"])
        for row in negative
        if isinstance(row.get("negative_transfer"), (int, float))
        and not isinstance(row.get("negative_transfer"), bool)
    ]
    return [
        {
            "experiment": 7143,
            "protected_retention_row_count": len(retention),
            "protected_retention_rates": _rates_by_arm(retention, "exact_success"),
            "negative_transfer_row_count": len(negative),
            "negative_transfer_rate": sum(numeric) / len(numeric) if numeric else None,
            "measurement_available": bool(retention) and bool(negative),
        }
    ]


def _common_request_configuration(request_rows: list[dict[str, Any]]) -> bool | None:
    """Compare configuration fields while excluding outputs and arm identity."""

    if not request_rows:
        return None
    fields = ("budget", "model_hash", "model_repository", "tools_hash", "seed", "executable_environment_hash")
    signatures = {
        json.dumps({field: row.get(field) for field in fields}, sort_keys=True)
        for row in request_rows
    }
    arms = {row.get("arm") for row in request_rows}
    return len(arms) >= 2 and len(signatures) == 1


def recompute_arc(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Recompute ARC controls from arm, isolation, and execution rows."""

    if payload is None:
        return []
    arm_rows = payload.get("rows", [])
    levels = {
        row.get("arm"): row.get("levels")
        for row in arm_rows
        if isinstance(row.get("levels"), (int, float))
    }
    withheld = levels.get("adapter_withheld")
    visible = levels.get("adapter_visible_control")
    forbidden = [
        row for row in payload.get("forbidden_read_rows", [])
        if row.get("arm") == "adapter_withheld"
    ]
    imports = [
        row for row in payload.get("import_rows", [])
        if row.get("arm") == "adapter_withheld"
    ]
    isolation_clean = bool(forbidden) and bool(imports) and all(
        row.get("passed") is True and not row.get("forbidden_reads") for row in forbidden
    ) and all(
        row.get("target_adapter_module_loaded") is False
        and row.get("target_recipe_symbols_loaded") is False
        for row in imports
    )
    real_outputs = [
        row for row in payload.get("truncation_rows", []) if row.get("real_output") is True
    ]
    action_rows = payload.get("action_rows", [])
    return [
        {
            "experiment": 7144,
            "withheld_levels": withheld,
            "visible_control_levels": visible,
            "level_delta": withheld - visible
            if isinstance(withheld, (int, float)) and isinstance(visible, (int, float))
            else None,
            "visible_control_nonzero": visible > 0 if isinstance(visible, (int, float)) else None,
            "isolation_clean": isolation_clean,
            "common_arm_configuration": _common_request_configuration(payload.get("request_rows", [])),
            "real_output_count": len(real_outputs),
            "all_real_outputs_truncated": bool(real_outputs)
            and all(row.get("truncated") is True for row in real_outputs),
            "executed_action_count": sum(row.get("executed") is True for row in action_rows),
            "input_difference_measured": any(
                row.get("measured_difference") is True
                for row in payload.get("input_difference_rows", [])
            ),
            "policy_difference_measured": any(
                row.get("measured_difference") is True
                for row in payload.get("policy_difference_rows", [])
            ),
            "measurement_available": bool(arm_rows),
        }
    ]


def recompute_rust(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Require every parity family and both benchmark implementations."""

    if payload is None:
        return []
    parity_fields = (
        "fixture_rows",
        "normalization_rows",
        "support_rows",
        "detailed_balance_rows",
        "stationarity_rows",
        "chain_rows",
    )
    parity_rows = [payload.get(field, []) for field in parity_fields]
    exact_parity = None
    if all(rows for rows in parity_rows):
        exact_parity = all(row.get("passed") is True for rows in parity_rows for row in rows)
    benchmarks: dict[str, list[float]] = defaultdict(list)
    for row in payload.get("benchmark_rows", []):
        implementation = row.get("implementation")
        throughput = row.get("updates_per_s")
        if isinstance(implementation, str) and isinstance(throughput, (int, float)):
            benchmarks[implementation].append(float(throughput))
    ratio = None
    if benchmarks.get("python") and benchmarks.get("rust"):
        python_rate = sum(benchmarks["python"]) / len(benchmarks["python"])
        rust_rate = sum(benchmarks["rust"]) / len(benchmarks["rust"])
        ratio = rust_rate / python_rate if python_rate else None
    return [
        {
            "experiment": 7145,
            "parity_row_counts": {
                field: len(rows) for field, rows in zip(parity_fields, parity_rows)
            },
            "exact_parity": exact_parity,
            "benchmark_row_count": sum(len(rows) for rows in benchmarks.values()),
            "rust_over_python_throughput": ratio,
            "measurement_available": exact_parity is not None and ratio is not None,
        }
    ]


def recompute_gatemate(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Count receipts and commands without executing either hardware path."""

    if payload is None:
        return []
    receipts = payload.get("receipt_rows", [])
    commands = payload.get("command_rows", [])
    valid_receipts = [row for row in receipts if row.get("valid") is True]
    successful_commands = [
        row for row in commands if row.get("returncode") == 0 or row.get("passed") is True
    ]
    return [
        {
            "experiment": 7146,
            "receipt_count": len(receipts),
            "valid_receipt_count": len(valid_receipts),
            "command_count": len(commands),
            "successful_command_count": len(successful_commands),
            "changed_state_available": bool(valid_receipts),
            "terminal_external_block": not valid_receipts,
        }
    ]


def _citation(number: int, field: str, selector: str, value: Any) -> dict[str, Any]:
    """Name the exact evidence location behind one disposition."""

    return {
        "artifact_path": UPSTREAM_SPECS[number]["path"],
        "field": field,
        "row_selector": selector,
        "observed_value": value,
    }


def _branch_dispositions(
    records: dict[int, dict[str, Any]], inventory: list[dict[str, Any]],
    arc_rows: list[dict[str, Any]], rust_rows: list[dict[str, Any]],
    gatemate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign one evidence-cited action to every V627 branch."""

    by_number = {row["experiment"]: row for row in inventory}
    contract_ok = by_number[7136]["completion_value"] == 1
    source_done = by_number[7137]["completion_value"] == 1
    symbolic_positive = by_number[7140]["recomputed_verdict_class"] == "positive"
    csl_positive = by_number[7143]["recomputed_verdict_class"] == "positive"
    arc = arc_rows[0] if arc_rows else {}
    rust = rust_rows[0] if rust_rows else {}
    gatemate = gatemate_rows[0] if gatemate_rows else {}
    arc_class = by_number[7144]["recomputed_verdict_class"]
    return [
        {
            "branch": "contract_intake",
            "task_complete": contract_ok,
            "scientific_promotion": False,
            "disposition": "promote" if contract_ok else "repair",
            "citation": _citation(7136, UPSTREAM_SPECS[7136]["completion_field"], "top_level", by_number[7136]["completion_value"]),
        },
        {
            "branch": "source_intake",
            "task_complete": source_done,
            "scientific_promotion": False,
            "disposition": "continue" if source_done else "repair",
            "citation": _citation(7137, UPSTREAM_SPECS[7137]["completion_field"], "top_level", by_number[7137]["completion_value"]),
        },
        {
            "branch": "source_grounded_verification",
            "task_complete": by_number[7139]["task_complete"] and by_number[7140]["task_complete"],
            "scientific_promotion": symbolic_positive,
            "disposition": "promote" if symbolic_positive else "repair",
            "citation": _citation(7139, "gate_check_summary", "top_level.failed_check", (records[7139].get("payload") or {}).get("gate_check_summary", {}).get("failed_check")),
        },
        {
            "branch": "continuous_self_learning",
            "task_complete": by_number[7142]["task_complete"] and by_number[7143]["task_complete"],
            "scientific_promotion": csl_positive,
            "disposition": "promote" if csl_positive else "repair",
            "citation": _citation(7142, "gate_check_summary", "top_level.failed_check", (records[7142].get("payload") or {}).get("gate_check_summary", {}).get("failed_check")),
        },
        {
            "branch": "arc_generalization",
            "task_complete": by_number[7144]["task_complete"],
            "scientific_promotion": arc_class == "positive" and arc.get("isolation_clean") is True,
            "disposition": "promote" if arc_class == "positive" and arc.get("isolation_clean") is True else ("retire" if arc_class == "disqualified" else "repair"),
            "citation": _citation(7144, "forbidden_read_rows", "arm=adapter_withheld", arc.get("isolation_clean")),
        },
        {
            "branch": "rust_sampling",
            "task_complete": by_number[7145]["task_complete"],
            "scientific_promotion": rust.get("exact_parity") is True,
            "disposition": "promote" if rust.get("exact_parity") is True else "repair",
            "citation": _citation(7145, "fixture_rows", "all parity families", rust.get("exact_parity")),
        },
        {
            "branch": "gatemate_continuity",
            "task_complete": by_number[7146]["task_complete"],
            "scientific_promotion": False,
            "disposition": "blocked_pending_external_state" if gatemate.get("terminal_external_block") else ("promote" if gatemate.get("successful_command_count", 0) > 0 else "repair"),
            "citation": _citation(7146, "receipt_rows", "valid=true", gatemate.get("valid_receipt_count")),
        },
    ]


def _prd_gaps(
    source_rows: list[dict[str, Any]], csl_rows: list[dict[str, Any]],
    arc_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the three named PRD gaps separate from task completion."""

    source_measurement = any(row.get("experiment") == 7140 and row.get("measurement_available") for row in source_rows)
    csl_measurement = any(row.get("experiment") == 7142 and row.get("measurement_available") for row in csl_rows)
    arc = arc_rows[0] if arc_rows else {}
    return [
        {
            "gap": "useful_source_grounded_verification",
            "status": "closed" if source_measurement else "repair",
            "citation": _citation(7140, "useful_intervention_rows", "all rows", source_measurement),
        },
        {
            "gap": "model_facing_continuous_self_learning",
            "status": "closed" if csl_measurement else "repair",
            "citation": _citation(7142, "future_success_rows", "all arms", csl_measurement),
        },
        {
            "gap": "nondegenerate_arc_generalization",
            "status": "closed" if arc.get("isolation_clean") is True else "retire_current_cell",
            "citation": _citation(7144, "forbidden_read_rows", "arm=adapter_withheld", arc.get("isolation_clean")),
        },
    ]


def _deferral_rows() -> list[dict[str, Any]]:
    """Preserve every explicit deferral without manufacturing a result."""

    return [
        {
            "deferral": name,
            "status": "deferred",
            "evidence_result_created": False,
            "reopened": False,
            "source": "openspec/change-proposals/research-roadmap-vNEXT.md",
        }
        for name in EXPLICIT_DEFERRALS
    ]


def _matrix_complete(artifact: dict[str, Any]) -> bool:
    """Measure matrix structure without treating it as branch promotion."""

    return (
        len(artifact.get("artifact_inventory_rows", [])) == 11
        and len(artifact.get("artifact_hash_rows", [])) == 11
        and len(artifact.get("completion_rows", [])) == 11
        and len(artifact.get("verdict_class_rows", [])) == 11
        and len(artifact.get("row_count_rows", [])) == 11
        and len(artifact.get("gate_recompute_rows", [])) == 4
        and len(artifact.get("branch_disposition_rows", [])) == 7
        and len(artifact.get("prd_gap_rows", [])) == 3
        and len(artifact.get("deferral_rows", [])) == len(EXPLICIT_DEFERRALS)
        and all(
            row.get("disposition") in DISPOSITIONS
            and isinstance(row.get("citation"), dict)
            and bool(row["citation"].get("field"))
            and bool(row["citation"].get("row_selector"))
            for row in artifact.get("branch_disposition_rows", [])
        )
    )


def build_artifact(
    root: Path,
    run_date: str,
    output: Path,
    *,
    loader: Callable[[Path], dict[int, dict[str, Any]]] = load_upstreams,
) -> dict[str, Any]:
    """Initialize, independently aggregate, validate, and atomically finish."""

    started = time.monotonic()
    artifact = initialize_artifact(run_date)
    write_atomic(output, artifact)
    records = loader(root)
    gates = recompute_gates(records)
    inventory = build_inventory_rows(records, gates)
    payload = lambda number: records[number].get("payload")
    source_rows = recompute_source_grounding(payload(7138), payload(7139))
    symbolic_rows = recompute_symbolic_interventions(payload(7140))
    csl_rows = recompute_csl(payload(7141), payload(7142))
    cold_rows = recompute_cold_retention(payload(7143))
    arc_rows = recompute_arc(payload(7144))
    rust_rows = recompute_rust(payload(7145))
    gatemate_rows = recompute_gatemate(payload(7146))

    artifact.update(
        {
            "preconditions_checked": [
                {"check": "complete_schema_written_before_load", "passed": True},
                *[
                    {
                        "check": "exact_upstream_path_readable",
                        "experiment": number,
                        "path": records[number]["path"],
                        "passed": records[number].get("payload") is not None,
                        "observed_value": records[number].get("read_error") or "readable",
                    }
                    for number in UPSTREAM_ORDER
                ],
            ],
            "source_artifact_hashes": {
                row["path"]: row["sha256"] for row in inventory
            },
            "artifact_inventory_rows": inventory,
            "artifact_hash_rows": [
                {
                    "experiment": row["experiment"],
                    "path": row["path"],
                    "present": row["present"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
                for row in inventory
            ],
            "gate_recompute_rows": gates,
            "completion_rows": [
                {
                    "experiment": row["experiment"],
                    "field": row["completion_field"],
                    "value": row["completion_value"],
                    "task_complete": row["task_complete"],
                }
                for row in inventory
            ],
            "verdict_class_rows": [
                {
                    "experiment": row["experiment"],
                    "declared": row["declared_verdict_class"],
                    "recomputed": row["recomputed_verdict_class"],
                    "validation_status": row["validation_status"],
                    "scientific_promotion_eligible": row["scientific_promotion_eligible"],
                }
                for row in inventory
            ],
            "row_count_rows": [
                {"experiment": row["experiment"], "row_count": row["row_count"]}
                for row in inventory
            ],
            "source_grounding_rows": source_rows,
            "symbolic_intervention_rows": symbolic_rows,
            "csl_rows": csl_rows,
            "cold_retention_rows": cold_rows,
            "arc_rows": arc_rows,
            "rust_sampler_rows": rust_rows,
            "gatemate_rows": gatemate_rows,
            "deferral_rows": _deferral_rows(),
        }
    )
    artifact["branch_disposition_rows"] = _branch_dispositions(
        records, inventory, arc_rows, rust_rows, gatemate_rows
    )
    artifact["prd_gap_rows"] = _prd_gaps(symbolic_rows, csl_rows, arc_rows)
    artifact["rows"] = [
        {
            "matrix_slot_coverage_rate": len(inventory) / len(UPSTREAM_ORDER),
            "present_artifact_rate": sum(row["present"] for row in inventory)
            / len(UPSTREAM_ORDER),
            "dependency_gate_pass_rate": sum(row["passed"] for row in gates)
            / len(GATE_SPECS),
            "scientific_branch_promotion_rate": sum(
                row["scientific_promotion"]
                for row in artifact["branch_disposition_rows"]
            )
            / 7,
        }
    ]
    artifact["v627_capstone_complete_score"] = int(_matrix_complete(artifact))

    missing = [row["experiment"] for row in inventory if not row["present"]]
    unreadable = [
        row["experiment"]
        for row in inventory
        if row["present"] and row["validation_status"] == "unreadable"
    ]
    external_blocks = [
        "gatemate_operator_receipt"
        for row in gatemate_rows
        if row.get("terminal_external_block") is True
        and by_number_class(inventory, 7146) == "blocked"
    ]
    if missing or unreadable or external_blocks:
        artifact.update(
            {
                "status": "blocked",
                "inference_substrate_class": "blocked_no_run",
                "verdict_class": "blocked",
                "honest_verdict": "blocked_upstream_terminal_availability_v627_matrix_complete",
                "gate_check_summary": {
                    "passed": False,
                    "failed_check": "upstream_terminal_availability",
                    "expected_value": {
                        "missing_artifacts": [],
                        "unreadable_artifacts": [],
                        "external_state_blocks": [],
                    },
                    "observed_value": {
                        "missing_artifacts": missing,
                        "unreadable_artifacts": unreadable,
                        "external_state_blocks": external_blocks,
                    },
                },
            }
        )
    else:
        artifact.update(
            {
                "status": "complete",
                "inference_substrate_class": "aggregation",
                "verdict_class": "positive",
                "honest_verdict": "complete_positive_v627_evidence_matrix_without_branch_promotion",
                "gate_check_summary": {
                    "passed": True,
                    "failed_check": None,
                    "expected_value": "eleven_readable_upstream_artifacts",
                    "observed_value": "eleven_readable_upstream_artifacts",
                },
            }
        )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_atomic(output, artifact)
    return artifact


def by_number_class(inventory: list[dict[str, Any]], number: int) -> str | None:
    """Return one matrix verdict without trusting the producer headline."""

    return next(
        (row["recomputed_verdict_class"] for row in inventory if row["experiment"] == number),
        None,
    )


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Independently check the completed capstone's derived structure."""

    failures: list[str] = []
    missing_fields = sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
    failures.extend(f"missing_field:{field}" for field in missing_fields)
    if missing_fields:
        return failures
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or any(
        not isinstance(principles.get(field), str) or not principles[field].strip()
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        failures.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        failures.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") not in {"aggregation", "blocked_no_run"}:
        failures.append("inference_substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        failures.append("execution_venue_mismatch")
    for field in ("inference_rerun_count", "hardware_command_count"):
        if artifact.get(field) != 0:
            failures.append(f"{field}_nonzero")
    inventory = artifact.get("artifact_inventory_rows")
    gates = artifact.get("gate_recompute_rows")
    branches = artifact.get("branch_disposition_rows")
    expected_primary_rows = None
    if isinstance(inventory, list) and isinstance(gates, list) and isinstance(branches, list):
        expected_primary_rows = [
            {
                "matrix_slot_coverage_rate": len(inventory) / len(UPSTREAM_ORDER),
                "present_artifact_rate": sum(row.get("present") is True for row in inventory)
                / len(UPSTREAM_ORDER),
                "dependency_gate_pass_rate": sum(row.get("passed") is True for row in gates)
                / len(GATE_SPECS),
                "scientific_branch_promotion_rate": sum(
                    row.get("scientific_promotion") is True for row in branches
                )
                / 7,
            }
        ]
    if artifact.get("rows") != expected_primary_rows:
        failures.append("primary_rows_mismatch")
    if not isinstance(inventory, list) or [row.get("experiment") for row in inventory] != list(UPSTREAM_ORDER):
        failures.append("artifact_inventory_slots_invalid")
    expected_lengths = {
        "artifact_hash_rows": 11,
        "completion_rows": 11,
        "verdict_class_rows": 11,
        "row_count_rows": 11,
        "gate_recompute_rows": 4,
        "branch_disposition_rows": 7,
        "prd_gap_rows": 3,
        "deferral_rows": len(EXPLICIT_DEFERRALS),
    }
    for field, expected in expected_lengths.items():
        rows = artifact.get(field)
        if not isinstance(rows, list) or len(rows) != expected:
            failures.append(f"{field}_count_mismatch")
    branches = artifact.get("branch_disposition_rows", [])
    if isinstance(branches, list):
        names = [row.get("branch") for row in branches if isinstance(row, dict)]
        if len(names) != len(set(names)):
            failures.append("branch_dispositions_not_unique")
        for row in branches:
            citation = row.get("citation", {}) if isinstance(row, dict) else {}
            if row.get("disposition") not in DISPOSITIONS:
                failures.append("branch_disposition_invalid")
            if not citation.get("field") or not citation.get("row_selector"):
                failures.append("branch_citation_incomplete")
    expected_score = int(_matrix_complete(artifact))
    if artifact.get("v627_capstone_complete_score") != expected_score:
        failures.append("v627_capstone_complete_score_mismatch")
    verdict = artifact.get("verdict_class")
    honest = str(artifact.get("honest_verdict", ""))
    if verdict not in LEGAL_VERDICTS:
        failures.append("verdict_class_invalid")
    if verdict == "blocked":
        summary = artifact.get("gate_check_summary", {})
        if not honest.startswith("blocked_"):
            failures.append("blocked_honest_verdict_prefix_mismatch")
        if not isinstance(summary, dict) or summary.get("passed") is not False:
            failures.append("blocked_gate_summary_invalid")
        elif any(summary.get(field) is None for field in ("failed_check", "expected_value", "observed_value")):
            failures.append("blocked_gate_summary_incomplete")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            failures.append("blocked_substrate_class_mismatch")
    elif verdict == "positive" and not honest.startswith("complete_positive"):
        failures.append("positive_honest_verdict_prefix_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        failures.append("capstone_oracle_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        failures.append("reproducibility_checksum_mismatch")
    return failures


def _parser() -> argparse.ArgumentParser:
    """Define the generation and offline-validation command surface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260908")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write a capstone or validate one already present on disk."""

    args = _parser().parse_args(argv)
    output = args.output or args.root / OUTPUT_PATH
    if args.validate:
        try:
            artifact = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(json.dumps({"status": "invalid", "failures": [str(exc)]}, sort_keys=True))
            return 1
        failures = validate_artifact(artifact) if isinstance(artifact, dict) else ["json_root_not_object"]
        print(json.dumps({"status": "valid" if not failures else "invalid", "failures": failures}, sort_keys=True))
        return int(bool(failures))
    artifact = build_artifact(args.root, args.date, output)
    failures = validate_artifact(artifact)
    print(json.dumps({"status": "valid" if not failures else "invalid", "failures": failures}, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

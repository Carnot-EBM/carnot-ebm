"""Measure native whole-response and lossless-window option transport.

The pilot uses the sealed V656 prompts and records native final-position logits.
It measures transport and cost only. It does not treat predictions as benefit.

Spec refs: REQ-VERIFY-7492 and SCENARIO-VERIFY-7492-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_7462_v654_option_protocol as option_protocol
from carnot import experiment_7463_v654_semif_e0_logprob_parity as parity
from carnot import experiment_7477_v655_native_readout_pilot as native_pilot
from carnot import experiment_7491_v656_window_protocol as window_protocol
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model, current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7492-v656-window-pilot"
SCHEMA = "carnot.exp7492.v656.window_pilot.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
model_specs = [MODEL_HF_ID]
INFERENCE_SUBSTRATE = "live_native_llama_cpp_whole_and_lossless_window_raw_logit_readout"
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
FORWARD_BUDGET = 72
CAPTURE_HARD_CAP_S = 3300.0
FORECAST_MULTIPLIER = 1.25
MAX_LIVE_SECONDS = 3300.0
OPTION_IDS = option_protocol.OPTION_IDS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7492_v656_window_pilot.json")
RAW_DIR = Path("results/raw/experiment_7492_v656_window_pilot")
MODULE_PATH = Path("python/carnot/experiment_7492_v656_window_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7492_v656_window_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7492_v656_window_pilot.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
UPSTREAM_PATH = REPO_ROOT / "results/experiment_7491_v656_window_protocol.json"
UPSTREAM_RAW_DIR = Path("results/raw/experiment_7491_v656_window_protocol")
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))

# These identities were frozen from predictor-safe Exp7491 rows. They span one
# through 25 sentences and 177 through 1,737 response bytes.
PILOT_GROUP_IDS = (
    "group-09d602e208b1ba374f1ce750",
    "group-82c77df1d61cdbc0db52f496",
    "group-52ae6c79dc729baad26b73ee",
    "group-3b67d3bfc240d517bf7d7c15",
    "group-0eea728263b0feff7547e8a2",
    "group-4651d6036d7a81745d98755b",
    "group-a3bd6000f235dbcdb010a05e",
    "group-e519c22db6da8d28998851ef",
)

UPSTREAM_EXPECTED = {
    "schema": "carnot.exp7491.v656.window_protocol.v1",
    "experiment_id": "exp7491-v656-window-protocol",
    "milestone": MILESTONE,
    "terminal_status": "complete",
    "honest_verdict": "complete_null_structural_window_protocol_ready",
    "verdict_class": "null",
    "flagged_adversarial": False,
    "window_protocol_ready_score": 1,
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7477_v655_native_readout_pilot.py"),
    Path("python/carnot/experiment_7462_v654_option_protocol.py"),
    Path("python/carnot/experiment_7449_v653_source_protocol.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("results/experiment_7477_v655_native_readout_pilot.json"),
    Path("results/experiment_7491_v656_window_protocol.json"),
    UPSTREAM_RAW_DIR / "predictors.jsonl",
    UPSTREAM_RAW_DIR / "groups.jsonl",
    UPSTREAM_RAW_DIR / "windows.jsonl",
    UPSTREAM_RAW_DIR / "requests.jsonl",
    Path("openspec/capabilities/verification/spec.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


class WindowPilotError(ValueError):
    """Reject changed protocol identity or incomplete native evidence."""


def _principle_row(
    check: str, expected: Any, observed: Any, *, field_path: str, upstream: str
) -> JsonDict:
    """Attach the exact operand and the failure that this check prevents."""

    return {
        "check": check,
        "upstream": upstream,
        "field_path": field_path,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "principle": "Exact prerequisite identity prevents guessed or laundered producer evidence.",
    }


def reduce_upstream_gate(value: Mapping[str, Any], *, validator_errors: Sequence[str]) -> JsonDict:
    """Require the complete Exp7491 structure before model work can start."""

    checks = [
        _principle_row(
            f"upstream_{field}",
            expected,
            value.get(field),
            field_path=field,
            upstream="results/experiment_7491_v656_window_protocol.json",
        )
        for field, expected in UPSTREAM_EXPECTED.items()
    ]
    gates = value.get("acceptance_gate_results")
    gates_passed = (
        isinstance(gates, list)
        and bool(gates)
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in gates)
    )
    checks.extend(
        (
            _principle_row(
                "upstream_acceptance_gates",
                True,
                gates_passed,
                field_path="acceptance_gate_results[*].passed",
                upstream="results/experiment_7491_v656_window_protocol.json",
            ),
            _principle_row(
                "upstream_gate_summary",
                True,
                (value.get("gate_check_summary") or {}).get("passed"),
                field_path="gate_check_summary.passed",
                upstream="results/experiment_7491_v656_window_protocol.json",
            ),
            _principle_row(
                "upstream_cold_validator",
                [],
                list(validator_errors),
                field_path="cold_validator_errors",
                upstream="carnot.experiment_7491_v656_window_protocol.validate_artifact",
            ),
        )
    )
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def _request_key(row: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], int | None]:
    return (
        str(row.get("group_id")),
        str(row.get("arm")),
        tuple(str(item) for item in row.get("option_order") or []),
        None if row.get("window_index") is None else int(row["window_index"]),
    )


def build_pilot_schedule(
    predictors: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Rebuild the eight frozen groups and match every prompt to Exp7491."""

    predictor_by_id = {str(row.get("group_id")): row for row in predictors}
    group_by_id = {str(row.get("group_id")): row for row in group_rows}
    windows_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in window_rows:
        windows_by_id.setdefault(str(row.get("group_id")), []).append(row)
    requests = {_request_key(row): row for row in request_rows}
    schedule: list[JsonDict] = []
    orders = (OPTION_IDS, tuple(reversed(OPTION_IDS)))
    for group_id in PILOT_GROUP_IDS:
        predictor = predictor_by_id.get(group_id)
        group = group_by_id.get(group_id)
        if predictor is None or group is None:
            raise WindowPilotError(f"frozen_group_missing:{group_id}")
        if group.get("role") != "training" or group.get("eligible") is not True:
            raise WindowPilotError(f"frozen_group_not_eligible_training:{group_id}")
        source = str(predictor.get("source_text"))
        response = str(predictor.get("response_text"))
        windows = sorted(windows_by_id.get(group_id, []), key=lambda row: row["window_index"])
        rebuilt_windows = window_protocol.build_lossless_windows(response)
        if [dict(row, group_id=group_id) for row in rebuilt_windows] != [
            dict(row) for row in windows
        ]:
            raise WindowPilotError(f"sealed_windows_mismatch:{group_id}")
        for order_index, order in enumerate(orders):
            prompt_cells: list[tuple[str, int | None, str]] = [
                (
                    "whole_response",
                    None,
                    window_protocol.build_whole_prompt(source, response, order),
                )
            ]
            for window in windows:
                focused = window_protocol.build_focused_prompt(source, response, order, window)
                prompt_cells.append(
                    ("focused_window", int(window["window_index"]), str(focused["prompt"]))
                )
            for arm, window_index, prompt in prompt_cells:
                key = (group_id, arm, tuple(order), window_index)
                sealed = requests.get(key)
                if sealed is None:
                    raise WindowPilotError(f"sealed_request_missing:{key}")
                prompt_sha256 = window_protocol.sha256_text(prompt)
                if prompt_sha256 != sealed.get("prompt_sha256"):
                    raise WindowPilotError(
                        f"sealed_prompt_hash_mismatch:{sealed.get('request_id')}"
                    )
                schedule.append(
                    {
                        "call_id": f"pilot-{len(schedule):03d}-{sealed['request_id']}",
                        "request_id": sealed["request_id"],
                        "group_id": group_id,
                        "role": "training",
                        "arm": arm,
                        "window_index": window_index,
                        "option_order": list(order),
                        "order_index": order_index,
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256,
                        "sealed_prompt_sha256": sealed["prompt_sha256"],
                        "sealed_prompt_token_count": int(sealed["prompt_token_count"]),
                        "source_sha256": window_protocol.sha256_text(source),
                        "response_sha256": window_protocol.sha256_text(response),
                        "response_byte_count": len(response.encode("utf-8")),
                        "sentence_count": sum(int(row["sentence_count"]) for row in windows),
                    }
                )
    if len(schedule) > FORWARD_BUDGET:
        raise WindowPilotError(f"forward_budget_exceeded:{len(schedule)}")
    return schedule


def schedule_span_summary(schedule: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Show that fixed groups cover different response and sentence sizes."""

    by_group = {str(row["group_id"]): row for row in schedule}
    response_bytes = [int(row["response_byte_count"]) for row in by_group.values()]
    sentence_counts = [int(row["sentence_count"]) for row in by_group.values()]
    return {
        "group_count": len(by_group),
        "response_byte_min": min(response_bytes),
        "response_byte_max": max(response_bytes),
        "sentence_count_min": min(sentence_counts),
        "sentence_count_max": max(sentence_counts),
        "schedule_sha256": canonical_hash(
            [{key: row[key] for key in row if key != "prompt"} for row in schedule]
        ),
    }


def _softmax_pair(raw: Mapping[str, Any]) -> dict[str, float]:
    values = {key: float(raw[key]) for key in OPTION_IDS}
    maximum = max(values.values())
    shifted = {key: math.exp(value - maximum) for key, value in values.items()}
    denominator = sum(shifted.values())
    return {key: value / denominator for key, value in shifted.items()}


def analytic_token_mapping_fixture() -> list[JsonDict]:
    """Create known display-label rows that are separate from model outcomes."""

    rows: list[JsonDict] = []
    for order in (OPTION_IDS, tuple(reversed(OPTION_IDS))):
        mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=True))
        stable_logits = {"supported": 2.0, "contains_unsupported": 0.0}
        display_logits = {label: stable_logits[option_id] for label, option_id in mapping.items()}
        rows.append(
            {
                "option_order": list(order),
                "label_to_option_id": mapping,
                "raw_logits_by_display_label": display_logits,
                "expected_probabilities_by_option_id": _softmax_pair(stable_logits),
                "semantic_observation": False,
            }
        )
    return rows


def reduce_analytic_fixture(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Verify known label remapping without treating the fixture as semantics."""

    mapping_valid = len(rows) == 2
    probabilities_valid = len(rows) == 2
    for row in rows:
        order = tuple(str(item) for item in row.get("option_order") or [])
        expected_mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=False))
        mapping = row.get("label_to_option_id")
        mapping_valid &= set(order) == set(OPTION_IDS) and mapping == expected_mapping
        display = row.get("raw_logits_by_display_label")
        if not isinstance(display, Mapping) or not isinstance(mapping, Mapping):
            probabilities_valid = False
            continue
        stable = {
            str(mapping[label]): float(display[label]) for label in option_protocol.DISPLAY_LABELS
        }
        probabilities = _softmax_pair(stable)
        expected = row.get("expected_probabilities_by_option_id")
        probabilities_valid &= isinstance(expected, Mapping) and all(
            math.isclose(probabilities[key], float(expected.get(key, math.nan)), abs_tol=1e-12)
            for key in OPTION_IDS
        )
    return {
        "passed": bool(mapping_valid and probabilities_valid),
        "stable_mapping_valid": bool(mapping_valid),
        "expected_probabilities_valid": bool(probabilities_valid),
        "semantic_observation": False,
    }


def controlled_schedule_fixture() -> list[JsonDict]:
    """Build two arms in both orders for deterministic reducer tests."""

    schedule: list[JsonDict] = []
    for arm_index, (arm, window_index) in enumerate(
        (("whole_response", None), ("focused_window", 0))
    ):
        for order_index, order in enumerate((OPTION_IDS, tuple(reversed(OPTION_IDS)))):
            schedule.append(
                {
                    "call_id": f"fixture-{arm_index}-{order_index}",
                    "request_id": f"request-{arm_index}-{order_index}",
                    "group_id": "fixture-group",
                    "role": "training",
                    "arm": arm,
                    "window_index": window_index,
                    "option_order": list(order),
                    "order_index": order_index,
                    "prompt_sha256": f"sha256:fixture-{arm_index}-{order_index}",
                    "sealed_prompt_sha256": f"sha256:fixture-{arm_index}-{order_index}",
                    "sealed_prompt_token_count": 3,
                    "response_byte_count": 40,
                    "sentence_count": 2,
                }
            )
    return schedule


def controlled_call_fixture(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create valid raw call rows with deliberately visible order sensitivity."""

    rows: list[JsonDict] = []
    for index, cell in enumerate(schedule):
        order = tuple(str(item) for item in cell["option_order"])
        mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=True))
        stable_logits = {
            "supported": 2.0 + 0.1 * index,
            "contains_unsupported": 0.5 + 0.2 * index,
        }
        display_logits = {label: stable_logits[option_id] for label, option_id in mapping.items()}
        probabilities = _softmax_pair(stable_logits)
        rows.append(
            {
                **deepcopy(dict(cell)),
                "disposition": "complete",
                "prompt_token_ids": [1, 2, 10 + index],
                "prompt_token_count": 3,
                "native_option_token_ids": {" A": 11, " B": 17},
                "label_to_option_id": mapping,
                "requested_score_position": 2,
                "actual_last_evaluated_position": 2,
                "token_boundary_receipts": {" A": True, " B": True},
                "raw_logits_by_display_label": display_logits,
                "raw_logits_by_option_id": stable_logits,
                "probabilities_by_option_id": probabilities,
                "log_odds_contains_unsupported": math.log(
                    probabilities["contains_unsupported"] / probabilities["supported"]
                ),
                "normalization": "two_option_softmax_float64",
                "order_remapping": mapping,
                "state_reset": True,
                "kv_state_id": f"fresh-{index}",
                "generated_tokens": 0,
                "tokenize_s": 0.001,
                "prefill_s": 0.01 + index / 1000,
                "readout_s": 0.0001,
            }
        )
    return rows


def mutate_call_fixture(rows: Sequence[Mapping[str, Any]], mutation: str) -> list[JsonDict]:
    """Apply one named transport defect to production-shaped fixture rows."""

    altered = deepcopy([dict(row) for row in rows])
    if mutation == "label_swap":
        mapping = altered[0]["label_to_option_id"]
        altered[0]["label_to_option_id"] = {
            " A": mapping[" B"],
            " B": mapping[" A"],
        }
    elif mutation == "token_position":
        altered[0]["actual_last_evaluated_position"] += 1
    elif mutation == "state_reuse":
        altered[1]["kv_state_id"] = altered[0]["kv_state_id"]
    elif mutation == "prompt_hash":
        altered[0]["prompt_sha256"] = "sha256:changed"
    elif mutation == "generation":
        altered[0]["generated_tokens"] = 1
    elif mutation == "nonfinite":
        altered[0]["raw_logits_by_option_id"]["supported"] = math.inf
    else:
        raise ValueError(f"unknown_mutation:{mutation}")
    return altered


def reduce_pilot_calls(
    rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce native receipts without imposing a semantic or order-equality gate."""

    expected = {str(row["call_id"]): row for row in schedule}
    observed = {str(row.get("call_id")): row for row in rows}
    complete = [row for row in rows if row.get("disposition") == "complete"]
    completeness_valid = len(rows) == len(schedule) == len(complete) and set(observed) == set(
        expected
    )
    label_mapping_valid = True
    final_position_valid = True
    fresh_kv_valid = True
    sealed_prompt_identity_valid = True
    zero_generation_valid = True
    finite_logits_valid = True
    native_option_tokens_valid = True
    normalization_valid = True
    timing_valid = True
    states: list[str] = []
    vectors: list[tuple[float, float]] = []
    for row in complete:
        cell = expected.get(str(row.get("call_id")), {})
        order = tuple(str(item) for item in row.get("option_order") or [])
        mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=False))
        label_mapping_valid &= (
            set(order) == set(OPTION_IDS) and row.get("label_to_option_id") == mapping
        )
        count = int(row.get("prompt_token_count", -1))
        final_position_valid &= (
            row.get("requested_score_position") == count - 1
            and row.get("actual_last_evaluated_position") == count - 1
            and len(row.get("prompt_token_ids") or []) == count
        )
        states.append(str(row.get("kv_state_id")))
        fresh_kv_valid &= row.get("state_reset") is True
        sealed_prompt_identity_valid &= (
            row.get("request_id") == cell.get("request_id")
            and row.get("prompt_sha256") == cell.get("sealed_prompt_sha256")
            and row.get("sealed_prompt_token_count") == cell.get("sealed_prompt_token_count")
            and count == cell.get("sealed_prompt_token_count")
        )
        zero_generation_valid &= row.get("generated_tokens") == 0
        raw = row.get("raw_logits_by_option_id")
        display = row.get("raw_logits_by_display_label")
        if isinstance(raw, Mapping) and set(raw) == set(OPTION_IDS):
            values = tuple(float(raw[key]) for key in OPTION_IDS)
            finite_logits_valid &= all(math.isfinite(value) for value in values)
            vectors.append(values)
        else:
            finite_logits_valid = False
        native_ids = row.get("native_option_token_ids")
        boundaries = row.get("token_boundary_receipts")
        native_option_tokens_valid &= (
            isinstance(native_ids, Mapping)
            and set(native_ids) == set(option_protocol.DISPLAY_LABELS)
            and len(set(native_ids.values())) == 2
            and all(isinstance(value, int) and value >= 0 for value in native_ids.values())
            and isinstance(boundaries, Mapping)
            and all(boundaries.get(label) is True for label in option_protocol.DISPLAY_LABELS)
        )
        probabilities = row.get("probabilities_by_option_id")
        if (
            isinstance(raw, Mapping)
            and isinstance(display, Mapping)
            and isinstance(probabilities, Mapping)
        ):
            remapped = {option_id: float(display[label]) for label, option_id in mapping.items()}
            expected_probabilities = _softmax_pair(remapped)
            normalization_valid &= all(
                math.isclose(float(raw[key]), remapped[key], abs_tol=1e-12)
                and math.isclose(
                    float(probabilities.get(key, math.nan)),
                    expected_probabilities[key],
                    abs_tol=1e-12,
                )
                for key in OPTION_IDS
            )
        else:
            normalization_valid = False
        timing_valid &= all(
            isinstance(row.get(field), (int, float))
            and math.isfinite(float(row[field]))
            and float(row[field]) >= 0
            for field in ("tokenize_s", "prefill_s", "readout_s")
        )
    fresh_kv_valid &= len(states) == len(set(states)) == len(complete)
    analytic = reduce_analytic_fixture(analytic_token_mapping_fixture())
    checks = {
        "completeness_valid": bool(completeness_valid),
        "label_mapping_valid": bool(label_mapping_valid),
        "final_position_valid": bool(final_position_valid),
        "fresh_kv_valid": bool(fresh_kv_valid),
        "sealed_prompt_identity_valid": bool(sealed_prompt_identity_valid),
        "zero_generation_valid": bool(zero_generation_valid),
        "finite_logits_valid": bool(finite_logits_valid),
        "native_option_tokens_valid": bool(native_option_tokens_valid),
        "normalization_valid": bool(normalization_valid),
        "timing_valid": bool(timing_valid),
        "analytic_mapping_valid": analytic["passed"],
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "planned_forwards": len(schedule),
        "observed_rows": len(rows),
        "complete_forwards": len(complete),
        "failed_forwards": sum(row.get("disposition") == "failed" for row in rows),
        "real_vector_diversity_observed": len(set(vectors)) > 1,
        "predictive_score_is_gate": False,
        "order_equality_is_gate": False,
    }


def reduce_group_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report order sensitivity per group while keeping it observational."""

    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("group_id")), []).append(row)
    result: list[JsonDict] = []
    for group_id, group_calls in sorted(by_group.items()):
        pairs: dict[tuple[str, int | None], list[Mapping[str, Any]]] = {}
        for row in group_calls:
            key = (
                str(row.get("arm")),
                None if row.get("window_index") is None else int(row["window_index"]),
            )
            pairs.setdefault(key, []).append(row)
        probability_deltas: list[float] = []
        log_odds_deltas: list[float] = []
        for pair in pairs.values():
            complete = [row for row in pair if row.get("disposition") == "complete"]
            if len(complete) != 2:
                continue
            probability_deltas.append(
                abs(
                    float(complete[0]["probabilities_by_option_id"]["contains_unsupported"])
                    - float(complete[1]["probabilities_by_option_id"]["contains_unsupported"])
                )
            )
            log_odds_deltas.append(
                abs(
                    float(complete[0]["log_odds_contains_unsupported"])
                    - float(complete[1]["log_odds_contains_unsupported"])
                )
            )
        first = group_calls[0]
        result.append(
            {
                "group_id": group_id,
                "disposition": "complete_transport_measurement"
                if all(row.get("disposition") == "complete" for row in group_calls)
                else "failed_transport_measurement",
                "planned_calls": len(group_calls),
                "complete_calls": sum(row.get("disposition") == "complete" for row in group_calls),
                "failed_calls": sum(row.get("disposition") == "failed" for row in group_calls),
                "response_byte_count": first.get("response_byte_count"),
                "sentence_count": first.get("sentence_count"),
                "paired_arms": len(probability_deltas),
                "maximum_order_probability_delta": max(probability_deltas, default=None),
                "mean_order_probability_delta": statistics.mean(probability_deltas)
                if probability_deltas
                else None,
                "maximum_order_log_odds_delta": max(log_odds_deltas, default=None),
                "order_equality_is_gate": False,
            }
        )
    return result


def build_capture_budget_forecasts(
    samples: Sequence[Mapping[str, Any]],
    *,
    load_s: float,
    manifests: Mapping[str, Sequence[int]],
) -> list[JsonDict]:
    """Forecast both fixed manifests from a conservative measured token rate."""

    rates = [
        float(row["prefill_s"]) / int(row["prompt_token_count"])
        for row in samples
        if int(row.get("prompt_token_count", 0)) > 0 and float(row.get("prefill_s", 0)) > 0
    ]
    if not rates or any(not math.isfinite(value) or value <= 0 for value in rates):
        raise ValueError("positive_prefill_samples_required")
    ordered = sorted(rates)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    p90 = percentile(0.90)
    order = [name for name in ("fit_capture", "evaluation_capture") if name in manifests]
    order.extend(sorted(set(manifests) - set(order)))
    forecasts: list[JsonDict] = []
    for name in order:
        lengths = [int(value) for value in manifests[name]]
        planned_tokens = sum(lengths)
        forecast_s = float(load_s + planned_tokens * p90 * FORECAST_MULTIPLIER)
        feasible = forecast_s <= CAPTURE_HARD_CAP_S
        forecasts.append(
            {
                "capture": name,
                "planned_calls": len(lengths),
                "planned_tokens": planned_tokens,
                "planned_call_token_min": min(lengths, default=0),
                "planned_call_token_max": max(lengths, default=0),
                "load_s": float(load_s),
                "p90_seconds_per_token": p90,
                "safety_multiplier": FORECAST_MULTIPLIER,
                "forecast_s": forecast_s,
                "hard_cap_s": CAPTURE_HARD_CAP_S,
                "feasible": feasible,
                "unstarted_calls": 0 if feasible else len(lengths),
                "roster_or_text_changed": False,
                "uncertainty": {
                    "sample_size": len(ordered),
                    "estimator": "nearest_rank_empirical_prefill_seconds_per_token",
                    "minimum_seconds_per_token": min(ordered),
                    "median_seconds_per_token": statistics.median(ordered),
                    "p90_seconds_per_token": p90,
                    "maximum_seconds_per_token": max(ordered),
                    "interpretation": "The observed range is descriptive; the budget uses the p90 with a 1.25 safety multiplier.",
                },
            }
        )
    return forecasts


fixture_invocation_events = native_pilot.fixture_invocation_events
reduce_invocation_events = native_pilot.reduce_invocation_events
invocation_counts_balanced = native_pilot.invocation_counts_balanced


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    principle: str,
    *,
    upstream: str,
    field_path: str,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "upstream": upstream,
        "field_path": field_path,
        "principle": principle,
    }


def build_acceptance_gates(
    *,
    validity_passed: bool,
    transport_passed: bool,
    forecasts_passed: bool,
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep validity, readiness, and unmeasured benefit as separate claims."""

    validity_principle = "A favorable metric cannot excuse invalid evidence."
    readiness_principle = "A valid scientific null must not block independent measurements."
    benefit_principle = (
        "A favorable seed, fixture or low-support result cannot replace held-out value."
    )
    return [
        _gate(
            "authenticated_identity_and_accounting",
            "validity",
            True,
            validity_passed,
            validity_principle,
            upstream="preconditions_checked",
            field_path="all",
        ),
        _gate(
            "native_window_transport",
            "validity",
            True,
            transport_passed,
            validity_principle,
            upstream="transport_reduction",
            field_path="passed",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            validation_passed,
            validity_principle,
            upstream="validation_receipts",
            field_path="required",
        ),
        _gate(
            "capture_forecasts_fit",
            "readiness",
            True,
            forecasts_passed,
            readiness_principle,
            upstream="capture_budget_forecasts",
            field_path="fit_capture.feasible",
        ),
        _gate(
            "capture_forecasts_evaluation",
            "readiness",
            True,
            forecasts_passed,
            readiness_principle,
            upstream="capture_budget_forecasts",
            field_path="evaluation_capture.feasible",
        ),
        _gate(
            "predictive_benefit_not_claimed",
            "benefit",
            False,
            False,
            benefit_principle,
            upstream="honest_verdict",
            field_path="predictive_benefit_claimed",
        ),
    ]


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed field and retain the first failure for routing."""

    failed = [
        {
            key: row.get(key)
            for key in ("check", "category", "upstream", "field_path", "expected", "observed")
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _seal_call_rows(root: Path, path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Write one bounded raw shard atomically and return its exact identity."""

    target = path if path.is_absolute() else root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonl_bytes(rows)
    if len(payload) >= 20 * 1024 * 1024:
        raise WindowPilotError("raw_call_shard_exceeds_20_mib")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    label = target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target)
    return {
        "path": label,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(rows),
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind the full terminal record without hashing its checksum into itself."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


_FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
    "run_date": "Use 20260921; measured UTC and monotonic process identity prevent replay ambiguity.",
    "preconditions_checked": "Exact paths, observed values, ownership and input validity prevent guessed prerequisites.",
    "MODEL_SPECS": "The current Qwen model with resolved provenance prevents a legacy substitute.",
    "model_specs": "The lowercase model list prevents compatibility readers from losing model identity.",
    "model_invoked": "Attempted live work remains distinct from historical or scripted events to prevent false execution claims.",
    "invocation_counts": "Balanced attempted and terminal counters prevent missing model work.",
    "inference_substrate": "The actual native readout name prevents aggregation or generation from being implied.",
    "inference_substrate_class": "The model_load_no_generation class prevents padded duration or generated-token claims.",
    "execution_venue": "Host CPU and owned CUDA identity remain separate to prevent archived hardware substitution.",
    "duration_s": "Measured work and separate components prevent a synthetic time floor.",
    "phase_spans": "Flushed checkpoints prevent unfinished operations and stalls from disappearing.",
    "random_seed": "Frozen role, audit and order choices prevent outcome-guided retries.",
    "reproducibility_checksum": "A bound checksum prevents code, prompts, roles, raw shards or scope drift.",
    "source_artifact_hashes": "Original bytes, verdicts and flags prevent upstream evidence laundering.",
    "rows": "Per-group failures and sensitivity prevent difficult units from disappearing.",
    "sample_size_budget": "Separate planned and terminal units prevent silent roster shrinkage.",
    "acceptance_gate_results": "Typed gate operands and principles prevent one favorable metric from bypassing a gate.",
    "gate_check_summary": "Exact upstream field failures prevent blocked evidence from being hidden.",
    "honest_verdict": "A complete terminal finding prevents a measured null from looking unfinished.",
    "verdict_class": "The closed verdict enum prevents retryable work from masquerading as a null.",
    "verifier_is_oracle": "Declaring no evaluation oracle prevents analytic controls from becoming positive evidence.",
    "flagged_adversarial": "Actual reader flags remain visible to prevent a gate from being opened by deletion.",
    "validation_receipts": "Exact commands, exits and log hashes prevent validation scope drift.",
    "field_principles": "A purpose for every field prevents unexplained evidence from becoming authoritative.",
    "window_native_ready_score": "Transport plus both full forecasts prevent an infeasible capture from opening.",
    "pilot_complete_score": "Measured infeasibility remains complete to prevent scientific nulls from blocking measurements.",
    "pilot_call_rows": "Raw order-specific tokens and logits prevent summary-only transport claims.",
    "capture_budget_forecasts": "Measured token rates and load cost prevent fictional full-capture budgets.",
    "scored_runtime_parity_score": "A fixed zero prevents this native pilot from implying scored-server parity.",
}


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    return {
        key: _FIELD_PRINCIPLES.get(
            key,
            f"The {key} field preserves measured supporting evidence to prevent silent record drift.",
        )
        for key in keys
    }


def _validation_names_passed(receipts: object, names: Sequence[str]) -> bool:
    if not isinstance(receipts, list):
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in rows
        )
        == 1
        for name in names
    )


def _build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    call_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    capture_manifests: Mapping[str, Sequence[int]],
    model_identity: Mapping[str, Any],
    device_identity: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    started_at: str,
    ended_at: str,
    started_ns: int,
    ended_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    duration_components: Mapping[str, float],
) -> JsonDict:
    reduction = reduce_pilot_calls(call_rows, schedule)
    group_rows = reduce_group_rows(call_rows)
    counts = reduce_invocation_events(events)
    forecasts = build_capture_budget_forecasts(
        call_rows,
        load_s=float(duration_components["model_load"]),
        manifests=capture_manifests,
    )
    forecasts_passed = len(forecasts) == 2 and all(row["feasible"] for row in forecasts)
    identity_valid = (
        all(row.get("passed") is True for row in preconditions)
        and invocation_counts_balanced(counts)
        and model_identity.get("identity_authenticated") is True
    )
    gates = build_acceptance_gates(
        validity_passed=identity_valid,
        transport_passed=reduction["passed"],
        forecasts_passed=forecasts_passed,
        validation_passed=validation_passed,
    )
    ready = int(identity_valid and reduction["passed"] and forecasts_passed and validation_passed)
    complete = int(identity_valid and reduction["passed"] and validation_passed)
    if not identity_valid or not reduction["passed"] or not validation_passed:
        verdict = "complete_disqualified_window_pilot_invalid_evidence"
        verdict_class = "disqualified"
    elif forecasts_passed:
        verdict = "complete_null_window_native_transport_and_capture_forecasts_ready"
        verdict_class = "null"
    else:
        verdict = "complete_null_window_native_capture_forecast_infeasible"
        verdict_class = "null"
    sample_budget = {
        "planned": len(schedule),
        "attempted": counts["forward_calls"]["attempted"],
        "complete": counts["forward_calls"]["completed"],
        "failed": counts["forward_calls"]["failed"],
        "cancelled": counts["forward_calls"]["cancelled"],
        "censored": counts["forward_calls"]["cancelled"],
        "excluded": 0,
        "unstarted": len(schedule) - counts["forward_calls"]["attempted"],
        "forward_ceiling": FORWARD_BUDGET,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "measured_at_utc": ended_at,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "process_identity": {
            "pid": int(model_identity.get("owned_pid", os.getpid())),
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "clock": "time.monotonic_ns",
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": deepcopy(model_specs),
        "model_invoked": counts["model_loads"]["attempted"] > 0,
        "invocation_counts": counts,
        "current_invocation_events": [deepcopy(dict(row)) for row in events],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(device_identity)),
        "model_identity": deepcopy(dict(model_identity)),
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_breakdown_s": deepcopy(dict(duration_components)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "role_seed": 656001,
            "pilot_group_selection": "fixed_predictor_safe_ids_no_random_draw",
            "option_order": "both_frozen_orders",
            "audit_seed": 656092,
            "interval_seed": None,
            "deterministic_null_seed_reason": "No predictive interval or outcome-selected retry is used.",
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "pilot_schedule": [
            {key: deepcopy(value) for key, value in row.items() if key != "prompt"}
            for row in schedule
        ],
        "pilot_schedule_span": schedule_span_summary(schedule),
        "rows": group_rows,
        "pilot_call_rows": [deepcopy(dict(row)) for row in call_rows],
        "raw_shards": {"pilot_call_rows": deepcopy(dict(raw_receipt))},
        "sample_size_budget": sample_budget,
        "transport_reduction": reduction,
        "analytic_token_mapping_fixture": {
            "rows": analytic_token_mapping_fixture(),
            "reduction": reduce_analytic_fixture(analytic_token_mapping_fixture()),
            "semantic_observation": False,
        },
        "capture_request_lengths": {
            key: [int(value) for value in values] for key, values in capture_manifests.items()
        },
        "capture_budget_forecasts": forecasts,
        "full_capture_started": False,
        "full_capture_blocked": not forecasts_passed,
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "window_native_ready_score": ready,
        "pilot_complete_score": complete,
        "scored_runtime_parity_score": 0,
        "predictive_benefit_claimed": False,
        "server_deployment_changed": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "affected_validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "passed" if validation_passed else "pending",
            "independent_raw_reduction": "passed" if validation_passed else "pending",
            "numbered_runtime_e2e": "not_applicable_isolated_experiment_no_shared_runtime_change",
        },
        "methodology": (
            "One native Qwen load evaluates both option orders for whole responses and every "
            "lossless focus window. Raw final-position logits are remapped to stable option IDs. "
            "Forecasts use 1.25 times the empirical per-token prefill p90 plus measured load time."
        ),
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    tmp_path: Path, *, feasible: bool = True, validity: bool = True
) -> JsonDict:
    """Build a circular fixture through the same reducers without model work."""

    schedule = controlled_schedule_fixture()
    rows = controlled_call_fixture(schedule)
    events = fixture_invocation_events(forwards=len(rows))
    raw_receipt = _seal_call_rows(tmp_path, Path("pilot_call_rows.jsonl"), rows)
    manifests = (
        {"fit_capture": [3, 3], "evaluation_capture": [3, 3]}
        if feasible
        else {"fit_capture": [1_000_000], "evaluation_capture": [1_000_000]}
    )
    model_identity = {
        "identity_authenticated": validity,
        "model_id": MODEL_HF_ID,
        "model_path": "/fixture/Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256": "sha256:fixture",
        "quantization": "Q4_K_M",
        "owned_pid": os.getpid(),
        "gpu_uuid": "GPU-fixture",
        "peak_owned_vram_mb": 16000,
        "actual_layer_placement": {"placement_authenticated": True},
        "llama_cpp_build": {"version": "fixture", "module_sha256": "sha256:fixture"},
    }
    return _build_artifact(
        root=tmp_path,
        preconditions=[
            {
                "check": "fixture",
                "expected": True,
                "observed": validity,
                "passed": validity,
                "principle": "The fixture identity prevents unbounded test evidence.",
            }
        ],
        source_hashes=[],
        schedule=schedule,
        call_rows=rows,
        events=events,
        capture_manifests=manifests,
        model_identity=model_identity,
        device_identity={"cpu": "fixture", "selected_cuda": {"uuid": "GPU-fixture"}},
        raw_receipt=raw_receipt,
        validation_receipts=[],
        validation_passed=True,
        started_at="2026-09-21T00:00:00Z",
        ended_at="2026-09-21T00:00:03Z",
        started_ns=1_000_000_000,
        ended_ns=4_000_000_000,
        phase_spans=[{"phase": "fixture", "start_s": 0.0, "end_s": 3.0, "completed_units": 4}],
        duration_components={
            "model_load": 1.0,
            "tokenize": 0.004,
            "prefill": 0.046,
            "readout": 0.0004,
            "generation": 0.0,
            "validation": 0.0,
        },
    )


def _raw_shard_valid(artifact: Mapping[str, Any], root: Path) -> bool:
    receipt = (artifact.get("raw_shards") or {}).get("pilot_call_rows")
    if not isinstance(receipt, Mapping):
        return False
    path = Path(str(receipt.get("path")))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != receipt.get("sha256"):
        return False
    try:
        loaded = [json.loads(line) for line in resolved.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return False
    return loaded == artifact.get("pilot_call_rows") and len(loaded) == receipt.get("rows")


def validate_artifact(value: object, *, root: Path, require_validation: bool = True) -> list[str]:
    """Cold-check identity, raw rows, forecasts, scores, gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    expected_identity = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": model_specs,
        "model_invoked": True,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
        "scored_runtime_parity_score": 0,
        "predictive_benefit_claimed": False,
        "server_deployment_changed": False,
    }
    for field, expected in expected_identity.items():
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    schedule = artifact.get("pilot_schedule")
    call_rows = artifact.get("pilot_call_rows")
    if not isinstance(schedule, list) or not isinstance(call_rows, list):
        errors.append("pilot_rows_invalid")
        reduction = {"passed": False}
    else:
        reduction = reduce_pilot_calls(call_rows, schedule)
        if reduction != artifact.get("transport_reduction"):
            errors.append("pilot_call_reduction_mismatch")
        if reduce_group_rows(call_rows) != artifact.get("rows"):
            errors.append("group_reduction_mismatch")
    if not _raw_shard_valid(artifact, root):
        errors.append("raw_call_shard_invalid")
    events = artifact.get("current_invocation_events")
    counts = artifact.get("invocation_counts")
    if not isinstance(events, list) or not isinstance(counts, Mapping):
        errors.append("invocation_evidence_invalid")
        counts_balanced = False
    else:
        recomputed_counts = reduce_invocation_events(events)
        if recomputed_counts != counts:
            errors.append("invocation_counts_mismatch")
        counts_balanced = invocation_counts_balanced(counts)
        if not counts_balanced:
            errors.append("invocation_counts_unbalanced")
    manifests = artifact.get("capture_request_lengths")
    components = artifact.get("duration_breakdown_s")
    if (
        isinstance(manifests, Mapping)
        and isinstance(components, Mapping)
        and isinstance(call_rows, list)
    ):
        try:
            forecasts = build_capture_budget_forecasts(
                call_rows,
                load_s=float(components["model_load"]),
                manifests=manifests,
            )
        except (KeyError, TypeError, ValueError):
            forecasts = []
        if forecasts != artifact.get("capture_budget_forecasts"):
            errors.append("capture_forecast_mismatch")
    else:
        forecasts = []
        errors.append("capture_forecast_inputs_invalid")
    forecasts_passed = len(forecasts) == 2 and all(row.get("feasible") is True for row in forecasts)
    identity_valid = (
        all(row.get("passed") is True for row in artifact.get("preconditions_checked") or [])
        and counts_balanced
        and (artifact.get("model_identity") or {}).get("identity_authenticated") is True
    )
    validation_passed = True
    if require_validation:
        validation_passed = _validation_names_passed(
            artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
        )
        if not validation_passed:
            errors.append("required_validation_failed")
    ready = int(
        identity_valid
        and reduction.get("passed") is True
        and forecasts_passed
        and validation_passed
    )
    complete = int(identity_valid and reduction.get("passed") is True and validation_passed)
    if artifact.get("window_native_ready_score") != ready:
        errors.append("window_native_ready_score_mismatch")
    if artifact.get("pilot_complete_score") != complete:
        errors.append("pilot_complete_score_mismatch")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping) or not row.get("principle") for row in gates
    ):
        errors.append("acceptance_gates_invalid")
    elif gate_check_summary(gates) != artifact.get("gate_check_summary"):
        errors.append("gate_summary_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    elif any("prevent" not in str(text) for text in principles.values()):
        errors.append("field_principles_missing_failure_mode")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:  # pragma: no cover - fresh-process capability reader.
    """Reload the raw shard and recompute terminal scores from per-call rows."""

    errors = validate_artifact(artifact, root=root, require_validation=require_terminal)
    return {
        "passed": not errors,
        "errors": errors,
        "transport_reduction": artifact.get("transport_reduction"),
        "window_native_ready_score": artifact.get("window_native_ready_score"),
        "pilot_complete_score": artifact.get("pilot_complete_score"),
    }


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - live progress boundary.
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7492] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - live input.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - live input.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover - live input.
    label = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    receipt: JsonDict = {"path": label, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.suffix == ".json":
        value = _load_object(path)
        receipt.update(
            {
                "original_honest_verdict": value.get("honest_verdict"),
                "original_verdict_class": value.get("verdict_class"),
                "original_flagged_adversarial": value.get("flagged_adversarial"),
            }
        )
    return receipt


def _call_with_heartbeats(
    operation: Callable[[], Any], *, started: float, phase: str, operation_name: str
) -> Any:  # pragma: no cover - slow-operation guard.
    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # noqa: BLE001 - re-raised in owner thread.
            error.append(exc)

    worker = threading.Thread(target=target, name=f"{EXPERIMENT_ID}-{phase}", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=60.0)
        if worker.is_alive():
            progress(started, phase, "pending", operation=operation_name)
    if error:
        raise error[0]
    return result[0] if result else None


def _gpu_process_rows(pid: int) -> list[JsonDict]:  # pragma: no cover - hardware evidence.
    completed = subprocess.run(  # noqa: S603 - fixed local diagnostic argv.
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) == pid:
            rows.append(
                {"pid": int(parts[0]), "gpu_uuid": parts[1], "used_memory_mb": int(parts[2])}
            )
    return rows


def collect_preconditions(
    root: Path, *, started: float
) -> tuple[list[JsonDict], JsonDict, list[JsonDict]]:  # pragma: no cover - live gate.
    """Authenticate source bytes, Exp7491, cached Qwen, runtime, and idle GPU."""

    checks: list[JsonDict] = []
    sources: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"resource_exists:{relative.as_posix()}",
                "path": str(path),
                "expected": True,
                "observed": available,
                "op": "eq",
                "passed": available,
                "principle": "Exact resource paths prevent substituted or guessed inputs.",
            }
        )
        if available:
            sources.append(_source_receipt(path, root))

    upstream = _load_object(UPSTREAM_PATH)
    upstream_errors = (
        window_protocol.validate_artifact(upstream, root=root, require_terminal=True)
        if upstream
        else ["artifact_unreadable"]
    )
    upstream_gate = reduce_upstream_gate(upstream, validator_errors=upstream_errors)
    checks.extend(upstream_gate["checks"])

    cached = cached_current_model(gpu_index=0)
    model_path = Path(str(cached.get("model_path"))) if cached else Path("/absent-model")
    gpus = parity._gpu_inventory()
    idle = [
        row
        for row in gpus
        if row["memory_free_mb"] >= 20_000
        and row["memory_used_mb"] <= 1_024
        and row["utilization_pct"] <= 5
    ]
    selected = deepcopy(idle[0]) if idle else {}
    mandate = current_model()
    checks.extend(
        (
            _principle_row(
                "force_live",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
                field_path="CARNOT_FORCE_LIVE",
                upstream="environment",
            ),
            _principle_row(
                "model_hf_id",
                MODEL_HF_ID,
                cached.get("hf_id") if cached else None,
                field_path="hf_id",
                upstream=str(model_path),
            ),
            _principle_row(
                "model_quantization",
                "Q4_K_M",
                mandate["quantization"],
                field_path="quantization",
                upstream="carnot.inference.sota_models.current_model",
            ),
            _principle_row(
                "cached_gguf",
                True,
                model_path.is_file(),
                field_path="is_file",
                upstream=str(model_path),
            ),
            _principle_row(
                "idle_cuda_device",
                True,
                bool(idle),
                field_path="idle_candidate",
                upstream="nvidia-smi",
            ),
            _principle_row(
                "current_task_not_quarantined",
                False,
                "7492" in (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"),
                field_path=EXPERIMENT_ID,
                upstream="ops/exclusion_manifest.yaml",
            ),
            _principle_row(
                "driving_requirement",
                "REQ-VERIFY-7492",
                "REQ-VERIFY-7492"
                if "REQ-VERIFY-7492" in SPEC_PATH.read_text(encoding="utf-8")
                else None,
                field_path="REQ-*",
                upstream="openspec/capabilities/verification/spec.md",
            ),
        )
    )
    model_sha256 = None
    if model_path.is_file():
        progress(started, "preconditions", "before_model_hash", path=model_path)
        model_sha256 = _call_with_heartbeats(
            lambda: sha256_file(model_path),
            started=started,
            phase="preconditions",
            operation_name="model_sha256",
        )
        progress(started, "preconditions", "after_model_hash", sha256=model_sha256)
        historical = _load_object(root / "results/experiment_7477_v655_native_readout_pilot.json")
        expected_hash = (historical.get("prompt_identity") or {}).get("model_sha256")
        checks.append(
            _principle_row(
                "cached_weight_hash",
                expected_hash,
                model_sha256,
                field_path="model_sha256",
                upstream=str(model_path),
            )
        )
        sources.append(
            {
                "path": str(model_path),
                "sha256": model_sha256,
                "bytes": model_path.stat().st_size,
                "quantization": mandate["quantization"],
                "original_flagged_adversarial": None,
                "original_verdict_class": None,
            }
        )
    return (
        checks,
        {
            "upstream": upstream,
            "model_path": model_path,
            "model_sha256": model_sha256,
            "selected_gpu": selected,
            "gpu_inventory": gpus,
            "quantization": mandate["quantization"],
        },
        sources,
    )


def _score_native_prompt(
    runner: native_pilot.NativeReadoutRunner, cell: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - live native forward.
    """Measure tokenization, native prefill, and final-logit readout separately."""

    prompt = str(cell["prompt"])
    tokenize_started = time.monotonic()
    prompt_ids = runner.scorer.tokenize(prompt, add_bos=True)
    tokenize_s = time.monotonic() - tokenize_started
    native_ids = {
        label: runner.scorer.tokenize(label, add_bos=False)[0]
        for label in option_protocol.DISPLAY_LABELS
    }
    boundaries = {
        label: runner.scorer.tokenize(prompt + label, add_bos=True)
        == [*prompt_ids, native_ids[label]]
        for label in option_protocol.DISPLAY_LABELS
    }
    runner.scorer.llm.reset()
    prefill_started = time.monotonic()
    runner.scorer.llm.eval(prompt_ids)
    prefill_s = time.monotonic() - prefill_started
    readout_started = time.monotonic()
    logits = runner.scorer.llm.scores[len(prompt_ids) - 1]
    display_logits = {
        label: float(logits[native_ids[label]]) for label in option_protocol.DISPLAY_LABELS
    }
    order = tuple(str(item) for item in cell["option_order"])
    mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=True))
    stable_logits = {option_id: display_logits[label] for label, option_id in mapping.items()}
    probabilities = _softmax_pair(stable_logits)
    readout_s = time.monotonic() - readout_started
    actual_tokens = getattr(
        runner.scorer.llm,
        "n_tokens",
        getattr(runner.scorer.llm, "_n_tokens", len(prompt_ids)),
    )
    runner.reset_epoch += 1
    return {
        **{key: deepcopy(value) for key, value in cell.items() if key != "prompt"},
        "disposition": "complete",
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "native_option_token_ids": native_ids,
        "label_to_option_id": mapping,
        "requested_score_position": len(prompt_ids) - 1,
        "actual_last_evaluated_position": int(actual_tokens) - 1,
        "token_boundary_receipts": boundaries,
        "raw_logits_by_display_label": display_logits,
        "raw_logits_by_option_id": stable_logits,
        "probabilities_by_option_id": probabilities,
        "log_odds_contains_unsupported": math.log(
            probabilities["contains_unsupported"] / probabilities["supported"]
        ),
        "normalization": "two_option_softmax_float64",
        "order_remapping": mapping,
        "state_reset": True,
        "kv_state_id": f"reset-epoch-{runner.reset_epoch:03d}",
        "generated_tokens": 0,
        "tokenize_s": tokenize_s,
        "prefill_s": prefill_s,
        "readout_s": readout_s,
        "model_sha256": runner.identity["model_sha256"],
        "tokenizer_identity": deepcopy(runner.identity["tokenizer_identity"]),
    }


def _event(call_id: str, operation: str, state: str) -> JsonDict:  # pragma: no cover
    return {
        "call_id": call_id,
        "operation": operation,
        "state": state,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current_owned_process",
    }


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": time.monotonic() - run_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _capture_manifests(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:  # pragma: no cover
    return {
        "fit_capture": [
            int(row["prompt_token_count"])
            for row in requests
            if row.get("role") in {"training", "calibration_tuning"}
        ],
        "evaluation_capture": [
            int(row["prompt_token_count"])
            for row in requests
            if row.get("role") in {"test", "online"}
        ],
    }


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    relative = candidate.relative_to(root).as_posix()
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                relative,
            ),
            "completion",
            180,
        ),
    )
    return [
        PlannedCommand(spec, "safety" if spec.name == "adversarial_verify" else "completion", True)
        for spec in specs
    ]


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Run one owned Qwen load, bounded native forwards, and scoped readers."""

    if run_date != RUN_DATE:
        raise WindowPilotError(f"run_date_mismatch:{run_date}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    events: list[JsonDict] = []
    call_rows: list[JsonDict] = []
    durations: JsonDict = {
        "model_load": 0.0,
        "tokenize": 0.0,
        "prefill": 0.0,
        "readout": 0.0,
        "generation": 0.0,
        "validation": 0.0,
    }

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, context, source_hashes = collect_preconditions(root, started=run_started)
    if not all(row.get("passed") is True for row in preconditions):
        failed = next(row for row in preconditions if row.get("passed") is not True)
        raise WindowPilotError(f"precondition_failed:{failed['check']}:{failed.get('observed')}")
    spans.append(
        _phase_span(
            "preconditions", phase_started, run_started, len(preconditions), "authenticated"
        )
    )
    progress(run_started, "preconditions", "complete", checked=len(preconditions))

    progress(run_started, "schedule", "start")
    phase_started = time.monotonic()
    upstream_raw = root / UPSTREAM_RAW_DIR
    predictors = _load_jsonl(upstream_raw / "predictors.jsonl")
    groups = _load_jsonl(upstream_raw / "groups.jsonl")
    windows = _load_jsonl(upstream_raw / "windows.jsonl")
    requests = _load_jsonl(upstream_raw / "requests.jsonl")
    schedule = build_pilot_schedule(predictors, groups, windows, requests)
    capture_manifests = _capture_manifests(requests)
    if [len(capture_manifests[name]) for name in ("fit_capture", "evaluation_capture")] != [
        1936,
        2192,
    ]:
        raise WindowPilotError("sealed_capture_manifest_count_mismatch")
    spans.append(_phase_span("schedule", phase_started, run_started, len(schedule), "frozen"))
    progress(run_started, "schedule", "complete", forwards=len(schedule))

    manifest_path = root / RAW_DIR / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    )
    source_hashes.append(_source_receipt(manifest_path, root))

    gpu = dict(context["selected_gpu"])
    model_path = Path(context["model_path"])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=EXPERIMENT_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=MAX_LIVE_SECONDS + 300.0,
    )
    owner = lease.owner_receipt()
    runner = native_pilot.NativeReadoutRunner(model_path, int(gpu["index"]))
    model_identity: JsonDict = {}
    release: JsonDict = {}
    inference_ok = False
    peak_owned_vram_mb = 0
    live_started = time.monotonic()
    try:
        lease.transition("admitted")
        lease.transition("loading")
        phase_started = time.monotonic()
        progress(run_started, "model_load", "before_model_load", gpu_uuid=gpu["uuid"])
        events.append(_event("native-load", "model_loads", "attempted"))
        load_started = time.monotonic()
        try:
            model_identity = _call_with_heartbeats(
                lambda: runner.load(weight_sha256=str(context["model_sha256"])),
                started=run_started,
                phase="model_load",
                operation_name="native_qwen_load",
            )
        except BaseException:
            events.append(_event("native-load", "model_loads", "failed"))
            raise
        events.append(_event("native-load", "model_loads", "completed"))
        durations["model_load"] = time.monotonic() - load_started
        process_rows = _gpu_process_rows(os.getpid())
        selected_rows = [row for row in process_rows if row["gpu_uuid"] == gpu["uuid"]]
        peak_owned_vram_mb = max((row["used_memory_mb"] for row in selected_rows), default=0)
        resident_mb = parity._gpu_memory(int(gpu["index"]))
        metadata = dict(getattr(runner.scorer.llm, "metadata", {}) or {})
        block_counts = {
            key: value for key, value in metadata.items() if str(key).endswith(".block_count")
        }
        placement_authenticated = bool(selected_rows) and peak_owned_vram_mb > 1_000
        model_identity.update(
            {
                "quantization": context["quantization"],
                "owned_pid": os.getpid(),
                "gpu_uuid": gpu["uuid"],
                "peak_owned_vram_mb": peak_owned_vram_mb,
                "owned_gpu_process_rows": process_rows,
                "actual_layer_placement": {
                    "requested_n_gpu_layers": -1,
                    "model_block_counts": block_counts,
                    "selected_gpu_uuid": gpu["uuid"],
                    "owned_pid_resident_vram_mb": peak_owned_vram_mb,
                    "placement_evidence": "nvidia_smi_compute_app_pid_gpu_uuid_and_memory",
                    "placement_authenticated": placement_authenticated,
                },
                "observed_offload": {
                    "baseline_vram_mb": gpu["memory_used_mb"],
                    "resident_vram_mb": resident_mb,
                    "cuda_offload": placement_authenticated,
                },
                "identity_authenticated": (
                    context["model_sha256"] == model_identity.get("model_sha256")
                    and context["quantization"] == "Q4_K_M"
                    and model_identity.get("llama_cpp_build", {}).get("module_sha256")
                    and placement_authenticated
                ),
                "lease_id": owner["lease_id"],
            }
        )
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        spans.append(_phase_span("model_load", phase_started, run_started, 1, "resident"))
        progress(run_started, "model_load", "after_model_load", owned_vram_mb=peak_owned_vram_mb)

        checkpoint_path = root / RAW_DIR / "checkpoint.json"
        phase_started = time.monotonic()
        progress(run_started, "native_readout", "before_forward_loop", planned=len(schedule))
        for index, cell in enumerate(schedule):
            if time.monotonic() - live_started >= MAX_LIVE_SECONDS:
                for unstarted in schedule[index:]:
                    call_rows.append(
                        {
                            **{key: value for key, value in unstarted.items() if key != "prompt"},
                            "disposition": "unstarted_live_cap",
                            "generated_tokens": 0,
                        }
                    )
                break
            call_id = str(cell["call_id"])
            progress(
                run_started,
                "native_readout",
                "before_forward",
                unit=index + 1,
                planned=len(schedule),
            )
            events.append(_event(call_id, "forward_calls", "attempted"))
            try:
                row = _score_native_prompt(runner, cell)
            except BaseException as exc:  # preserve each attempted failure before continuing.
                events.append(_event(call_id, "forward_calls", "failed"))
                call_rows.append(
                    {
                        **{key: value for key, value in cell.items() if key != "prompt"},
                        "disposition": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "generated_tokens": 0,
                    }
                )
            else:
                events.append(_event(call_id, "forward_calls", "completed"))
                call_rows.append(row)
                durations["tokenize"] += float(row["tokenize_s"])
                durations["prefill"] += float(row["prefill_s"])
                durations["readout"] += float(row["readout_s"])
                owned = [
                    item
                    for item in _gpu_process_rows(os.getpid())
                    if item["gpu_uuid"] == gpu["uuid"]
                ]
                peak_owned_vram_mb = max(
                    peak_owned_vram_mb,
                    max((item["used_memory_mb"] for item in owned), default=0),
                )
            atomic_json(
                checkpoint_path,
                {
                    "schema": "carnot.exp7492.checkpoint.v1",
                    "schedule_sha256": schedule_span_summary(schedule)["schedule_sha256"],
                    "model_sha256": context["model_sha256"],
                    "completed_units": len(call_rows),
                    "rows_sha256": canonical_hash(call_rows),
                },
            )
            progress(
                run_started,
                "native_readout",
                "after_forward",
                completed=len(call_rows),
                planned=len(schedule),
            )
        model_identity["peak_owned_vram_mb"] = peak_owned_vram_mb
        model_identity["actual_layer_placement"]["peak_owned_vram_mb"] = peak_owned_vram_mb
        inference_ok = reduce_pilot_calls(call_rows, schedule)["passed"]
        spans.append(
            _phase_span(
                "native_readout", phase_started, run_started, len(call_rows), "forwards_terminal"
            )
        )
        progress(
            run_started,
            "native_readout",
            "after_forward_loop",
            completed=len(call_rows),
            passed=inference_ok,
        )
    finally:
        progress(run_started, "model_unload", "before_model_unload")
        runner.close()
        gc.collect()
        progress(run_started, "model_unload", "after_model_unload")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_mb = parity._gpu_memory(int(gpu["index"]))
            unload_observed = after_mb <= int(gpu["memory_used_mb"]) + 1_024
            lease.transition(
                "validating",
                vram_mb=after_mb,
                exit_code=0 if inference_ok else 1,
                unload_observed=unload_observed,
            )
            lease.transition(
                "terminal_complete" if inference_ok and unload_observed else "terminal_blocked"
            )
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
    model_identity["gpu_lease"] = {**owner, "release": release}

    raw_receipt = _seal_call_rows(root, RAW_DIR / "pilot_call_rows.jsonl", call_rows)
    device_identity = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "cuda_inventory": context["gpu_inventory"],
        "selected_cuda": gpu,
        "owned_pid": os.getpid(),
    }

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7492-"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise WindowPilotError("validation_plan_invalid:" + ",".join(plan_errors))
    validation_started = time.monotonic()
    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "validity", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "checks_terminal"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )
    if not affected_reduction["passed"]:
        raise WindowPilotError("affected_validation_failed")

    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        schedule=schedule,
        call_rows=call_rows,
        events=events,
        capture_manifests=capture_manifests,
        model_identity=model_identity,
        device_identity=device_identity,
        raw_receipt=raw_receipt,
        validation_receipts=affected,
        validation_passed=True,
        started_at=started_at,
        ended_at=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        phase_spans=spans,
        duration_components=durations,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "readers_terminal"
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise WindowPilotError("terminal_validation_failed")
    durations["validation"] = time.monotonic() - validation_started

    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        schedule=schedule,
        call_rows=call_rows,
        events=events,
        capture_manifests=capture_manifests,
        model_identity=model_identity,
        device_identity=device_identity,
        raw_receipt=raw_receipt,
        validation_receipts=[*affected, *terminal],
        validation_passed=True,
        started_at=started_at,
        ended_at=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        phase_spans=spans,
        duration_components=durations,
    )
    errors = validate_artifact(final, root=root, require_validation=True)
    if errors:
        raise WindowPilotError("terminal_artifact_invalid:" + ",".join(errors))
    progress(run_started, "publish", "before_atomic_publish", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_publish", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the live pilot or one read-only fresh-process terminal mode."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        path = args.cold_replay if args.cold_replay.is_absolute() else root / args.cold_replay
        artifact = _load_object(path)
        errors = (
            validate_artifact(artifact, root=root, require_validation=False)
            if artifact
            else ["artifact_unreadable"]
        )
        print(
            json.dumps(
                {"mode": "cold_replay", "passed": not errors, "errors": errors}, sort_keys=True
            ),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        path = (
            args.independent_reduce
            if args.independent_reduce.is_absolute()
            else root / args.independent_reduce
        )
        artifact = _load_object(path)
        reduced = (
            independent_reduce(artifact, root=root, require_terminal=False)
            if artifact
            else {"passed": False, "errors": ["artifact_unreadable"]}
        )
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return int(reduced.get("passed") is not True)
    result = run_experiment(root, args.date, output=args.output)
    print(
        json.dumps(
            {
                "artifact": str(root / args.output),
                "honest_verdict": result["honest_verdict"],
                "window_native_ready_score": result["window_native_ready_score"],
                "pilot_complete_score": result["pilot_complete_score"],
                "scored_runtime_parity_score": result["scored_runtime_parity_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

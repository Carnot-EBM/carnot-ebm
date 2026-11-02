"""Measure integer placement parity while preserving board evidence.

The current run uses host CPU arithmetic. It loads no language model and runs
no board command. Historical model and board claims keep their original scope.

Spec refs: REQ-HW-7513 and SCENARIO-HW-7513-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7505_v657_energy_fit import (
    expected_action,
    fit_spline_knots,
    load_fit_rows as upstream_load_fit_rows,
    spline_design_matrix,
    stable_sigmoid,
    validate_artifact as validate_fit_artifact,
)
from carnot.experiment_7506_v657_causal_prototype import (
    validate_artifact as validate_prototype_artifact,
)
from carnot.experiment_7504_v657_evidence_interface import FEATURE_NAMES
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7513-v657-placement-continuity"
SCHEMA = "carnot.exp7513.v657.placement_continuity.v1"
RESULT_PATH = Path("results/experiment_7513_v657_placement_continuity.json")
RAW_DIR = Path("results/raw/experiment_7513_v657_placement_continuity")
MODULE_PATH = Path("python/carnot/experiment_7513_v657_placement_continuity.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7513_v657_placement_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7513_v657_placement_continuity.py")
SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
NOTE_PATH = Path("docs/research-notes/v657-placement-continuity.md")

FIT_ARTIFACT = Path("results/experiment_7505_v657_energy_fit.json")
PROTOTYPE_ARTIFACT = Path("results/experiment_7506_v657_causal_prototype.json")
CHECKPOINT_PATH = Path("results/raw/experiment_7505_v657_energy_fit/frozen_checkpoints.json")
FEATURE_PATH = Path("results/raw/experiment_7504_v657_evidence_interface/features.jsonl")
EVALUATOR_PATH = Path("results/raw/experiment_7491_v656_window_protocol/evaluators.jsonl")
BOARD_PATH = Path("results/experiment_7473_v654_board_continuity.json")
GRADUATION_PATH = Path("results/experiment_7314_v642_board_continuity.json")
BOARD_ARTIFACT = REPO_ROOT / BOARD_PATH

MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
INTEGER_FORMATS = ("int16", "int8")
PROBABILITY_ERROR_LIMIT = 0.01
TRAINING_GROUPS = 176
CALIBRATION_GROUPS = 60
WARMUP_BATCHES = 3
TIMED_BATCHES = 30
MAX_CPU_MEASUREMENT_S = 300.0
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7487_v655_learning_placement.py"),
    Path("research-hardware-wishlist.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    FIT_ARTIFACT,
    PROTOTYPE_ARTIFACT,
    CHECKPOINT_PATH,
    FEATURE_PATH,
    EVALUATOR_PATH,
    BOARD_PATH,
    GRADUATION_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed external bytes unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes and retain the source artifact's terminal flags."""

    value = _load_object(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "original_honest_verdict": value.get("honest_verdict"),
        "original_verdict_class": value.get("verdict_class"),
        "original_flagged_adversarial": value.get("flagged_adversarial"),
    }


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
    branch: str = "required",
) -> JsonDict:
    """Record the exact observed prerequisite before dependent work starts."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else passed,
        "branch": branch,
    }


def load_fit_rows(root: Path) -> JsonDict:
    """Reuse the qualified fit reader without opening held-out labels."""

    return upstream_load_fit_rows(root)


def quantize_symmetric(values: Any, bits: int) -> tuple[np.ndarray, float]:
    """Quantize a finite tensor with one signed symmetric scale."""

    if bits not in (8, 16):
        raise ValueError("integer_bits_invalid")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("quantization_input_nonfinite")
    limit = (1 << (bits - 1)) - 1
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    scale = maximum / limit if maximum else 1.0
    dtype = np.int16 if bits == 16 else np.int8
    quantized = np.rint(array / scale).clip(-limit, limit).astype(dtype)
    return quantized, scale


def dequantize(values: Any, scale: float) -> np.ndarray:
    """Convert one integer tensor with its explicit scale to float64."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantization_scale_invalid")
    return np.asarray(values, dtype=np.float64) * scale


def load_checkpoint_bundle(root: Path) -> JsonDict:
    """Load the frozen head and reconstruct its hash-bound training transform."""

    checkpoint = _load_object(root / CHECKPOINT_PATH)
    fit_rows = load_fit_rows(root)
    training = fit_rows["training"]
    matrix = np.asarray([row["features"] for row in training], dtype=np.float64)
    mean = matrix.mean(axis=0)
    population_sd = matrix.std(axis=0)
    safe_scale = np.where(population_sd > 0.0, population_sd, 1.0)
    normalized = (matrix - mean) / safe_scale
    knots = fit_spline_knots(normalized)
    transform = {
        "feature_names": list(FEATURE_NAMES),
        "normalization": {
            "fit_role": "training",
            "mean": mean.tolist(),
            "population_sd": population_sd.tolist(),
            "safe_scale": safe_scale.tolist(),
        },
        "spline": {
            "basis": "open_clamped_cubic_quantile",
            "knots": knots.tolist(),
            "coefficients_per_input": 8,
            "trainable_parameter_count": 81,
        },
    }
    if canonical_hash(transform) != checkpoint.get("transform_sha256"):
        raise ValueError("frozen_transform_hash_mismatch")
    checkpoint["transform"] = transform
    return checkpoint


def frozen_action(
    probability: float,
    false_accept_cost: float,
    escalation_cost: float,
    false_reject_cost: float,
) -> str:
    """Apply the upstream minimum-cost rule with its frozen tie order."""

    return expected_action(
        probability,
        false_accept_cost=false_accept_cost,
        escalation_cost=escalation_cost,
        false_reject_cost=false_reject_cost,
    )


def _numeric_inputs(rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]) -> JsonDict:
    """Build float32 basis rows and retain only the nine selected policies."""

    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    transform = checkpoint["transform"]
    normalization = transform["normalization"]
    normalized = (features - np.asarray(normalization["mean"])) / np.asarray(
        normalization["safe_scale"]
    )
    knots = np.asarray(transform["spline"]["knots"], dtype=np.float64)
    design = spline_design_matrix(normalized, knots).astype(np.float32)
    head = checkpoint["selected_heads"]["window_gibbs"]
    coefficients = np.asarray(head["checkpoint"]["coefficient"], dtype=np.float32)
    bias = np.float32(head["checkpoint"]["bias"])
    policies = [
        deepcopy(dict(row))
        for row in checkpoint["frozen_policies"]["rows"]
        if row.get("probability_source") == "window_gibbs"
    ]
    if design.shape != (TRAINING_GROUPS + CALIBRATION_GROUPS, len(coefficients)):
        raise ValueError("placement_design_shape_invalid")
    if len(policies) != 9:
        raise ValueError("frozen_policy_count_invalid")
    return {
        "design": design,
        "coefficients": coefficients,
        "bias": bias,
        "knots": knots.astype(np.float32),
        "policies": policies,
    }


def _integer_kernel(
    design: np.ndarray,
    coefficients: np.ndarray,
    bias: np.float32,
    *,
    bits: int,
) -> JsonDict:
    """Run one integer dot product and expose all scales and overflow bounds."""

    block_size = 8
    block_count = design.shape[1] // block_size
    basis_blocks: list[np.ndarray] = []
    coefficient_blocks: list[np.ndarray] = []
    basis_scales: list[float] = []
    coefficient_scales: list[float] = []
    for block in range(block_count):
        block_slice = slice(block * block_size, (block + 1) * block_size)
        basis_integer, basis_scale = quantize_symmetric(design[:, block_slice], bits)
        coefficient_integer, coefficient_scale = quantize_symmetric(coefficients[block_slice], bits)
        basis_blocks.append(basis_integer)
        coefficient_blocks.append(coefficient_integer)
        basis_scales.append(basis_scale)
        coefficient_scales.append(coefficient_scale)
    accumulator_dtype = np.int64 if bits == 16 else np.int32
    accumulator_limit = np.iinfo(accumulator_dtype).max
    accumulators = np.column_stack(
        [
            basis.astype(accumulator_dtype) @ coefficient.astype(accumulator_dtype)
            for basis, coefficient in zip(basis_blocks, coefficient_blocks, strict=True)
        ]
    )
    product_scales = np.asarray(basis_scales) * np.asarray(coefficient_scales)
    bias_integer, bias_scale = quantize_symmetric(np.asarray([bias]), bits)
    overflow = np.abs(accumulators.astype(np.int64)) > accumulator_limit
    logits = accumulators.astype(np.float64) @ product_scales + float(
        dequantize(bias_integer, bias_scale)[0]
    )
    probabilities = stable_sigmoid(logits).astype(np.float32)
    theoretical_bound = block_size * ((1 << (bits - 1)) - 1) ** 2
    return {
        "basis_integer": np.column_stack(basis_blocks),
        "coefficient_integer": np.concatenate(coefficient_blocks),
        "basis_scales": basis_scales,
        "coefficient_scales": coefficient_scales,
        "product_scales": product_scales.tolist(),
        "bias_integer": int(bias_integer[0]),
        "bias_scale": bias_scale,
        "accumulator": accumulators,
        "accumulator_dtype": np.dtype(accumulator_dtype).name,
        "accumulator_bits": np.dtype(accumulator_dtype).itemsize * 8,
        "accumulator_limit": int(accumulator_limit),
        "theoretical_accumulator_bound": theoretical_bound,
        "block_size": block_size,
        "block_count": block_count,
        "overflow": overflow,
        "probabilities": probabilities,
    }


def _policy_comparisons(
    reference: float,
    candidate: float,
    policies: Sequence[Mapping[str, Any]],
    *,
    calibration_index: int | None,
) -> list[JsonDict]:
    """Compare one integer result with float32 across all frozen policies."""

    comparisons: list[JsonDict] = []
    for index, policy in enumerate(policies):
        costs = {
            "false_accept_cost": float(policy["false_accept_cost"]),
            "escalation_cost": float(policy["escalation_cost"]),
            "false_reject_cost": float(policy["false_reject_cost"]),
        }
        reference_action = frozen_action(reference, **costs)
        candidate_action = frozen_action(candidate, **costs)
        frozen_actions = policy.get("actions")
        upstream_action = (
            frozen_actions[calibration_index]
            if calibration_index is not None and isinstance(frozen_actions, list)
            else None
        )
        comparisons.append(
            {
                "policy_id": f"fa{costs['false_accept_cost']:g}_e{costs['escalation_cost']:g}",
                **costs,
                "float32_action": reference_action,
                "integer_action": candidate_action,
                "upstream_frozen_action": upstream_action,
                "float32_matches_upstream": (
                    None if upstream_action is None else reference_action == upstream_action
                ),
                "disagreed": reference_action != candidate_action,
            }
        )
    return comparisons


def _parameter_footprint(
    numeric: Mapping[str, Any], *, bits: int, coefficient_count: int, knot_count: int
) -> JsonDict:
    """Count retained parameters and one per-request basis vector separately."""

    integer_bytes = bits // 8
    accumulator_bytes = int(numeric["accumulator_bits"]) // 8
    return {
        "coefficient_bytes": coefficient_count * integer_bytes,
        "bias_integer_bytes": integer_bytes,
        "accumulator_working_bytes": int(numeric["block_count"]) * accumulator_bytes,
        "lookup_knot_bytes": knot_count * 4,
        "basis_row_bytes": coefficient_count * integer_bytes,
        "retained_parameter_bytes": (
            coefficient_count * integer_bytes + integer_bytes + knot_count * 4
        ),
        "float32_retained_parameter_bytes": coefficient_count * 4 + 4 + knot_count * 4,
        "includes_lookup_storage": True,
        "includes_basis_storage": True,
        "normalization_transform_bytes": len(FEATURE_NAMES) * 2 * 4,
    }


def _quantization_rows(
    source_rows: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any]
) -> list[JsonDict]:
    """Emit every training and calibration group for both integer formats."""

    design = np.asarray(inputs["design"], dtype=np.float32)
    coefficients = np.asarray(inputs["coefficients"], dtype=np.float32)
    bias = np.float32(inputs["bias"])
    policies = inputs["policies"]
    reference = stable_sigmoid(design @ coefficients + bias).astype(np.float32)
    rows: list[JsonDict] = []
    calibration_index = 0
    for arithmetic in INTEGER_FORMATS:
        bits = int(arithmetic.removeprefix("int"))
        numeric = _integer_kernel(design, coefficients, bias, bits=bits)
        footprint = _parameter_footprint(
            numeric,
            bits=bits,
            coefficient_count=len(coefficients),
            knot_count=np.asarray(inputs["knots"]).size,
        )
        for index, source in enumerate(source_rows):
            role = str(source["role"])
            current_calibration_index = calibration_index if role == "calibration_tuning" else None
            candidate = float(numeric["probabilities"][index])
            baseline = float(reference[index])
            comparisons = _policy_comparisons(
                baseline,
                candidate,
                policies,
                calibration_index=current_calibration_index,
            )
            overflow_count = int(bool(np.any(numeric["overflow"][index])))
            rows.append(
                {
                    "row_type": "quantization_group",
                    "unit_id": f"{role}:{source['group_id']}:{arithmetic}",
                    "group_id": source["group_id"],
                    "role": role,
                    "arithmetic": arithmetic,
                    "feature_count": len(FEATURE_NAMES),
                    "basis_value_count": design.shape[1],
                    "float32_probability": baseline,
                    "integer_probability": candidate,
                    "abs_probability_error": abs(candidate - baseline),
                    "action_disagreement_count": sum(
                        comparison["disagreed"] is True for comparison in comparisons
                    ),
                    "policy_comparisons": comparisons,
                    "overflow_count": overflow_count,
                    "accumulator_dtype": numeric["accumulator_dtype"],
                    "accumulator_values": [int(value) for value in numeric["accumulator"][index]],
                    "accumulator_limit": numeric["accumulator_limit"],
                    "theoretical_accumulator_bound": numeric["theoretical_accumulator_bound"],
                    "basis_scales": numeric["basis_scales"],
                    "coefficient_scales": numeric["coefficient_scales"],
                    "product_scales": numeric["product_scales"],
                    "bias_scale": numeric["bias_scale"],
                    "block_size": numeric["block_size"],
                    "parameter_footprint": deepcopy(footprint),
                    "execution_venue": "host_cpu_emulation",
                    "fpga_performance_measured": False,
                    "complete": True,
                    "failed": False,
                    "censored": False,
                    "excluded": False,
                }
            )
            if role == "calibration_tuning":
                calibration_index += 1
        calibration_index = 0
    return rows


def _time_component(operation: Any) -> tuple[Any, int]:
    """Measure one callable with a monotonic high-resolution CPU clock."""

    started = time.perf_counter_ns()
    value = operation()
    return value, time.perf_counter_ns() - started


def _benchmark_arm(inputs: Mapping[str, Any], arithmetic: str, *, timed_batches: int) -> JsonDict:
    """Time allocation, conversion, kernel, readout, and policy decisions."""

    design = np.asarray(inputs["design"], dtype=np.float32)
    coefficients = np.asarray(inputs["coefficients"], dtype=np.float32)
    bias = np.float32(inputs["bias"])
    policies = inputs["policies"]
    components = {
        name: []
        for name in (
            "allocation",
            "quantization",
            "kernel",
            "dequantization",
            "policy_decision",
        )
    }
    for _ in range(timed_batches):
        allocated, elapsed = _time_component(
            lambda: (np.empty_like(design), np.empty_like(coefficients))
        )
        components["allocation"].append(elapsed)
        del allocated
        if arithmetic == "float32":
            converted, elapsed = _time_component(
                lambda: (design.astype(np.float32), coefficients.astype(np.float32))
            )
            components["quantization"].append(elapsed)
            batch_design, batch_coefficients = converted
            logits, elapsed = _time_component(lambda: batch_design @ batch_coefficients + bias)
            components["kernel"].append(elapsed)
            probabilities, elapsed = _time_component(
                lambda: stable_sigmoid(logits).astype(np.float32)
            )
        else:
            bits = int(arithmetic.removeprefix("int"))
            converted, elapsed = _time_component(
                lambda: [
                    (
                        quantize_symmetric(design[:, block * 8 : (block + 1) * 8], bits),
                        quantize_symmetric(coefficients[block * 8 : (block + 1) * 8], bits),
                    )
                    for block in range(design.shape[1] // 8)
                ]
            )
            components["quantization"].append(elapsed)
            accumulator_dtype = np.int64 if bits == 16 else np.int32
            accumulators, elapsed = _time_component(
                lambda: np.column_stack(
                    [
                        basis[0].astype(accumulator_dtype)
                        @ coefficient[0].astype(accumulator_dtype)
                        for basis, coefficient in converted
                    ]
                )
            )
            components["kernel"].append(elapsed)
            product_scales = np.asarray(
                [basis[1] * coefficient[1] for basis, coefficient in converted]
            )
            bias_integer, bias_scale = quantize_symmetric(np.asarray([bias]), bits)
            probabilities, elapsed = _time_component(
                lambda: stable_sigmoid(
                    accumulators.astype(np.float64) @ product_scales
                    + dequantize(bias_integer, bias_scale)[0]
                ).astype(np.float32)
            )
        components["dequantization"].append(elapsed)
        _actions, elapsed = _time_component(
            lambda: [
                frozen_action(
                    float(probability),
                    float(policy["false_accept_cost"]),
                    float(policy["escalation_cost"]),
                    float(policy["false_reject_cost"]),
                )
                for probability in probabilities
                for policy in policies
            ]
        )
        components["policy_decision"].append(elapsed)
    return {
        "arithmetic": arithmetic,
        "timed_batches": timed_batches,
        "component_ns": components,
        "component_mean_ns": {name: float(np.mean(values)) for name, values in components.items()},
        "independent_batch_count": timed_batches,
        "execution_venue": "host_cpu_emulation",
        "fpga_performance_measured": False,
    }


def _shadow_training_benchmark(
    source_rows: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    *,
    timed_batches: int,
) -> JsonDict:
    """Measure sparse Brier updates on a disposable copy of the frozen head."""

    training_count = sum(row["role"] == "training" for row in source_rows)
    design = np.asarray(inputs["design"][:training_count], dtype=np.float32)
    labels = np.asarray([row["label"] for row in source_rows[:training_count]], dtype=np.float32)
    original = np.asarray(inputs["coefficients"], dtype=np.float32)
    component_ns = {name: [] for name in ("allocation", "prediction", "gradient", "update")}
    for _ in range(timed_batches):
        coefficient, elapsed = _time_component(lambda: original.copy())
        component_ns["allocation"].append(elapsed)
        probabilities, elapsed = _time_component(
            lambda: stable_sigmoid(design @ coefficient + np.float32(inputs["bias"])).astype(
                np.float32
            )
        )
        component_ns["prediction"].append(elapsed)
        gradient, elapsed = _time_component(
            lambda: (
                (
                    design.T
                    @ (
                        np.float32(2.0)
                        * (probabilities - labels)
                        * probabilities
                        * (np.float32(1.0) - probabilities)
                    )
                )
                / np.float32(training_count)
            )
        )
        component_ns["gradient"].append(elapsed)
        _updated, elapsed = _time_component(
            lambda: coefficient - np.float32(0.001) * gradient.astype(np.float32)
        )
        component_ns["update"].append(elapsed)
    return {
        "performed": True,
        "scope": "current_cpu_shadow_copy_only",
        "objective": "sparse_brier_update",
        "source_role": "training",
        "training_rows": training_count,
        "timed_batches": timed_batches,
        "component_ns": component_ns,
        "component_mean_ns": {
            name: float(np.mean(values)) for name, values in component_ns.items()
        },
        "upstream_checkpoint_mutated": False,
        "generator_weights_changed": False,
        "current_llm_calls": 0,
    }


def evaluate_numeric_placement(
    source_rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    *,
    warmup_batches: int = WARMUP_BATCHES,
    timed_batches: int = TIMED_BATCHES,
) -> tuple[list[JsonDict], JsonDict]:
    """Evaluate frozen parity and time independent host-emulation batches."""

    if warmup_batches < 0 or timed_batches < 1:
        raise ValueError("benchmark_batch_count_invalid")
    inputs = _numeric_inputs(source_rows, checkpoint)
    for _ in range(warmup_batches):
        _benchmark_arm(inputs, "float32", timed_batches=1)
        _benchmark_arm(inputs, "int16", timed_batches=1)
        _benchmark_arm(inputs, "int8", timed_batches=1)
    started = time.monotonic()
    benchmark_rows = [
        _benchmark_arm(inputs, arithmetic, timed_batches=timed_batches)
        for arithmetic in ("float32", *INTEGER_FORMATS)
    ]
    training = _shadow_training_benchmark(source_rows, inputs, timed_batches=timed_batches)
    rows = _quantization_rows(source_rows, inputs)
    duration_s = time.monotonic() - started
    return rows, {
        "execution_venue": "host_cpu_emulation",
        "warmup_batches": warmup_batches,
        "timed_batches": timed_batches,
        "rows": benchmark_rows,
        "small_head_training": training,
        "duration_s": duration_s,
        "duration_limit_s": MAX_CPU_MEASUREMENT_S,
        "within_duration_limit": duration_s < MAX_CPU_MEASUREMENT_S,
        "fpga_performance_measured": False,
        "whole_service_performance_measured": False,
    }


def reduce_quantization_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require every role, group, format, policy, error, and overflow operand."""

    groups_by_role: dict[str, set[str]] = {"training": set(), "calibration_tuning": set()}
    keys: list[tuple[str, str, str]] = []
    row_contract = True
    calibration_rows: list[Mapping[str, Any]] = []
    for row in rows:
        role = str(row.get("role"))
        group_id = str(row.get("group_id"))
        arithmetic = str(row.get("arithmetic"))
        if role in groups_by_role:
            groups_by_role[role].add(group_id)
        keys.append((role, group_id, arithmetic))
        comparisons = row.get("policy_comparisons")
        row_contract = row_contract and bool(
            arithmetic in INTEGER_FORMATS
            and isinstance(comparisons, list)
            and len(comparisons) == 9
            and isinstance(row.get("parameter_footprint"), Mapping)
            and row.get("complete") is True
            and row.get("failed") is False
        )
        if role == "calibration_tuning":
            calibration_rows.append(row)
    expected_count = (TRAINING_GROUPS + CALIBRATION_GROUPS) * len(INTEGER_FORMATS)
    matrix_complete = (
        len(rows) == expected_count
        and len(set(keys)) == expected_count
        and len(groups_by_role["training"]) == TRAINING_GROUPS
        and len(groups_by_role["calibration_tuning"]) == CALIBRATION_GROUPS
        and all(
            {key[2] for key in keys if key[0] == role and key[1] == group} == set(INTEGER_FORMATS)
            for role, groups in groups_by_role.items()
            for group in groups
        )
    )
    overflow_count = sum(int(row.get("overflow_count", 0)) for row in rows)
    action_disagreements = sum(
        int(row.get("action_disagreement_count", 0)) for row in calibration_rows
    )
    errors = [float(row.get("abs_probability_error", math.inf)) for row in rows]
    maximum_error = max(errors, default=None)
    upstream_parity = bool(calibration_rows) and all(
        comparison.get("float32_matches_upstream") is True
        for row in calibration_rows
        for comparison in row.get("policy_comparisons") or []
    )
    ready = bool(
        matrix_complete
        and row_contract
        and overflow_count == 0
        and action_disagreements == 0
        and maximum_error is not None
        and maximum_error <= PROBABILITY_ERROR_LIMIT
        and upstream_parity
    )
    return {
        "planned_row_count": expected_count,
        "observed_row_count": len(rows),
        "training_group_count": len(groups_by_role["training"]),
        "calibration_group_count": len(groups_by_role["calibration_tuning"]),
        "arithmetic_formats": list(INTEGER_FORMATS),
        "policy_count": 9,
        "overflow_count": overflow_count,
        "action_disagreement_count": action_disagreements,
        "max_abs_probability_error": maximum_error,
        "probability_error_limit": PROBABILITY_ERROR_LIMIT,
        "float32_matches_frozen_policies": upstream_parity,
        "matrix_complete": matrix_complete,
        "row_contract_complete": row_contract,
        "numeric_placement_ready_score": int(ready),
        "quantization_rows_sha256": canonical_hash(rows),
    }


def reduce_board_rows(
    board_artifact: Mapping[str, Any], changed_state: Mapping[str, Any] | None = None
) -> tuple[list[JsonDict], JsonDict]:
    """Preserve three venue-specific conclusions without asserting reachability."""

    source_rows = board_artifact.get("board_rows")
    by_board = {
        str(row.get("board")): deepcopy(dict(row))
        for row in source_rows or []
        if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    for board in ("KV260", "PolarFire", "GateMate"):
        if board not in by_board:
            continue
        row = by_board[board]
        row["current_reachability"] = "unknown_not_probed"
        row["current_reachability_probe_issued"] = False
        row["hardware_operations_issued"] = []
        row["hardware_operation_count"] = 0
        row["fresh_physical_attempt"] = False
        row["new_hardware_execution_claimed"] = False
        row["current_execution_venue"] = "host"
        if board == "KV260":
            row["exact_claim_scope"] = "historical_kv260_fpga_fabric_sampling_only"
            row["historical_execution_venue"] = "kv260_fpga_fabric"
            row["architecture_limit"] = "k_max<=5"
            row["future_access"] = "ssh kria only"
        elif board == "PolarFire":
            row["exact_claim_scope"] = "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
            row["historical_execution_venue"] = "polarfire_linux_cpu"
            row["fpga_sampling_claimed"] = False
        else:
            prior_disposition = str(row.get("current_disposition"))
            state = deepcopy(
                dict(changed_state or row.get("changed_state_evidence") or {"exists": False})
            )
            changed = (
                state.get("exists") is True or int(state.get("accepted_receipt_count", 0) or 0) > 0
            )
            row["changed_state_evidence"] = state
            row["previous_disposition"] = prior_disposition
            row["current_disposition"] = (
                "changed_physical_prerequisite_recorded_future_probe_only"
                if changed
                else "blocked_unchanged_physical_prerequisite"
            )
            row["disposition"] = "complete" if changed else "blocked"
            row["historical_execution_venue"] = "none_read_only"
            row["exact_claim_scope"] = "no_current_execution_changed_state_gate_only"
            row["gatemate_blocker_changed"] = row["current_disposition"] != prior_disposition
        rows.append(row)
    board_names = [row.get("board") for row in rows]
    complete = bool(
        board_names == ["KV260", "PolarFire", "GateMate"]
        and rows[0].get("exact_claim_scope") == "historical_kv260_fpga_fabric_sampling_only"
        and rows[0].get("historical_execution_venue") == "kv260_fpga_fabric"
        and rows[1].get("exact_claim_scope")
        == "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
        and rows[1].get("historical_execution_venue") == "polarfire_linux_cpu"
        and rows[2].get("current_disposition")
        in {
            "blocked_unchanged_physical_prerequisite",
            "changed_physical_prerequisite_recorded_future_probe_only",
        }
        and all(row.get("current_reachability") == "unknown_not_probed" for row in rows)
        and all(row.get("hardware_operations_issued") == [] for row in rows)
    )
    blocker_changed = bool(rows and rows[-1].get("gatemate_blocker_changed") is True)
    return rows, {
        "board_continuity_complete_score": int(complete),
        "gatemate_blocker_changed": blocker_changed,
        "gatemate_current_disposition": (
            rows[-1].get("current_disposition") if len(rows) == 3 else None
        ),
        "hardware_operations_issued": [],
        "current_reachability_probes": 0,
        "board_rows_sha256": canonical_hash(rows),
    }


def build_operation_inventory(parameter_count: int, policy_count: int) -> list[JsonDict]:
    """Inventory full service work without inventing transfer or device costs."""

    shared = {
        "transfer_cost": "unknown",
        "whole_service_denominator_available": False,
        "whole_service_speedup": None,
        "kernel_ceiling_class": "hypothetical_only",
    }
    return [
        {
            "operation": "prediction",
            "operations": {
                "integer_multiply": parameter_count,
                "integer_add": parameter_count,
                "sigmoid_lookup_or_host_math": 1,
            },
            "bytes": {
                "feature_basis": "format_dependent",
                "coefficients": "format_dependent",
                "probability_output": 4,
            },
            "state": "frozen_head_and_one_feature_group",
            "conditional_binary_energy_normalization": "exact_two_state",
            "tsu_sampling_required": False,
            "current_tsu_access": "unverified",
            **shared,
        },
        {
            "operation": "sparse_brier_update",
            "operations": {
                "prediction": 1,
                "active_gradient_terms": parameter_count,
                "coefficient_updates": parameter_count,
            },
            "bytes": {
                "read_state": "coefficients_plus_feature_basis",
                "write_state": "shadow_coefficients_only",
            },
            "state": "current_host_cpu_shadow_copy_not_upstream_checkpoint",
            **shared,
        },
        {
            "operation": "guard",
            "operations": {
                "overflow_checks": 1,
                "probability_error_checks": 1,
                "policy_action_checks": policy_count,
            },
            "bytes": {"policy_cells": policy_count, "gate_outputs": 3},
            "state": "row_local_then_independent_reduction",
            **shared,
        },
        {
            "operation": "serialization",
            "operations": {"json_encode": 1, "checksum_update": 1},
            "bytes": {"payload": "measured_at_runtime"},
            "state": "candidate_then_terminal_artifact",
            **shared,
        },
        {
            "operation": "durable_acknowledgement",
            "operations": {"write": 1, "flush": 1, "fsync": 1, "atomic_replace": 1},
            "bytes": {"terminal_artifact": "measured_at_runtime"},
            "state": "published_only_after_required_readers_pass",
            **shared,
        },
    ]


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate required bytes and classify numeric and board branches."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = repo / relative
        available = path.is_file() and path.stat().st_size > 0
        branch = (
            "numeric"
            if relative
            in {FIT_ARTIFACT, PROTOTYPE_ARTIFACT, CHECKPOINT_PATH, FEATURE_PATH, EVALUATOR_PATH}
            else "board"
            if relative in {BOARD_PATH, GRADUATION_PATH}
            else "required"
        )
        rows.append(
            _precondition(
                "resource_readable",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                branch=branch,
            )
        )
        if available:
            sources[relative.as_posix()] = _source_row(path, repo)

    spec_text = (
        (repo / SPEC_PATH).read_text(encoding="utf-8") if (repo / SPEC_PATH).is_file() else ""
    )
    rows.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-HW-7513",
            "REQ-HW-7513" if "REQ-HW-7513" in spec_text else None,
        )
    )
    exclusion_text = (
        (repo / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (repo / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    excluded = "experiment_id: 7513" in exclusion_text
    rows.append(
        _precondition(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            "experiment_id",
            False,
            excluded,
        )
    )

    fit = _load_object(repo / FIT_ARTIFACT)
    prototype = _load_object(repo / PROTOTYPE_ARTIFACT)
    board = _load_object(repo / BOARD_PATH)
    graduation = _load_object(repo / GRADUATION_PATH)
    fit_errors = (
        validate_fit_artifact(fit, root=repo, verify_sources=False) if fit else ["artifact_missing"]
    )
    prototype_errors = (
        validate_prototype_artifact(prototype, root=repo, verify_sources=False)
        if prototype
        else ["artifact_missing"]
    )
    numeric_checks = (
        (
            "fit_identity",
            FIT_ARTIFACT,
            "experiment_id",
            "exp7505-v657-energy-fit",
            fit.get("experiment_id"),
        ),
        ("fit_ready", FIT_ARTIFACT, "energy_fit_ready_score", 1, fit.get("energy_fit_ready_score")),
        (
            "fit_unflagged",
            FIT_ARTIFACT,
            "flagged_adversarial",
            False,
            fit.get("flagged_adversarial"),
        ),
        ("fit_valid", FIT_ARTIFACT, "cold_reader_errors", [], fit_errors),
        (
            "prototype_identity",
            PROTOTYPE_ARTIFACT,
            "experiment_id",
            "exp7506-v657-causal-prototype",
            prototype.get("experiment_id"),
        ),
        (
            "prototype_ready",
            PROTOTYPE_ARTIFACT,
            "causal_update_ready_score",
            1,
            prototype.get("causal_update_ready_score"),
        ),
        (
            "prototype_unflagged",
            PROTOTYPE_ARTIFACT,
            "flagged_adversarial",
            False,
            prototype.get("flagged_adversarial"),
        ),
        ("prototype_valid", PROTOTYPE_ARTIFACT, "cold_reader_errors", [], prototype_errors),
    )
    for check, path, field, expected, observed in numeric_checks:
        rows.append(
            _precondition(
                check,
                path.as_posix(),
                field,
                expected,
                observed,
                branch="numeric",
            )
        )
    checkpoint_receipt = fit.get("raw_sidecars", {}).get("checkpoints", {})
    checkpoint_hash = (
        sha256_file(repo / CHECKPOINT_PATH) if (repo / CHECKPOINT_PATH).is_file() else None
    )
    rows.append(
        _precondition(
            "checkpoint_hash",
            CHECKPOINT_PATH.as_posix(),
            "sha256",
            checkpoint_receipt.get("sha256"),
            checkpoint_hash,
            branch="numeric",
        )
    )

    board_checks = (
        (
            "board_identity",
            BOARD_PATH,
            "experiment_id",
            "exp7473-v654-board-continuity",
            board.get("experiment_id"),
        ),
        (
            "board_unflagged",
            BOARD_PATH,
            "flagged_adversarial",
            False,
            board.get("flagged_adversarial"),
        ),
        (
            "graduation_identity",
            GRADUATION_PATH,
            "experiment_id",
            7314,
            graduation.get("experiment_id"),
        ),
    )
    for check, path, field, expected, observed in board_checks:
        rows.append(
            _precondition(
                check,
                path.as_posix(),
                field,
                expected,
                observed,
                branch="board",
            )
        )
    graduation_hash = (
        sha256_file(repo / GRADUATION_PATH) if (repo / GRADUATION_PATH).is_file() else None
    )
    evidence_hashes = {
        row.get("evidence_sha256")
        for row in board.get("board_rows") or []
        if isinstance(row, Mapping) and row.get("board") in {"KV260", "PolarFire"}
    }
    rows.append(
        _precondition(
            "graduation_hash_preserved",
            BOARD_PATH.as_posix(),
            "board_rows[].evidence_sha256",
            [graduation_hash],
            sorted(evidence_hashes),
            branch="board",
        )
    )
    required_ready = all(row["passed"] for row in rows if row["branch"] == "required")
    numeric_ready = required_ready and all(
        row["passed"] for row in rows if row["branch"] == "numeric"
    )
    board_ready = required_ready and all(row["passed"] for row in rows if row["branch"] == "board")
    return {
        "rows": rows,
        "source_artifact_hashes": sources,
        "required_ready": required_ready,
        "numeric_ready": numeric_ready,
        "board_ready": board_ready,
        "fit": fit,
        "prototype": prototype,
        "board": board,
        "graduation": graduation,
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one clean receipt for each affected and terminal command."""

    required = set(AFFECTED_CHECK_NAMES) | set(TERMINAL_CHECK_NAMES)
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in receipts:
        by_name.setdefault(str(row.get("name")), []).append(row)
    return all(
        len(by_name.get(name, [])) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is not True
        for name in required
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute independent placement, board, and validation outcomes."""

    numeric = reduce_quantization_rows(value.get("quantization_rows") or [])
    _board_rows, board = reduce_board_rows(
        {"board_rows": value.get("board_rows") or []}, value.get("gatemate_changed_state")
    )
    declarations = bool(
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and value.get("inference_substrate") == INFERENCE_SUBSTRATE
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("execution_venue") == EXECUTION_VENUE
    )
    benchmark = value.get("cpu_benchmark") or {}
    benchmark_valid = bool(
        value.get("numeric_branch_status") == "blocked_external_prerequisite"
        or (
            benchmark.get("timed_batches", 0) >= 30
            and benchmark.get("within_duration_limit") is True
            and len(benchmark.get("rows") or []) == 3
            and all(
                set(row.get("component_ns") or {})
                == {
                    "allocation",
                    "quantization",
                    "kernel",
                    "dequantization",
                    "policy_decision",
                }
                and all(len(values) >= 30 for values in row["component_ns"].values())
                for row in benchmark.get("rows") or []
            )
        )
    )
    return {
        "numeric": numeric,
        "board": board,
        "current_inference_declarations_valid": declarations,
        "benchmark_valid": benchmark_valid,
        "required_validation_passed": _validation_passed(value.get("validation_receipts") or []),
        "numeric_placement_ready_score": numeric["numeric_placement_ready_score"],
        "board_continuity_complete_score": board["board_continuity_complete_score"],
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    path: str,
    field: str,
) -> JsonDict:
    """Attach exact operands and a reason to one acceptance decision."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def _acceptance_gates(value: Mapping[str, Any], reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Keep external readiness, evidence validity, and scientific benefit separate."""

    failed_numeric = next(
        (
            row
            for row in value.get("preconditions_checked") or []
            if row.get("branch") == "numeric" and row.get("passed") is not True
        ),
        None,
    )
    numeric_input_ready = failed_numeric is None
    numeric = reduction["numeric"]
    board = reduction["board"]
    return [
        _gate(
            "numeric_inputs_ready",
            "readiness",
            failed_numeric.get("expected") if failed_numeric else True,
            failed_numeric.get("observed") if failed_numeric else True,
            "eq",
            numeric_input_ready,
            "External absence blocks dependent arithmetic but not board accounting.",
            upstream=str(failed_numeric.get("upstream") if failed_numeric else "Exp7505+Exp7506"),
            path=str(failed_numeric.get("path") if failed_numeric else FIT_ARTIFACT.as_posix()),
            field=str(failed_numeric.get("artifact_field") if failed_numeric else "ready"),
        ),
        _gate(
            "current_inference_declarations",
            "validity",
            True,
            reduction["current_inference_declarations_valid"],
            "is",
            reduction["current_inference_declarations_valid"] is True,
            "Historical model provenance cannot become a current model call.",
            upstream="current_work",
            path="invocation_counts",
            field="inference_declarations",
        ),
        _gate(
            "board_continuity_complete",
            "readiness",
            1,
            board["board_continuity_complete_score"],
            "eq",
            board["board_continuity_complete_score"] == 1,
            "All three venue-specific dispositions remain visible even if learning is blocked.",
            upstream="Exp7473+Exp7314",
            path="board_rows",
            field="board_continuity_complete_score",
        ),
        _gate(
            "numeric_overflow",
            "readiness",
            0,
            numeric["overflow_count"],
            "eq",
            numeric_input_ready and numeric["overflow_count"] == 0,
            "Accumulator overflow invalidates an integer placement even when actions match.",
            upstream="quantization_rows",
            path="quantization_rows",
            field="overflow_count",
        ),
        _gate(
            "numeric_probability_error",
            "readiness",
            PROBABILITY_ERROR_LIMIT,
            numeric["max_abs_probability_error"],
            "<=",
            bool(
                numeric_input_ready
                and numeric["max_abs_probability_error"] is not None
                and numeric["max_abs_probability_error"] <= PROBABILITY_ERROR_LIMIT
            ),
            "The frozen limit prevents a favorable action tie from hiding numeric drift.",
            upstream="quantization_rows",
            path="quantization_rows",
            field="max_abs_probability_error",
        ),
        _gate(
            "numeric_action_parity",
            "readiness",
            0,
            numeric["action_disagreement_count"],
            "eq",
            numeric_input_ready and numeric["action_disagreement_count"] == 0,
            "All nine policies must preserve calibration decisions.",
            upstream="quantization_rows",
            path="quantization_rows[].policy_comparisons",
            field="action_disagreement_count",
        ),
        _gate(
            "host_benchmark_protocol",
            "validity",
            True,
            reduction["benchmark_valid"],
            "is",
            reduction["benchmark_valid"] is True,
            "Warmup and independent batches prevent one setup call from posing as performance.",
            upstream="cpu_benchmark",
            path="cpu_benchmark",
            field="timed_batches",
        ),
        _gate(
            "scoped_terminal_validation",
            "validity",
            True,
            reduction["required_validation_passed"],
            "is",
            reduction["required_validation_passed"] is True,
            "A favorable numeric result cannot excuse failed implementation checks.",
            upstream="validation_receipts",
            path="validation_receipts",
            field="required_checks_passed",
        ),
        _gate(
            "whole_service_denominator_measured",
            "benefit",
            True,
            False,
            "is",
            False,
            "Exp7514 must measure the denominator before any service speedup claim.",
            upstream="future_Exp7514",
            path="whole_service_speedup",
            field="denominator_available",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """List every failed operand with its exact upstream and path."""

    failed = [
        {
            "check": row.get("check"),
            "category": row.get("category"),
            "upstream": row.get("upstream"),
            "path": row.get("path"),
            "field": row.get("field"),
            "expected": row.get("expected"),
            "observed": row.get("observed"),
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {"passed": not failed, "failed_count": len(failed), "failed_checks": failed}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain how each field prevents one class of evidence failure."""

    specific = {
        "schema": "Versioned identity prevents reader drift.",
        "run_date": "The fixed date stays separate from actual UTC and monotonic clocks.",
        "preconditions_checked": "Exact paths and observed failures prevent invented prerequisites.",
        "MODEL_SPECS": "An empty list proves the current task did not load a model.",
        "model_specs": "The lowercase mirror prevents aliases from hiding a model call.",
        "model_invoked": "False separates current work from historical Qwen evidence.",
        "invocation_counts": "Balanced zero counters expose any concealed current model operation.",
        "inference_substrate": "The exact aggregation label prevents cached evidence from posing as live inference.",
        "inference_substrate_class": "No-model-load selects the correct validation floor.",
        "execution_venue": "Host work stays distinct from historical FPGA and board-CPU evidence.",
        "duration_s": "Measured elapsed work cannot be padded into plausibility.",
        "phase_spans": "Flushed boundaries and checkpoints expose silent unfinished work.",
        "random_seed": "Frozen upstream seeds do not multiply the sample size.",
        "reproducibility_checksum": "One checksum binds sources, settings, rows, and validation.",
        "source_artifact_hashes": "Exact bytes preserve original flags and venue-specific claims.",
        "rows": "Per-unit rows retain failures, censoring, and independent board evidence.",
        "sample_size_budget": "Planned, attempted, completed, and blocked units stay distinct.",
        "acceptance_gate_results": "Expected and observed operands keep validity separate from benefit.",
        "gate_check_summary": "Every block names the exact upstream field and observed value.",
        "honest_verdict": "A complete prefix prevents a terminal null from becoming a retry.",
        "verdict_class": "The closed class separates null, blocked, and disqualified outcomes.",
        "verifier_is_oracle": "False prevents protocol readers from becoming scientific labels.",
        "flagged_adversarial": "A real guard finding cannot be cleared to open a gate.",
        "validation_receipts": "Commands, exits, and log hashes bind the exact checked scope.",
        "field_principles": "Field reasons keep the terminal record self-explanatory.",
        "board_continuity_complete_score": "A bare score retains all three dispositions independently of numeric work.",
        "numeric_placement_ready_score": "A bare score requires valid host equivalence and is not device performance.",
        "board_rows": "Venue, history, and physical prerequisites prevent fabricated integration.",
        "quantization_rows": "Every group exposes error, action parity, overflow, and footprint.",
        "operation_inventory": "Bytes, operations, state, transfers, and unknowns support a later service bound.",
        "whole_service_speedup": "None prevents a kernel benchmark from becoming a service claim.",
    }
    return {
        field: specific.get(
            field, "This field preserves one auditable part of the terminal experiment record."
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind protocol, exact sources, raw rows, declarations, and validation."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "milestone": value.get("milestone"),
            "run_date": value.get("run_date"),
            "protocol": value.get("protocol"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "quantization_rows": value.get("quantization_rows"),
            "board_rows": value.get("board_rows"),
            "operation_inventory": value.get("operation_inventory"),
            "evidence_sidecars": value.get("evidence_sidecars"),
            "MODEL_SPECS": value.get("MODEL_SPECS"),
            "model_specs": value.get("model_specs"),
            "model_invoked": value.get("model_invoked"),
            "invocation_counts": value.get("invocation_counts"),
            "validation_manifest": value.get("validation_manifest"),
            "validation_receipts": value.get("validation_receipts"),
        }
    )


def _finalize(value: JsonDict) -> JsonDict:
    """Derive scores, gates, terminal class, principles, and checksum."""

    reduction = independent_reduce(value)
    value["independent_reduction"] = reduction
    value["numeric_placement_ready_score"] = reduction["numeric_placement_ready_score"]
    value["board_continuity_complete_score"] = reduction["board_continuity_complete_score"]
    value["gatemate_blocker_changed"] = reduction["board"]["gatemate_blocker_changed"]
    gates = _acceptance_gates(value, reduction)
    value["acceptance_gate_results"] = gates
    value["gate_check_summary"] = _gate_summary(gates)
    validity_passed = all(gate["passed"] for gate in gates if gate["category"] == "validity")
    if not validity_passed:
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_required_validation_failed"
    elif value.get("external_blocker") is not None:
        reason = str(value.get("external_blocker", {}).get("check") or "external_prerequisite")
        value["verdict_class"] = "blocked"
        value["honest_verdict"] = f"complete_blocked_{reason}"
    elif reduction["numeric_placement_ready_score"] == 1:
        value["verdict_class"] = "null"
        value["honest_verdict"] = (
            "complete_null_numeric_placement_ready_whole_service_speedup_unmeasured"
        )
    else:
        value["verdict_class"] = "null"
        value["honest_verdict"] = (
            "complete_null_numeric_placement_not_ready_board_continuity_preserved"
        )
    value["status"] = value["honest_verdict"]
    value["rows"] = [*deepcopy(value["quantization_rows"]), *deepcopy(value["board_rows"])]
    value["field_principles"] = _field_principles(
        (*value.keys(), "field_principles", "reproducibility_checksum")
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    quantization_rows: Sequence[Mapping[str, Any]],
    cpu_benchmark: Mapping[str, Any],
    board_rows: Sequence[Mapping[str, Any]],
    board_summary: Mapping[str, Any],
    changed_state: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    historical_model_provenance: Any = None,
    evidence_sidecars: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble a terminal candidate from independent numeric and board branches."""

    failed_numeric = next(
        (
            deepcopy(dict(row))
            for row in preconditions
            if row.get("branch") == "numeric" and row.get("passed") is not True
        ),
        None,
    )
    external_blocker = next(
        (
            deepcopy(dict(row))
            for row in preconditions
            if row.get("passed") is not True
            and row.get("branch") in {"required", "numeric", "board"}
        ),
        None,
    )
    numeric_status = "complete" if failed_numeric is None else "blocked_external_prerequisite"
    validation_manifest = {
        "experiment_id": VALIDATION_MANIFEST.experiment_id,
        "test_paths": list(VALIDATION_MANIFEST.test_paths),
        "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
        "static_paths": list(VALIDATION_MANIFEST.static_paths),
        "affected_files": [
            MODULE_PATH.as_posix(),
            WRAPPER_PATH.as_posix(),
            TEST_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            NOTE_PATH.as_posix(),
        ],
    }
    validation_manifest["manifest_sha256"] = canonical_hash(validation_manifest)
    numeric_count = len(quantization_rows)
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000,
        "process_identity": {"pid": os.getpid(), "hostname": platform.node()},
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": float(cpu_benchmark.get("duration_s", 0.0)),
            "validation": sum(float(row.get("duration_s", 0.0)) for row in validation_receipts),
            "historical_capture": 0.0,
        },
        "random_seed": {
            "fit_seeds": [656101, 656102, 656103, 656104, 656105],
            "arrival_seed": None,
            "audit_seed": None,
            "bootstrap_seed": None,
            "note": "Upstream seeds freeze fitting and do not multiply group count.",
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": deepcopy(dict(sources)),
        "evidence_sidecars": deepcopy(dict(evidence_sidecars or {})),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "historical_model_provenance": deepcopy(historical_model_provenance or []),
        "historical_qwen_evidence_only": True,
        "protocol": {
            "head": "Exp7505 frozen window_gibbs",
            "roles": ["training", "calibration_tuning"],
            "heldout_labels_opened": False,
            "reference_arithmetic": "float32",
            "integer_arithmetic": list(INTEGER_FORMATS),
            "probability_error_limit": PROBABILITY_ERROR_LIMIT,
            "overflow_limit": 0,
            "calibration_action_disagreement_limit": 0,
            "policy_count": 9,
            "warmup_batches": cpu_benchmark.get("warmup_batches", 0),
            "timed_batches": cpu_benchmark.get("timed_batches", 0),
            "cpu_measurement_limit_s": MAX_CPU_MEASUREMENT_S,
        },
        "numeric_branch_status": numeric_status,
        "numeric_blocker": failed_numeric,
        "external_blocker": external_blocker,
        "quantization_rows": [deepcopy(dict(row)) for row in quantization_rows],
        "cpu_benchmark": deepcopy(dict(cpu_benchmark)),
        "small_ebm_training": deepcopy(
            dict(cpu_benchmark.get("small_head_training") or {"performed": False})
        ),
        "board_rows": [deepcopy(dict(row)) for row in board_rows],
        "board_source_summary": deepcopy(dict(board_summary)),
        "gatemate_changed_state": deepcopy(dict(changed_state)),
        "hardware_operations_issued": [],
        "operation_inventory": build_operation_inventory(80, 9),
        "whole_service_speedup": None,
        "whole_service_denominator_available": False,
        "whole_service_speedup_deferred_to": "Exp7514",
        "hypothetical_kernel_placement_ceiling": {
            "classification": "hypothetical_not_measured_service_speedup",
            "transfer_costs": "unknown",
            "value": None,
        },
        "fpga_performance_measured": False,
        "hardware_milestone_graduated": False,
        "tsu_sampling_task": False,
        "conditional_binary_energy_normalization": "exact_two_state",
        "current_tsu_access": "unverified",
        "sample_size_budget": {
            "numeric_groups": {
                "planned": (TRAINING_GROUPS + CALIBRATION_GROUPS) * len(INTEGER_FORMATS),
                "attempted": numeric_count,
                "completed": sum(row.get("complete") is True for row in quantization_rows),
                "excluded": sum(row.get("excluded") is True for row in quantization_rows),
                "failed": sum(row.get("failed") is True for row in quantization_rows),
                "censored": sum(row.get("censored") is True for row in quantization_rows),
                "unstarted": max(
                    0,
                    (TRAINING_GROUPS + CALIBRATION_GROUPS) * len(INTEGER_FORMATS) - numeric_count,
                ),
            },
            "boards": {
                "planned": 3,
                "attempted": len(board_rows),
                "completed": len(board_rows),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": max(0, 3 - len(board_rows)),
            },
        },
        "validation_manifest": validation_manifest,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "capability_e2e": {
            "entrypoint_run": True,
            "fresh_process_cold_replay_required": True,
            "numbered_runtime_e2e": "not_applicable_reporting_only_host_emulation",
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
        "methodology_note": (
            "Zero action disagreement is deterministic numeric parity on training and calibration "
            "rows. It is not held-out predictive benefit or FPGA performance."
        ),
    }
    return _finalize(value)


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, raw reductions, source bytes, gates, and checksum."""

    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
        "fpga_performance_measured": False,
        "whole_service_speedup": None,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(
                "current_inference_declaration_invalid"
                if field
                in {
                    "MODEL_SPECS",
                    "model_specs",
                    "model_invoked",
                    "invocation_counts",
                    "inference_substrate",
                    "inference_substrate_class",
                    "execution_venue",
                }
                else f"field_invalid:{field}"
            )
    try:
        reduction = independent_reduce(value)
    except (KeyError, TypeError, ValueError):
        reduction = {}
    if reduction != value.get("independent_reduction"):
        errors.append("independent_reduction_mismatch")
    if reduction:
        gates = _acceptance_gates(value, reduction)
        if gates != value.get("acceptance_gate_results"):
            errors.append("acceptance_gate_results_mismatch")
        if _gate_summary(gates) != value.get("gate_check_summary"):
            errors.append("gate_check_summary_mismatch")
        if value.get("numeric_placement_ready_score") != reduction["numeric_placement_ready_score"]:
            if "independent_reduction_mismatch" not in errors:
                errors.append("independent_reduction_mismatch")
        if (
            value.get("board_continuity_complete_score")
            != reduction["board_continuity_complete_score"]
        ):
            if "independent_reduction_mismatch" not in errors:
                errors.append("independent_reduction_mismatch")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_invalid")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(value):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(not row.get("principle") for row in gates):
        errors.append("gate_principle_missing")
    if verify_sources:
        for label, row in dict(value.get("source_artifact_hashes") or {}).items():
            path = Path(str(row.get("path") or label))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_invalid:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic private receipts for pure artifact fixtures."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "log_sha256": "sha256:" + "0" * 64,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_fixture_artifact(path: Path, *, numeric_available: bool) -> JsonDict:
    """Build private complete or externally blocked evidence for mutation tests."""

    del path
    board_source = _load_object(BOARD_ARTIFACT)
    board_rows, board_summary = reduce_board_rows(board_source)
    changed_state = deepcopy(dict(board_rows[-1].get("changed_state_evidence") or {}))
    preconditions = [
        _precondition(
            "required_fixture",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-HW-7513",
            "REQ-HW-7513",
        ),
        _precondition(
            "board_fixture",
            BOARD_PATH.as_posix(),
            "presence",
            True,
            True,
            branch="board",
        ),
    ]
    quantization_rows: list[JsonDict] = []
    benchmark: JsonDict = {
        "execution_venue": "host_cpu_emulation",
        "warmup_batches": 0,
        "timed_batches": 0,
        "rows": [],
        "small_head_training": {"performed": False},
        "duration_s": 0.0,
        "duration_limit_s": MAX_CPU_MEASUREMENT_S,
        "within_duration_limit": True,
        "fpga_performance_measured": False,
    }
    if numeric_available:
        fit_rows = load_fit_rows(REPO_ROOT)
        checkpoint = load_checkpoint_bundle(REPO_ROOT)
        quantization_rows, benchmark = evaluate_numeric_placement(
            [*fit_rows["training"], *fit_rows["calibration"]],
            checkpoint,
            warmup_batches=0,
            timed_batches=30,
        )
    else:
        preconditions.append(
            _precondition(
                "fit_artifact_missing",
                FIT_ARTIFACT.as_posix(),
                "presence",
                True,
                None,
                branch="numeric",
            )
        )
    return build_artifact(
        preconditions=preconditions,
        sources={},
        quantization_rows=quantization_rows,
        cpu_benchmark=benchmark,
        board_rows=board_rows,
        board_summary=board_summary,
        changed_state=changed_state,
        validation_receipts=_passing_receipts(),
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00+00:00",
        completed_at_utc="2026-09-22T00:00:01+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000_000,
    )


def build_validation_commands(private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the affected checks with private pytest and coverage state."""

    private_root.mkdir(parents=True, exist_ok=True)
    return build_command_plan(REPO_ROOT, VALIDATION_MANIFEST, private_root)


def terminal_commands(candidate: Path) -> list[PlannedCommand]:
    """Build fresh replay, independent reduction, and strict artifact readers."""

    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "capability_end_to_end",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "quantization_and_board_rows",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one measured UTC boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and slow-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7513] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - real monotonic boundaries.
    """Close one phase with a completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _scan_gatemate(root: Path, raw_path: Path) -> JsonDict:  # pragma: no cover
    """Run the approved local receipt parser without a hardware command."""

    from carnot.experiment_7473_v654_board_continuity import (
        normalize_changed_state,
        search_changed_state_evidence,
    )

    return normalize_changed_state(search_changed_state_evidence(root, raw_path))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Write measured rows atomically and return their exact byte receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised through capability E2E.
    """Authenticate, measure, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    repo = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = utc_now()
    spans: list[JsonDict] = []
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            len(context["rows"]),
            "exact_paths_hashes_and_branch_readiness",
        )
    )
    progress(
        started,
        "preconditions",
        "complete",
        required=context["required_ready"],
        numeric=context["numeric_ready"],
        board=context["board_ready"],
    )

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0, f"no_{phase}"))
        progress(started, phase, "after", completed=0)

    progress(started, "board_ledger", "before_local_receipt_scan", planned=3)
    phase_started = time.monotonic()
    changed_state: JsonDict = {}
    if context["board_ready"]:
        changed_state = _scan_gatemate(repo, raw_dir / "gatemate_changed_state_evidence.json")
    board_rows, board_summary = reduce_board_rows(context["board"], changed_state or None)
    spans.append(_span("board_ledger", phase_started, started, len(board_rows), "three_board_rows"))
    progress(
        started,
        "board_ledger",
        "after_local_receipt_scan",
        completed=len(board_rows),
        score=board_summary["board_continuity_complete_score"],
        hardware_operations=0,
    )

    quantization_rows: list[JsonDict] = []
    benchmark: JsonDict = {
        "execution_venue": "host_cpu_emulation",
        "warmup_batches": 0,
        "timed_batches": 0,
        "rows": [],
        "small_head_training": {"performed": False},
        "duration_s": 0.0,
        "duration_limit_s": MAX_CPU_MEASUREMENT_S,
        "within_duration_limit": True,
        "fpga_performance_measured": False,
    }
    progress(started, "numeric_placement", "before_benchmark", planned=472)
    phase_started = time.monotonic()
    if context["numeric_ready"]:
        fit_rows = load_fit_rows(repo)
        checkpoint = load_checkpoint_bundle(repo)
        quantization_rows, benchmark = evaluate_numeric_placement(
            [*fit_rows["training"], *fit_rows["calibration"]], checkpoint
        )
    spans.append(
        _span(
            "numeric_placement",
            phase_started,
            started,
            len(quantization_rows),
            "per_group_integer_rows",
        )
    )
    progress(
        started,
        "numeric_placement",
        "after_benchmark",
        completed=len(quantization_rows),
        duration_s=benchmark["duration_s"],
    )
    evidence_sidecars: dict[str, JsonDict] = {}
    if quantization_rows:
        evidence_sidecars["quantization_rows"] = _write_jsonl(
            raw_dir / "quantization_rows.jsonl", quantization_rows
        )

    sources = deepcopy(dict(context["source_artifact_hashes"]))
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH, NOTE_PATH):
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = _source_row(path, repo)

    private = Path(tempfile.mkdtemp(prefix="exp7513-validation-", dir="/tmp"))
    commands = build_validation_commands(private)
    plan_errors = validate_command_plan(repo, VALIDATION_MANIFEST, commands)
    progress(
        started,
        "affected_validation",
        "before_subprocesses",
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            repo,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(repo, VALIDATION_MANIFEST, affected)
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            started,
            len(affected),
            "scoped_commands",
        )
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    if plan_errors or not affected_reduction["passed"]:
        raise RuntimeError(f"affected_validation_failed:{plan_errors}:{affected_reduction}")

    provisional = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]
    candidate = build_artifact(
        preconditions=context["rows"],
        sources=sources,
        quantization_rows=quantization_rows,
        cpu_benchmark=benchmark,
        board_rows=board_rows,
        board_summary=board_summary,
        changed_state=changed_state,
        validation_receipts=[*affected, *provisional],
        phase_spans=spans,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        historical_model_provenance=context["fit"].get("historical_model_provenance", []),
        evidence_sidecars=evidence_sidecars,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_serialization", path=candidate_path)
    atomic_json(candidate_path, candidate)
    evidence_sidecars["measured_terminal_candidate"] = {
        "path": candidate_path.relative_to(repo).as_posix(),
        "sha256": sha256_file(candidate_path),
        "bytes": candidate_path.stat().st_size,
    }
    progress(started, "candidate", "after_serialization")

    planned_terminal = terminal_commands(candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(planned_terminal))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        repo, planned_terminal, log_dir=raw_dir / "validation/terminal"
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal),
            "cold_replay_reduction_and_strict_readers",
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        preconditions=context["rows"],
        sources=sources,
        quantization_rows=quantization_rows,
        cpu_benchmark=benchmark,
        board_rows=board_rows,
        board_summary=board_summary,
        changed_state=changed_state,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        historical_model_provenance=context["fit"].get("historical_model_provenance", []),
        evidence_sidecars={
            key: value
            for key, value in evidence_sidecars.items()
            if key != "measured_terminal_candidate"
        },
    )
    errors = validate_artifact(final, root=repo, verify_sources=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(repo / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or one strict fresh-process reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=root, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

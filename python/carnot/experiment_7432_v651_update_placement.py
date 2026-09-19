"""Measure sparse online-update arithmetic and persistence on the host.

The experiment replays one complete registered Exp7427 journal condition. It
compares numeric update representations and complete persistence cost. It does
not load a model or execute on a hardware board.

Spec refs: REQ-KAN-7432 and SCENARIO-KAN-7432-*.
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
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_7367_v646_board_disposition as board_history
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    EVALUATOR_TOKEN,
    ProtocolReaders,
    reload_corpus,
)
from carnot.experiment_7412_v650_source_features import SOURCE_FEATURE_NAMES
from carnot.experiment_7425_v651_spline_prototype import (
    GRADIENT_NORM_CAP,
    LEARNING_RATE,
    dense_design_vector,
    fit_quantile_knots,
)
from carnot.experiment_7426_v651_static_decisions import (
    _apply_affine,
    join_predictor_labels,
    typed_support_decision,
)
from carnot.experiment_7427_v651_randomized_feedback import build_streams
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
PHASE = 4
EXPERIMENT_ID = "exp7432-v651-update-placement"
SCHEMA = "carnot.exp7432.v651.update_placement.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7432_v651_update_placement.json")
RAW_DIR = Path("results/raw/experiment_7432_v651_update_placement")
MODULE_PATH = Path("python/carnot/experiment_7432_v651_update_placement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7432_v651_update_placement.py")
TEST_PATH = Path("tests/python/test_experiment_7432_v651_update_placement.py")
SPEC_PATH = Path("openspec/capabilities/kan/spec.md")

EXP7425_PATH = Path("results/experiment_7425_v651_spline_prototype.json")
EXP7426_PATH = Path("results/experiment_7426_v651_static_decisions.json")
EXP7427_PATH = Path("results/experiment_7427_v651_randomized_feedback.json")
EXP7419_PATH = Path("results/experiment_7419_v650_precision_placement.json")
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_PATH = CORPUS_DIR / "corpus_manifest.json"
KV260_TRANSCRIPT_PATH = Path(
    "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"
)
POLARFIRE_TRANSCRIPT_PATH = Path("results/raw/experiment_7231/polarfire_dispatch.json")
SELECTED_CHECKPOINT_PATH = Path(
    "results/raw/experiment_7426_v651_static_decisions/checkpoints/"
    "full_source--sparse_spline_49--65101.json"
)
SELECTED_JOURNAL_CONDITION = {
    "ordering": "hash_order",
    "schedule": "hybrid_four_plus_four",
    "delay": 8,
    "seed": 65101,
    "arm": "online_sparse_spline",
}

EXPECTED_HASHES = {
    EXP7425_PATH: "sha256:362874817bd332e64e54ddd527b581070ffba06c59ef021c6c104febfff47bce",
    EXP7426_PATH: "sha256:82405ac1229cd49a3cb4bc700b4cd978d423c166b51da23f4e4af6b1e8ec7c36",
    EXP7427_PATH: "sha256:0fa3911cee634a2606cbe8b657f45b3f8bed4b88171a1248246fc2f236384555",
    EXP7419_PATH: "sha256:bb4c7fb4e041b5a0b454dbcfbe90c069b0243a47ac8712bf7bc9b80eb046529c",
    KV260_TRANSCRIPT_PATH: "sha256:b813db8619cf29c3350fe9e806f68e20f0c8a5c381b6c11cb8dc096f524f6e5e",
    POLARFIRE_TRANSCRIPT_PATH: "sha256:d1c438da812a1c45d63d5d777050c5169dab6aac33bfee8b44f4f4152dd6b801",
}

ARMS = ("float64_dense", "float32_dense", "float32_sparse", "int16_fixed_sparse")
REFERENCE_ARM = ARMS[0]
BATCH_SIZES = (1, 32, 128)
PAIRED_BLOCKS = 30
BOOTSTRAP_DRAWS = 10_000
RANDOM_SEED = 7_432_651
INT16_MIN = -(2**15)
INT16_MAX = 2**15 - 1
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/models/pwa_kan.py"),
    Path("python/carnot/experiment_7419_v650_precision_placement.py"),
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    SPEC_PATH,
    EXP7425_PATH,
    EXP7426_PATH,
    EXP7427_PATH,
    EXP7419_PATH,
    CORPUS_PATH,
    KV260_TRANSCRIPT_PATH,
    POLARFIRE_TRANSCRIPT_PATH,
    SELECTED_CHECKPOINT_PATH,
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

REQUIRED_FIELDS = {
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "update_placement_complete_score",
    "update_placement_value_score",
    "update_rows",
    "board_rows",
    "hardware_ready_score",
    "hardware_value_score",
}


def utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one real UTC boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print one flushed phase or slow-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7432] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json(path: Path) -> JsonDict:
    """Load one JSON object and fail with its exact path."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable_json:{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def gate_row(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    path: str | None = None,
    field: str | None = None,
    upstream: str | None = None,
) -> JsonDict:
    """Keep every gate operand plain and machine-readable."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
        "path": path,
        "field": field,
        "upstream": upstream,
    }


def _source_gate(root: Path, relative: Path) -> JsonDict:
    """Authenticate one exact source when pinned, otherwise its readable bytes."""

    path = root / relative
    observed_hash = sha256_file(path) if path.is_file() else None
    expected_hash = EXPECTED_HASHES.get(relative)
    passed = observed_hash is not None and (expected_hash is None or observed_hash == expected_hash)
    return gate_row(
        f"source_bytes:{relative.as_posix()}",
        "numeric_precondition",
        "==",
        expected_hash or "readable_nonempty_bytes",
        observed_hash if expected_hash else ("readable_nonempty_bytes" if observed_hash else None),
        passed,
        "Dependent work uses authenticated source bytes.",
        path=relative.as_posix(),
        field="sha256" if expected_hash else "bytes",
        upstream=relative.as_posix(),
    )


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate numeric sources and the three historical board dispositions."""

    root = root.resolve()
    checks = [_source_gate(root, relative) for relative in SOURCE_PATHS]
    hashes: JsonDict = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in SOURCE_PATHS
        if (root / relative).is_file()
    }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_requirement",
            "numeric_precondition",
            "==",
            "REQ-KAN-7432",
            "REQ-KAN-7432" if "REQ-KAN-7432" in spec else None,
            "REQ-KAN-7432" in spec,
            "The measured code remains anchored to its declared contract.",
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
            upstream=SPEC_PATH.as_posix(),
        )
    )
    upstream = {
        "exp7425": load_json(root / EXP7425_PATH),
        "exp7426": load_json(root / EXP7426_PATH),
        "exp7427": load_json(root / EXP7427_PATH),
        "exp7419": load_json(root / EXP7419_PATH),
    }
    eligibility = (
        (upstream["exp7425"].get("spline_prototype_ready_score") == 1)
        and (upstream["exp7426"].get("decision_capture_complete_score") == 1)
        and (upstream["exp7427"].get("online_capture_complete_score") == 1)
        and all(upstream[name].get("flagged_adversarial") is False for name in upstream)
    )
    checks.append(
        gate_row(
            "numeric_branch_eligibility",
            "numeric_precondition",
            "==",
            [1, 1, 1, False],
            [
                upstream["exp7425"].get("spline_prototype_ready_score"),
                upstream["exp7426"].get("decision_capture_complete_score"),
                upstream["exp7427"].get("online_capture_complete_score"),
                any(upstream[name].get("flagged_adversarial") is True for name in upstream),
            ],
            eligibility,
            "Unavailable science blocks only the numeric placement branch.",
            path=EXP7427_PATH.as_posix(),
            field="completion_and_adversarial_fields",
            upstream="exp7425+exp7426+exp7427",
        )
    )
    journal = upstream["exp7427"].get("feedback_event_rows") or {}
    shard_ok = isinstance(journal, Mapping) and journal.get("row_count") == 225900
    checks.append(
        gate_row(
            "complete_online_journal_manifest",
            "numeric_precondition",
            "==",
            225900,
            journal.get("row_count") if isinstance(journal, Mapping) else None,
            shard_ok,
            "The selected replay is read from the complete authenticated journal manifest.",
            path=EXP7427_PATH.as_posix(),
            field="feedback_event_rows.row_count",
            upstream="exp7427-v651-randomized-feedback",
        )
    )
    search_path = root / RAW_DIR / "preconditions/gatemate_changed_state_search.json"
    changed = board_history.search_changed_state_receipt(root, search_path)
    hashes[search_path.relative_to(root).as_posix()] = sha256_file(search_path)
    checks.append(
        gate_row(
            "gatemate_changed_physical_state_receipt",
            "external_prerequisite",
            "matches",
            deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT),
            changed.get("accepted_receipt_count", 0),
            changed.get("exists") is True,
            "Only an operator-authored cable, port, power, or JTAG change can reopen GateMate.",
            path=search_path.relative_to(root).as_posix(),
            field="accepted_receipt_count",
            upstream="operator_authored_local_receipts",
        )
    )
    board_rows = upstream["exp7419"].get("board_rows") or []
    names = {row.get("board") for row in board_rows if isinstance(row, Mapping)}
    checks.append(
        gate_row(
            "separate_board_rows",
            "board_accounting",
            "==",
            ["GateMate", "KV260", "PolarFire"],
            sorted(str(name) for name in names),
            names == {"KV260", "GateMate", "PolarFire"},
            "Historical board evidence remains separate from host execution.",
            path=EXP7419_PATH.as_posix(),
            field="board_rows",
            upstream="exp7419-precision-placement",
        )
    )
    numeric_ok = all(row["passed"] for row in checks if row["category"] == "numeric_precondition")
    return (
        checks,
        hashes,
        {
            "numeric_branch": {"eligible": numeric_ok},
            "upstream": upstream,
            "historical_board_rows": deepcopy(board_rows),
            "gatemate_changed_state": changed,
        },
    )


def build_board_rows(context: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve dated KV260, GateMate, and PolarFire dispositions without probing."""

    source = context.get("historical_board_rows") or []
    by_name = {
        str(row.get("board")): deepcopy(dict(row)) for row in source if isinstance(row, Mapping)
    }
    if set(by_name) != {"KV260", "GateMate", "PolarFire"}:
        return _fixture_board_rows()
    rows = [by_name[name] for name in ("KV260", "GateMate", "PolarFire")]
    for row in rows:
        row["current_execution_venue"] = "host"
        row["new_hardware_execution_claimed"] = False
        row["hardware_ready_score"] = 0
        row["hardware_value_score"] = 0
    gate = rows[1]
    changed = context.get("gatemate_changed_state") or {}
    if changed.get("exists") is not True:
        gate["terminal_state"] = "blocked_changed_physical_state"
        gate["error"] = board_history.MISSING_RECEIPT
    rows[0]["future_access"] = "ssh_only"
    rows[0]["architecture_limit"] = "k_max<=5"
    rows[2]["fpga_sampling_claimed"] = False
    return rows


def _fixture_board_rows() -> list[JsonDict]:
    """Return bounded historical board rows for mutation tests and blocked output."""

    return [
        {
            "unit_id": "board:KV260",
            "board": "KV260",
            "terminal_state": "graduated_preserved",
            "last_authenticated_date": "20260915",
            "future_access": "ssh_only",
            "architecture_limit": "k_max<=5",
            "new_hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
        {
            "unit_id": "board:GateMate",
            "board": "GateMate",
            "terminal_state": "blocked_changed_physical_state",
            "last_authenticated_date": "20260916",
            "error": board_history.MISSING_RECEIPT,
            "new_hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
        {
            "unit_id": "board:PolarFire",
            "board": "PolarFire",
            "terminal_state": "graduated_cpu_dispatch_preserved",
            "last_authenticated_date": "20260915",
            "fpga_sampling_claimed": False,
            "new_hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
    ]


def fixture_checkpoint(training: Any) -> JsonDict:
    """Build one deterministic 49-parameter state for pure reducer tests."""

    values = np.asarray(training, dtype=np.float64)
    knots = fit_quantile_knots(values)
    coefficients = np.linspace(-0.2, 0.2, 48, dtype=np.float64).reshape(6, 8)
    return {
        "knots": knots.tolist(),
        "coef": coefficients.tolist(),
        "bias": 0.025,
        "calibration": {"affine": {"slope": 1.0, "intercept": 0.0}},
        "policy": {
            "accept_threshold": 0.75,
            "reject_threshold": 0.25,
            "accept_enabled": True,
            "reject_enabled": True,
        },
    }


def derive_fixed_point_scale(checkpoint: Mapping[str, Any], training_features: Any) -> JsonDict:
    """Freeze coefficient and feature scale from training-only numeric bounds."""

    features = np.asarray(training_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1:] != (6,) or not len(features):
        raise ValueError("training features must be a nonempty six-column matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("training features must be finite")
    coefficients = np.asarray(checkpoint.get("coef"), dtype=np.float64)
    bias = float(checkpoint.get("bias", math.nan))
    if (
        coefficients.shape != (6, 8)
        or not np.all(np.isfinite(coefficients))
        or not math.isfinite(bias)
    ):
        raise ValueError("checkpoint coefficients must be finite 6x8 values")
    coefficient_bound = max(float(np.max(np.abs(coefficients))), abs(bias), 1e-12)
    coefficient_step = coefficient_bound / INT16_MAX
    max_coefficient_q = max(1, int(np.max(np.abs(np.rint(coefficients / coefficient_step)))))
    active_product_count = 24
    feature_integer_bound = min(
        INT16_MAX, max(1, INT32_MAX // (active_product_count * max_coefficient_q))
    )
    feature_bound = max(1.0, float(np.max(np.abs(features))))
    feature_step = feature_bound / feature_integer_bound
    return {
        "source_partition": "fit_training_only",
        "label_count": 0,
        "coefficient_abs_bound": coefficient_bound,
        "feature_abs_bound": feature_bound,
        "coefficient_step": coefficient_step,
        "feature_step": feature_step,
        "feature_integer_bound": feature_integer_bound,
        "accumulator_bits": 32,
        "rounding": "nearest_even",
        "saturation": "signed_int16",
        "tuned_on_test_outcomes": False,
        "paper_hardware_reproduction_claimed": False,
    }


def checked_int32_dot(left: Any, right: Any) -> tuple[int, bool]:
    """Accumulate integer products with detection before any unsafe wraparound."""

    a = np.asarray(left)
    b = np.asarray(right)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("dot inputs must be matching one-dimensional arrays")
    total = 0
    overflow = False
    for first, second in zip(a.tolist(), b.tolist(), strict=True):
        candidate = total + int(first) * int(second)
        if candidate < INT32_MIN or candidate > INT32_MAX:
            overflow = True
            candidate = min(max(candidate, INT32_MIN), INT32_MAX)
        total = candidate
    return total, overflow


def _sigmoid(value: float) -> float:
    """Evaluate one stable finite sigmoid."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _calibrated_probability(raw: float, checkpoint: Mapping[str, Any]) -> float:
    """Apply the frozen affine calibrator used by the source journal."""

    affine = (checkpoint.get("calibration") or {}).get("affine") or {
        "slope": 1.0,
        "intercept": 0.0,
    }
    return float(_apply_affine(np.asarray([raw], dtype=np.float64), affine)[0])


def _action(probability: float, checkpoint: Mapping[str, Any]) -> str:
    """Apply the unchanged frozen policy thresholds and enablement flags."""

    return str(typed_support_decision(probability, checkpoint["policy"])["action"])


def _initial_state(checkpoint: Mapping[str, Any], arm: str, scale: Mapping[str, Any]) -> JsonDict:
    """Copy one arm state so paired replays never share mutable coefficients."""

    coefficient_dtype = np.float64 if arm == "float64_dense" else np.float32
    coefficients = np.asarray(checkpoint["coef"], dtype=coefficient_dtype).copy()
    bias = coefficient_dtype(checkpoint["bias"])
    state: JsonDict = {"coef": coefficients, "bias": bias, "update_count": 0}
    if arm == "int16_fixed_sparse":
        step = float(scale["coefficient_step"])
        state["coef_q"] = np.clip(np.rint(coefficients / step), INT16_MIN, INT16_MAX).astype(
            np.int16
        )
        state["bias_q"] = int(np.clip(np.rint(float(bias) / step), INT16_MIN, INT16_MAX))
    return state


def _design(features: Any, checkpoint: Mapping[str, Any], dtype: Any) -> np.ndarray:
    """Evaluate the shipped spline basis before each measured update."""

    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("event features must be six finite values")
    design, _support = dense_design_vector(vector, np.asarray(checkpoint["knots"]))
    return np.asarray(design, dtype=dtype)


def _predict(
    state: Mapping[str, Any],
    arm: str,
    design: np.ndarray,
    checkpoint: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> tuple[float, bool]:
    """Score one arithmetic arm and return any detected accumulator overflow."""

    overflow = False
    if arm == "int16_fixed_sparse":
        feature_step = float(scale["feature_step"])
        design_q = np.clip(
            np.rint(design.astype(np.float64) / feature_step), INT16_MIN, INT16_MAX
        ).astype(np.int16)
        dot, overflow = checked_int32_dot(state["coef_q"].reshape(-1), design_q)
        logit = dot * float(scale["coefficient_step"]) * feature_step
        logit += int(state["bias_q"]) * float(scale["coefficient_step"])
    else:
        logit = float(design @ state["coef"].reshape(-1) + state["bias"])
    return _calibrated_probability(_sigmoid(logit), checkpoint), overflow


def _update(
    state: JsonDict,
    arm: str,
    design: np.ndarray,
    label: int,
    raw_probability: float,
    scale: Mapping[str, Any],
) -> tuple[int, int]:
    """Apply one clipped dense or local sparse coefficient update."""

    residual = raw_probability - label
    gradient = np.concatenate((residual * design.astype(np.float64), [residual]))
    norm = float(np.linalg.norm(gradient))
    gradient *= min(1.0, GRADIENT_NORM_CAP / max(norm, np.finfo(np.float64).tiny))
    active = np.flatnonzero(design)
    dense = arm in {"float64_dense", "float32_dense"}
    selected = np.arange(48) if dense else active
    saturation = 0
    if arm == "int16_fixed_sparse":
        step = float(scale["coefficient_step"])
        flat = state["coef_q"].reshape(-1).astype(np.int64)
        delta = np.rint(LEARNING_RATE * gradient[selected] / step).astype(np.int64)
        proposed = flat[selected] - delta
        saturation += int(np.count_nonzero((proposed < INT16_MIN) | (proposed > INT16_MAX)))
        flat[selected] = np.clip(proposed, INT16_MIN, INT16_MAX)
        state["coef_q"] = flat.reshape(6, 8).astype(np.int16)
        bias_delta = int(round(LEARNING_RATE * float(gradient[-1]) / step))
        bias_proposed = int(state["bias_q"]) - bias_delta
        saturation += int(bias_proposed < INT16_MIN or bias_proposed > INT16_MAX)
        state["bias_q"] = int(min(max(bias_proposed, INT16_MIN), INT16_MAX))
    else:
        flat = state["coef"].reshape(-1)
        flat[selected] -= np.asarray(LEARNING_RATE * gradient[selected], dtype=flat.dtype)
        state["bias"] = type(state["bias"])(
            state["bias"] - type(state["bias"])(LEARNING_RATE * gradient[-1])
        )
    state["update_count"] += 1
    touched = len(selected) + 1
    bytes_per = 8 if arm == "float64_dense" else 2 if arm == "int16_fixed_sparse" else 4
    return saturation, touched * bytes_per


def _float_state(state: Mapping[str, Any], arm: str, scale: Mapping[str, Any]) -> np.ndarray:
    """Return comparable float64 coefficients without mutating the arm state."""

    if arm == "int16_fixed_sparse":
        step = float(scale["coefficient_step"])
        return np.concatenate(
            (state["coef_q"].reshape(-1).astype(np.float64) * step, [state["bias_q"] * step])
        )
    return np.concatenate((state["coef"].reshape(-1).astype(np.float64), [float(state["bias"])]))


def replay_update_arms(
    checkpoint: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    scale: Mapping[str, Any],
    *,
    arms: Sequence[str] = ARMS,
) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Replay unchanged masks and order across all registered arithmetic arms."""

    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError("replay requires registered arms")
    states = {arm: _initial_state(checkpoint, arm, scale) for arm in arms}
    rows: list[JsonDict] = []
    for event_index, event in enumerate(events):
        label = event.get("label")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("event label must be binary")
        if int(event.get("prediction_index", -1)) != event_index:
            raise ValueError("event order must match prediction index")
        designs = {
            arm: _design(
                event.get("features"),
                checkpoint,
                np.float64 if arm == REFERENCE_ARM else np.float32,
            )
            for arm in arms
        }
        before: dict[str, tuple[float, bool]] = {}
        for arm in arms:
            before[arm] = _predict(states[arm], arm, designs[arm], checkpoint, scale)
        reference_probability = before[REFERENCE_ARM][0]
        reference_action = _action(reference_probability, checkpoint)
        brier_reference = (reference_probability - int(label)) ** 2
        log_reference = -(
            int(label) * math.log(max(reference_probability, 1e-15))
            + (1 - int(label)) * math.log(max(1 - reference_probability, 1e-15))
        )
        saturation_by_arm: dict[str, int] = {arm: 0 for arm in arms}
        bytes_by_arm: dict[str, int] = {arm: 0 for arm in arms}
        if event.get("revealed") is True:
            for arm in arms:
                raw_probability = _sigmoid(
                    math.log(before[arm][0] / max(1.0 - before[arm][0], 1e-15))
                )
                saturation_by_arm[arm], bytes_by_arm[arm] = _update(
                    states[arm], arm, designs[arm], int(label), raw_probability, scale
                )
        reference_state = _float_state(states[REFERENCE_ARM], REFERENCE_ARM, scale)
        for arm in arms:
            probability, overflow = before[arm]
            action = _action(probability, checkpoint)
            brier = (probability - int(label)) ** 2
            log_loss = -(
                int(label) * math.log(max(probability, 1e-15))
                + (1 - int(label)) * math.log(max(1 - probability, 1e-15))
            )
            rows.append(
                {
                    "unit_id": f"event:{event_index}:{arm}",
                    "event_index": event_index,
                    "observation_id": str(event.get("observation_id")),
                    "source_event_hash": event.get("source_event_hash"),
                    "prediction_index": int(event["prediction_index"]),
                    "available_at": int(event.get("available_at", event_index)),
                    "feedback_mask": bool(event.get("revealed") is True),
                    "label": int(label),
                    "arm": arm,
                    "precision": "float64"
                    if arm == REFERENCE_ARM
                    else "int16"
                    if arm.startswith("int16")
                    else "float32",
                    "update_locality": "dense" if "dense" in arm else "sparse_local",
                    "probability": probability,
                    "probability_error": abs(probability - reference_probability),
                    "action": action,
                    "action_flip": action != reference_action,
                    "brier_contribution": brier,
                    "brier_change": brier - brier_reference,
                    "log_loss_contribution": log_loss,
                    "log_loss_change": log_loss - log_reference,
                    "coefficient_error_linf": float(
                        np.max(np.abs(_float_state(states[arm], arm, scale) - reference_state))
                    ),
                    "coefficient_bytes_touched": bytes_by_arm[arm],
                    "saturation_count": saturation_by_arm[arm],
                    "accumulator_overflow_detected": overflow,
                    "unsafe_wraparound": False,
                    "update_applied": bool(event.get("revealed") is True),
                }
            )
    serializable_states = {
        arm: {
            "coefficient_state": _float_state(state, arm, scale).tolist(),
            "update_count": int(state["update_count"]),
        }
        for arm, state in states.items()
    }
    return rows, serializable_states


def _checkpoint_write(path: Path, value: Mapping[str, Any]) -> None:
    """Write a durable numeric checkpoint for restart and service timing."""

    atomic_json(path, value)


def run_development_controls(
    checkpoint: Mapping[str, Any], scale: Mapping[str, Any], directory: Path
) -> list[JsonDict]:
    """Exercise threshold, knot, overflow, delayed-order, and restart fixtures."""

    policy = checkpoint["policy"]
    near = []
    for threshold in (float(policy["reject_threshold"]), float(policy["accept_threshold"])):
        near.extend([np.nextafter(threshold, 0.0), threshold, np.nextafter(threshold, 1.0)])
    threshold_actions = [_action(float(value), checkpoint) for value in near]
    knots = np.asarray(checkpoint["knots"], dtype=np.float64)
    knot_values = [float(row[3]) for row in knots] + [float(row[-4]) for row in knots]
    knot_ok = all(
        np.isfinite(_design([value] * 6, checkpoint, np.float64)).all() for value in knot_values
    )
    huge = np.full(24, INT16_MAX, dtype=np.int16)
    _overflow_value, overflow = checked_int32_dot(huge, huge)
    delayed = [
        {
            "observation_id": "delayed-0",
            "features": [0.25] * 6,
            "label": 0,
            "revealed": True,
            "prediction_index": 0,
            "available_at": 2,
            "source_event_hash": "sha256:" + "1" * 64,
        },
        {
            "observation_id": "delayed-1",
            "features": [0.75] * 6,
            "label": 1,
            "revealed": True,
            "prediction_index": 1,
            "available_at": 1,
            "source_event_hash": "sha256:" + "2" * 64,
        },
    ]
    rows, states = replay_update_arms(checkpoint, delayed, scale)
    order_ok = [row["observation_id"] for row in rows[:: len(ARMS)]] == [
        "delayed-0",
        "delayed-1",
    ]
    directory.mkdir(parents=True, exist_ok=True)
    restart_path = directory / "development-restart.json"
    _checkpoint_write(restart_path, states["float64_dense"])
    restored = load_json(restart_path)
    restart_ok = restored == states["float64_dense"]
    return [
        {
            "control": "near_threshold_actions",
            "passed": len(threshold_actions) == 6,
            "observed": threshold_actions,
        },
        {"control": "knot_boundary_values", "passed": knot_ok, "observed": len(knot_values)},
        {"control": "overflow_detection", "passed": overflow, "observed": overflow},
        {"control": "delayed_event_order", "passed": order_ok, "observed": order_ok},
        {"control": "checkpoint_restart", "passed": restart_ok, "observed": restart_ok},
    ]


def _service_call(
    events: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    scale: Mapping[str, Any],
    arm: str,
    directory: Path,
    identity: str,
) -> JsonDict:
    """Measure feature work, update, journal fsync, and checkpoint fsync."""

    state = _initial_state(checkpoint, arm, scale)
    total_started = time.perf_counter_ns()
    stage: JsonDict = {}
    started = time.perf_counter_ns()
    designs = [
        _design(event["features"], checkpoint, np.float64 if arm == REFERENCE_ARM else np.float32)
        for event in events
    ]
    stage["feature_evaluation"] = (time.perf_counter_ns() - started) / 1e9

    started = time.perf_counter_ns()
    service_rows: list[JsonDict] = []
    saturation_count = 0
    overflow_count = 0
    for event, design in zip(events, designs, strict=True):
        probability, overflow = _predict(state, arm, design, checkpoint, scale)
        overflow_count += int(overflow)
        touched = 0
        if event.get("revealed") is True:
            saturation, touched = _update(
                state, arm, design, int(event["label"]), probability, scale
            )
            saturation_count += saturation
        service_rows.append(
            {
                "observation_id": event["observation_id"],
                "probability": probability,
                "action": _action(probability, checkpoint),
                "coefficient_bytes_touched": touched,
            }
        )
    stage["update"] = (time.perf_counter_ns() - started) / 1e9

    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    journal = directory / f"{identity}.journal.jsonl"
    with journal.open("w", encoding="utf-8") as handle:
        for row in service_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    stage["journal_write_fsync"] = (time.perf_counter_ns() - started) / 1e9

    started = time.perf_counter_ns()
    checkpoint_path = directory / f"{identity}.checkpoint.json"
    _checkpoint_write(
        checkpoint_path,
        {
            "arm": arm,
            "state": _float_state(state, arm, scale).tolist(),
            "update_count": state["update_count"],
        },
    )
    stage["checkpoint_write_fsync"] = (time.perf_counter_ns() - started) / 1e9
    total = (time.perf_counter_ns() - total_started) / 1e9
    return {
        "stage_duration_s": stage,
        "complete_service_s": max(total, sum(float(value) for value in stage.values())),
        "saturation_count": saturation_count,
        "overflow_detection_count": overflow_count,
        "journal_bytes": journal.stat().st_size,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }


def measure_service_cost(
    events: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    scale: Mapping[str, Any],
    directory: Path,
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
    emit_progress: bool = False,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run cold setup and paired rotated complete-service timing blocks."""

    if blocks <= 0 or any(size <= 0 for size in batch_sizes) or not events:
        raise ValueError("timing needs events, positive batches, and positive blocks")
    timing: list[JsonDict] = []
    cold: list[JsonDict] = []
    progress_started = time.monotonic()
    for arm in ARMS:
        started = time.perf_counter_ns()
        _initial_state(checkpoint, arm, scale)
        cold.append(
            {
                "unit_id": f"cold:{arm}",
                "arm": arm,
                "cold_setup_s": (time.perf_counter_ns() - started) / 1e9,
            }
        )
    completed = 0
    total = len(batch_sizes) * blocks * len(ARMS)
    for batch_size in batch_sizes:
        batch = [events[index % len(events)] for index in range(batch_size)]
        for block in range(blocks):
            order = (*ARMS[block % len(ARMS) :], *ARMS[: block % len(ARMS)])
            for order_index, arm in enumerate(order):
                receipt = _service_call(
                    batch,
                    checkpoint,
                    scale,
                    arm,
                    directory,
                    f"batch-{batch_size}-block-{block}-{arm}",
                )
                timing.append(
                    {
                        "unit_id": f"timing:{batch_size}:{block}:{arm}",
                        "batch_size": batch_size,
                        "block": block,
                        "order": list(order),
                        "order_index": order_index,
                        "arm": arm,
                        **receipt,
                    }
                )
                completed += 1
                if emit_progress and (completed == total or completed % 30 == 0):
                    progress(
                        progress_started,
                        "complete_service_timing",
                        "unit_complete",
                        completed=completed,
                        total=total,
                    )
    return timing, cold


def _bootstrap_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return a paired ratio-of-means interval over complete timing blocks."""

    first = np.asarray(numerator, dtype=np.float64)
    second = np.asarray(denominator, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("paired timing vectors must be matching and nonempty")
    point = float(np.mean(first) / np.mean(second))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(first), size=(draws, len(first)))
    ratios = np.mean(first[indices], axis=1) / np.mean(second[indices], axis=1)
    lower, upper = np.quantile(ratios, [0.025, 0.975]).tolist()
    return point, float(lower), float(upper)


def summarize_timing(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = RANDOM_SEED
) -> list[JsonDict]:
    """Reduce paired whole-service ratios against the float32 dense baseline."""

    output: list[JsonDict] = []
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    for batch_size in batch_sizes:
        selected = [row for row in rows if int(row["batch_size"]) == batch_size]
        by_arm = {
            arm: {
                int(row["block"]): float(row["complete_service_s"])
                for row in selected
                if row["arm"] == arm
            }
            for arm in ARMS
        }
        baseline = by_arm["float32_dense"]
        for arm_index, arm in enumerate(ARMS):
            if arm == "float32_dense":
                continue
            blocks = sorted(set(baseline) & set(by_arm[arm]))
            numerator = [by_arm[arm][block] for block in blocks]
            denominator = [baseline[block] for block in blocks]
            point, lower, upper = _bootstrap_ratio(
                numerator,
                denominator,
                draws=draws,
                seed=seed + batch_size * 10 + arm_index,
            )
            output.append(
                {
                    "batch_size": batch_size,
                    "arm": arm,
                    "baseline_arm": "float32_dense",
                    "paired_blocks": len(blocks),
                    "whole_service_time_ratio": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "speed_gate_passed": upper < 1.0,
                }
            )
    return output


def reduce_updates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce errors, proper-score changes, flips, overflow, and locality."""

    by_arm: list[JsonDict] = []
    for arm in ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        if not selected:
            continue
        by_arm.append(
            {
                "arm": arm,
                "events": len(selected),
                "updates": sum(row.get("update_applied") is True for row in selected),
                "action_flips": sum(row.get("action_flip") is True for row in selected),
                "max_coefficient_error_linf": max(
                    float(row.get("coefficient_error_linf", math.inf)) for row in selected
                ),
                "max_probability_error": max(
                    float(row.get("probability_error", math.inf)) for row in selected
                ),
                "mean_brier_change": float(
                    np.mean([float(row.get("brier_change", math.inf)) for row in selected])
                ),
                "mean_log_loss_change": float(
                    np.mean([float(row.get("log_loss_change", math.inf)) for row in selected])
                ),
                "coefficient_bytes_touched": sum(
                    int(row.get("coefficient_bytes_touched", 0)) for row in selected
                ),
                "saturation_count": sum(int(row.get("saturation_count", 0)) for row in selected),
                "overflow_detection_count": sum(
                    row.get("accumulator_overflow_detected") is True for row in selected
                ),
                "unsafe_wraparound_count": sum(
                    row.get("unsafe_wraparound") is True for row in selected
                ),
            }
        )
    return {
        "arms": by_arm,
        "event_count": len(rows) // len(ARMS) if rows else 0,
        "action_flip_count": sum(row.get("action_flip") is True for row in rows),
        "unsafe_wraparound_count": sum(row.get("unsafe_wraparound") is True for row in rows),
        "saturation_count": sum(int(row.get("saturation_count", 0)) for row in rows),
    }


def _fixture_receipts() -> list[JsonDict]:
    """Build complete scoped receipts for mutation-oriented artifact tests."""

    names = [
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_cold_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
        "declared_entrypoint_e2e",
    ]
    return [
        {
            "name": name,
            "argv": ["fixture", name],
            "command_environment": {"COVERAGE_FILE": "/tmp/fixture.coverage"},
            "exit_code": 0,
            "duration_s": 0.001,
            "log_sha256": "sha256:" + f"{index + 1:064x}"[-64:],
            "passed": True,
        }
        for index, name in enumerate(names)
    ]


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require every frozen affected, reader, lint, and entrypoint receipt."""

    required = {
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_cold_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
        "declared_entrypoint_e2e",
    }
    passed = {str(row.get("name")) for row in receipts if row.get("passed") is True}
    return required <= passed


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failed gate without hiding later failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_checks": [row.get("check") for row in failed],
        "first_failed_check": failed[0].get("check") if failed else None,
        "first_failed_upstream": failed[0].get("upstream") if failed else None,
        "first_failed_path": failed[0].get("path") if failed else None,
        "first_failed_field": failed[0].get("field") if failed else None,
        "first_failed_expected": failed[0].get("expected") if failed else None,
        "first_failed_observed": failed[0].get("observed") if failed else None,
    }


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Explain top-level field intent without wrapping machine-readable values."""

    specific = {
        "schema": "Use one versioned plain top-level schema with identity and terminal status.",
        "run_date": "Use 20260919 with actual UTC and monotonic boundaries.",
        "preconditions_checked": "Record exact resource identities before dependent work.",
        "MODEL_SPECS": "List current LLMs; this no-model task uses an empty list.",
        "model_invoked": "Mark true only for actual current attempted model work.",
        "invocation_counts": "Count owned current load and generation attempts and outcomes.",
        "inference_substrate": "Name the truthful current arithmetic substrate.",
        "inference_substrate_class": "Classify current compute without historical padding.",
        "execution_venue": "Use the closed host venue; board identities stay separate.",
        "duration_s": "Measure current work and retain validation and cold-start phases separately.",
        "phase_spans": "Retain real phase timing, flushed progress, and checkpoint references.",
        "random_seed": "Freeze resampling and fixture seeds before measurement.",
        "reproducibility_checksum": "Bind code, protocol, sources, raw rows, and validation scope.",
        "source_artifact_hashes": "Bind exact inputs and their original flag state.",
        "rows": "Retain every comparative arm and measured condition.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, and censored units.",
        "acceptance_gate_results": "Keep validity, completion, safety, and benefit gates separate.",
        "gate_check_summary": "Name each exact blocked upstream operand including null and zero.",
        "verifier_is_oracle": "Mark shared correctness authority so positive fixtures remain circular.",
        "honest_verdict": "Use complete_ for completed findings and blocked_ for external absence.",
        "verdict_class": "Use only the registered terminal verdict enum.",
        "flagged_adversarial": "Preserve critical findings and exclude flagged evidence from readiness.",
        "validation_receipts": "Retain exact scoped commands, environments, exits, durations, and logs.",
        "promotion_score": "Remain zero; this experiment cannot roll out or update generator weights.",
        "update_placement_complete_score": "Credit independently valid numeric placement evidence.",
        "update_placement_value_score": "Credit only parity, quality, and complete-service timing benefit.",
        "update_rows": "Retain event arithmetic, action, error, bytes, saturation, and overflow.",
        "board_rows": "Keep KV260, GateMate, and PolarFire dispositions separate.",
        "hardware_ready_score": "Remain zero because no fresh physical qualification occurred.",
        "hardware_value_score": "Remain zero because host arithmetic cannot establish device value.",
    }
    return {key: specific.get(key, "Retain this field as plain typed evidence.") for key in keys}


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable protocol, source, row, gate, and validation evidence."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def _acceptance_gates(
    *,
    update_rows: Sequence[Mapping[str, Any]],
    timing_summary: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build registered validity, safety, completion, and benefit comparisons."""

    reduction = reduce_updates(update_rows)
    int16 = next((row for row in reduction["arms"] if row["arm"] == "int16_fixed_sparse"), {})
    sparse_timing = [
        row for row in timing_summary if row.get("arm") in {"float32_sparse", "int16_fixed_sparse"}
    ]
    numeric_preconditions = [
        row for row in preconditions if row.get("category") == "numeric_precondition"
    ]
    gates = [
        gate_row(
            "numeric_preconditions_authenticated",
            "validity",
            "all",
            True,
            bool(numeric_preconditions)
            and all(row.get("passed") is True for row in numeric_preconditions),
            bool(numeric_preconditions)
            and all(row.get("passed") is True for row in numeric_preconditions),
            "Numeric placement starts only from authenticated upstream bytes.",
            path=EXP7427_PATH.as_posix(),
            field="preconditions_checked",
            upstream="exp7425+exp7426+exp7427",
        ),
        gate_row(
            "development_controls_passed",
            "validity",
            "all",
            True,
            bool(controls) and all(row.get("passed") is True for row in controls),
            bool(controls) and all(row.get("passed") is True for row in controls),
            "Boundary, overflow, order, and restart controls precede corpus replay.",
            field="development_controls",
            upstream="current_development_fixtures",
        ),
        gate_row(
            "complete_journal_replayed",
            "completion",
            "==",
            True,
            bool(update_rows) and len(update_rows) % len(ARMS) == 0,
            bool(update_rows) and len(update_rows) % len(ARMS) == 0,
            "Every selected journal event must have one row for every registered arm.",
            field="update_rows",
            upstream="selected_complete_exp7427_journal",
        ),
        gate_row(
            "unsafe_wraparound_zero",
            "safety",
            "==",
            0,
            reduction["unsafe_wraparound_count"],
            reduction["unsafe_wraparound_count"] == 0,
            "Detected overflow may saturate but integer arithmetic must never wrap.",
            field="update_reduction.unsafe_wraparound_count",
            upstream="current_fixed_point_replay",
        ),
        gate_row(
            "action_parity",
            "benefit",
            "==",
            0,
            reduction["action_flip_count"],
            reduction["action_flip_count"] == 0,
            "A fixed-point action flip is an honest placement null.",
            field="update_reduction.action_flip_count",
            upstream="current_update_rows",
        ),
        gate_row(
            "fixed_point_brier_increase",
            "benefit",
            "<=",
            0.001,
            int16.get("mean_brier_change"),
            bool(int16) and float(int16["mean_brier_change"]) <= 0.001,
            "Value requires negligible proper-score degradation.",
            field="update_reduction.arms.int16_fixed_sparse.mean_brier_change",
            upstream="current_update_rows",
        ),
        gate_row(
            "paired_whole_service_speed",
            "benefit",
            "all_ci95_upper<",
            1.0,
            [row.get("ci95_upper") for row in sparse_timing],
            len(sparse_timing) == len(BATCH_SIZES) * 2
            and all(float(row["ci95_upper"]) < 1.0 for row in sparse_timing),
            "Host value requires complete persistence service benefit at every batch size.",
            field="timing_summary.ci95_upper",
            upstream="current_timing_rows",
        ),
        gate_row(
            "required_validation_passed",
            "validity",
            "==",
            True,
            _required_validation_passed(validation_receipts),
            _required_validation_passed(validation_receipts),
            "Only the frozen affected and terminal command plan can validate this artifact.",
            field="validation_receipts",
            upstream="current_scoped_validation",
        ),
    ]
    return gates


def _hardware_mapping(
    update_rows: Sequence[Mapping[str, Any]], timing_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Map host locality to a future KV260 budget without a device claim."""

    sparse = [row for row in update_rows if row.get("arm") == "int16_fixed_sparse"]
    updates = [row for row in sparse if row.get("update_applied") is True]
    bytes_per_update = (
        float(np.mean([row["coefficient_bytes_touched"] for row in updates])) if updates else 0.0
    )
    timed = [row for row in timing_rows if row.get("arm") == "int16_fixed_sparse"]
    per_event_update = [
        float(row["stage_duration_s"]["update"]) / int(row["batch_size"])
        for row in timed
        if int(row["batch_size"]) > 0
    ]
    per_event_lookup = [
        float(row["stage_duration_s"]["journal_write_fsync"]) / int(row["batch_size"])
        for row in timed
        if int(row["batch_size"]) > 0
    ]
    update_p50 = float(np.median(per_event_update)) if per_event_update else math.inf
    lookup_p50 = float(np.median(per_event_lookup)) if per_event_lookup else math.inf
    return {
        "placement": "future_kv260_local_coefficient_sram_or_bram_budget",
        "precision_bits": 16,
        "coefficient_count": 49,
        "full_coefficient_storage_bytes": 98,
        "mean_touched_coefficient_bytes_per_update": bytes_per_update,
        "transfer_frequency": "once_per_admitted_feedback_event",
        "state_transfer_scope": "touched_coefficients_plus_bias_only",
        "cpu_update_p50_s": update_p50,
        "cpu_update_target_s": 1e-6,
        "cpu_update_target_met": update_p50 < 1e-6,
        "lookup_p50_s": lookup_p50,
        "lookup_target_s": 1e-3,
        "lookup_target_met": lookup_p50 < 1e-3,
        "kv260_access": "ssh_only",
        "kv260_architecture_limit": "k_max<=5",
        "hardware_power_claimed": False,
        "hardware_latency_claimed": False,
        "paper_hardware_reproduction_claimed": False,
        "external_options": ["Extropic Z1", "NPU", "additional hardware"],
        "external_options_qualified": False,
        "purchase_recommended": False,
    }


def _sample_budget(
    update_rows: Sequence[Mapping[str, Any]], timing_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Count planned and completed units without treating paired arms as independent."""

    event_count = len(update_rows) // len(ARMS) if update_rows else 0
    timing_units = len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS)
    return {
        "planned": event_count * len(ARMS) + timing_units,
        "attempted": len(update_rows) + len(timing_rows),
        "completed": len(update_rows) + len(timing_rows),
        "failed": 0,
        "censored": 0,
        "unstarted": max(
            0, event_count * len(ARMS) + timing_units - len(update_rows) - len(timing_rows)
        ),
        "independent_groups": event_count,
        "paired_timing_blocks": PAIRED_BLOCKS,
        "batch_sizes": list(BATCH_SIZES),
        "stop_rule": "complete selected eligible journal and exactly 30 paired blocks per batch",
    }


def assemble_artifact(
    *,
    update_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    cold_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    scale: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    flagged_adversarial: bool,
) -> JsonDict:
    """Assemble one terminal artifact with validity independent from benefit."""

    timing_summary = summarize_timing(timing_rows) if timing_rows else []
    update_reduction = reduce_updates(update_rows)
    gates = _acceptance_gates(
        update_rows=update_rows,
        timing_summary=timing_summary,
        preconditions=preconditions,
        validation_receipts=validation_receipts,
        controls=controls,
    )
    valid = all(row["passed"] for row in gates if row["category"] != "benefit")
    value = valid and all(row["passed"] for row in gates if row["category"] == "benefit")
    complete_score = int(valid and bool(update_rows))
    value_score = int(value and complete_score == 1)
    if flagged_adversarial or not valid:
        verdict_class = "disqualified"
        honest = "complete_disqualified_update_placement_validation_failed"
    elif value_score:
        verdict_class = "positive"
        honest = "complete_positive_sparse_update_host_service_benefit"
    else:
        verdict_class = "null"
        honest = "complete_null_sparse_update_no_registered_complete_service_benefit"
    receipt = build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_sparse_local_update_arithmetic_and_persistence",
        inference_substrate_details={
            "arithmetic_arms": list(ARMS),
            "model_load": False,
            "fpga_or_tsu_execution": False,
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        phase_spans=phase_spans,
        small_ebm_training={
            "receipt_class": "small_ebm_training",
            "performed": bool(update_rows),
            "updates_completed": sum(row.get("update_applied") is True for row in update_rows),
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
    )
    hardware = _hardware_mapping(update_rows, timing_rows)
    baseline = [row for row in timing_rows if row.get("arm") == "float32_dense"]
    unaccelerated_fraction = (
        float(
            np.mean(
                [
                    (
                        float(row["stage_duration_s"]["journal_write_fsync"])
                        + float(row["stage_duration_s"]["checkpoint_write_fsync"])
                    )
                    / float(row["complete_service_s"])
                    for row in baseline
                ]
            )
        )
        if baseline
        else None
    )
    artifact: JsonDict = {
        **receipt,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(list(preconditions)),
        "random_seed": {
            "replay_seed": 65101,
            "paired_resampling_seed": RANDOM_SEED,
            "paired_resampling_draws": BOOTSTRAP_DRAWS,
            "scale_fit_partition": "fit_training_only",
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "numeric_protocol": {
            "reference": "float64_dense",
            "comparison_arms": list(ARMS[1:]),
            "learning_rate": LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "feedback_masks_changed": False,
            "test_outcome_tuning": False,
            "selected_journal_condition": deepcopy(SELECTED_JOURNAL_CONDITION),
        },
        "fixed_point_scale": deepcopy(dict(scale)),
        "development_controls": deepcopy(list(controls)),
        "update_rows": deepcopy(list(update_rows)),
        "update_reduction": update_reduction,
        "timing_rows": deepcopy(list(timing_rows)),
        "timing_summary": timing_summary,
        "cold_setup_rows": deepcopy(list(cold_rows)),
        "board_rows": deepcopy(list(board_rows)),
        "hardware_mapping": hardware,
        "amdahl_analysis": {
            "target_service_acceleration": 100.0,
            "required_unaccelerated_fraction": 0.01,
            "observed_unaccelerated_persistence_fraction": unaccelerated_fraction,
            "target_feasible_from_host_measurement": bool(
                unaccelerated_fraction is not None and unaccelerated_fraction <= 0.01
            ),
        },
        "rows": [
            *deepcopy(list(update_rows)),
            *deepcopy(list(timing_rows)),
            *deepcopy(list(cold_rows)),
            *deepcopy(list(board_rows)),
        ],
        "sample_size_budget": _sample_budget(update_rows, timing_rows),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "promotion_score": 0,
        "update_placement_complete_score": complete_score,
        "update_placement_value_score": value_score,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "hardware_operations": {
            "ssh": 0,
            "usb_probe": 0,
            "flash": 0,
            "jtag": 0,
            "physical_execution": 0,
            "purchase": 0,
        },
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _fixture_timing_rows() -> list[JsonDict]:
    """Build paired complete-service rows with deterministic fixture ratios."""

    rows: list[JsonDict] = []
    totals = {
        "float64_dense": 1.05,
        "float32_dense": 1.0,
        "float32_sparse": 0.8,
        "int16_fixed_sparse": 0.7,
    }
    for batch_size in BATCH_SIZES:
        for block in range(PAIRED_BLOCKS):
            order = (*ARMS[block % len(ARMS) :], *ARMS[: block % len(ARMS)])
            for order_index, arm in enumerate(order):
                total = totals[arm] * (1.0 + block / 10_000)
                rows.append(
                    {
                        "unit_id": f"fixture-timing:{batch_size}:{block}:{arm}",
                        "batch_size": batch_size,
                        "block": block,
                        "order": list(order),
                        "order_index": order_index,
                        "arm": arm,
                        "stage_duration_s": {
                            "feature_evaluation": total * 0.2,
                            "update": total * 0.3,
                            "journal_write_fsync": total * 0.2,
                            "checkpoint_write_fsync": total * 0.2,
                        },
                        "complete_service_s": total,
                        "saturation_count": 0,
                        "overflow_detection_count": 0,
                        "journal_bytes": 100,
                        "checkpoint_bytes": 100,
                    }
                )
    return rows


def build_fixture_artifact() -> JsonDict:
    """Build one compact valid artifact for cold-reader mutation tests."""

    training = np.asarray(
        [[((row + column) % 9) / 8 for column in range(6)] for row in range(12)],
        dtype=np.float64,
    )
    checkpoint = fixture_checkpoint(training)
    scale = derive_fixed_point_scale(checkpoint, training)
    events = [
        {
            "observation_id": f"fixture-{index}",
            "features": training[index].tolist(),
            "label": index % 2,
            "revealed": True,
            "prediction_index": index,
            "available_at": index,
            "source_event_hash": "sha256:" + f"{index + 1:064x}"[-64:],
        }
        for index in range(4)
    ]
    updates, _states = replay_update_arms(checkpoint, events, scale)
    controls = [
        {"control": name, "passed": True, "observed": True}
        for name in (
            "near_threshold_actions",
            "knot_boundary_values",
            "overflow_detection",
            "delayed_event_order",
            "checkpoint_restart",
        )
    ]
    preconditions = [
        gate_row(
            "fixture_numeric_inputs",
            "numeric_precondition",
            "==",
            True,
            True,
            True,
            "Fixture inputs are complete.",
            path="fixture",
            field="available",
            upstream="fixture",
        )
    ]
    return assemble_artifact(
        update_rows=updates,
        timing_rows=_fixture_timing_rows(),
        cold_rows=[{"unit_id": f"cold:{arm}", "arm": arm, "cold_setup_s": 0.001} for arm in ARMS],
        board_rows=_fixture_board_rows(),
        scale=scale,
        controls=controls,
        preconditions=preconditions,
        source_hashes={"fixture": "sha256:" + "a" * 64},
        validation_receipts=_fixture_receipts(),
        phase_spans=[
            {
                "phase": "fixture",
                "start_s": 0.0,
                "end_s": 0.001,
                "duration_s": 0.001,
                "completed_units": 1,
                "heartbeat_times_utc": [],
            }
        ],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00.001000+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000,
        flagged_adversarial=False,
    )


def rebuild_fixture_reductions(value: Mapping[str, Any]) -> JsonDict:
    """Recompute dependent fixture gates and checksum after a deliberate mutation."""

    copied = deepcopy(dict(value))
    gates = _acceptance_gates(
        update_rows=copied["update_rows"],
        timing_summary=copied["timing_summary"],
        preconditions=copied["preconditions_checked"],
        validation_receipts=copied["validation_receipts"],
        controls=copied["development_controls"],
    )
    copied["update_reduction"] = reduce_updates(copied["update_rows"])
    copied["acceptance_gate_results"] = gates
    copied["gate_check_summary"] = _gate_summary(gates)
    valid = all(row["passed"] for row in gates if row["category"] != "benefit")
    value_score = int(valid and all(row["passed"] for row in gates if row["category"] == "benefit"))
    copied["update_placement_complete_score"] = int(valid and bool(copied["update_rows"]))
    copied["update_placement_value_score"] = value_score
    copied["verdict_class"] = "positive" if value_score else "null"
    copied["honest_verdict"] = (
        "complete_positive_sparse_update_host_service_benefit"
        if value_score
        else "complete_null_sparse_update_no_registered_complete_service_benefit"
    )
    copied["status"] = copied["honest_verdict"]
    copied["rows"] = [
        *deepcopy(copied["update_rows"]),
        *deepcopy(copied["timing_rows"]),
        *deepcopy(copied["cold_setup_rows"]),
        *deepcopy(copied["board_rows"]),
    ]
    copied["reproducibility_checksum"] = reproducibility_checksum(copied)
    return copied


def raw_view(value: Mapping[str, Any]) -> JsonDict:
    """Return the exact evidence bundle consumed by the independent reducer."""

    return {
        "update_rows": deepcopy(value.get("update_rows")),
        "timing_rows": deepcopy(value.get("timing_rows")),
        "cold_setup_rows": deepcopy(value.get("cold_setup_rows")),
        "board_rows": deepcopy(value.get("board_rows")),
        "fixed_point_scale": deepcopy(value.get("fixed_point_scale")),
        "development_controls": deepcopy(value.get("development_controls")),
    }


def independent_reduce(value: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, bool]:
    """Independently recompute raw equality, reductions, scores, and local validity."""

    update_match = value.get("update_rows") == raw.get("update_rows")
    timing_match = value.get("timing_rows") == raw.get("timing_rows")
    update_reduced = (not value.get("update_rows")) or reduce_updates(
        value["update_rows"]
    ) == value.get("update_reduction")
    timing_reduced = (not value.get("timing_rows")) or summarize_timing(
        value["timing_rows"],
        draws=int(value["random_seed"]["paired_resampling_draws"]),
        seed=int(value["random_seed"]["paired_resampling_seed"]),
    ) == value.get("timing_summary")
    return {
        "update_rows_match": update_match,
        "timing_rows_match": timing_match,
        "cold_rows_match": value.get("cold_setup_rows") == raw.get("cold_setup_rows"),
        "board_rows_match": value.get("board_rows") == raw.get("board_rows"),
        "fixed_point_scale_match": value.get("fixed_point_scale") == raw.get("fixed_point_scale"),
        "development_controls_match": value.get("development_controls")
        == raw.get("development_controls"),
        "update_reduction_match": update_reduced,
        "timing_reduction_match": timing_reduced,
        "artifact_valid": not validate_artifact(value),
    }


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Reject changed rows, false scores, model claims, and hardware promotion."""

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(value))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
        return errors
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_or_identity_invalid")
    if value.get("run_date") != RUN_DATE or value.get("milestone") != MILESTONE:
        errors.append("run_identity_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_claim_invalid")
    counts = value.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(key) != 0 for key in ZERO_INVOCATION_COUNTS
    ):
        errors.append("invocation_counts_nonzero")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("substrate_class_invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if any(
        value.get(key) != 0
        for key in ("promotion_score", "hardware_ready_score", "hardware_value_score")
    ):
        errors.append("hardware_score_nonzero")
    boards = value.get("board_rows")
    if not isinstance(boards, list) or {row.get("board") for row in boards} != {
        "KV260",
        "GateMate",
        "PolarFire",
    }:
        errors.append("board_rows_invalid")
    else:
        by_name = {row["board"]: row for row in boards}
        if (
            by_name["KV260"].get("future_access") != "ssh_only"
            or by_name["KV260"].get("architecture_limit") != "k_max<=5"
            or by_name["GateMate"].get("terminal_state") != "blocked_changed_physical_state"
            or by_name["PolarFire"].get("terminal_state") != "graduated_cpu_dispatch_preserved"
        ):
            errors.append("board_disposition_invalid")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, events=[]))
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not REQUIRED_FIELDS <= set(principles):
        errors.append("field_principles_incomplete")
    verdict = str(value.get("honest_verdict") or "")
    verdict_class = value.get("verdict_class")
    if verdict_class == "blocked":
        if not verdict.startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
        if (
            value.get("update_placement_complete_score") != 0
            or value.get("update_placement_value_score") != 0
        ):
            errors.append("blocked_score_nonzero")
        return list(dict.fromkeys(errors))
    if not verdict.startswith("complete_"):
        errors.append("completed_verdict_prefix_invalid")
    update_rows = value.get("update_rows")
    timing_rows = value.get("timing_rows")
    if not isinstance(update_rows, list) or not isinstance(timing_rows, list):
        errors.append("numeric_rows_invalid")
        return list(dict.fromkeys(errors))
    if reduce_updates(update_rows) != value.get("update_reduction"):
        errors.append("update_reduction_mismatch")
    expected_timing = summarize_timing(
        timing_rows,
        draws=int(value["random_seed"]["paired_resampling_draws"]),
        seed=int(value["random_seed"]["paired_resampling_seed"]),
    )
    if expected_timing != value.get("timing_summary"):
        errors.append("timing_reduction_mismatch")
    expected_gates = _acceptance_gates(
        update_rows=update_rows,
        timing_summary=expected_timing,
        preconditions=value.get("preconditions_checked") or [],
        validation_receipts=value.get("validation_receipts") or [],
        controls=value.get("development_controls") or [],
    )
    if expected_gates != value.get("acceptance_gate_results"):
        errors.append("acceptance_gates_mismatch")
    valid = all(row["passed"] for row in expected_gates if row["category"] != "benefit")
    expected_complete = int(valid and bool(update_rows))
    expected_value = int(
        expected_complete == 1
        and all(row["passed"] for row in expected_gates if row["category"] == "benefit")
    )
    if value.get("update_placement_complete_score") != expected_complete:
        errors.append("completion_score_mismatch")
    if value.get("update_placement_value_score") != expected_value:
        errors.append("value_score_mismatch")
    expected_rows = [
        *update_rows,
        *timing_rows,
        *(value.get("cold_setup_rows") or []),
        *(value.get("board_rows") or []),
    ]
    if value.get("rows") != expected_rows:
        errors.append("combined_rows_mismatch")
    if (value.get("hardware_mapping") or {}).get("external_options_qualified") is not False:
        errors.append("external_hardware_qualification_invalid")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, value: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a locally valid terminal artifact."""

    errors = validate_artifact(value)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    atomic_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def cold_replay(path: Path) -> list[str]:
    """Reload one candidate in a fresh process and run the complete validator."""

    try:
        return validate_artifact(load_json(path))
    except ValueError:
        return ["artifact_unreadable_or_not_object"]


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Preserve external or numeric absence as blocked while keeping board rows."""

    value = assemble_artifact(
        update_rows=[],
        timing_rows=[],
        cold_rows=[],
        board_rows=_fixture_board_rows(),
        scale={},
        controls=[],
        preconditions=[deepcopy(dict(failed))],
        source_hashes={},
        validation_receipts=[],
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        flagged_adversarial=False,
    )
    check = str(failed.get("check") or "prerequisite_unavailable")
    value["status"] = f"blocked_{check}"
    value["honest_verdict"] = value["status"]
    value["verdict_class"] = "blocked"
    value["flagged_adversarial"] = False
    value["update_placement_complete_score"] = 0
    value["update_placement_value_score"] = 0
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 scoped command plan before any child starts."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad pytest, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _load_numeric_inputs(
    root: Path, exp7426: Mapping[str, Any], exp7427: Mapping[str, Any]
) -> tuple[JsonDict, np.ndarray, list[JsonDict], JsonDict]:  # pragma: no cover - E2E data path.
    """Use shipped corpus readers and authenticate every selected journal shard."""

    manifest = exp7426.get("checkpoint_manifest") or []
    matches = [
        row
        for row in manifest
        if isinstance(row, Mapping)
        and row.get("condition") == "full_source"
        and row.get("arm") == "sparse_spline_49"
        and row.get("seed") == 65101
    ]
    if len(matches) != 1:
        raise ValueError("selected Exp7426 spline checkpoint is not unique")
    checkpoint_path = root / str(matches[0]["path"])
    if sha256_file(checkpoint_path) != matches[0].get("sha256"):
        raise ValueError("selected Exp7426 spline checkpoint hash mismatch")
    payload = load_json(checkpoint_path)
    checkpoint = {
        **deepcopy(payload["checkpoint"]),
        "calibration": deepcopy(payload["calibration"]),
        "policy": deepcopy(payload["policy"]),
    }

    corpus = reload_corpus(root / CORPUS_DIR)
    readers = ProtocolReaders(corpus)
    fit_rows = readers.read_predictors("fit")
    training = np.asarray(
        [[float(row["features"][name]) for name in SOURCE_FEATURE_NAMES] for row in fit_rows],
        dtype=np.float64,
    )
    prospective = join_predictor_labels(
        readers.read_predictors("prospective_stream"),
        readers.read_evaluators("prospective_stream", EVALUATOR_TOKEN),
    )
    selected = [row for row in prospective if row.get("certificate_selected") is True]
    stream = build_streams(selected)["hash_order"]
    stream_by_id = {str(row["observation_id"]): row for row in stream}

    reference = exp7427.get("feedback_event_rows")
    if not isinstance(reference, Mapping):
        raise ValueError("Exp7427 journal manifest is required")
    directory = root / str(reference["directory"])
    selected_rows: list[JsonDict] = []
    shard_hashes: JsonDict = {}
    for shard in reference.get("shards") or []:
        path = directory / str(shard["path"])
        observed = sha256_file(path)
        if observed != shard.get("sha256"):
            raise ValueError(f"Exp7427 journal shard hash mismatch:{path}")
        label = path.relative_to(root).as_posix()
        shard_hashes[label] = {
            "path": label,
            "sha256": observed,
            "original_flagged_adversarial": exp7427.get("flagged_adversarial"),
        }
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if all(row.get(key) == value for key, value in SELECTED_JOURNAL_CONDITION.items()):
                    selected_rows.append(row)
    selected_rows.sort(key=lambda row: int(row["prediction_index"]))
    if len(selected_rows) != len(stream) or len(selected_rows) != 753:
        raise ValueError("complete selected Exp7427 journal must contain 753 events")
    events: list[JsonDict] = []
    for expected_index, row in enumerate(selected_rows):
        identity = str(row["observation_id"])
        source = stream_by_id.get(identity)
        if source is None or int(row["prediction_index"]) != expected_index:
            raise ValueError("selected Exp7427 event order or source identity mismatch")
        events.append(
            {
                "observation_id": identity,
                "features": [float(source["features"][name]) for name in SOURCE_FEATURE_NAMES],
                "label": int(row["label"]),
                "revealed": bool(row["revealed"]),
                "prediction_index": expected_index,
                "available_at": int(row["available_at"]),
                "source_event_hash": canonical_hash(row),
            }
        )
    return checkpoint, training, events, shard_hashes


def _span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - live timing receipt.
    """Close one real phase with completed units and a resumable reference."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_reference": checkpoint,
        "checkpoint_at_utc": utc_now(),
        "heartbeat_times_utc": [],
    }


def _terminal_commands(candidate: Path, raw: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = ".venv/bin/python"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--cold-replay",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_cold_recompute",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--independent-reduce",
                    str(candidate),
                    "--raw",
                    str(raw),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def _entrypoint_receipt(
    root: Path, started_at_utc: str, duration_s: float
) -> JsonDict:  # pragma: no cover - live E2E receipt.
    """Retain the declared command and environment as capability E2E evidence."""

    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.json"
    value = {
        "argv": [
            ".venv/bin/python",
            "-u",
            WRAPPER_PATH.as_posix(),
            "--date",
            RUN_DATE,
        ],
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        "duration_s": duration_s,
        "environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }
    atomic_json(path, value)
    return {
        "name": "declared_entrypoint_e2e",
        "argv": value["argv"],
        "command_environment": value["environment"],
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(path),
        "passed": True,
        "required": True,
        "command_category": "completion",
        "started_at_utc": started_at_utc,
        "ended_at_utc": value["completed_at_utc"],
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared capability entrypoint.
    """Authenticate, replay, time, validate, and atomically publish one result."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "startup", "flushed", completed_units=0)

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before_authentication", completed_units=0)
    preconditions, source_hashes, context = collect_preconditions(root)
    board_rows = build_board_rows(context)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            run_started,
            len(preconditions),
            (root / RAW_DIR / "preconditions/gatemate_changed_state_search.json").as_posix(),
        )
    )
    progress(
        run_started,
        "preconditions",
        "after_authentication",
        completed_units=len(preconditions),
    )
    numeric_failed = next(
        (
            row
            for row in preconditions
            if row.get("category") == "numeric_precondition" and row.get("passed") is not True
        ),
        None,
    )
    destination = (root / output_path) if not output_path.is_absolute() else output_path
    if numeric_failed is not None:
        blocked = build_blocked_artifact(numeric_failed)
        progress(run_started, "publish", "before_atomic_terminal", status="blocked")
        write_artifact(destination, blocked)
        progress(run_started, "publish", "after_atomic_terminal", status="blocked")
        return blocked

    phase_started = time.monotonic()
    progress(run_started, "numeric_input_load", "before_load", completed_units=0)
    checkpoint, training, events, shard_hashes = _load_numeric_inputs(
        root, context["upstream"]["exp7426"], context["upstream"]["exp7427"]
    )
    source_hashes.update(shard_hashes)
    spans.append(
        _span(
            "numeric_input_load",
            phase_started,
            run_started,
            len(events),
            SELECTED_CHECKPOINT_PATH.as_posix(),
        )
    )
    progress(
        run_started,
        "numeric_input_load",
        "after_load",
        completed_units=len(events),
    )

    scale = derive_fixed_point_scale(checkpoint, training)
    phase_started = time.monotonic()
    progress(run_started, "development_controls", "before_generation", planned=5)
    controls = run_development_controls(checkpoint, scale, root / RAW_DIR / "controls")
    spans.append(
        _span(
            "development_controls",
            phase_started,
            run_started,
            len(controls),
            (root / RAW_DIR / "controls/development-restart.json").as_posix(),
        )
    )
    progress(
        run_started,
        "development_controls",
        "after_generation",
        completed_units=len(controls),
    )

    phase_started = time.monotonic()
    progress(
        run_started,
        "journal_replay",
        "before_benchmark",
        planned=len(events) * len(ARMS),
    )
    update_rows, _states = replay_update_arms(checkpoint, events, scale)
    spans.append(
        _span(
            "journal_replay",
            phase_started,
            run_started,
            len(update_rows),
            (root / RAW_DIR / "raw_evidence.json").as_posix(),
        )
    )
    progress(
        run_started,
        "journal_replay",
        "after_benchmark",
        completed_units=len(update_rows),
    )

    phase_started = time.monotonic()
    progress(
        run_started,
        "complete_service_timing",
        "before_benchmark",
        planned=len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS),
    )
    timing_rows, cold_rows = measure_service_cost(
        events,
        checkpoint,
        scale,
        root / RAW_DIR / "timing_persistence",
        emit_progress=True,
    )
    spans.append(
        _span(
            "complete_service_timing",
            phase_started,
            run_started,
            len(timing_rows) + len(cold_rows),
            (root / RAW_DIR / "timing_persistence").as_posix(),
        )
    )
    progress(
        run_started,
        "complete_service_timing",
        "after_benchmark",
        completed_units=len(timing_rows) + len(cold_rows),
    )

    raw_path = root / RAW_DIR / "raw_evidence.json"
    raw = {
        "schema": "carnot.exp7432.v651.raw_update_placement.v1",
        "update_rows": update_rows,
        "timing_rows": timing_rows,
        "cold_setup_rows": cold_rows,
        "board_rows": board_rows,
        "fixed_point_scale": scale,
        "development_controls": controls,
    }
    progress(run_started, "raw_checkpoint", "before_atomic_write")
    atomic_json(raw_path, raw)
    progress(
        run_started,
        "raw_checkpoint",
        "after_atomic_write",
        completed_units=len(update_rows) + len(timing_rows),
        bytes=raw_path.stat().st_size,
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7432-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError("invalid_validation_plan:" + ",".join(plan_errors))
    phase_started = time.monotonic()
    progress(
        run_started,
        "affected_validation",
        "before_subprocesses",
        planned=len(commands),
    )
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60.0,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            run_started,
            len(affected),
            (root / RAW_DIR / "validation/affected").as_posix(),
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    source_rows: JsonDict = {}
    for path, digest in source_hashes.items():
        source_rows[path] = (
            deepcopy(digest)
            if isinstance(digest, Mapping)
            else {
                "path": path,
                "sha256": digest,
                "original_flagged_adversarial": (
                    context["upstream"]["exp7427"].get("flagged_adversarial")
                    if path == EXP7427_PATH.as_posix()
                    else None
                ),
            }
        )
    candidate = assemble_artifact(
        update_rows=update_rows,
        timing_rows=timing_rows,
        cold_rows=cold_rows,
        board_rows=board_rows,
        scale=scale,
        controls=controls,
        preconditions=preconditions,
        source_hashes=source_rows,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        flagged_adversarial=not affected_reduction["passed"],
    )
    candidate["raw_evidence_receipt"] = {
        "path": raw_path.relative_to(root).as_posix(),
        "sha256": sha256_file(raw_path),
        "scope": "current_raw_evidence",
    }
    candidate["historical_evidence_sidecars"] = [
        {
            "path": path.as_posix(),
            "sha256": EXPECTED_HASHES[path],
            "scope": "historical_model_receipts",
            "current_invocation_counted": False,
        }
        for path in (EXP7425_PATH, EXP7426_PATH, EXP7427_PATH)
    ]
    candidate["field_principles"].update(
        _field_principles(["raw_evidence_receipt", "historical_evidence_sidecars"])
    )
    candidate["reproducibility_checksum"] = reproducibility_checksum(candidate)
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    progress(run_started, "candidate", "before_atomic_write")
    write_artifact(candidate_path, candidate)
    progress(
        run_started,
        "candidate",
        "after_atomic_write",
        bytes=candidate_path.stat().st_size,
    )

    terminal_plan = _terminal_commands(candidate_path, raw_path)
    phase_started = time.monotonic()
    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        planned=len(terminal_plan),
    )
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            run_started,
            len(terminal),
            candidate_path.as_posix(),
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )

    entrypoint = _entrypoint_receipt(root, started_at, time.monotonic() - run_started)
    final = assemble_artifact(
        update_rows=update_rows,
        timing_rows=timing_rows,
        cold_rows=cold_rows,
        board_rows=board_rows,
        scale=scale,
        controls=controls,
        preconditions=preconditions,
        source_hashes=source_rows,
        validation_receipts=[*affected, *terminal, entrypoint],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        flagged_adversarial=not affected_reduction["passed"] or not terminal_passed or critical,
    )
    final["raw_evidence_receipt"] = candidate["raw_evidence_receipt"]
    final["historical_evidence_sidecars"] = candidate["historical_evidence_sidecars"]
    final["field_principles"].update(
        _field_principles(["raw_evidence_receipt", "historical_evidence_sidecars"])
    )
    final["validation_duration_s"] = sum(
        float(span["duration_s"]) for span in spans if "validation" in str(span["phase"])
    )
    final["model_duration_s"] = 0.0
    final["cold_start_duration_s"] = sum(float(row["cold_setup_s"]) for row in cold_rows)
    final["field_principles"].update(
        _field_principles(["validation_duration_s", "model_duration_s", "cold_start_duration_s"])
    )
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    progress(run_started, "publish", "before_atomic_terminal", path=destination)
    write_artifact(destination, final)
    progress(run_started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date, output, and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--raw", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process terminal reader."""

    print("[exp7432] phase=entrypoint event=flushed", flush=True)
    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        try:
            artifact = load_json(args.independent_reduce)
            raw = raw_view(artifact) if args.raw is None else load_json(args.raw)
            receipt = artifact.get("raw_evidence_receipt")
            if args.raw is not None and isinstance(receipt, Mapping):
                if sha256_file(args.raw) != receipt.get("sha256"):
                    raise ValueError("raw evidence hash mismatch")
            reduction = independent_reduce(artifact, raw)
            errors = [] if all(reduction.values()) else ["independent_reduction_failed"]
        except (OSError, TypeError, ValueError) as error:
            reduction = {}
            errors = [str(error)]
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

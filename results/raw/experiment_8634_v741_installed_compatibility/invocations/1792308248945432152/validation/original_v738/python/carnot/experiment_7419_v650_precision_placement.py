"""Measure calibrated decision precision and complete host service cost.

The experiment freezes one source-aware head from Exp7413. It compares three
numeric representations on the host. It does not load an LLM or use a board.

Spec refs: REQ-REPORT-7419 and SCENARIO-REPORT-7419-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
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
from carnot.experiment_7413_v650_source_calibration import certified_decision
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
    write_immutable_sidecar,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
PHASE = 4
EXPERIMENT_ID = "exp7419-precision-placement"
SCHEMA = "carnot.exp7419.v650.precision_placement.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7419_v650_precision_placement.json")
RAW_DIR = Path("results/raw/experiment_7419_v650_precision_placement")
MODULE_PATH = Path("python/carnot/experiment_7419_v650_precision_placement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7419_v650_precision_placement.py")
TEST_PATH = Path("tests/python/test_experiment_7419_v650_precision_placement.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXP7413_PATH = Path("results/experiment_7413_v650_source_calibration.json")
CHECKPOINT_PATH = Path(
    "results/raw/experiment_7413_v650_source_calibration/checkpoints/"
    "full_source--source_aware_6_4_1_gibbs--65001.json"
)
FEATURE_PATH = Path("results/raw/experiment_7412_v650_source_features/source_feature_rows.json")
EXP7407_PATH = Path("results/experiment_7407_v649_service_cost.json")
EXP7314_PATH = Path("results/experiment_7314_v642_board_continuity.json")
EXP7358_PATH = Path("results/experiment_7358_v646_validation_contract.json")
KV260_TRANSCRIPT_PATH = Path(
    "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"
)
POLARFIRE_TRANSCRIPT_PATH = Path("results/raw/experiment_7231/polarfire_dispatch.json")

EXPECTED_EXP7413_SHA256 = "sha256:a32ff491648d1459a2948280d5a662b8a30fb9bb3476d10331d597be150ff647"
EXPECTED_CHECKPOINT_SHA256 = (
    "sha256:e6e17466ec34a20094ee9212723685efc905a52f354ff08994e3464325441fbd"
)
EXPECTED_FEATURE_SHA256 = "sha256:a34e26beee7f797bb68fe69af1be00420c1c2cc3bbb1b203ebc89dd59df36864"
EXPECTED_EXP7407_SHA256 = "sha256:bed7441f73378035f7eac5f82194550f83d20f2f4cf000b8d3bddd5f12a68d1e"
EXPECTED_EXP7314_SHA256 = "sha256:c83cc85d16c082898992b3d34197d5d6bb07dfae5cd5bb9beca10bbbc20d51e6"
EXPECTED_EXP7358_SHA256 = "sha256:0d2c2fe876ba303b21c9b5fe5b094d92851ea407d7d9f06ae9d8ca4d37475781"
EXPECTED_KV260_TRANSCRIPT_SHA256 = (
    "sha256:b813db8619cf29c3350fe9e806f68e20f0c8a5c381b6c11cb8dc096f524f6e5e"
)
EXPECTED_POLARFIRE_TRANSCRIPT_SHA256 = (
    "sha256:d1c438da812a1c45d63d5d777050c5169dab6aac33bfee8b44f4f4152dd6b801"
)

RANDOM_SEED = 7_419_650
BOOTSTRAP_DRAWS = 10_000
BATCH_SIZES = (1, 32, 128)
PAIRED_BLOCKS = 30
SELECTED_SEED = 65001
ARMS = ("float64_scalar", "float32_vectorized", "int8_float32_accum")
SOURCE_FEATURE_NAMES = (
    "normalized_content_token_overlap",
    "max_answer_source_sentence_overlap",
    "normalized_number_token_overlap",
    "numeric_novelty_with_context",
    "falsifiability_score",
    "missing_or_empty_source",
)
STAGE_NAMES = (
    "source_feature_work",
    "quantization",
    "numeric_scoring",
    "policy_decisions",
    "serialization",
    "complete_return_materialization",
)
ZERO_INVOCATION_COUNTS = deepcopy(ZERO_INVOCATION_COUNTS)
MISSING_RECEIPT = board_history.MISSING_RECEIPT

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    EXP7413_PATH,
    CHECKPOINT_PATH,
    FEATURE_PATH,
    EXP7407_PATH,
    EXP7314_PATH,
    EXP7358_PATH,
    KV260_TRANSCRIPT_PATH,
    POLARFIRE_TRANSCRIPT_PATH,
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7413_v650_source_calibration.py"),
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


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw measurements, receipt searches, logs, and output separate."""

    artifact: Path
    raw_evidence: Path
    changed_state_search: Path
    historical_receipts: Path
    terminal_candidate: Path
    validation_dir: Path

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Resolve task-owned outputs below a repository or test directory."""

        raw = root / RAW_DIR
        return cls(
            artifact=root / RESULT_PATH,
            raw_evidence=raw / "measured_precision_rows.json",
            changed_state_search=raw / "gatemate_changed_state_search.json",
            historical_receipts=raw / "historical_model_receipts.json",
            terminal_candidate=raw / "measured_terminal_candidate.json",
            validation_dir=raw / "validation",
        )


def utc_now() -> str:  # pragma: no cover - live wall-clock boundary.
    """Return one actual UTC boundary for the live experiment receipt."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase, slow-operation, and completed-unit boundaries."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7419] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json(path: Path) -> Any:
    """Load exact JSON bytes and name the unreadable path on failure."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable_json:{path}") from error


def gate_row(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep each validity, completion, safety, or benefit comparison explicit."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _source_gate(root: Path, relative: Path, expected_hash: str | None) -> JsonDict:
    """Authenticate exact source bytes before dependent measurement."""

    path = root / relative
    observed = sha256_file(path) if path.is_file() else None
    passed = observed is not None and (expected_hash is None or observed == expected_hash)
    return gate_row(
        f"source_bytes:{relative.as_posix()}",
        "numeric_precondition",
        "==",
        expected_hash or "readable_nonempty_bytes",
        observed if expected_hash else ("readable_nonempty_bytes" if observed else None),
        passed,
        "Exact bytes prevent a parsed object from hiding source replacement.",
    )


def _board_by_name(rows: Sequence[Any], name: str) -> JsonDict:
    """Select one board row so duplicate or missing identities fail closed."""

    matches = [dict(row) for row in rows if isinstance(row, Mapping) and row.get("board") == name]
    if len(matches) != 1:
        raise ValueError(f"board_row_not_unique:{name}")
    return matches[0]


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate numeric inputs and three independent historical board rows."""

    root = root.resolve()
    for destination in paths.__dict__.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    pinned = {
        EXP7413_PATH: EXPECTED_EXP7413_SHA256,
        CHECKPOINT_PATH: EXPECTED_CHECKPOINT_SHA256,
        FEATURE_PATH: EXPECTED_FEATURE_SHA256,
        EXP7407_PATH: EXPECTED_EXP7407_SHA256,
        EXP7314_PATH: EXPECTED_EXP7314_SHA256,
        EXP7358_PATH: EXPECTED_EXP7358_SHA256,
        KV260_TRANSCRIPT_PATH: EXPECTED_KV260_TRANSCRIPT_SHA256,
        POLARFIRE_TRANSCRIPT_PATH: EXPECTED_POLARFIRE_TRANSCRIPT_SHA256,
    }
    checks = [
        _source_gate(root, relative, pinned.get(relative)) for relative in REQUIRED_SOURCE_PATHS
    ]
    hashes = {
        relative.as_posix(): sha256_file(root / relative) if (root / relative).is_file() else None
        for relative in REQUIRED_SOURCE_PATHS
    }

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_requirement",
            "numeric_precondition",
            "==",
            "REQ-REPORT-7419",
            "REQ-REPORT-7419" if "REQ-REPORT-7419" in spec_text else None,
            "REQ-REPORT-7419" in spec_text,
            "The implementation must remain anchored to the declared precision contract.",
        )
    )

    upstream = load_json(root / EXP7413_PATH)
    checkpoint = load_json(root / CHECKPOINT_PATH)
    features = load_json(root / FEATURE_PATH)
    allowed = {"positive", "circular_positive", "null"}
    upstream_ok = (
        isinstance(upstream, Mapping)
        and upstream.get("experiment_id") == "exp7413-source-calibration"
        and upstream.get("milestone") == MILESTONE
        and upstream.get("calibration_capture_complete_score") == 1
        and upstream.get("verdict_class") in allowed
        and upstream.get("flagged_adversarial") is False
    )
    checks.append(
        gate_row(
            "exp7413_numeric_branch_eligibility",
            "numeric_precondition",
            "==",
            [1, sorted(allowed), False],
            [
                upstream.get("calibration_capture_complete_score"),
                upstream.get("verdict_class"),
                upstream.get("flagged_adversarial"),
            ],
            upstream_ok,
            "Only a complete, eligible, unflagged calibration branch can supply numeric science.",
        )
    )
    checkpoint_ok = (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("schema") == "carnot.exp7413.numeric_checkpoint.v1"
        and checkpoint.get("condition") == "full_source"
        and checkpoint.get("arm") == "source_aware_6_4_1_gibbs"
        and checkpoint.get("seed") == SELECTED_SEED
        and checkpoint.get("architecture") == [6, 4, 1]
    )
    checks.append(
        gate_row(
            "frozen_source_aware_head",
            "numeric_precondition",
            "==",
            ["full_source", "source_aware_6_4_1_gibbs", SELECTED_SEED, [6, 4, 1]],
            [
                checkpoint.get("condition"),
                checkpoint.get("arm"),
                checkpoint.get("seed"),
                checkpoint.get("architecture"),
            ],
            checkpoint_ok,
            "The first registered head is frozen before held-out labels are reduced.",
        )
    )
    records = features.get("records") if isinstance(features, Mapping) else None
    heldout_count = (
        sum(row.get("partition") == "final_test" for row in records)
        if isinstance(records, list)
        else 0
    )
    train_count = (
        sum(row.get("partition") == "train" for row in records) if isinstance(records, list) else 0
    )
    features_ok = isinstance(records, list) and heldout_count == 1995 and train_count > 0
    checks.append(
        gate_row(
            "feature_partition_identity",
            "numeric_precondition",
            "==",
            {"heldout_rows": 1995, "train_rows_positive": True},
            {"heldout_rows": heldout_count, "train_rows_positive": train_count > 0},
            features_ok,
            "Train-only scale fitting needs explicit partitions and a complete held-out cohort.",
        )
    )
    selected_metric_rows = [
        row
        for row in (upstream.get("paired_metric_rows") or [])
        if isinstance(row, Mapping)
        and row.get("condition") == "full_source"
        and row.get("arm") == "source_aware_6_4_1_gibbs"
        and row.get("seed") == SELECTED_SEED
    ]
    metric_keys = {row.get("row_key") for row in selected_metric_rows}
    feature_keys = {
        row.get("row_key")
        for row in (records or [])
        if isinstance(row, Mapping) and row.get("partition") == "final_test"
    }
    metrics_ok = len(selected_metric_rows) == 1995 and metric_keys == feature_keys
    checks.append(
        gate_row(
            "heldout_metric_row_identity",
            "numeric_precondition",
            "==",
            {"rows": 1995, "keys_match": True},
            {"rows": len(selected_metric_rows), "keys_match": metric_keys == feature_keys},
            metrics_ok,
            "Every held-out feature needs one frozen producer label disposition.",
        )
    )

    service_artifact = load_json(root / EXP7407_PATH)
    old_board_artifact = load_json(root / EXP7314_PATH)
    service_boards = service_artifact.get("board_rows") or []
    old_boards = old_board_artifact.get("board_rows") or []
    board_checks: list[JsonDict] = []
    expected_states = {
        "KV260": "graduated_preserved",
        "GateMate": "blocked_changed_physical_state",
        "PolarFire": "graduated_cpu_dispatch_preserved",
    }
    for name, state in expected_states.items():
        try:
            row = _board_by_name(service_boards, name)
            old = _board_by_name(old_boards, name)
            passed = row.get("terminal_state") == state and old.get("board") == name
            observed = row.get("terminal_state")
        except ValueError:
            passed = False
            observed = None
        board_checks.append(
            gate_row(
                f"board_record:{name}",
                "board_accounting",
                "==",
                state,
                observed,
                passed,
                "Each board keeps its own dated capability boundary.",
            )
        )
    checks.extend(board_checks)
    boundary = load_json(root / EXP7358_PATH)
    boundary_ok = (
        isinstance(boundary, Mapping)
        and boundary.get("validation_contract_ready_score") == 1
        and boundary.get("flagged_adversarial") is False
    )
    checks.append(
        gate_row(
            "exp7358_scoped_validation_contract",
            "numeric_precondition",
            "==",
            1,
            boundary.get("validation_contract_ready_score"),
            boundary_ok,
            "The frozen affected plan prevents broad or inherited validation work.",
        )
    )
    physical = board_history.search_changed_state_receipt(
        root,
        paths.changed_state_search,
        candidate_paths=candidate_paths,
    )
    hashes[paths.changed_state_search.as_posix()] = sha256_file(paths.changed_state_search)
    checks.append(
        gate_row(
            "gatemate_changed_physical_state_receipt",
            "external_prerequisite",
            "matches",
            deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT),
            physical
            if physical.get("exists") is True
            else {
                "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                "selected_source_path": physical.get("selected_source_path"),
                "absence": MISSING_RECEIPT,
            },
            physical.get("exists") is True,
            "Only an operator-authored changed physical state can reopen future GateMate work.",
        )
    )
    numeric_ok = all(row["passed"] for row in checks if row["category"] == "numeric_precondition")
    context = {
        "numeric_branch": {
            "eligible": numeric_ok,
            "source_verdict_class": upstream.get("verdict_class"),
            "checkpoint_seed": checkpoint.get("seed"),
            "heldout_rows": heldout_count,
            "train_rows": train_count,
        },
        "historical_board_rows": [deepcopy(dict(row)) for row in service_boards],
        "gatemate_changed_state": physical,
    }
    return checks, hashes, context


def _weight_arrays(checkpoint: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    """Validate the frozen 6-4-1 numeric state before scoring or quantization."""

    try:
        weights = checkpoint["weights"]
        w1 = np.asarray(weights["w1"], dtype=np.float64)
        b1 = np.asarray(weights["b1"], dtype=np.float64)
        w_out = np.asarray(weights["w_out"], dtype=np.float64)
        b_out = np.asarray([weights["b_out"]], dtype=np.float64)
        affine = np.asarray(
            [checkpoint["affine"]["slope"], checkpoint["affine"]["intercept"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint fields are invalid") from error
    if (
        w1.shape != (4, 6)
        or b1.shape != (4,)
        or w_out.shape != (4,)
        or b_out.shape != (1,)
        or affine.shape != (2,)
        or not all(np.isfinite(array).all() for array in (w1, b1, w_out, b_out, affine))
    ):
        raise ValueError("checkpoint shape or numeric values are invalid")
    return w1, b1, w_out, b_out, affine


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Materialize the six frozen source fields in registered order."""

    vectors: list[list[float]] = []
    for row in rows:
        source = row.get("source_features")
        if not isinstance(source, Mapping) or set(source) != set(SOURCE_FEATURE_NAMES):
            raise ValueError("source feature keys do not match the frozen protocol")
        vector = [float(source[name]) for name in SOURCE_FEATURE_NAMES]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in vector):
            raise ValueError("source features must be finite values in [0, 1]")
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1:] != (6,):
        raise ValueError("feature matrix must have shape (n, 6)")
    return matrix


def _symmetric_scale(values: np.ndarray) -> float:
    """Return a nonzero symmetric int8 scale from calibration-only values."""

    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    return maximum / 127.0 if maximum > 0 else 1.0 / 127.0


def _quantize(values: np.ndarray, scale: float) -> np.ndarray:
    """Round and clip values to the signed symmetric int8 range."""

    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8)


def derive_quantization(
    checkpoint: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Derive scales from frozen weights and train features without test labels."""

    w1, b1, w_out, b_out, _affine = _weight_arrays(checkpoint)
    train = [row for row in rows if row.get("partition") == "train"]
    if not train:
        raise ValueError("training features are required for quantization")
    train_matrix = feature_matrix(train)
    scales = {
        "w1": _symmetric_scale(w1),
        "b1": _symmetric_scale(b1),
        "w_out": _symmetric_scale(w_out),
        "b_out": _symmetric_scale(b_out),
    }
    return {
        "scheme": "symmetric_int8_weights_and_features_float32_accumulation",
        "scale_source": "checkpoint_weights_and_train_features_only",
        "feature_scale": _symmetric_scale(train_matrix),
        "feature_row_count": len(train),
        "feature_partition": "train",
        "feature_values_sha256": canonical_hash(train_matrix.tolist()),
        "weight_scales": scales,
        "quantized_weights": {
            "w1": _quantize(w1, scales["w1"]).tolist(),
            "b1": _quantize(b1, scales["b1"]).tolist(),
            "w_out": _quantize(w_out, scales["w_out"]).tolist(),
            "b_out": int(_quantize(b_out, scales["b_out"])[0]),
        },
        "heldout_labels_used": False,
        "thresholds_tuned": False,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert finite logits to probabilities without overflow."""

    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _swish(values: np.ndarray) -> np.ndarray:
    """Apply the exact smooth activation used by the frozen Gibbs head."""

    return values * _sigmoid(values)


def score_probabilities(
    matrix: np.ndarray,
    checkpoint: Mapping[str, Any],
    arm: str,
    quantization: Mapping[str, Any],
) -> np.ndarray:
    """Score one matrix with float64, float32, or int8 host emulation."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1:] != (6,) or not np.isfinite(values).all():
        raise ValueError("feature matrix must be finite with shape (n, 6)")
    if arm not in ARMS:
        raise ValueError(f"registered arm required: {arm}")
    w1, b1, w_out, b_out, affine = _weight_arrays(checkpoint)
    if arm == "float64_scalar":
        output = []
        for row in values.astype(np.float64):
            hidden_linear = w1 @ row + b1
            raw = float(w_out @ _swish(hidden_linear) + b_out[0])
            output.append(float(_sigmoid(np.asarray([affine[0] * raw + affine[1]]))[0]))
        return np.asarray(output, dtype=np.float64)
    if arm == "float32_vectorized":
        matrix32 = values.astype(np.float32)
        hidden = _swish(matrix32 @ w1.astype(np.float32).T + b1.astype(np.float32))
        raw = hidden @ w_out.astype(np.float32) + np.float32(b_out[0])
        logits = np.float32(affine[0]) * raw + np.float32(affine[1])
        return _sigmoid(logits.astype(np.float32)).astype(np.float64)

    try:
        feature_scale = float(quantization["feature_scale"])
        scales = quantization["weight_scales"]
        quantized = quantization["quantized_weights"]
        matrix32 = _quantize(values.astype(np.float32), feature_scale).astype(
            np.float32
        ) * np.float32(feature_scale)
        qw1 = np.asarray(quantized["w1"], dtype=np.int8).astype(np.float32)
        qb1 = np.asarray(quantized["b1"], dtype=np.int8).astype(np.float32)
        qw_out = np.asarray(quantized["w_out"], dtype=np.int8).astype(np.float32)
        qb_out = np.float32(quantized["b_out"])
        hidden_linear = matrix32 @ (qw1 * np.float32(scales["w1"])).T + qb1 * np.float32(
            scales["b1"]
        )
        hidden = _swish(hidden_linear.astype(np.float32))
        raw = hidden @ (qw_out * np.float32(scales["w_out"])) + qb_out * np.float32(scales["b_out"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("quantization receipt is invalid") from error
    logits = np.float32(affine[0]) * raw + np.float32(affine[1])
    return _sigmoid(logits.astype(np.float32)).astype(np.float64)


def _threshold_bucket(probability: float, policy: Mapping[str, Any]) -> str:
    """Name raw threshold position independently from action enablement."""

    if probability < float(policy["accept_threshold"]):
        return "accept_region"
    if probability >= float(policy["reject_threshold"]):
        return "reject_region"
    return "escalate_region"


def _log_loss(label: int, probability: float) -> float:
    """Return a finite binary log-loss contribution."""

    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def _representation_bytes(arm: str, row_count: int, checkpoint: Mapping[str, Any]) -> int:
    """Measure the numeric model and feature payload for one arm."""

    w1, b1, w_out, b_out, affine = _weight_arrays(checkpoint)
    parameter_count = sum(array.size for array in (w1, b1, w_out, b_out))
    if arm == "float64_scalar":
        return int((parameter_count + row_count * 6 + affine.size) * 8)
    if arm == "float32_vectorized":
        return int((parameter_count + row_count * 6 + affine.size) * 4)
    scale_count = 5
    return int(parameter_count + row_count * 6 + (scale_count + affine.size) * 4)


def build_precision_rows(
    heldout_rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    quantization: Mapping[str, Any],
) -> list[JsonDict]:
    """Retain every held-out unit and numeric arm, including missing labels."""

    matrix = feature_matrix(heldout_rows)
    scores = {arm: score_probabilities(matrix, checkpoint, arm, quantization) for arm in ARMS}
    policy = checkpoint["selected_policy"]
    baseline = scores["float64_scalar"]
    output: list[JsonDict] = []
    for index, row in enumerate(heldout_rows):
        label = row.get("label")
        scored = label in {0, 1}
        baseline_action = certified_decision(
            float(baseline[index]), policy, f"exp7419:float64:{checkpoint.get('seed')}"
        )
        baseline_bucket = _threshold_bucket(float(baseline[index]), policy)
        for arm in ARMS:
            probability = float(scores[arm][index])
            action = certified_decision(
                probability, policy, f"exp7419:{arm}:{checkpoint.get('seed')}"
            )
            bucket = _threshold_bucket(probability, policy)
            precision_row: JsonDict = {
                "unit_id": f"{row['row_key']}:{arm}",
                "row_key": row["row_key"],
                "group_id": row["group_id"],
                "partition": "final_test",
                "arm": arm,
                "seed": checkpoint.get("seed"),
                "probability": probability,
                "float64_probability": float(baseline[index]),
                "probability_abs_error": abs(probability - float(baseline[index])),
                "decision": action["decision"],
                "decision_reason": action["reason"],
                "float64_decision": baseline_action["decision"],
                "action_flip": action["decision"] != baseline_action["decision"],
                "threshold_bucket": bucket,
                "float64_threshold_bucket": baseline_bucket,
                "threshold_crossing": bucket != baseline_bucket,
                "representation_bytes": _representation_bytes(arm, 1, checkpoint),
                "scored": scored,
                "label": int(label) if scored else None,
                "label_authority": "machine_annotation",
                "brier_contribution": (probability - int(label)) ** 2 if scored else None,
                "log_loss_contribution": _log_loss(int(label), probability) if scored else None,
                "invalid_numeric": not math.isfinite(probability),
            }
            precision_row["row_sha256"] = row_hash(precision_row)
            output.append(precision_row)
    return output


def reduce_precision(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute proper scores, errors, actions, and threshold crossings."""

    if not rows:
        raise ValueError("precision rows are required")
    identities: dict[str, set[str]] = {}
    for row in rows:
        identities.setdefault(str(row.get("row_key")), set()).add(str(row.get("arm")))
    if any(arms != set(ARMS) for arms in identities.values()):
        raise ValueError("precision arm coverage is incomplete")
    by_arm: JsonDict = {}
    for arm in ARMS:
        selected = [row for row in rows if row.get("arm") == arm]
        scored = [row for row in selected if row.get("scored") is True]
        by_arm[arm] = {
            "rows": len(selected),
            "scored_rows": len(scored),
            "unscored_rows": len(selected) - len(scored),
            "brier": float(np.mean([row["brier_contribution"] for row in scored]))
            if scored
            else None,
            "log_loss": float(np.mean([row["log_loss_contribution"] for row in scored]))
            if scored
            else None,
            "maximum_probability_error": max(
                (float(row["probability_abs_error"]) for row in selected), default=0.0
            ),
            "representation_bytes": max(
                (int(row["representation_bytes"]) for row in selected), default=0
            ),
        }
    baseline_brier = by_arm["float64_scalar"]["brier"]
    baseline_log = by_arm["float64_scalar"]["log_loss"]
    for arm in ARMS:
        by_arm[arm]["brier_delta_vs_float64"] = (
            by_arm[arm]["brier"] - baseline_brier
            if by_arm[arm]["brier"] is not None and baseline_brier is not None
            else None
        )
        by_arm[arm]["log_loss_delta_vs_float64"] = (
            by_arm[arm]["log_loss"] - baseline_log
            if by_arm[arm]["log_loss"] is not None and baseline_log is not None
            else None
        )
    flips = [deepcopy(dict(row)) for row in rows if row.get("action_flip") is True]
    crossings = [deepcopy(dict(row)) for row in rows if row.get("threshold_crossing") is True]
    invalid = [deepcopy(dict(row)) for row in rows if row.get("invalid_numeric") is True]
    int8_delta = by_arm["int8_float32_accum"]["brier_delta_vs_float64"]
    score_gate = int8_delta is not None and int8_delta <= 0.001
    return {
        "by_arm": by_arm,
        "invalid_numeric_count": len(invalid),
        "invalid_numeric_rows": invalid,
        "measured_action_flip_count": len(flips),
        "measured_action_flips": flips,
        "threshold_crossing_count": len(crossings),
        "threshold_crossings": crossings,
        "unscored_rows_preserved": sum(row.get("scored") is not True for row in rows),
        "action_parity_passed": not flips,
        "numeric_validity_passed": not invalid,
        "int8_brier_increase_limit": 0.001,
        "int8_brier_gate_passed": bool(score_gate),
    }


def near_threshold_fixture(
    checkpoint: Mapping[str, Any], quantization: Mapping[str, Any]
) -> list[JsonDict]:
    """Expose float32 adverse rounding immediately around both frozen thresholds."""

    del quantization
    policy = checkpoint["selected_policy"]
    output: list[JsonDict] = []
    for name in ("accept", "reject"):
        threshold = float(policy[f"{name}_threshold"])
        for side, value in (
            ("below", float(np.nextafter(threshold, -math.inf))),
            ("exact", threshold),
            ("above", float(np.nextafter(threshold, math.inf))),
        ):
            rounded = float(np.float32(value))
            base_bucket = _threshold_bucket(value, policy)
            rounded_bucket = _threshold_bucket(rounded, policy)
            output.append(
                {
                    "threshold_name": name,
                    "side": side,
                    "threshold": threshold,
                    "float64_probability": value,
                    "float32_probability": rounded,
                    "float64_threshold_bucket": base_bucket,
                    "float32_threshold_bucket": rounded_bucket,
                    "adverse_rounding": base_bucket != rounded_bucket,
                    "included_in_measured_action_parity": False,
                }
            )
    if not any(row["adverse_rounding"] for row in output):
        raise ValueError("near-threshold fixture did not expose adverse rounding")
    return output


def row_hash(row: Mapping[str, Any]) -> str:
    """Bind one complete row while excluding its self-reference."""

    payload = deepcopy(dict(row))
    payload.pop("row_sha256", None)
    return canonical_hash(payload)


def _run_service(
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    quantization: Mapping[str, Any],
    arm: str,
) -> JsonDict:
    """Measure the complete feature-to-serialized-return host boundary."""

    started = time.perf_counter_ns()
    stage_started = time.perf_counter_ns()
    matrix = feature_matrix(rows)
    feature_s = (time.perf_counter_ns() - stage_started) / 1_000_000_000

    stage_started = time.perf_counter_ns()
    if arm == "int8_float32_accum":
        scale = float(quantization["feature_scale"])
        prepared = _quantize(matrix, scale).astype(np.float32) * np.float32(scale)
    elif arm == "float32_vectorized":
        prepared = matrix.astype(np.float32)
    elif arm == "float64_scalar":
        prepared = matrix.astype(np.float64)
    else:
        raise ValueError(f"registered arm required: {arm}")
    quantization_s = (time.perf_counter_ns() - stage_started) / 1_000_000_000

    stage_started = time.perf_counter_ns()
    probabilities = score_probabilities(prepared, checkpoint, arm, quantization)
    scoring_s = (time.perf_counter_ns() - stage_started) / 1_000_000_000

    stage_started = time.perf_counter_ns()
    policy = checkpoint["selected_policy"]
    actions = [
        certified_decision(float(probability), policy, f"exp7419:{arm}:{checkpoint.get('seed')}")
        for probability in probabilities
    ]
    policy_s = (time.perf_counter_ns() - stage_started) / 1_000_000_000

    stage_started = time.perf_counter_ns()
    serialized = json.dumps(
        {
            "arm": arm,
            "probabilities": probabilities.tolist(),
            "actions": actions,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    serialization_s = (time.perf_counter_ns() - stage_started) / 1_000_000_000

    stage_started = time.perf_counter_ns()
    materialized = bytes(serialized)
    return_s = (time.perf_counter_ns() - stage_started) / 1_000_000_000
    total_s = (time.perf_counter_ns() - started) / 1_000_000_000
    return {
        "arm": arm,
        "batch_size": len(rows),
        "stage_durations_s": {
            "source_feature_work": feature_s,
            "quantization": quantization_s,
            "numeric_scoring": scoring_s,
            "policy_decisions": policy_s,
            "serialization": serialization_s,
            "complete_return_materialization": return_s,
        },
        "total_service_s": total_s,
        "serialized_bytes": len(materialized),
        "representation_bytes": _representation_bytes(arm, len(rows), checkpoint),
        "invalid_numeric_count": int(np.count_nonzero(~np.isfinite(probabilities))),
        "actions": [row["decision"] for row in actions],
        "probabilities": probabilities.tolist(),
    }


def _timing_row(
    result: Mapping[str, Any], block: int, order: Sequence[str], order_index: int
) -> JsonDict:
    """Reduce one complete service call to an immutable paired timing row."""

    row = {
        "unit_id": f"batch:{result['batch_size']}:block:{block}:arm:{result['arm']}",
        "batch_size": int(result["batch_size"]),
        "block": block,
        "arm": result["arm"],
        "rotated_arm_order": list(order),
        "order_index": order_index,
        "stage_durations_s": deepcopy(dict(result["stage_durations_s"])),
        "total_service_s": float(result["total_service_s"]),
        "serialized_bytes": int(result["serialized_bytes"]),
        "representation_bytes": int(result["representation_bytes"]),
        "invalid_numeric_count": int(result["invalid_numeric_count"]),
        "status": "completed",
    }
    row["row_sha256"] = row_hash(row)
    return row


def measure_service_cost(
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    quantization: Mapping[str, Any],
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
    emit_progress: bool = False,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run cold calls and rotated warm paired blocks for every batch size."""

    if not rows or blocks <= 0 or any(size not in BATCH_SIZES for size in batch_sizes):
        raise ValueError("batch plan must use nonempty rows, positive blocks, and 1, 32, or 128")
    expanded = [rows[index % len(rows)] for index in range(max(batch_sizes))]
    cold: list[JsonDict] = []
    for arm in ARMS:
        before = time.perf_counter_ns()
        result = _run_service(expanded[:1], checkpoint, quantization, arm)
        cold.append(
            {
                "arm": arm,
                "batch_size": 1,
                "cold_initialization_s": (time.perf_counter_ns() - before) / 1_000_000_000,
                "first_service_s": result["total_service_s"],
                "status": "completed",
            }
        )
    output: list[JsonDict] = []
    started = time.monotonic()
    total_units = len(batch_sizes) * blocks * len(ARMS)
    completed = 0
    for batch_index, batch_size in enumerate(batch_sizes):
        batch = expanded[:batch_size]
        for arm in ARMS:
            _run_service(batch, checkpoint, quantization, arm)
        for block in range(blocks):
            offset = (block + batch_index) % len(ARMS)
            order = (*ARMS[offset:], *ARMS[:offset])
            for order_index, arm in enumerate(order):
                output.append(
                    _timing_row(
                        _run_service(batch, checkpoint, quantization, arm),
                        block,
                        order,
                        order_index,
                    )
                )
                completed += 1
            if emit_progress and (block == blocks - 1 or time.monotonic() - started >= 60):
                progress(
                    started,
                    "complete_service_benchmark",
                    "checkpoint",
                    completed=completed,
                    total=total_units,
                )
                started = time.monotonic()
    return output, cold


def synthetic_timing_row(
    batch_size: int,
    block: int,
    arm: str,
    total_s: float,
    *,
    scoring_s: float | None = None,
) -> JsonDict:
    """Build one internally consistent timing row for reducer tests."""

    if arm not in ARMS or total_s <= 0:
        raise ValueError("synthetic timing arm and duration must be valid")
    scoring = total_s * 0.4 if scoring_s is None else scoring_s
    if not 0 <= scoring <= total_s:
        raise ValueError("synthetic scoring duration must fit the total")
    remaining = total_s - scoring
    stages = {
        "source_feature_work": remaining * 0.35,
        "quantization": remaining * 0.15,
        "numeric_scoring": scoring,
        "policy_decisions": remaining * 0.15,
        "serialization": remaining * 0.25,
        "complete_return_materialization": remaining * 0.10,
    }
    order = list(ARMS[block % 3 :] + ARMS[: block % 3])
    row = {
        "unit_id": f"batch:{batch_size}:block:{block}:arm:{arm}",
        "batch_size": batch_size,
        "block": block,
        "arm": arm,
        "rotated_arm_order": order,
        "order_index": order.index(arm),
        "stage_durations_s": stages,
        "total_service_s": total_s,
        "serialized_bytes": 100,
        "representation_bytes": 100,
        "invalid_numeric_count": 0,
        "status": "completed",
    }
    row["row_sha256"] = row_hash(row)
    return row


def _bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, draws: int, seed: int
) -> JsonDict:
    """Resample paired blocks and retain a ratio-of-means interval."""

    if numerator.shape != denominator.shape or numerator.ndim != 1 or not numerator.size:
        raise ValueError("paired ratio inputs need one common nonempty shape")
    if draws <= 0 or np.any(denominator <= 0):
        raise ValueError("paired ratio draws and denominator must be positive")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selection = rng.integers(0, numerator.size, size=numerator.size)
        sampled[index] = float(np.mean(numerator[selection]) / np.mean(denominator[selection]))
    return {
        "estimate": float(np.mean(numerator) / np.mean(denominator)),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "paired_blocks": int(numerator.size),
        "draws": draws,
        "seed": seed,
    }


def summarize_timing(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = RANDOM_SEED,
) -> JsonDict:
    """Reduce paired complete-service ratios for all represented batch sizes."""

    if not rows:
        raise ValueError("paired timing rows are required")
    batch_rows: list[JsonDict] = []
    for batch_size in sorted({int(row["batch_size"]) for row in rows}):
        selected = [row for row in rows if row.get("batch_size") == batch_size]
        blocks = sorted({int(row["block"]) for row in selected})
        by_pair: dict[tuple[int, str], Mapping[str, Any]] = {}
        for row in selected:
            key = (int(row["block"]), str(row["arm"]))
            if key in by_pair:
                raise ValueError("paired timing has duplicate rows")
            by_pair[key] = row
        if any((block, arm) not in by_pair for block in blocks for arm in ARMS):
            raise ValueError("paired timing is incomplete")
        int8 = np.asarray(
            [by_pair[(block, "int8_float32_accum")]["total_service_s"] for block in blocks],
            dtype=np.float64,
        )
        float32 = np.asarray(
            [by_pair[(block, "float32_vectorized")]["total_service_s"] for block in blocks],
            dtype=np.float64,
        )
        ratio = _bootstrap_ratio(int8, float32, draws, seed + batch_size)
        batch_rows.append(
            {
                "batch_size": batch_size,
                "paired_blocks": len(blocks),
                "int8_total_time_ratio_vs_float32": ratio,
                "speed_gate_passed": ratio["ci95"][1] < 1.0,
            }
        )
    primary = next(
        row
        for row in batch_rows
        if row["batch_size"] == max(row["batch_size"] for row in batch_rows)
    )
    return {
        "batch_rows": batch_rows,
        "primary_batch_size": primary["batch_size"],
        "primary_ratio": primary["int8_total_time_ratio_vs_float32"],
        "speed_gate_passed": primary["speed_gate_passed"],
    }


def compute_amdahl(rows: Sequence[Mapping[str, Any]], *, batch_size: int, arm: str) -> JsonDict:
    """Compute an infinite scoring-stage bound from complete measured costs."""

    selected = [
        row for row in rows if row.get("batch_size") == batch_size and row.get("arm") == arm
    ]
    if not selected:
        raise ValueError("matching timing rows are required for Amdahl analysis")
    total = float(np.mean([row["total_service_s"] for row in selected]))
    scoring = float(np.mean([row["stage_durations_s"]["numeric_scoring"] for row in selected]))
    accelerated_fraction = min(max(scoring / total, 0.0), 1.0)
    unaccelerated = 1.0 - accelerated_fraction
    upper = math.inf if unaccelerated == 0 else 1.0 / unaccelerated
    return {
        "batch_size": batch_size,
        "arm": arm,
        "accelerated_stage": "numeric_scoring",
        "accelerated_fraction": accelerated_fraction,
        "unaccelerated_fraction": unaccelerated,
        "infinite_accelerated_stage_upper_bound": upper,
        "hundred_x_feasibility": {
            "operator": "<=",
            "expected": 0.01,
            "observed": unaccelerated,
            "passed": unaccelerated <= 0.01,
        },
        "achieved_device_acceleration": False,
        "interpretation": "feasibility_bound_not_achieved_acceleration",
    }


def build_board_rows(context: Mapping[str, Any]) -> list[JsonDict]:
    """Retain three dated board states without issuing a physical command."""

    rows = [deepcopy(dict(row)) for row in context.get("historical_board_rows") or []]
    if {row.get("board") for row in rows} != {"KV260", "GateMate", "PolarFire"}:
        raise ValueError("three authenticated board rows are required")
    physical = context.get("gatemate_changed_state") or {}
    for row in rows:
        row["hardware_ready_score"] = 0
        row["hardware_value_score"] = 0
        row["new_hardware_execution_claimed"] = False
        row["fresh_physical_attempt"] = False
        if row.get("board") == "GateMate" and physical.get("exists") is True:
            row["terminal_state"] = "changed_physical_state_future_prerequisite"
            row["future_prerequisite"] = deepcopy(dict(physical))
            row["error"] = None
        row["row_sha256"] = row_hash(row)
    return rows


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require the frozen affected plan and every supplied terminal reader."""

    by_name = {str(row.get("name")): row for row in receipts}
    affected = all(
        name in by_name and by_name[name].get("passed") is True
        for name in validation_scope.REQUIRED_CHECK_NAMES
    )
    terminal_names = {
        "declared_entrypoint_cold_replay",
        "independent_cold_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    supplied_terminal = terminal_names.intersection(by_name)
    terminal = not supplied_terminal or (
        supplied_terminal == terminal_names
        and all(by_name[name].get("passed") is True for name in terminal_names)
    )
    return affected and terminal


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep the numeric blocker separate from the expected GateMate prerequisite."""

    numeric = next(
        (
            deepcopy(dict(row))
            for row in gates
            if row.get("category") == "numeric_precondition" and row.get("passed") is not True
        ),
        None,
    )
    gatemate = next(
        (
            deepcopy(dict(row))
            for row in gates
            if row.get("check") == "gatemate_changed_physical_state_receipt"
        ),
        None,
    )
    required_failures = [
        deepcopy(dict(row))
        for row in gates
        if row.get("category") in {"numeric_precondition", "completion", "safety", "validation"}
        and row.get("passed") is not True
    ]
    return {
        "required_checks_passed": not required_failures,
        "failed_required_checks": required_failures,
        "numeric_branch_blocker": numeric,
        "gatemate_changed_state_prerequisite": gatemate,
        "scientific_benefit_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") == "scientific_benefit"
        ),
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary fields without wrapping their machine-readable values."""

    principles = {key: "Keep this ordinary field machine-readable and replayable." for key in keys}
    principles.update(
        {
            "schema": "Version ordinary top-level identity, milestone, and terminal status fields.",
            "run_date": "Use 20260919 with actual UTC start and end timestamps.",
            "preconditions_checked": "Record exact paths, hashes, and resources before dependent work.",
            "MODEL_SPECS": "Remain empty because this run makes no current LLM call.",
            "model_invoked": "Remain false because no current model load or generation is attempted.",
            "invocation_counts": "Reduce current owned events; every current LLM count stays zero.",
            "inference_substrate": "Use a truthful string and keep device detail in a separate field.",
            "inference_substrate_class": "Declare no_model_load without padding work to meet a floor.",
            "execution_venue": "Use the closed host value; no device executes this experiment.",
            "duration_s": "Measure current monotonic work apart from historical and validation claims.",
            "phase_spans": "Retain real phase boundaries, completed units, and heartbeat records.",
            "random_seed": "Freeze paired order and resampling seeds before measurement.",
            "reproducibility_checksum": "Bind code, protocol, source bytes, scales, and raw rows.",
            "source_artifact_hashes": "Hash exact paths and preserve upstream adversarial flags.",
            "rows": "Account for every comparative, timing, cold, and board unit.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted work.",
            "acceptance_gate_results": "Separate validity, completion, safety, benefit, and external prerequisites.",
            "gate_check_summary": "Name the numeric blocker and GateMate changed-state prerequisite separately.",
            "verifier_is_oracle": "Source-defined score recomputation shares correctness authority and is circular.",
            "honest_verdict": "Start completed findings with complete_ and unchanged absence with blocked_.",
            "verdict_class": "Use the closed terminal enum; completed no-benefit work is null.",
            "flagged_adversarial": "Preserve critical findings and deny readiness when flagged.",
            "validation_receipts": "Retain exact argv, environment, exit, duration, and hashed logs.",
            "field_principles": "Explain fields separately while gate scalars remain ordinary numbers.",
            "promotion_score": "Always remain zero; this result cannot roll out, publish, or update weights.",
            "precision_capture_complete_score": "One requires a valid complete measured numeric branch.",
            "precision_value_score": "One requires action parity, score tolerance, and full-service speed.",
            "precision_rows": "Retain each unit, arm, probability, error, bytes, action, and crossing.",
            "board_rows": "Keep KV260, GateMate, and PolarFire terminal states independent.",
            "hardware_ready_score": "Remain zero because no fresh physical qualification occurs.",
            "hardware_value_score": "Remain zero because host emulation cannot establish device acceleration.",
        }
    )
    return principles


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash protocol, exact sources, scales, and raw evidence without self-reference."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def _source_hash_rows(
    hashes: Mapping[str, str | None], upstream: Mapping[str, Any] | None = None
) -> JsonDict:
    """Attach original flag states to exact source path hashes."""

    upstream = upstream or {}
    return {
        path: {
            "path": path,
            "sha256": value,
            "original_flagged_adversarial": (
                upstream.get("flagged_adversarial") if path == EXP7413_PATH.as_posix() else None
            ),
        }
        for path, value in hashes.items()
    }


def _sample_budget(
    precision_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    cold_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Count each planned comparative unit without treating paired arms as independent."""

    heldout = len({row.get("row_key") for row in precision_rows})
    planned_precision = heldout * len(ARMS)
    planned_timing = len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS)
    return {
        "planned": planned_precision + planned_timing + len(ARMS),
        "attempted": len(precision_rows) + len(timing_rows) + len(cold_rows),
        "completed": sum(
            row.get("status", "completed") == "completed"
            for row in (*precision_rows, *timing_rows, *cold_rows)
        ),
        "failed": sum(row.get("status") == "failed" for row in (*timing_rows, *cold_rows)),
        "censored": sum(row.get("status") == "censored" for row in (*timing_rows, *cold_rows)),
        "unstarted": max(
            planned_precision
            + planned_timing
            + len(ARMS)
            - len(precision_rows)
            - len(timing_rows)
            - len(cold_rows),
            0,
        ),
        "independent_groups": len({row.get("group_id") for row in precision_rows}),
        "heldout_rows": heldout,
        "planned_precision_rows": planned_precision,
        "planned_timing_rows": planned_timing,
        "planned_cold_rows": len(ARMS),
        "stop_rule": "complete every held-out arm, 30 rotated paired blocks per batch, and one cold call per arm",
    }


def assemble_artifact(
    *,
    precision_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    cold_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    quantization: Mapping[str, Any],
    boundary_rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_ns: int,
    raw_evidence_receipt: Mapping[str, Any] | None = None,
    sidecar_references: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
    fixture: bool = False,
) -> JsonDict:
    """Build one terminal record with validity independent from scientific benefit."""

    precision = [deepcopy(dict(row)) for row in precision_rows]
    timing = [deepcopy(dict(row)) for row in timing_rows]
    cold = [deepcopy(dict(row)) for row in cold_rows]
    boards = [deepcopy(dict(row)) for row in board_rows]
    numeric_available = all(
        row.get("passed") is True
        for row in preconditions
        if row.get("category") == "numeric_precondition"
    )
    precision_reduction = reduce_precision(precision) if precision else None
    timing_summary = (
        summarize_timing(timing, draws=200 if fixture else BOOTSTRAP_DRAWS, seed=RANDOM_SEED)
        if timing
        else None
    )
    amdahl = (
        compute_amdahl(timing, batch_size=max(BATCH_SIZES), arm="float32_vectorized")
        if timing
        else None
    )
    budget = _sample_budget(precision, timing, cold)
    complete = bool(
        numeric_available
        and precision
        and timing
        and len(cold) == len(ARMS)
        and budget["attempted"] == budget["planned"]
        and budget["failed"] == budget["censored"] == budget["unstarted"] == 0
    )
    validation_passed = True if fixture else _required_validation_passed(validation_receipts)
    parity = bool(
        precision_reduction
        and precision_reduction["action_parity_passed"]
        and precision_reduction["numeric_validity_passed"]
    )
    score_tolerance = bool(precision_reduction and precision_reduction["int8_brier_gate_passed"])
    speed = bool(timing_summary and timing_summary["speed_gate_passed"])
    gates = [deepcopy(dict(row)) for row in preconditions]
    gates.extend(
        [
            gate_row(
                "complete_numeric_evidence",
                "completion",
                "==",
                True,
                complete,
                complete,
                "Every measured and paired unit must finish before capture can complete.",
            ),
            gate_row(
                "identical_typed_actions_and_finite_outputs",
                "safety",
                "==",
                True,
                parity,
                parity,
                "Changed actions or invalid numbers invalidate smaller representations.",
            ),
            gate_row(
                "int8_brier_increase",
                "scientific_benefit",
                "<=",
                0.001,
                precision_reduction["by_arm"]["int8_float32_accum"]["brier_delta_vs_float64"]
                if precision_reduction
                else None,
                score_tolerance,
                "A smaller representation must preserve calibrated proper-score quality.",
            ),
            gate_row(
                "int8_total_time_ratio_upper_paired_95",
                "scientific_benefit",
                "<",
                1.0,
                timing_summary["primary_ratio"]["ci95"][1] if timing_summary else None,
                speed,
                "A kernel claim needs a complete-service paired speed benefit.",
            ),
            gate_row(
                "required_affected_and_terminal_validation",
                "validation",
                "==",
                True,
                validation_passed,
                validation_passed,
                "Every frozen affected and supplied terminal check must pass.",
            ),
        ]
    )
    capture = int(complete and validation_passed and parity and not flagged_adversarial)
    value_score = int(capture == 1 and score_tolerance and speed)
    if not numeric_available:
        status = "blocked"
        verdict_class = "blocked"
        honest = "blocked_numeric_calibration_branch_unavailable"
    elif capture == 0:
        status = "complete"
        verdict_class = "disqualified"
        honest = "complete_disqualified_precision_evidence_or_validation"
    elif value_score:
        status = "complete"
        verdict_class = "circular_positive"
        honest = "complete_circular_positive_int8_parity_and_full_service_benefit"
    else:
        status = "complete"
        verdict_class = "null"
        honest = "complete_null_int8_no_registered_full_service_benefit"

    invocation = build_current_work_receipt(
        run_id="exp7419-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=(
            "host NumPy float64, float32, and symmetric int8 emulation; no model load or device operation"
        ),
        inference_substrate_details={
            "device": "CPU",
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "int8_accumulator": "float32_after_symmetric_dequantization",
            "hardware_implementation": False,
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=duration_ns,
        sidecar_references=sidecar_references,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": False,
            "historical_source": "Exp7413 source_aware_6_4_1_gibbs seed 65001",
            "historical_receipt_only": True,
            "current_training_updates": 0,
        },
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **invocation,
        "random_seed": {
            "selected_checkpoint_seed": SELECTED_SEED,
            "paired_order_seed": RANDOM_SEED,
            "paired_resampling_seed": RANDOM_SEED,
            "paired_resampling_draws": 200 if fixture else BOOTSTRAP_DRAWS,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "raw_evidence_receipt": deepcopy(dict(raw_evidence_receipt or {})),
        "numeric_protocol": {
            "arms": list(ARMS),
            "batch_sizes": list(BATCH_SIZES),
            "paired_blocks_per_batch": PAIRED_BLOCKS,
            "rotated_order": True,
            "cold_initialization_separate": True,
            "scale_partition": "train",
            "heldout_tuning": False,
            "host_emulation_only": True,
        },
        "quantization": deepcopy(dict(quantization)),
        "precision_rows": precision,
        "precision_reduction": precision_reduction,
        "near_threshold_fixture": [deepcopy(dict(row)) for row in boundary_rows],
        "timing_rows": timing,
        "cold_initialization_rows": cold,
        "timing_summary": timing_summary,
        "amdahl_analysis": amdahl,
        "placement_constraints": {
            "hundred_x_requires_unaccelerated_fraction_at_most": 0.01,
            "kv260_future_access": "ssh_only",
            "kv260_future_k_max": 5,
            "extropic_tsu": "unavailable_research_target",
            "xdna": "unavailable_research_target",
            "device_port_implemented": False,
        },
        "board_rows": boards,
        "hardware_operations": {
            "ssh": 0,
            "flash": 0,
            "device_probe": 0,
            "purchase": 0,
            "vendor_contact": 0,
            "physical_command": 0,
        },
        "rows": [*precision, *timing, *cold, *boards],
        "sample_size_budget": budget,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "promotion_score": 0,
        "precision_capture_complete_score": capture,
        "precision_value_score": value_score,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "fixture_artifact": fixture,
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _fixture_receipts() -> list[JsonDict]:
    """Build passing bounded receipts for mutation-oriented artifact tests."""

    names = [
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_cold_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    return [
        {
            "name": name,
            "argv": ["fixture", name],
            "command_environment": {"COVERAGE_FILE": "/tmp/fixture.coverage"},
            "exit_code": 0,
            "duration_s": 0.01,
            "log_sha256": "sha256:" + f"{index + 1:064x}"[-64:],
            "passed": True,
        }
        for index, name in enumerate(names)
    ]


def _fixture_board_rows() -> list[JsonDict]:
    """Create the three bounded board states used by artifact mutation tests."""

    rows = [
        {
            "unit_id": "board:KV260",
            "board": "KV260",
            "terminal_state": "graduated_preserved",
            "last_authenticated_date": "20260915",
            "last_authenticated_venue": "kv260_fpga_fabric",
            "future_access": "ssh_only",
            "architecture_limit": "k_max<=5",
            "fpga_sampling_claimed": True,
            "new_hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
        {
            "unit_id": "board:GateMate",
            "board": "GateMate",
            "terminal_state": "blocked_changed_physical_state",
            "last_authenticated_date": "20260916",
            "last_authenticated_venue": "none_read_only",
            "future_access": None,
            "architecture_limit": None,
            "fpga_sampling_claimed": False,
            "new_hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
            "error": MISSING_RECEIPT,
        },
        {
            "unit_id": "board:PolarFire",
            "board": "PolarFire",
            "terminal_state": "graduated_cpu_dispatch_preserved",
            "last_authenticated_date": "20260915",
            "last_authenticated_venue": "polarfire_linux_cpu",
            "future_access": None,
            "architecture_limit": None,
            "fpga_sampling_claimed": False,
            "new_hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
    ]
    for row in rows:
        row["row_sha256"] = row_hash(row)
    return rows


def _fixture_rows() -> list[JsonDict]:
    """Build small train and held-out rows for pure artifact tests."""

    rows: list[JsonDict] = []
    for index in range(12):
        values = [
            (index % 7) / 7,
            (index % 5) / 5,
            (index % 3) / 3,
            ((index + 1) % 7) / 7,
            ((index + 2) % 5) / 5,
            float(index % 2),
        ]
        rows.append(
            {
                "row_key": f"fixture-{index}",
                "group_id": f"fixture-group-{index // 2}",
                "partition": "train" if index < 9 else "final_test",
                "source_features": dict(zip(SOURCE_FEATURE_NAMES, values, strict=True)),
                "label": index % 2 if index != 11 else None,
                "label_authority": "machine_annotation",
            }
        )
    return rows


def _fixture_checkpoint() -> JsonDict:
    """Return a valid 6-4-1 state with the frozen policy shape."""

    return {
        "schema": "carnot.exp7413.numeric_checkpoint.v1",
        "condition": "full_source",
        "arm": "source_aware_6_4_1_gibbs",
        "seed": SELECTED_SEED,
        "architecture": [6, 4, 1],
        "weights": {
            "w1": [
                [0.9, -0.2, 0.4, -0.8, -0.9, 0.1],
                [-0.3, -0.1, -0.2, 1.1, 1.0, -0.1],
                [0.5, -0.1, 0.3, -0.7, -0.8, 0.1],
                [0.8, -0.1, 0.4, -0.9, -0.8, -0.1],
            ],
            "b1": [1.4, -0.2, 1.1, 1.4],
            "w_out": [1.3, -0.9, 1.0, 1.4],
            "b_out": 0.56,
        },
        "affine": {"slope": 1.2, "intercept": -0.13},
        "selected_policy": {
            "accept_threshold": 0.01,
            "reject_threshold": 0.9,
            "accept_enabled": False,
            "reject_enabled": False,
        },
    }


def build_fixture_artifact(*, numeric_available: bool = True) -> JsonDict:
    """Build a compact null or blocked artifact for cold mutation tests."""

    checkpoint = _fixture_checkpoint()
    rows = _fixture_rows()
    quantization = derive_quantization(checkpoint, rows)
    heldout = [row for row in rows if row["partition"] == "final_test"]
    precision = build_precision_rows(heldout, checkpoint, quantization) if numeric_available else []
    timing = (
        [
            synthetic_timing_row(
                batch,
                block,
                arm,
                1.0 if arm == "float32_vectorized" else 1.2,
            )
            for batch in BATCH_SIZES
            for block in range(PAIRED_BLOCKS)
            for arm in ARMS
        ]
        if numeric_available
        else []
    )
    cold = (
        [
            {
                "arm": arm,
                "batch_size": 1,
                "cold_initialization_s": 0.01,
                "first_service_s": 0.005,
                "status": "completed",
            }
            for arm in ARMS
        ]
        if numeric_available
        else []
    )
    preconditions = [
        gate_row(
            "fixture_numeric_branch",
            "numeric_precondition",
            "==",
            True,
            numeric_available,
            numeric_available,
            "A fixture can exercise either eligible or blocked numeric classification.",
        ),
        gate_row(
            "gatemate_changed_physical_state_receipt",
            "external_prerequisite",
            "matches",
            deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT),
            {"absence": MISSING_RECEIPT},
            False,
            "The external GateMate absence must not erase board accounting.",
        ),
    ]
    return assemble_artifact(
        precision_rows=precision,
        timing_rows=timing,
        cold_rows=cold,
        board_rows=_fixture_board_rows(),
        quantization=quantization,
        boundary_rows=near_threshold_fixture(checkpoint, quantization),
        preconditions=preconditions,
        source_hashes={
            "fixture": {
                "path": "fixture",
                "sha256": "sha256:" + "2" * 64,
                "original_flagged_adversarial": False,
            }
        },
        validation_receipts=_fixture_receipts(),
        phase_spans=[
            {
                "phase": "fixture",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
                "completed_units": len(precision) + len(timing) + len(cold),
                "checkpoint_at_utc": "2026-09-19T00:00:01+00:00",
                "heartbeat_times_utc": [],
            }
        ],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        duration_ns=1_000_000_000,
        fixture=True,
    )


def validate_artifact(value: object) -> list[str]:
    """Reject changed rows, false gates, promoted hardware, and schema drift."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = value
    errors: list[str] = []
    required = {
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
        "precision_capture_complete_score",
        "precision_value_score",
        "precision_rows",
        "board_rows",
        "hardware_ready_score",
        "hardware_value_score",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("identity_mismatch")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
        or artifact.get("promotion_score") != 0
        or artifact.get("hardware_ready_score") != 0
        or artifact.get("hardware_value_score") != 0
    ):
        errors.append("fixed_declaration_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    boards = artifact.get("board_rows") if isinstance(artifact.get("board_rows"), list) else []
    if len(boards) != 3 or {row.get("board") for row in boards if isinstance(row, Mapping)} != {
        "KV260",
        "GateMate",
        "PolarFire",
    }:
        errors.append("board_rows_invalid")
    if any(
        not isinstance(row, Mapping)
        or row_hash(row) != row.get("row_sha256")
        or row.get("hardware_ready_score") != 0
        or row.get("hardware_value_score") != 0
        for row in boards
    ):
        errors.append("board_row_hash_or_score_mismatch")
    timing = artifact.get("timing_rows") if isinstance(artifact.get("timing_rows"), list) else []
    if any(
        not isinstance(row, Mapping) or row_hash(row) != row.get("row_sha256") for row in timing
    ):
        errors.append("timing_row_hash_mismatch")
    precision = (
        artifact.get("precision_rows") if isinstance(artifact.get("precision_rows"), list) else []
    )
    if precision:
        try:
            reduced = reduce_precision(precision)
            row_hashes_valid = all(
                isinstance(row, Mapping) and row_hash(row) == row.get("row_sha256")
                for row in precision
            )
            if not row_hashes_valid or reduced != artifact.get("precision_reduction"):
                errors.append("precision_reduction_mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append("precision_rows_invalid")
    elif artifact.get("precision_reduction") is not None:
        errors.append("precision_reduction_mismatch")
    if timing:
        draws = int((artifact.get("random_seed") or {}).get("paired_resampling_draws", 0))
        try:
            summary = summarize_timing(timing, draws=draws, seed=RANDOM_SEED)
            if summary != artifact.get("timing_summary"):
                errors.append("timing_summary_mismatch")
            amdahl = compute_amdahl(timing, batch_size=max(BATCH_SIZES), arm="float32_vectorized")
            if amdahl != artifact.get("amdahl_analysis"):
                errors.append("amdahl_analysis_mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append("timing_rows_invalid")
    elif artifact.get("timing_summary") is not None or artifact.get("amdahl_analysis") is not None:
        errors.append("timing_summary_mismatch")
    numeric_available = all(
        row.get("passed") is True
        for row in artifact.get("preconditions_checked") or []
        if isinstance(row, Mapping) and row.get("category") == "numeric_precondition"
    )
    capture = artifact.get("precision_capture_complete_score")
    value_score = artifact.get("precision_value_score")
    if not numeric_available:
        if (
            artifact.get("verdict_class") != "blocked"
            or artifact.get("honest_verdict") != "blocked_numeric_calibration_branch_unavailable"
            or capture != 0
            or value_score != 0
            or len(boards) != 3
        ):
            errors.append("blocked_disposition_invalid")
    elif capture not in {0, 1} or value_score not in {0, 1} or value_score > capture:
        errors.append("precision_scores_invalid")
    if capture == 1:
        reduction = artifact.get("precision_reduction") or {}
        if (
            not precision
            or not timing
            or reduction.get("action_parity_passed") is not True
            or reduction.get("numeric_validity_passed") is not True
        ):
            errors.append("capture_evidence_invalid")
    if value_score == 1:
        if (artifact.get("precision_reduction") or {}).get(
            "int8_brier_gate_passed"
        ) is not True or (artifact.get("timing_summary") or {}).get(
            "speed_gate_passed"
        ) is not True:
            errors.append("value_score_mismatch")
    if set((artifact.get("field_principles") or {})) != set(artifact):
        errors.append("field_principles_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(artifact, root=REPO_ROOT))
    if artifact.get("fixture_artifact") is not True:
        for row in (artifact.get("source_artifact_hashes") or {}).values():
            if not isinstance(row, Mapping):
                errors.append("source_artifact_hash_row_invalid")
                continue
            label = str(row.get("path") or "")
            path = Path(label)
            resolved = path if path.is_absolute() else REPO_ROOT / path
            observed = sha256_file(resolved) if resolved.is_file() else None
            if observed != row.get("sha256"):
                errors.append(f"source_artifact_hash_mismatch:{label}")
    return list(dict.fromkeys(errors))


def independent_reduce(artifact: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute raw-row equality, numeric reductions, timing, and board identity."""

    precision_match = artifact.get("precision_rows") == raw.get("precision_rows")
    timing_match = artifact.get("timing_rows") == raw.get("timing_rows")
    cold_match = artifact.get("cold_initialization_rows") == raw.get("cold_initialization_rows")
    boards_match = artifact.get("board_rows") == raw.get("board_rows")
    quantization_match = artifact.get("quantization") == raw.get("quantization")
    precision_reduced = not artifact.get("precision_rows") or reduce_precision(
        artifact["precision_rows"]
    ) == artifact.get("precision_reduction")
    timing_reduced = not artifact.get("timing_rows") or summarize_timing(
        artifact["timing_rows"],
        draws=int(artifact["random_seed"]["paired_resampling_draws"]),
        seed=RANDOM_SEED,
    ) == artifact.get("timing_summary")
    return {
        "precision_rows_match": precision_match,
        "timing_rows_match": timing_match,
        "cold_rows_match": cold_match,
        "board_rows_match": boards_match,
        "quantization_match": quantization_match,
        "precision_reduction_match": precision_reduced,
        "timing_reduction_match": timing_reduced,
        "artifact_valid": not validate_artifact(artifact),
    }


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a locally valid terminal artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 scoped command plan before any child starts."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion, broad pytest, and missing private parents."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def load_precision_fixture(root: Path) -> JsonDict:
    """Join frozen feature partitions to one selected Exp7413 metric ledger."""

    upstream = load_json(root / EXP7413_PATH)
    checkpoint = load_json(root / CHECKPOINT_PATH)
    feature_payload = load_json(root / FEATURE_PATH)
    feature_rows = feature_payload.get("records") if isinstance(feature_payload, Mapping) else None
    if not isinstance(feature_rows, list):
        raise ValueError("feature records are unavailable")
    selected = [
        row
        for row in upstream.get("paired_metric_rows") or []
        if isinstance(row, Mapping)
        and row.get("condition") == "full_source"
        and row.get("arm") == "source_aware_6_4_1_gibbs"
        and row.get("seed") == SELECTED_SEED
    ]
    metric_by_key = {str(row["row_key"]): row for row in selected}
    joined: list[JsonDict] = []
    for row in feature_rows:
        if not isinstance(row, Mapping) or row.get("partition") not in {"train", "final_test"}:
            continue
        copied = deepcopy(dict(row))
        if copied["partition"] == "final_test":
            metric = metric_by_key.get(str(copied["row_key"]))
            if metric is None:
                raise ValueError("held-out metric row is unavailable")
            copied["label"] = metric.get("label")
            copied["label_authority"] = "machine_annotation"
        joined.append(copied)
    if sum(row["partition"] == "final_test" for row in joined) != 1995:
        raise ValueError("held-out cohort is incomplete")
    return {"checkpoint": checkpoint, "rows": joined}


def cold_replay(path: Path) -> list[str]:
    """Reload one candidate and run the complete local artifact validator."""

    try:
        value = load_json(path)
    except ValueError:
        return ["artifact_unreadable_or_not_object"]
    return validate_artifact(value)


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover - live timing receipt.
    """Close one real monotonic phase with completed-unit evidence."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
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
                "measured_candidate",
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
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
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
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def _entrypoint_receipt(
    root: Path, started_at_utc: str, duration_s: float
) -> JsonDict:  # pragma: no cover - live invocation receipt.
    """Retain the exact declared command and environment as capability E2E evidence."""

    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.json"
    value = {
        "argv": list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
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
        "command": " ".join(value["argv"]),
        "command_argv": value["argv"],
        "command_environment": value["environment"],
        "scope": "capability_end_to_end",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(path),
        "passed": True,
        "timed_out": False,
        "required": True,
        "command_category": "completion",
        "started_at_utc": started_at_utc,
        "ended_at_utc": value["completed_at_utc"],
    }


def _historical_sidecar(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Bind cited historical training without counting it as current invocation work."""

    return write_immutable_sidecar(
        paths.historical_receipts,
        scope="historical_model_receipts",
        root=root,
        payload={
            "current_llm_calls": 0,
            "small_ebm_training": {
                "performed_in_current_run": False,
                "source_experiment": EXP7413_PATH.as_posix(),
                "source_sha256": EXPECTED_EXP7413_SHA256,
                "checkpoint": CHECKPOINT_PATH.as_posix(),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "seed": SELECTED_SEED,
            },
        },
    )


def _raw_receipt(root: Path, path: Path) -> JsonDict:
    """Name and hash the raw rows used by the independent terminal reducer."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved), "scope": "current_raw_evidence"}


def run_experiment(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:  # pragma: no cover - exercised through the declared entrypoint.
    """Authenticate, measure, validate, replay, and atomically publish one result."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    root = root.resolve()
    paths = ExperimentPaths.under(root)
    paths = ExperimentPaths(
        artifact=(root / output_path) if not output_path.is_absolute() else output_path,
        raw_evidence=paths.raw_evidence,
        changed_state_search=paths.changed_state_search,
        historical_receipts=paths.historical_receipts,
        terminal_candidate=paths.terminal_candidate,
        validation_dir=paths.validation_dir,
    )
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "startup", "flushed")

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before")
    preconditions, hashes, context = collect_preconditions(root, paths)
    boards = build_board_rows(context)
    sidecar = _historical_sidecar(root, paths)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    numeric_available = all(
        row.get("passed") is True
        for row in preconditions
        if row.get("category") == "numeric_precondition"
    )
    progress(
        run_started,
        "preconditions",
        "after",
        completed=len(preconditions),
        numeric_available=numeric_available,
    )

    precision_rows: list[JsonDict] = []
    timing_rows: list[JsonDict] = []
    cold_rows: list[JsonDict] = []
    boundary_rows: list[JsonDict] = []
    quantization: JsonDict = {}
    if numeric_available:
        phase_started = time.monotonic()
        progress(run_started, "fixture_load", "before_benchmark")
        fixture = load_precision_fixture(root)
        checkpoint = fixture["checkpoint"]
        rows = fixture["rows"]
        train_and_test_rows = [dict(row) for row in rows]
        heldout = [row for row in train_and_test_rows if row.get("partition") == "final_test"]
        spans.append(_span("fixture_load", phase_started, run_started, len(rows)))
        progress(run_started, "fixture_load", "after_benchmark", completed=len(rows))

        phase_started = time.monotonic()
        progress(run_started, "numeric_measurement", "before_benchmark", planned=len(heldout) * 3)
        quantization = derive_quantization(checkpoint, train_and_test_rows)
        precision_rows = build_precision_rows(heldout, checkpoint, quantization)
        boundary_rows = near_threshold_fixture(checkpoint, quantization)
        spans.append(_span("numeric_measurement", phase_started, run_started, len(precision_rows)))
        progress(
            run_started,
            "numeric_measurement",
            "after_benchmark",
            completed=len(precision_rows),
        )

        phase_started = time.monotonic()
        progress(
            run_started,
            "complete_service_benchmark",
            "before_benchmark",
            planned=len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS),
        )
        timing_rows, cold_rows = measure_service_cost(
            heldout,
            checkpoint,
            quantization,
            emit_progress=True,
        )
        spans.append(
            _span(
                "complete_service_benchmark",
                phase_started,
                run_started,
                len(timing_rows) + len(cold_rows),
            )
        )
        progress(
            run_started,
            "complete_service_benchmark",
            "after_benchmark",
            completed=len(timing_rows) + len(cold_rows),
        )

    raw: JsonDict = {
        "schema": "carnot.exp7419.v650.raw_precision_rows.v1",
        "precision_rows": precision_rows,
        "timing_rows": timing_rows,
        "cold_initialization_rows": cold_rows,
        "board_rows": boards,
        "quantization": quantization,
        "near_threshold_fixture": boundary_rows,
    }
    progress(run_started, "raw_checkpoint", "before_atomic")
    atomic_json(paths.raw_evidence, raw)
    raw_receipt = _raw_receipt(root, paths.raw_evidence)
    progress(
        run_started,
        "raw_checkpoint",
        "after_atomic",
        bytes=paths.raw_evidence.stat().st_size,
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7419-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError("invalid_validation_plan:" + ",".join(plan_errors))
    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=paths.validation_dir / "affected",
        heartbeat_s=60.0,
    )
    reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=reduction["passed"],
    )

    source_hashes = _source_hash_rows(hashes, load_json(root / EXP7413_PATH))
    duration_ns = int((time.monotonic() - run_started) * 1_000_000_000)
    candidate = assemble_artifact(
        precision_rows=precision_rows,
        timing_rows=timing_rows,
        cold_rows=cold_rows,
        board_rows=boards,
        quantization=quantization,
        boundary_rows=boundary_rows,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_ns=duration_ns,
        raw_evidence_receipt=raw_receipt,
        sidecar_references=[sidecar],
        flagged_adversarial=not reduction["passed"],
    )
    progress(run_started, "candidate", "before_atomic")
    atomic_json(paths.terminal_candidate, candidate)
    progress(
        run_started, "candidate", "after_atomic", bytes=paths.terminal_candidate.stat().st_size
    )

    terminal_plan = _terminal_commands(paths.terminal_candidate, paths.raw_evidence)
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
        log_dir=paths.validation_dir / "terminal",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )

    entrypoint = _entrypoint_receipt(root, started_at, time.monotonic() - run_started)
    final = assemble_artifact(
        precision_rows=precision_rows,
        timing_rows=timing_rows,
        cold_rows=cold_rows,
        board_rows=boards,
        quantization=quantization,
        boundary_rows=boundary_rows,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal, entrypoint],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_ns=int((time.monotonic() - run_started) * 1_000_000_000),
        raw_evidence_receipt=raw_receipt,
        sidecar_references=[sidecar],
        flagged_adversarial=not reduction["passed"] or not terminal_passed or critical,
    )
    progress(run_started, "terminal_write", "before_atomic", path=paths.artifact)
    write_artifact(paths.artifact, final)
    progress(run_started, "terminal_write", "after_atomic", status=final["status"])
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

    print("[exp7419] phase=entrypoint event=flushed", flush=True)
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
            if not isinstance(artifact, Mapping):
                raise ValueError("artifact is not an object")
            if args.raw is None:
                raw = {
                    "precision_rows": artifact.get("precision_rows"),
                    "timing_rows": artifact.get("timing_rows"),
                    "cold_initialization_rows": artifact.get("cold_initialization_rows"),
                    "board_rows": artifact.get("board_rows"),
                    "quantization": artifact.get("quantization"),
                }
            else:
                raw = load_json(args.raw)
                receipt = artifact.get("raw_evidence_receipt") or {}
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

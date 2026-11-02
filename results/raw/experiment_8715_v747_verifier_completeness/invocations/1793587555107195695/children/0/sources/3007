"""Build a label-separated interface over historical V656 native readouts.

The module reads immutable captures and produces small CPU feature rows. It
does not load Qwen, fit an optimizer, or decide whether the proposed features
predict human labels.

Spec refs: REQ-VERIFY-7504 and SCENARIO-VERIFY-7504-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7504-v657-evidence-interface"
SCHEMA = "carnot.exp7504.v657.evidence_interface.v1"

SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
PROTOCOL_ARTIFACT = Path("results/experiment_7491_v656_window_protocol.json")
FIT_ARTIFACT = Path("results/experiment_7493_v656_window_fit_capture.json")
EVAL_ARTIFACT = Path("results/experiment_7494_v656_window_eval_capture.json")
RESULT_PATH = Path("results/experiment_7504_v657_evidence_interface.json")
RAW_DIR = Path("results/raw/experiment_7504_v657_evidence_interface")
MODULE_PATH = Path("python/carnot/experiment_7504_v657_evidence_interface.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7504_v657_evidence_interface.py")
TEST_PATH = Path("tests/python/test_experiment_7504_v657_evidence_interface.py")

PROTOCOL_RAW = Path("results/raw/experiment_7491_v656_window_protocol")
FIT_RAW = Path("results/raw/experiment_7493_v656_window_fit_capture")
EVAL_RAW = Path("results/raw/experiment_7494_v656_window_eval_capture")
FEATURE_PATH = RAW_DIR / "features.jsonl"
ACCESS_PATH = RAW_DIR / "access_exposure_manifest.json"
NORMALIZATION_PATH = RAW_DIR / "training_normalization.json"
REQUIRED_INPUT_PATHS = (
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
    Path("python/carnot/experiment_7491_v656_window_protocol.py"),
    Path("python/carnot/experiment_7493_v656_window_fit_capture.py"),
    Path("python/carnot/experiment_7494_v656_window_eval_capture.py"),
    PROTOCOL_ARTIFACT,
    FIT_ARTIFACT,
    EVAL_ARTIFACT,
    Path("tests/python/test_experiment_7495_v656_window_calibration.py"),
    SPEC_PATH,
)
EXPECTED_ROLE_COUNTS = {
    "training": 176,
    "calibration_tuning": 60,
    "test": 116,
    "online": 159,
}
OPTION_ORDERS = (
    ("supported", "contains_unsupported"),
    ("contains_unsupported", "supported"),
)
FEATURE_NAMES = (
    "mean_whole_log_odds",
    "min_window_unsupported_probability",
    "mean_window_unsupported_probability",
    "max_window_unsupported_probability",
    "population_sd_window_unsupported_probability",
    "max_window_order_disagreement",
    "whole_order_disagreement",
    "log1p_window_count",
    "log1p_response_bytes",
    "log1p_source_bytes",
)
FEATURE_VERSION = "carnot-v657-window-evidence-v1"
FROZEN_SETTINGS: JsonDict = {
    "static_primary": {
        "metric": "Brier",
        "candidate": "conditional_gibbs",
        "controls": ["identical_feature_logistic", "whole_only_gibbs"],
        "minimum_brier_improvement": 0.01,
    },
    "online_primary": {
        "delay": 8,
        "audit_fraction": 0.25,
        "block_release": 8,
        "seeds": [656201, 656202, 656203, 656204, 656205],
    },
}
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class EvidenceInterfaceError(ValueError):
    """Reject evidence that could mix roles, labels, identities, or call cells."""


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so a text or sidecar mutation cannot be hidden."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash one UTF-8 string without normalization or implicit stripping."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large historical shards stay bounded."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: object) -> str:
    """Hash stable JSON bytes so frozen settings have one portable identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def load_json(path: Path) -> JsonDict:
    """Read one JSON object and reject arrays or malformed producer bytes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # pragma: no cover - malformed external JSON.
        raise EvidenceInterfaceError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read nonblank JSONL objects without altering any string value."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():  # pragma: no cover - production shards have no blanks.
                continue
            value = json.loads(line)
            if not isinstance(value, dict):  # pragma: no cover - malformed external JSONL.
                raise EvidenceInterfaceError(f"jsonl_object_required:{path}:{number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write canonical rows so later readers can bind exact bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _probability(row: Mapping[str, Any]) -> float:
    """Return the semantic unsupported probability after mapping display order."""

    order = row.get("option_order")
    if not isinstance(order, list) or tuple(order) not in OPTION_ORDERS:  # pragma: no cover
        raise EvidenceInterfaceError("option_order_invalid")
    mapping = row.get("label_to_option_id")
    expected_mapping = {" A": order[0], " B": order[1]}
    if mapping is not None and mapping != expected_mapping:  # pragma: no cover
        raise EvidenceInterfaceError("option_mapping_invalid")
    probabilities = row.get("probabilities_by_option_id")
    if not isinstance(probabilities, Mapping):  # pragma: no cover
        raise EvidenceInterfaceError("semantic_probabilities_missing")
    if set(probabilities) != {"supported", "contains_unsupported"}:  # pragma: no cover
        raise EvidenceInterfaceError("semantic_option_ids_invalid")
    probability = probabilities.get("contains_unsupported")
    supported = probabilities.get("supported")
    if not isinstance(probability, (int, float)) or not isinstance(  # pragma: no cover
        supported, (int, float)
    ):
        raise EvidenceInterfaceError("semantic_probability_invalid")
    if not math.isfinite(probability) or not math.isfinite(supported):  # pragma: no cover
        raise EvidenceInterfaceError("semantic_probability_nonfinite")
    if not 0.0 < probability < 1.0 or not math.isclose(  # pragma: no cover
        probability + supported, 1.0, abs_tol=1e-9
    ):
        raise EvidenceInterfaceError("semantic_probability_invalid")
    return float(probability)


def _validate_window(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    response_text: str,
) -> None:
    """Bind a focused call to its original qualifier-preserving byte slice."""

    fields = ("window_index", "byte_start", "byte_end", "window_sha256")
    if any(row.get(field) != expected.get(field) for field in fields):  # pragma: no cover
        raise EvidenceInterfaceError("window_join_mismatch")
    start = expected.get("byte_start")
    end = expected.get("byte_end")
    if (
        not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end
    ):  # pragma: no cover
        raise EvidenceInterfaceError("window_offsets_invalid")
    response = response_text.encode("utf-8")
    if end > len(response) or sha256_bytes(response[start:end]) != expected.get(  # pragma: no cover
        "window_sha256"
    ):
        raise EvidenceInterfaceError("window_bytes_mismatch")


def _pair_probabilities(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Require one call for each order and return probabilities in frozen order."""

    by_order: dict[tuple[str, str], float] = {}
    for row in rows:
        order_value = row.get("option_order")
        order = tuple(order_value) if isinstance(order_value, list) else ()
        if order in by_order:  # pragma: no cover - global call IDs reject this first.
            raise EvidenceInterfaceError("duplicate_call")
        by_order[order] = _probability(row)
    missing = set(OPTION_ORDERS) - set(by_order)
    if missing or len(by_order) != 2:
        raise EvidenceInterfaceError("missing_option_order")
    return by_order[OPTION_ORDERS[0]], by_order[OPTION_ORDERS[1]]


def _feature_vector(
    whole: tuple[float, float], windows: Sequence[tuple[float, float]], predictor: Mapping[str, Any]
) -> list[float]:
    """Apply the frozen ten-feature equation without consulting any label."""

    if not windows:  # pragma: no cover - a sealed eligible group always has a window.
        raise EvidenceInterfaceError("focused_windows_required")
    window_means = [(left + right) / 2.0 for left, right in windows]
    whole_logits = [math.log(value / (1.0 - value)) for value in whole]
    response = str(predictor.get("response_text", "")).encode("utf-8")
    source = str(predictor.get("source_text", "")).encode("utf-8")
    return [
        statistics.fmean(whole_logits),
        min(window_means),
        statistics.fmean(window_means),
        max(window_means),
        statistics.pstdev(window_means),
        max(abs(left - right) for left, right in windows),
        abs(whole[0] - whole[1]),
        math.log1p(len(windows)),
        math.log1p(len(response)),
        math.log1p(len(source)),
    ]


def build_feature_rows(
    native_rows: Sequence[Mapping[str, Any]],
    predictors: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join complete label-free calls into one proposed feature row per source."""

    predictor_by_group: dict[str, Mapping[str, Any]] = {}
    for predictor in predictors:
        group = predictor.get("group_id")
        if not isinstance(group, str) or group in predictor_by_group:  # pragma: no cover
            raise EvidenceInterfaceError("predictor_group_duplicate")
        predictor_by_group[group] = predictor
    windows_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for window in window_rows:
        group = window.get("group_id")
        index = window.get("window_index")
        if not isinstance(group, str) or not isinstance(index, int):  # pragma: no cover
            raise EvidenceInterfaceError("window_identity_invalid")
        key = (group, index)
        if key in windows_by_key:  # pragma: no cover
            raise EvidenceInterfaceError("window_duplicate")
        windows_by_key[key] = window

    calls_by_cell: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    control_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    call_ids: set[str] = set()
    for row in native_rows:
        if row.get("eligible") is not True or row.get("disposition") != "complete":
            continue
        if row.get("gold_label") is not None:  # pragma: no cover
            raise EvidenceInterfaceError("native_label_leakage")
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or call_id in call_ids:
            raise EvidenceInterfaceError("duplicate_call")
        call_ids.add(call_id)
        group = row.get("group_id")
        arm = row.get("arm")
        index = row.get("window_index")
        if not isinstance(group, str) or arm not in {  # pragma: no cover
            "whole_response",
            "focused_window",
            "source_derangement_control",
        }:
            raise EvidenceInterfaceError("call_identity_invalid")
        if arm == "source_derangement_control":
            source_group = row.get("source_group_id")
            if not isinstance(source_group, str) or index is not None:  # pragma: no cover
                raise EvidenceInterfaceError("derangement_identity_invalid")
            control_by_cell[(group, source_group)].append(row)
            continue
        if arm == "whole_response" and index is not None:  # pragma: no cover
            raise EvidenceInterfaceError("whole_window_index_invalid")
        if arm == "focused_window" and not isinstance(index, int):  # pragma: no cover
            raise EvidenceInterfaceError("focused_window_index_invalid")
        calls_by_cell[(group, str(arm), index)].append(row)

    output: list[JsonDict] = []
    source_hashes: set[str] = set()
    groups = sorted({key[0] for key in calls_by_cell})
    for group in groups:
        predictor = predictor_by_group.get(group)
        if predictor is None:  # pragma: no cover
            raise EvidenceInterfaceError("predictor_missing")
        role = predictor.get("role")
        if role not in EXPECTED_ROLE_COUNTS:  # pragma: no cover
            raise EvidenceInterfaceError("role_invalid")
        source_text = predictor.get("source_text")
        response_text = predictor.get("response_text")
        source_hash = predictor.get("source_hash")
        if not all(  # pragma: no cover
            isinstance(value, str) for value in (source_text, response_text, source_hash)
        ):
            raise EvidenceInterfaceError("predictor_text_identity_invalid")
        if source_hash in source_hashes:  # pragma: no cover
            raise EvidenceInterfaceError("normalized_source_duplicate")
        source_hashes.add(str(source_hash))
        group_calls = [
            row for key, rows in calls_by_cell.items() if key[0] == group for row in rows
        ]
        for row in group_calls:
            if row.get("role") != role or row.get("source_group_id") != group:
                raise EvidenceInterfaceError("role_mismatch")
            if row.get("source_sha256") != sha256_text(str(source_text)):
                raise EvidenceInterfaceError("source_hash_mismatch")
            if row.get("response_sha256") != sha256_text(str(response_text)):  # pragma: no cover
                raise EvidenceInterfaceError("response_hash_mismatch")
            if predictor.get("response_hash") != row.get("response_sha256"):  # pragma: no cover
                raise EvidenceInterfaceError("response_hash_mismatch")
        whole = _pair_probabilities(calls_by_cell.get((group, "whole_response", None), []))
        indices = sorted(
            key[2] for key in calls_by_cell if key[0] == group and key[1] == "focused_window"
        )
        if indices != list(range(len(indices))):  # pragma: no cover
            raise EvidenceInterfaceError("window_sequence_invalid")
        window_probabilities: list[tuple[float, float]] = []
        for index in indices:
            assert isinstance(index, int)
            rows = calls_by_cell[(group, "focused_window", index)]
            expected_window = windows_by_key.get((group, index))
            if expected_window is None:  # pragma: no cover
                raise EvidenceInterfaceError("window_join_missing")
            for row in rows:
                _validate_window(row, expected_window, str(response_text))
            window_probabilities.append(_pair_probabilities(rows))
        features = _feature_vector(whole, window_probabilities, predictor)
        output.append(
            {
                "group_id": group,
                "role": role,
                "source_hash": source_hash,
                "response_hash": predictor.get("response_hash"),
                "corpus": predictor.get("corpus"),
                "features": features,
                "raw_whole_expectation": statistics.fmean(whole),
                "raw_max_window_probability": max(
                    statistics.fmean(pair) for pair in window_probabilities
                ),
                "window_count": len(window_probabilities),
                "response_bytes": len(str(response_text).encode("utf-8")),
                "source_bytes": len(str(source_text).encode("utf-8")),
                "feature_version": FEATURE_VERSION,
                "proposed_features_not_findings": True,
            }
        )
    for (group, source_group), rows in control_by_cell.items():
        response_predictor = predictor_by_group.get(group)
        source_predictor = predictor_by_group.get(source_group)
        if response_predictor is None or source_predictor is None:  # pragma: no cover
            raise EvidenceInterfaceError("derangement_predictor_missing")
        for row in rows:
            if row.get("role") != response_predictor.get("role"):  # pragma: no cover
                raise EvidenceInterfaceError("role_mismatch")
            if row.get("source_sha256") != sha256_text(  # pragma: no cover
                str(source_predictor.get("source_text"))
            ):
                raise EvidenceInterfaceError("source_hash_mismatch")
            if row.get("response_sha256") != sha256_text(  # pragma: no cover
                str(response_predictor.get("response_text"))
            ):
                raise EvidenceInterfaceError("response_hash_mismatch")
        _pair_probabilities(rows)
    return output


def read_mode(
    feature_path: Path,
    evaluator_path: Path,
    *,
    mode: str,
    prediction_path: Path | None = None,
    prediction_sha256: str | None = None,
) -> JsonDict:
    """Expose labels only to the role set authorized for one reader mode."""

    if mode not in {"fit", "predict", "evaluate"}:  # pragma: no cover
        raise EvidenceInterfaceError("reader_mode_invalid")
    features = load_jsonl(feature_path)
    allowed_roles = {
        "fit": {"training", "calibration_tuning"},
        "predict": {"training", "calibration_tuning", "test", "online"},
        "evaluate": {"test"},
    }[mode]
    selected = [deepcopy(row) for row in features if row.get("role") in allowed_roles]
    labels_opened: list[str] = []
    freeze_hash: str | None = None
    if mode == "evaluate":
        if prediction_path is None or prediction_sha256 is None or not prediction_path.is_file():
            raise EvidenceInterfaceError("prediction_freeze_required")
        freeze_hash = sha256_file(prediction_path)
        if freeze_hash != prediction_sha256:
            raise EvidenceInterfaceError("prediction_hash_mismatch")
    if mode in {"fit", "evaluate"}:
        labels = {
            (row.get("group_id"), row.get("role")): row.get("label")
            for row in load_jsonl(evaluator_path)
            if row.get("role") in allowed_roles
        }
        for row in selected:
            key = (row.get("group_id"), row.get("role"))
            label = labels.get(key)
            if label not in {0, 1}:  # pragma: no cover
                raise EvidenceInterfaceError("authorized_label_missing")
            row["label"] = label
        labels_opened = sorted(allowed_roles)
    else:
        for row in selected:
            row.pop("label", None)
    return {
        "rows": selected,
        "freeze_sha256": canonical_hash(selected),
        "access_receipt": {
            "mode": mode,
            "feature_sha256": sha256_file(feature_path),
            "label_roles_opened": labels_opened,
            "held_out_labels_opened": bool(set(labels_opened) & {"test", "online"}),
            "prediction_freeze_sha256": freeze_hash,
        },
    }


def fit_normalization(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Fit location and scale only on training rows, before held-out labels."""

    training = [row for row in rows if row.get("role") == "training"]
    if not training:
        raise EvidenceInterfaceError("training_rows_required")
    matrix: list[list[float]] = []
    for row in training:
        values = row.get("features")
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):  # pragma: no cover
            raise EvidenceInterfaceError("feature_shape_invalid")
        if not all(  # pragma: no cover
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            raise EvidenceInterfaceError("feature_value_invalid")
        matrix.append([float(value) for value in values])
    columns = list(zip(*matrix, strict=True))
    means = [statistics.fmean(column) for column in columns]
    scales = [statistics.pstdev(column) for column in columns]
    safe_scales = [value if value > 0.0 else 1.0 for value in scales]
    value: JsonDict = {
        "version": "training-zscore-population-v1",
        "feature_names": list(FEATURE_NAMES),
        "fit_role": "training",
        "fit_group_count": len(training),
        "mean": means,
        "population_sd": scales,
        "safe_scale": safe_scales,
    }
    value["normalization_sha256"] = canonical_hash(value)
    return value


def feature_manifest() -> JsonDict:
    """Describe the frozen transform so downstream code cannot select features."""

    formulas = {
        FEATURE_NAMES[0]: "mean_o(log(p_unsupported[o] / (1 - p_unsupported[o])))",
        FEATURE_NAMES[1]: "min_w(mean_o(p_unsupported[w,o]))",
        FEATURE_NAMES[2]: "mean_w(mean_o(p_unsupported[w,o]))",
        FEATURE_NAMES[3]: "max_w(mean_o(p_unsupported[w,o]))",
        FEATURE_NAMES[4]: "population_sd_w(mean_o(p_unsupported[w,o]))",
        FEATURE_NAMES[5]: "max_w(abs(p_unsupported[w,order0] - p_unsupported[w,order1]))",
        FEATURE_NAMES[6]: "abs(p_unsupported[whole,order0] - p_unsupported[whole,order1])",
        FEATURE_NAMES[7]: "log1p(lossless_window_count)",
        FEATURE_NAMES[8]: "log1p(len(response_text.encode('utf-8')))",
        FEATURE_NAMES[9]: "log1p(len(source_text.encode('utf-8')))",
    }
    value: JsonDict = {
        "version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "formulas": formulas,
        "pair_reduction": "arithmetic_mean_probability_within_two_semantic_orders",
        "semantic_options": ["supported", "contains_unsupported"],
        "option_orders": [list(order) for order in OPTION_ORDERS],
        "raw_controls": ["raw_whole_expectation", "raw_max_window_probability"],
        "status": "proposed_features_not_findings",
    }
    value["transform_sha256"] = canonical_hash(value)
    return value


def _row_count(path: Path) -> int:
    """Count physical nonblank rows without parsing sealed evaluator content."""

    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _verify_shards(
    root: Path, raw_root: Path, shards: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Rehash each declared shard and retain exact observed operands."""

    receipts: list[JsonDict] = []
    for shard in shards:
        relative = shard.get("path")
        if not isinstance(relative, str):  # pragma: no cover
            raise EvidenceInterfaceError("raw_shard_path_invalid")
        path = root / raw_root / relative
        if not path.is_file():  # pragma: no cover
            raise EvidenceInterfaceError(f"raw_shard_missing:{path}")
        observed_hash = sha256_file(path)
        observed_rows = _row_count(path)
        expected_size = shard.get("size_bytes", shard.get("bytes"))
        passed = (
            observed_hash == shard.get("sha256")
            and observed_rows == shard.get("rows")
            and path.stat().st_size == expected_size
        )
        receipts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": shard.get("kind"),
                "expected_sha256": shard.get("sha256"),
                "observed_sha256": observed_hash,
                "expected_rows": shard.get("rows"),
                "observed_rows": observed_rows,
                "expected_bytes": expected_size,
                "observed_bytes": path.stat().st_size,
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in receipts):  # pragma: no cover
        raise EvidenceInterfaceError("raw_shard_authentication_failed")
    return receipts


def authenticate_capture_inputs(root: Path) -> JsonDict:
    """Authenticate terminal flags, original shards, roles, and model receipts."""

    paths = (PROTOCOL_ARTIFACT, FIT_ARTIFACT, EVAL_ARTIFACT)
    if missing := [  # pragma: no cover - the runner publishes this as blocked first.
        path.as_posix() for path in REQUIRED_INPUT_PATHS if not (root / path).is_file()
    ]:
        raise EvidenceInterfaceError(f"external_prerequisite_missing:{missing[0]}")
    protocol, fit, evaluation = (load_json(root / path) for path in paths)
    terminal_checks = {
        "protocol_ready": protocol.get("window_protocol_ready_score") == 1,
        "fit_ready": fit.get("window_fit_ready_score") == 1,
        "evaluation_ready": evaluation.get("window_evaluation_ready_score") == 1,
        "unflagged": all(
            value.get("flagged_adversarial") is False for value in (protocol, fit, evaluation)
        ),
        "terminal": all(
            str(value.get("honest_verdict", "")).startswith("complete_")
            for value in (protocol, fit, evaluation)
        ),
    }
    role_manifest = protocol.get("role_manifest")
    if not isinstance(role_manifest, Mapping):  # pragma: no cover
        raise EvidenceInterfaceError("role_manifest_missing")
    terminal_checks["role_counts"] = (
        role_manifest.get("eligible_role_counts") == EXPECTED_ROLE_COUNTS
    )
    model_keys = ("model_id", "model_sha256", "quantization")
    fit_model = fit.get("model_identity")
    eval_model = evaluation.get("model_identity")
    if not isinstance(fit_model, Mapping) or not isinstance(
        eval_model, Mapping
    ):  # pragma: no cover
        raise EvidenceInterfaceError("historical_model_receipt_missing")
    terminal_checks["model_identity"] = (
        all(fit_model.get(key) == eval_model.get(key) for key in model_keys)
        and fit_model.get("identity_authenticated") is True
        and eval_model.get("identity_authenticated") is True
        and fit_model.get("quantization") == "Q4_K_M"
    )
    terminal_checks["historical_forward_counts"] = (
        fit.get("invocation_counts", {}).get("forward_calls", {}).get("completed") == 1904
        and evaluation.get("invocation_counts", {}).get("forward_calls", {}).get("completed")
        == 2152
    )
    if not all(terminal_checks.values()):  # pragma: no cover
        failed = next(key for key, passed in terminal_checks.items() if not passed)
        raise EvidenceInterfaceError(f"upstream_contract_failed:{failed}")
    protocol_shards = role_manifest.get("raw_shards")
    if not isinstance(protocol_shards, Mapping):  # pragma: no cover
        raise EvidenceInterfaceError("protocol_raw_manifest_missing")
    protocol_receipts = _verify_shards(root, PROTOCOL_RAW, list(protocol_shards.values()))
    fit_shards = fit.get("raw_logit_shards")
    eval_shards = evaluation.get("raw_logit_shards")
    if not isinstance(fit_shards, list) or not isinstance(eval_shards, list):  # pragma: no cover
        raise EvidenceInterfaceError("capture_raw_manifest_missing")
    fit_receipts = _verify_shards(root, FIT_RAW, fit_shards)
    eval_receipts = _verify_shards(root, EVAL_RAW, eval_shards)
    return {
        "terminal_checks": terminal_checks,
        "resource_receipts": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(root / path),
                "bytes": (root / path).stat().st_size,
                "owner": "repository_or_declared_historical_input",
                "passed": True,
            }
            for path in REQUIRED_INPUT_PATHS
        ],
        "artifact_receipts": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(root / path),
                "bytes": (root / path).stat().st_size,
            }
            for path in paths
        ],
        "raw_shard_receipts": [*protocol_receipts, *fit_receipts, *eval_receipts],
        "historical_model_receipts": {
            "fit": deepcopy(dict(fit_model)),
            "evaluation": deepcopy(dict(eval_model)),
        },
        "prior_exposure_exclusions": deepcopy(protocol.get("prior_exposure_exclusions")),
    }


def _capture_rows(root: Path, raw_root: Path, artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Read only native logit shards after their artifact manifests pass."""

    output: list[JsonDict] = []
    for shard in artifact.get("raw_logit_shards", []):
        if isinstance(shard, Mapping) and shard.get("kind") == "raw_logits":
            output.extend(load_jsonl(root / raw_root / str(shard["path"])))
    return output


def build_real_feature_rows(root: Path) -> JsonDict:
    """Reduce historical captures while leaving the evaluator JSONL unopened."""

    authentication = authenticate_capture_inputs(root)
    fit = load_json(root / FIT_ARTIFACT)
    evaluation = load_json(root / EVAL_ARTIFACT)
    predictors = load_jsonl(root / PROTOCOL_RAW / "predictors.jsonl")
    windows = load_jsonl(root / PROTOCOL_RAW / "windows.jsonl")
    native = [
        *_capture_rows(root, FIT_RAW, fit),
        *_capture_rows(root, EVAL_RAW, evaluation),
    ]
    rows = build_feature_rows(native, predictors, windows)
    role_counts = dict(Counter(str(row["role"]) for row in rows))
    if len(rows) != 511 or role_counts != EXPECTED_ROLE_COUNTS:  # pragma: no cover
        raise EvidenceInterfaceError("eligible_role_reduction_mismatch")
    if len({row["source_hash"] for row in rows}) != len(rows):  # pragma: no cover
        raise EvidenceInterfaceError("normalized_source_duplicate")
    return {
        "rows": rows,
        "planned_groups": 520,
        "excluded_groups": 9,
        "role_counts": role_counts,
        "unique_normalized_source_hashes": len(rows),
        "evaluation_labels_opened": False,
        "authentication": authentication,
    }


def _sidecar(path: Path, root: Path, *, rows: int | None = None) -> JsonDict:
    """Describe exact sidecar bytes without copying those bytes into terminal JSON."""

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **({"rows": rows} if rows is not None else {}),
    }


def write_interface_sidecars(root: Path, result: Mapping[str, Any]) -> JsonDict:
    """Publish bounded feature, normalization, and access/exposure sidecars."""

    rows = result.get("rows")
    if not isinstance(rows, list):  # pragma: no cover
        raise EvidenceInterfaceError("feature_rows_missing")
    feature_path = root / FEATURE_PATH
    normalization_path = root / NORMALIZATION_PATH
    access_path = root / ACCESS_PATH
    write_jsonl(feature_path, rows)
    normalization = fit_normalization(rows)
    atomic_json(normalization_path, normalization)
    authentication = result.get("authentication")
    if not isinstance(authentication, Mapping):  # pragma: no cover
        raise EvidenceInterfaceError("authentication_missing")
    access_manifest: JsonDict = {
        "schema": "carnot.exp7504.v657.access_exposure.v1",
        "feature_sha256": sha256_file(feature_path),
        "reader_modes": {
            "fit": {
                "feature_roles": ["training", "calibration_tuning"],
                "label_roles": ["training", "calibration_tuning"],
                "forbidden_label_roles": ["test", "online"],
            },
            "predict": {
                "feature_roles": ["training", "calibration_tuning", "test", "online"],
                "label_roles": [],
                "forbidden_label_roles": ["training", "calibration_tuning", "test", "online"],
            },
            "evaluate": {
                "feature_roles": ["test", "online"],
                "label_roles": ["test", "online"],
                "requires_prediction_or_checkpoint_hash": True,
            },
        },
        "label_access_receipts": [
            {
                "mode": "predict",
                "feature_sha256": sha256_file(feature_path),
                "labels_opened": False,
                "label_roles_opened": [],
                "held_out_labels_opened": False,
            },
            {"mode": "fit", "executed": False, "policy_qualified_by_tests": True},
            {"mode": "evaluate", "executed": False, "prediction_freeze_required": True},
        ],
        "equal_access": {
            "arms": ["conditional_gibbs", "identical_feature_logistic", "whole_only_gibbs"],
            "identical_feature_sidecar": FEATURE_PATH.as_posix(),
            "identical_roles": True,
            "hidden_arm_specific_fields": [],
        },
        "exposure_audit": {
            "source": "Exp7491 prior exposure search",
            "search_complete": True,
            "earlier_ragtruth_exposure_found": True,
            "claim_scope": "exploratory_support_only",
            "fresh_confirmatory_claim_allowed": False,
            "faithbench_status": "previously_exposed_outside_new_primary_claims",
            "raw_receipt": deepcopy(authentication.get("prior_exposure_exclusions")),
        },
        "evaluator_separation": {
            "predictor_store": (PROTOCOL_RAW / "predictors.jsonl").as_posix(),
            "evaluator_store": (PROTOCOL_RAW / "evaluators.jsonl").as_posix(),
            "evaluator_store_parsed_during_feature_build": False,
        },
    }
    access_manifest["manifest_sha256"] = canonical_hash(access_manifest)
    atomic_json(access_path, access_manifest)
    for path in (feature_path, normalization_path, access_path):
        if path.stat().st_size >= 20 * 1024 * 1024:  # pragma: no cover
            raise EvidenceInterfaceError(f"sidecar_size_limit:{path}")
    return {
        "features": _sidecar(feature_path, root, rows=len(rows)),
        "training_normalization": _sidecar(normalization_path, root),
        "access_exposure_manifest": _sidecar(access_path, root),
    }


def _gate(check: str, category: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Attach exact operands and the category-specific failure principle."""

    principles = {
        "validity": "A favorable metric cannot excuse invalid evidence; this check prevents corrupted inputs from being promoted.",
        "readiness": "A valid scientific null must not block independent measurements; this check keeps readiness separate from efficacy.",
        "benefit": "A favorable seed, fixture, or low-support result cannot replace held-out value; this check prevents an unmeasured benefit claim.",
    }
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": passed,
        "principle": principles[category],
    }


def _validation_passed(  # pragma: no cover - exercised by the entrypoint subprocess plan.
    receipts: Sequence[Mapping[str, Any]],
) -> bool:
    """Require every receipt explicitly marked required to exit successfully."""

    required = [row for row in receipts if row.get("required", True)]
    return bool(required) and all(
        row.get("exit_code") == 0 and row.get("passed") is True for row in required
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed gate and retain the first exact observed operand."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "all_required_passed": not failed,
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind the complete artifact while excluding its self-referential hash."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the concrete drift or claim error prevented by every field."""

    specific = {
        "schema": "Version and identity prevent a reader from applying a different contract.",
        "run_date": "The fixed date and actual clocks separate the planned run from a later replay.",
        "preconditions_checked": "Exact observed resources prevent missing inputs from becoming invented evidence.",
        "MODEL_SPECS": "An empty current model list prevents historical Qwen work from becoming a current call.",
        "model_specs": "The duplicate standard spelling prevents consumers from guessing model use.",
        "model_invoked": "The false value separates current CPU transformation from historical inference.",
        "invocation_counts": "Balanced zero counters prevent cached calls from becoming current calls.",
        "inference_substrate": "The exact substrate prevents cached aggregation from being read as inference.",
        "inference_substrate_class": "The no-load class prevents duration padding or a hidden model claim.",
        "duration_s": "Measured elapsed time helps detect missing or implausible work.",
        "source_artifact_hashes": "Original byte hashes prevent upstream capture drift.",
        "rows": "One row per source prevents windows from multiplying independent support.",
        "sample_size_budget": "Complete accounting prevents exclusions from disappearing.",
        "acceptance_gate_results": "Exact operands prevent a favorable metric from excusing invalid evidence.",
        "gate_check_summary": "Named failures prevent blocked prerequisites from being hidden.",
        "honest_verdict": "A complete terminal prefix prevents retry and completion states from being confused.",
        "verdict_class": "A closed class prevents structural readiness from becoming a positive finding.",
        "flagged_adversarial": "The retained guard state prevents flagged evidence from opening a gate.",
        "validation_receipts": "Exact commands and exits prevent unrun checks from being claimed.",
        "field_principles": "Per-field reasons make future schema drift reviewable.",
        "evidence_ready_score": "A bare structural score stays independent of predictive quality.",
        "feature_manifest": "Frozen formulas prevent feature selection after labels.",
        "role_manifest": "Unique source roles prevent leakage and pseudoreplication.",
        "label_access_receipts": "Mode receipts establish which labels could enter each process.",
        "exposure_audit": "Prior exposure limits confirmatory scope without discarding descriptive rows.",
    }
    return {
        field: specific.get(field, f"This field prevents silent drift in {field}.")
        for field in fields
    }


def build_artifact(
    root: Path,
    result: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    validation_complete: bool | None = None,
) -> JsonDict:
    """Assemble one terminal evidence-interface record from raw reductions."""

    rows = result.get("rows")
    authentication = result.get("authentication")
    if not isinstance(rows, list) or not isinstance(authentication, Mapping):  # pragma: no cover
        raise EvidenceInterfaceError("artifact_input_invalid")
    access = load_json(root / ACCESS_PATH)
    normalization = load_json(root / NORMALIZATION_PATH)
    validation_ok = (
        _validation_passed(validation_receipts)
        if validation_complete is None
        else validation_complete
    )
    identity_ok = all(authentication.get("terminal_checks", {}).values())
    roles_ok = dict(Counter(str(row.get("role")) for row in rows)) == result.get("role_counts")
    separation_ok = (
        result.get("evaluation_labels_opened") is False
        and access.get("evaluator_separation", {}).get(
            "evaluator_store_parsed_during_feature_build"
        )
        is False
    )
    ready = identity_ok and roles_ok and separation_ok and validation_ok
    gates = [
        _gate("historical_identity", "validity", True, identity_ok, identity_ok),
        _gate("reader_role_separation", "validity", True, separation_ok, separation_ok),
        _gate("required_validation", "validity", True, validation_ok, validation_ok),
        _gate("complete_unique_source_rows", "readiness", True, roles_ok, roles_ok),
        _gate("evidence_ready", "readiness", 1, int(ready), ready),
        _gate("predictive_benefit_not_measured", "benefit", False, False, True),
    ]
    historical_models = authentication.get("historical_model_receipts", {})
    model_rows = {
        name: {
            "model_id": receipt.get("model_id"),
            "model_sha256": receipt.get("model_sha256"),
            "quantization": receipt.get("quantization"),
            "runtime": receipt.get("llama_cpp_build", {}).get("version"),
            "gpu_uuid": receipt.get("gpu_uuid"),
            "identity_authenticated": receipt.get("identity_authenticated"),
            "placement_authenticated": receipt.get("actual_layer_placement", {}).get(
                "placement_authenticated"
            ),
            "provenance": "historical_completed_capture",
        }
        for name, receipt in historical_models.items()
        if isinstance(receipt, Mapping)
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7504,
        "title": "Assemble sealed window features and qualify separate fit and evaluation readers",
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "preconditions_checked": [
            *[
                {
                    "check": "resource_readable",
                    "path": row.get("path"),
                    "expected": "readable_nonempty_bytes",
                    "observed": {"bytes": row.get("bytes"), "sha256": row.get("sha256")},
                    "owner": row.get("owner"),
                    "passed": row.get("passed"),
                }
                for row in authentication.get("resource_receipts", [])
            ],
            *[
                {"check": name, "expected": True, "observed": observed, "passed": observed}
                for name, observed in authentication.get("terminal_checks", {}).items()
            ],
        ],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "device_identity": {
            "host": platform.node(),
            "machine": platform.machine(),
            "device": "cpu",
            "cuda_used": False,
        },
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": duration_s,
            "validation": 0.0,
            "historical_capture": 0.0,
        },
        "random_seed": {
            "fitting": None,
            "arrival": None,
            "audit": 7504657,
            "bootstrap": None,
            "online": FROZEN_SETTINGS["online_primary"]["seeds"],
        },
        "source_artifact_hashes": [
            *deepcopy(authentication.get("resource_receipts", [])),
            *deepcopy(authentication.get("raw_shard_receipts", [])),
        ],
        "historical_raw_shard_receipts": deepcopy(authentication.get("raw_shard_receipts", [])),
        "historical_model_provenance": model_rows,
        "feature_manifest": feature_manifest(),
        "training_normalization": normalization,
        "frozen_settings": deepcopy(FROZEN_SETTINGS),
        "raw_sidecars": deepcopy(dict(sidecars)),
        "role_manifest": {
            "planned_groups": result.get("planned_groups"),
            "excluded_groups": result.get("excluded_groups"),
            "eligible_role_counts": deepcopy(result.get("role_counts")),
            "unique_normalized_source_hashes": result.get("unique_normalized_source_hashes"),
            "independent_unit": "unique_normalized_source_hash",
            "windows_are_independent_groups": False,
        },
        "label_access_receipts": deepcopy(access.get("label_access_receipts")),
        "equal_access_metadata": deepcopy(access.get("equal_access")),
        "exposure_audit": deepcopy(access.get("exposure_audit")),
        "rows": [
            {
                "group_id": row.get("group_id"),
                "role": row.get("role"),
                "source_hash": row.get("source_hash"),
                "status": "complete",
                "failed": False,
                "censored": False,
            }
            for row in rows
        ],
        "sample_size_budget": {
            "planned": result.get("planned_groups"),
            "attempted": len(rows),
            "complete": len(rows),
            "completed": len(rows),
            "excluded": result.get("excluded_groups"),
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "unique_normalized_source_hash",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "evidence_ready_score": int(ready),
        "small_ebm_training": {"performed": False, "optimizer_used": False, "parameter_count": 0},
        "predictive_benefit_measured": False,
        "honest_verdict": (
            "complete_null_evidence_interface_ready_predictive_benefit_unmeasured"
            if ready
            else "complete_disqualified_evidence_interface_validation_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "affected_validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "full_python_suite_run": False,
            "numbered_runtime_e2e_applicable": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": True,
            "numbered_runtime_e2e_applicable": False,
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _internal_hash(value: Mapping[str, Any], field: str) -> str:
    """Recompute a mapping hash after removing its stored hash field."""

    payload = deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_hash(payload)


def independent_reduce(
    artifact: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Reload sidecars and recompute identity, role separation, and readiness."""

    sidecars = artifact.get("raw_sidecars")
    if not isinstance(sidecars, Mapping):  # pragma: no cover
        raise EvidenceInterfaceError("raw_sidecars_missing")
    feature_ref = sidecars.get("features")
    access_ref = sidecars.get("access_exposure_manifest")
    normalization_ref = sidecars.get("training_normalization")
    if not all(  # pragma: no cover
        isinstance(value, Mapping) for value in (feature_ref, access_ref, normalization_ref)
    ):
        raise EvidenceInterfaceError("raw_sidecar_reference_invalid")
    feature_path = root / str(feature_ref["path"])
    access_path = root / str(access_ref["path"])
    normalization_path = root / str(normalization_ref["path"])
    rows = load_jsonl(feature_path)
    access = load_json(access_path)
    normalization = load_json(normalization_path)
    sidecars_ok = all(
        sha256_file(path) == reference.get("sha256")
        for path, reference in (
            (feature_path, feature_ref),
            (access_path, access_ref),
            (normalization_path, normalization_ref),
        )
    )
    role_counts = dict(Counter(str(row.get("role")) for row in rows))
    role_manifest = artifact.get("role_manifest")
    roles_ok = (
        isinstance(role_manifest, Mapping)
        and role_counts == role_manifest.get("eligible_role_counts")
        and len({row.get("source_hash") for row in rows}) == len(rows)
        and role_manifest.get("unique_normalized_source_hashes") == len(rows)
        and role_manifest.get("windows_are_independent_groups") is False
    )
    feature = artifact.get("feature_manifest")
    transform_ok = (
        isinstance(feature, Mapping)
        and feature.get("feature_names") == list(FEATURE_NAMES)
        and feature.get("transform_sha256") == _internal_hash(feature, "transform_sha256")
    )
    access_ok = (
        access.get("manifest_sha256") == _internal_hash(access, "manifest_sha256")
        and access.get("evaluator_separation", {}).get(
            "evaluator_store_parsed_during_feature_build"
        )
        is False
        and access.get("reader_modes", {}).get("predict", {}).get("label_roles") == []
        and set(access.get("reader_modes", {}).get("fit", {}).get("forbidden_label_roles", []))
        == {"test", "online"}
    )
    normalization_ok = normalization.get("fit_role") == "training" and normalization.get(
        "normalization_sha256"
    ) == _internal_hash(normalization, "normalization_sha256")
    preconditions = artifact.get("preconditions_checked")
    identity_ok = (
        isinstance(preconditions, list)
        and bool(preconditions)
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in preconditions)
    )
    validation_ok = (
        _validation_passed(artifact.get("validation_receipts", [])) if require_validation else True
    )
    model_ok = (
        artifact.get("MODEL_SPECS") == []
        and artifact.get("model_specs") == []
        and artifact.get("model_invoked") is False
        and artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and artifact.get("inference_substrate_class") == "no_model_load"
    )
    ready = all(
        (
            sidecars_ok,
            roles_ok,
            transform_ok,
            access_ok,
            normalization_ok,
            identity_ok,
            validation_ok,
            model_ok,
        )
    )
    return {
        "sidecars_authenticated": sidecars_ok,
        "role_counts": role_counts,
        "unique_normalized_source_hashes": len(rows),
        "reader_separation_passed": access_ok,
        "training_normalization_passed": normalization_ok,
        "evidence_ready_score": int(ready),
        "honest_verdict": (
            "complete_null_evidence_interface_ready_predictive_benefit_unmeasured"
            if ready
            else "complete_disqualified_evidence_interface_validation_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
    }


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> list[str]:
    """Cold-check schema, sidecars, reduction, principles, and checksum."""

    if not isinstance(value, Mapping):  # pragma: no cover
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    if (  # pragma: no cover
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):  # pragma: no cover
        errors.append("field_principles_incomplete")
    try:
        reduced = independent_reduce(artifact, root=root, require_validation=require_validation)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:  # pragma: no cover
        errors.append(f"independent_reduction_failed:{error}")
    else:
        for field in ("evidence_ready_score", "honest_verdict", "verdict_class"):
            if artifact.get(field) != reduced.get(field):
                errors.append("independent_reduction_mismatch")
                break
        manifest = artifact.get("role_manifest", {})
        if (
            isinstance(manifest, Mapping)
            and manifest.get("eligible_role_counts") != reduced["role_counts"]
        ):
            errors.append("independent_reduction_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if require_validation and not _validation_passed(  # pragma: no cover
        artifact.get("validation_receipts", [])
    ):
        errors.append("required_validation_failed")
    return list(dict.fromkeys(errors))


def build_artifact_for_test(root: Path) -> JsonDict:
    """Build a small circular fixture that tests structure, never efficacy."""

    roles = ("training", "calibration_tuning", "test", "online")
    rows = [
        {
            "group_id": f"fixture-{role}",
            "role": role,
            "source_hash": f"sha256:fixture-{role}",
            "response_hash": f"sha256:response-{role}",
            "corpus": "fixture",
            "features": [float(index)] * len(FEATURE_NAMES),
            "raw_whole_expectation": 0.5,
            "raw_max_window_probability": 0.5,
            "window_count": 1,
            "response_bytes": 8,
            "source_bytes": 8,
            "feature_version": FEATURE_VERSION,
            "proposed_features_not_findings": True,
        }
        for index, role in enumerate(roles, 1)
    ]
    result: JsonDict = {
        "rows": rows,
        "planned_groups": 4,
        "excluded_groups": 0,
        "role_counts": dict(Counter(row["role"] for row in rows)),
        "unique_normalized_source_hashes": 4,
        "evaluation_labels_opened": False,
        "authentication": {
            "terminal_checks": {"fixture_identity": True},
            "artifact_receipts": [],
            "raw_shard_receipts": [],
            "historical_model_receipts": {},
            "prior_exposure_exclusions": {"search_complete": True},
        },
    }
    sidecars = write_interface_sidecars(root, result)
    receipt = {
        "name": "fixture_validation",
        "command": "fixture only",
        "command_argv": ["fixture"],
        "exit_code": 0,
        "passed": True,
        "required": True,
    }
    return build_artifact(
        root,
        result,
        sidecars,
        [receipt],
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        duration_s=1.0,
        phase_spans=[],
        validation_complete=True,
    )


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC timestamp for a durable run receipt."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase or child boundary so the owned process stays visible."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7504] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase_span(  # pragma: no cover - measured runtime boundary.
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:
    """Close one measured phase and name its durable output."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
        "principle": "A measured span exposes unfinished work and prevents duration padding.",
    }


def _terminal_commands(  # pragma: no cover - capability E2E child plan.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, and exact terminal readers."""

    python = ".venv/bin/python"
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7504_v657_evidence_interface import independent_reduce;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "r=independent_reduce(v,require_validation=False);"
        "print(json.dumps(r,sort_keys=True),flush=True);"
        "raise SystemExit(r['evidence_ready_score']!=1)"
    )
    specifications = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_row_consistency",
        ),
    )
    return [
        PlannedCommand(specification, "required_validation", True)
        for specification in specifications
    ]


def _blocked_artifact(  # pragma: no cover - external absence boundary.
    missing: Path, started_at: str, started: float
) -> JsonDict:
    """Publish an exact external absence without running dependent work."""

    observed = None
    gate = {
        "check": "external_prerequisite_available",
        "category": "validity",
        "upstream": missing.as_posix(),
        "field_path": "bytes",
        "expected": "readable_nonempty_bytes",
        "observed": observed,
        "op": "eq",
        "passed": False,
        "principle": "A missing external input blocks dependent reduction instead of permitting an invented result.",
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete_blocked_external_prerequisite_missing",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "duration_s": time.monotonic() - started,
        "phase_spans": [],
        "preconditions_checked": [dict(gate)],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": {"device": "cpu", "cuda_used": False},
        "random_seed": {},
        "source_artifact_hashes": [],
        "rows": [],
        "sample_size_budget": {
            "planned": 520,
            "attempted": 0,
            "complete": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 520,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "all_required_passed": False,
            "failed_checks": [gate["check"]],
            "first_failure": gate,
        },
        "evidence_ready_score": 0,
        "honest_verdict": "complete_blocked_external_prerequisite_missing",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Authenticate, transform, validate, cold-replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise EvidenceInterfaceError(f"run_date_invalid:{run_date}")
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    missing = next((path for path in REQUIRED_INPUT_PATHS if not (root / path).is_file()), None)
    if missing is not None:
        artifact = _blocked_artifact(missing, started_at, started)
        atomic_json(root / RESULT_PATH, artifact)
        progress(started, "preconditions", "complete_blocked", missing=missing)
        return artifact
    phase_started = time.monotonic()
    authentication = authenticate_capture_inputs(root)
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            len(REQUIRED_INPUT_PATHS),
            "authenticated_inputs",
        )
    )
    progress(started, "preconditions", "complete", resources=len(REQUIRED_INPUT_PATHS))

    phase_started = time.monotonic()
    progress(started, "feature_transform", "start")
    result = build_real_feature_rows(root)
    result["authentication"] = authentication
    sidecars = write_interface_sidecars(root, result)
    spans.append(
        _phase_span(
            "feature_transform",
            phase_started,
            started,
            len(result["rows"]),
            FEATURE_PATH.as_posix(),
        )
    )
    progress(started, "feature_transform", "complete", groups=len(result["rows"]))

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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7504-validation-"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise EvidenceInterfaceError(f"validation_plan_invalid:{plan_errors}")
    phase_started = time.monotonic()
    progress(started, "affected_validation", "before_subprocesses", commands=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation" / "affected",
    )
    progress(
        started, "affected_validation", "after_subprocesses", passed=_validation_passed(affected)
    )
    spans.append(
        _phase_span(
            "affected_validation", phase_started, started, len(affected), "affected_validation_logs"
        )
    )

    candidate = root / RAW_DIR / "measured_terminal_candidate.json"
    artifact = build_artifact(
        root,
        result,
        sidecars,
        affected,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        validation_complete=_validation_passed(affected),
    )
    atomic_json(candidate, artifact)
    phase_started = time.monotonic()
    terminal_plan = _terminal_commands(root, candidate)
    progress(started, "terminal_validation", "before_subprocesses", commands=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=root / RAW_DIR / "validation" / "terminal",
    )
    progress(
        started, "terminal_validation", "after_subprocesses", passed=_validation_passed(terminal)
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, started, len(terminal), "terminal_validation_logs"
        )
    )

    receipts = [*affected, *terminal]
    artifact = build_artifact(
        root,
        result,
        sidecars,
        receipts,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        validation_complete=_validation_passed(receipts),
    )
    artifact["terminal_candidate"] = _sidecar(candidate, root)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, root=root, require_validation=True)
    if errors:
        raise EvidenceInterfaceError(f"terminal_artifact_invalid:{errors}")
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        started, "publish", "complete", path=RESULT_PATH, ready=artifact["evidence_ready_score"]
    )
    return artifact


def parse_args(  # pragma: no cover - thin CLI parser.
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the fixed run date and read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - capability E2E.
    """Run the interface or one read-only cold validator through a thin wrapper."""

    arguments = parse_args(argv)
    if arguments.validate is not None:
        value = load_json(arguments.validate)
        errors = validate_artifact(value, require_validation=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if arguments.independent_reduce is not None:
        value = load_json(arguments.independent_reduce)
        print(
            json.dumps(independent_reduce(value, require_validation=False), sort_keys=True),
            flush=True,
        )
        return 0
    run_experiment(REPO_ROOT, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

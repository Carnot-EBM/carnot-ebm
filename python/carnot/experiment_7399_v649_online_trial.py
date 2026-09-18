"""Measure the sealed delayed affine adapter against adaptive controls.

The trial reuses the tested Exp7397 update authority. It fits only on the
sealed initialization groups and records every later prediction before its
label becomes available. The archive orders are controlled replays, not real
chronology or a deployment claim.

Spec refs: REQ-CL-7399 and SCENARIO-CL-7399-GATE through
SCENARIO-CL-7399-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
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

from carnot import experiment_7397_v649_delayed_adapter as sealed
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.learning.delayed_energy_calibration import DelayedEnergyCalibrator
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
EXPERIMENT_ID = "exp7399-online-trial"
SCHEMA = "carnot.exp7399.v649.online_trial.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7399_v649_online_trial.json")
RAW_DIR = Path("results/raw/experiment_7399_v649_online_trial")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7399_v649_online_trial")
MODULE_PATH = Path("python/carnot/experiment_7399_v649_online_trial.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7399_v649_online_trial.py")
TEST_PATH = Path("tests/python/test_experiment_7399_v649_online_trial.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
UPSTREAM_PATH = Path("results/experiment_7397_v649_delayed_adapter.json")
PROTOCOL_PATH = sealed.PROTOCOL_PATH
TRAINING_ARTIFACT_PATH = sealed.TRAINING_ARTIFACT_PATH
EXPECTED_UPSTREAM_SHA256 = "sha256:c4e5139f7f3c600bd129e7e8425f362bac458796ae6a7db6052d6895140b2e3b"
EXPECTED_PROTOCOL_SHA256 = sealed.EXPECTED_PROTOCOL_SHA256
EXPECTED_TRAINING_SHA256 = sealed.EXPECTED_TRAINING_SHA256
EXPECTED_ADAPTER_SHA256 = "sha256:bdd5b9296722e45773f901c035174485be19339a8fe51767fc58da685880aa09"
TRAINING_SEEDS = sealed.TRAINING_SEEDS
ARMS = sealed.ARMS
PRACTICAL_CONTROLS = sealed.PRACTICAL_CONTROLS
ORDERINGS = sealed.ORDERINGS
BOOTSTRAP_DRAWS = sealed.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = sealed.BOOTSTRAP_SEED
BLOCK_LENGTHS = sealed.BLOCK_LENGTHS
MAX_STEPS = sealed.MAX_STEPS
LABEL_AUTHORITY = sealed.LABEL_AUTHORITY
LATENCY_FIELDS = (
    "feature_extraction_latency_s",
    "energy_scoring_latency_s",
    "affine_prediction_latency_s",
    "update_latency_s",
    "durable_state_write_latency_s",
    "orchestration_latency_s",
    "full_cost_s",
)
ZERO_INVOCATION_COUNTS = deepcopy(sealed.ZERO_INVOCATION_COUNTS)
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/experiment_7386_v648_online_decisions.py"),
    Path("python/carnot/experiment_7397_v649_delayed_adapter.py"),
    Path("python/carnot/learning/delayed_energy_calibration.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("openspec/capabilities/autoresearch/spec.md"),
    SPEC_PATH,
    UPSTREAM_PATH,
    PROTOCOL_PATH,
    TRAINING_ARTIFACT_PATH,
    sealed.CORPUS_PATH,
)
V649_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so numeric and replay identities are portable."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a later reader can detect source drift."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON bytes so readers never see a partial result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _numeric_tree(value: Any) -> Any:
    """Copy finite numeric JSON data and reject executable or opaque values."""

    if isinstance(value, Mapping):
        return {str(key): _numeric_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numeric_tree(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("numeric checkpoint contains non-numeric state")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric checkpoint contains non-finite state")
    return number


def make_numeric_checkpoint(record: Mapping[str, Any]) -> JsonDict:
    """Strip a training record to immutable numbers needed for fresh scoring."""

    checkpoint = {
        "schema": "carnot.exp7399.numeric_checkpoint.v1",
        "seed": int(record["seed"]),
        "training_authority": "frozen_initialization_groups_only",
        "initial_group_count": int(record["initial_group_count"]),
        "initial_group_ids_sha256": str(record["initial_group_ids_sha256"]),
        "future_stream_groups_used": int(record["future_stream_groups_used"]),
        "gibbs_steps": int(record["gibbs_steps"]),
        "weights": _numeric_tree(record["weights"]),
        "affine": _numeric_tree(record["affine"]),
        "logistic_weights": _numeric_tree(record["logistic_weights"]),
    }
    errors = numeric_checkpoint_errors(checkpoint)
    if errors:
        raise ValueError(errors[0])
    return checkpoint


def numeric_checkpoint_errors(checkpoint: object) -> list[str]:
    """Validate the code-free shape used by NumPy and cold replay."""

    if not isinstance(checkpoint, Mapping):
        return ["numeric_checkpoint_not_object"]
    try:
        weights = _numeric_tree(checkpoint["weights"])
        affine = _numeric_tree(checkpoint["affine"])
        logistic = _numeric_tree(checkpoint["logistic_weights"])
        w1 = np.asarray(weights["w1"], dtype=np.float64)
        b1 = np.asarray(weights["b1"], dtype=np.float64)
        w_out = np.asarray(weights["w_out"], dtype=np.float64)
        coef = np.asarray(logistic["coef"], dtype=np.float64)
        valid = (
            checkpoint.get("schema") == "carnot.exp7399.numeric_checkpoint.v1"
            and checkpoint.get("training_authority") == "frozen_initialization_groups_only"
            and int(checkpoint.get("future_stream_groups_used", -1)) == 0
            and 0 <= int(checkpoint.get("gibbs_steps", -1)) <= MAX_STEPS
            and w1.shape == (4, 2)
            and b1.shape == (4,)
            and w_out.shape == (4,)
            and coef.shape == (2,)
            and 0.25 <= float(affine["a"]) <= 4.0
            and -8.0 <= float(affine["b"]) <= 8.0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    return [] if valid else ["numeric_checkpoint_invalid"]


def checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    """Bind every checkpoint field with one canonical digest."""

    return canonical_hash(checkpoint)


def score_numeric_checkpoint(checkpoint: Mapping[str, Any], features: Sequence[float]) -> JsonDict:
    """Score a plain checkpoint with NumPy and no training-reducer import."""

    errors = numeric_checkpoint_errors(checkpoint)
    if errors:
        raise ValueError(errors[0])
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("checkpoint scoring requires two finite features")
    weights = checkpoint["weights"]
    hidden_linear = np.asarray(weights["w1"], dtype=np.float64) @ values + np.asarray(
        weights["b1"], dtype=np.float64
    )
    hidden = hidden_linear / (1.0 + np.exp(-hidden_linear))
    energy = float(
        np.asarray(weights["w_out"], dtype=np.float64) @ hidden + float(weights["b_out"])
    )
    affine = checkpoint["affine"]
    logit = float(affine["a"]) * energy + float(affine["b"])
    probability = (
        1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
    )
    return {
        "energy": energy,
        "probability": probability,
        "checkpoint_hash": checkpoint_hash(checkpoint),
    }


def states_from_numeric_checkpoint(
    checkpoint: Mapping[str, Any], initial_labels: Sequence[int]
) -> dict[str, JsonDict]:
    """Rebuild all five initial arms from code-free checkpoint numbers."""

    if numeric_checkpoint_errors(checkpoint):
        raise ValueError("numeric checkpoint invalid")
    adapter = DelayedEnergyCalibrator(
        checkpoint["weights"],
        a=float(checkpoint["affine"]["a"]),
        b=float(checkpoint["affine"]["b"]),
    ).to_dict()
    labels = [int(label) for label in initial_labels][-sealed.RECENT_WINDOW :]
    return {
        "static_affine_gibbs": {"adapter": deepcopy(adapter), "update_count": 0},
        "adaptive_affine_gibbs": {"adapter": deepcopy(adapter), "update_count": 0},
        "online_logistic_raw_features": {
            "weights": deepcopy(checkpoint["logistic_weights"]),
            "update_count": 0,
        },
        "recent_frequency_beta_binomial": {"recent_labels": labels, "update_count": 0},
        "no_feedback_adaptive_control": {"adapter": deepcopy(adapter), "update_count": 0},
    }


def _arm_state_hash(arm: str, state: Mapping[str, Any]) -> str:
    if "adapter" in state:
        return DelayedEnergyCalibrator.from_dict(state["adapter"]).state_hash
    if arm == "online_logistic_raw_features":
        return sealed._state_hash({**dict(state), "arm": arm})
    labels = list(state.get("recent_labels") or [])
    numeric = {
        "alpha": sum(labels) + 1.0,
        "beta": len(labels) - sum(labels) + 1.0,
        "window_size": len(labels),
        "update_count": int(state.get("update_count", 0)),
    }
    return sealed._state_hash(numeric)


def _compact_durable_state(states: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    compact: JsonDict = {}
    for arm, state in states.items():
        if "adapter" in state:
            adapter = DelayedEnergyCalibrator.from_dict(state["adapter"])
            compact[arm] = adapter.numeric_state()
        elif arm == "online_logistic_raw_features":
            compact[arm] = {
                "weights": deepcopy(state["weights"]),
                "update_count": int(state["update_count"]),
            }
        else:
            labels = list(state.get("recent_labels") or [])
            compact[arm] = {
                "recent_labels": labels,
                "update_count": int(state.get("update_count", 0)),
            }
    return compact


def _feedback_status(ledger: Mapping[str, Any], arm: str) -> tuple[str, Mapping[str, Any]]:
    feedback = ledger.get("feedback")
    if not isinstance(feedback, Mapping):
        return "pending", {}
    result = (feedback.get("arm_results") or {}).get(arm) or {}
    return str(result.get("status", "unscoreable")), result


def _compact_ledger(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    compact: list[JsonDict] = []
    for row in rows:
        feedback = row.get("feedback")
        compact_feedback = None
        if isinstance(feedback, Mapping):
            arm_results = {}
            for arm, result in (feedback.get("arm_results") or {}).items():
                arm_results[str(arm)] = {
                    key: deepcopy(result.get(key))
                    for key in (
                        "status",
                        "update_admitted",
                        "state_hash_before",
                        "state_hash_after",
                        "gibbs_weights_unchanged",
                    )
                    if key in result
                }
            compact_feedback = {
                "label": feedback.get("label"),
                "visible_at": feedback.get("visible_at"),
                "arm_results": arm_results,
            }
        compact.append(
            {
                key: deepcopy(row.get(key))
                for key in (
                    "event_id",
                    "ordering",
                    "feedback_condition",
                    "seed",
                    "stream_index",
                    "group_id",
                    "prediction_recorded_at",
                    "feedback_available_at",
                    "label_authority",
                    "prediction_before_feedback",
                    "missing",
                )
            }
            | {"feedback": compact_feedback}
        )
    return compact


def measure_replay_unit(
    initial_states: Mapping[str, Mapping[str, Any]],
    stream: Sequence[Mapping[str, Any]],
    *,
    ordering: str,
    condition: Mapping[str, Any],
    seed: int,
    thresholds: Mapping[str, Any],
    missing_group_ids: set[str],
    checkpoint: Mapping[str, Any],
    durable_path: Path,
) -> JsonDict:
    """Run one tested replay unit and add stage, lineage, and write evidence."""

    unit_started = time.perf_counter()
    restart_path = durable_path.with_name(f"{durable_path.stem}.restart.json")
    replay = sealed.replay_condition(
        initial_states,
        stream,
        ordering=ordering,
        condition=condition,
        seed=seed,
        thresholds=thresholds,
        missing_group_ids=missing_group_ids,
        checkpoint_path=restart_path,
    )
    source_by_id = {str(row["group_id"]): row for row in stream}
    ledger_by_id = {str(row["event_id"]): row for row in replay["event_ledger"]}
    rows_by_arm_index = {(str(row["arm"]), int(row["stream_index"])): row for row in replay["rows"]}
    final_hashes = {arm: _arm_state_hash(arm, replay["final_states"][arm]) for arm in ARMS}
    measured: list[JsonDict] = []
    unscoreable = 0
    update_operations = 0
    for original in replay["rows"]:
        row = deepcopy(dict(original))
        event_id = f"{ordering}:{condition['name']}:{seed}:{row['group_id']}"
        ledger = ledger_by_id[event_id]
        disposition, feedback_result = _feedback_status(ledger, str(row["arm"]))
        update_operations += int(feedback_result.get("update_admitted") is True)

        feature_started = time.perf_counter()
        source = source_by_id[str(row["group_id"])]
        features = [float(source["entity_uptake"]), float(source["falsifiability_score"])]
        feature_latency = time.perf_counter() - feature_started
        energy_latency = 0.0
        affine_latency = 0.0
        if str(row["arm"]) in {
            "static_affine_gibbs",
            "adaptive_affine_gibbs",
            "no_feedback_adaptive_control",
        }:
            energy_started = time.perf_counter()
            base = score_numeric_checkpoint(checkpoint, features)
            energy_latency = time.perf_counter() - energy_started
            affine_started = time.perf_counter()
            numeric_state = row["numeric_state"]
            logit = float(numeric_state["a"]) * float(base["energy"]) + float(numeric_state["b"])
            observed = (
                1.0 / (1.0 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1.0 + math.exp(logit))
            )
            affine_latency = time.perf_counter() - affine_started
            if not math.isclose(observed, float(row["probability"]), rel_tol=1e-12, abs_tol=1e-12):
                unscoreable += 1

        before_hash = str(feedback_result.get("state_hash_before") or row["state_hash"])
        after_hash = feedback_result.get("state_hash_after")
        if disposition == "committed" and after_hash is None:
            feedback = ledger.get("feedback") or {}
            next_index = int(feedback.get("visible_at", len(stream))) + 1
            next_row = rows_by_arm_index.get((str(row["arm"]), next_index))
            after_hash = next_row.get("state_hash") if next_row else final_hashes[str(row["arm"])]
        if after_hash is None:
            after_hash = before_hash

        orchestration = max(
            0.0,
            float(row["prediction_latency_s"]) - energy_latency - affine_latency,
        )
        row.update(
            {
                "pre_update_probability": float(row["probability"]),
                "typed_action": str(row["decision"]),
                "label_availability_time": int(row["feedback_available_at"]),
                "feedback_disposition": disposition,
                "state_hash_before_update": before_hash,
                "state_hash_after_update": str(after_hash),
                "feature_extraction_latency_s": feature_latency,
                "energy_scoring_latency_s": energy_latency,
                "affine_prediction_latency_s": affine_latency,
                "durable_state_write_latency_s": 0.0,
                "orchestration_latency_s": orchestration,
            }
        )
        row["full_cost_s"] = sum(float(row[field]) for field in LATENCY_FIELDS[:-1])
        row.pop("numeric_state", None)
        measured.append(row)

    durable_payload = {
        "schema": "carnot.exp7399.durable_unit.v1",
        "ordering": ordering,
        "feedback_condition": condition["name"],
        "seed": seed,
        "states": _compact_durable_state(replay["final_states"]),
    }
    write_started = time.perf_counter()
    atomic_json(durable_path, durable_payload)
    write_latency = time.perf_counter() - write_started
    if measured:
        measured[-1]["durable_state_write_latency_s"] = write_latency
        measured[-1]["full_cost_s"] += write_latency
    return {
        "rows": measured,
        "event_ledger": _compact_ledger(replay["event_ledger"]),
        "restart_receipt": replay["restart_receipt"],
        "pending_feedback_at_end": replay["pending_feedback_at_end"],
        "durable_state_path": durable_path.as_posix(),
        "durable_state_sha256": sha256_file(durable_path),
        "durable_state_bytes": durable_path.stat().st_size,
        "durable_state_write_latency_s": write_latency,
        "update_operation_count": update_operations,
        "failed_units": 0,
        "unscoreable_units": unscoreable,
        "orchestration_duration_s": time.perf_counter() - unit_started,
    }


def causal_erasure_probe(
    initial_states: Mapping[str, Mapping[str, Any]],
    stream: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    prefix_count: int,
) -> JsonDict:
    """Erase a verified prefix after restart and compare one unchanged query."""

    if prefix_count <= 0 or prefix_count >= len(stream):
        raise ValueError("prefix must leave one later query")
    adaptive = DelayedEnergyCalibrator.from_dict(initial_states["adaptive_affine_gibbs"]["adapter"])
    no_feedback = DelayedEnergyCalibrator.from_dict(
        initial_states["no_feedback_adaptive_control"]["adapter"]
    )
    committed: list[str] = []
    for index, source in enumerate(stream[:prefix_count]):
        event_id = f"erasure:{seed}:{source['group_id']}"
        features = [float(source["entity_uptake"]), float(source["falsifiability_score"])]
        adaptive.record_prediction(
            event_id,
            features,
            prediction_index=index,
            feedback_available_at=index + 1,
            label_authority=LABEL_AUTHORITY,
        )
        update = adaptive.commit_feedback(event_id, int(source["label"]), visible_at=index + 1)
        if update.get("update_admitted") is True:
            committed.append(event_id)
    restored = DelayedEnergyCalibrator.from_dict(json.loads(json.dumps(adaptive.to_dict())))
    query = stream[prefix_count]
    query_features = [float(query["entity_uptake"]), float(query["falsifiability_score"])]
    adapted_probability = restored.probability(query_features)[1]
    for event_id in committed:
        restored.erase_feedback(event_id)
    erased_probability = restored.probability(query_features)[1]
    no_feedback_probability = no_feedback.probability(query_features)[1]
    passed = (
        math.isclose(erased_probability, no_feedback_probability, rel_tol=0.0, abs_tol=1e-15)
        and restored.state_hash == no_feedback.state_hash
    )
    return {
        "seed": seed,
        "ordering": "fixed_hash_order",
        "feedback_condition": "primary_delay_1",
        "prefix_group_ids": [str(row["group_id"]) for row in stream[:prefix_count]],
        "query_group_id": str(query["group_id"]),
        "cold_restart_performed": True,
        "erased_prior_update_count": len(committed),
        "adapted_probability": adapted_probability,
        "erased_probability": erased_probability,
        "no_feedback_probability": no_feedback_probability,
        "attributed_prior_feedback_delta": adapted_probability - erased_probability,
        "erased_state_hash": restored.state_hash,
        "no_feedback_state_hash": no_feedback.state_hash,
        "current_query_label_used": False,
        "future_label_used": False,
        "sample_order_preserved": True,
        "passed": passed,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def latency_report(
    rows: Sequence[Mapping[str, Any]], *, serialized_bytes: int, update_operations: int
) -> JsonDict:
    """Reduce each measured stage without inferring device utilization."""

    stages = {}
    for field in LATENCY_FIELDS:
        values = [float(row[field]) for row in rows]
        stages[field] = {
            "count": len(values),
            "mean_s": float(np.mean(values)) if values else 0.0,
            "p50_s": _percentile(values, 0.50),
            "p95_s": _percentile(values, 0.95),
            "max_s": max(values, default=0.0),
        }
    return {
        "stage_distributions": stages,
        "serialized_bytes": int(serialized_bytes),
        "update_operation_count": int(update_operations),
        "measured_hardware_path": "bounded_cpu_scalar_updates",
        "future_hardware_path": "batched_device_score_computation_not_measured",
        "measured_100x_claim": False,
        "real_time_deployment_claim": False,
    }


def build_validation_commands(
    repo_root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    """Freeze the exact Exp7358 plan with private temp and coverage parents."""

    return build_command_plan(repo_root, V649_MANIFEST, private_root)


def precondition_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
) -> JsonDict:
    """Record one exact gate so missing values remain visible."""

    passed = observed in expected if operator == "in" else observed == expected
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], JsonDict, JsonDict]:  # pragma: no cover
    """Authenticate sources, structured eligibility, runtime, and entrypoint."""

    import jax

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in (*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            precondition_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                observed,
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = sha256_file(path)

    upstream = _load_object(root / UPSTREAM_PATH)
    protocol = _load_object(root / PROTOCOL_PATH)
    expected_rows = (
        (
            "upstream_bytes",
            UPSTREAM_PATH,
            "artifact_sha256",
            EXPECTED_UPSTREAM_SHA256,
            hashes.get(UPSTREAM_PATH.as_posix()),
            "==",
        ),
        (
            "upstream_ready",
            UPSTREAM_PATH,
            "delayed_adapter_ready_score",
            1,
            upstream.get("delayed_adapter_ready_score"),
            "==",
        ),
        (
            "upstream_verdict",
            UPSTREAM_PATH,
            "verdict_class",
            ["positive", "circular_positive", "null"],
            upstream.get("verdict_class"),
            "in",
        ),
        (
            "upstream_flag",
            UPSTREAM_PATH,
            "flagged_adversarial",
            False,
            upstream.get("flagged_adversarial"),
            "==",
        ),
        (
            "upstream_required_gates",
            UPSTREAM_PATH,
            "gate_check_summary.all_required_passed",
            True,
            (upstream.get("gate_check_summary") or {}).get("all_required_passed"),
            "==",
        ),
        (
            "protocol_bytes",
            PROTOCOL_PATH,
            "artifact_sha256",
            EXPECTED_PROTOCOL_SHA256,
            hashes.get(PROTOCOL_PATH.as_posix()),
            "==",
        ),
        (
            "training_bytes",
            TRAINING_ARTIFACT_PATH,
            "artifact_sha256",
            EXPECTED_TRAINING_SHA256,
            hashes.get(TRAINING_ARTIFACT_PATH.as_posix()),
            "==",
        ),
        (
            "tested_adapter_bytes",
            sealed.ADAPTER_PATH,
            "sha256",
            EXPECTED_ADAPTER_SHA256,
            hashes.get(sealed.ADAPTER_PATH.as_posix()),
            "==",
        ),
        (
            "upstream_protocol_binding",
            UPSTREAM_PATH,
            f"source_artifact_hashes.{PROTOCOL_PATH.as_posix()}",
            EXPECTED_PROTOCOL_SHA256,
            (upstream.get("source_artifact_hashes") or {}).get(PROTOCOL_PATH.as_posix()),
            "==",
        ),
        (
            "upstream_training_binding",
            UPSTREAM_PATH,
            f"source_artifact_hashes.{TRAINING_ARTIFACT_PATH.as_posix()}",
            EXPECTED_TRAINING_SHA256,
            (upstream.get("source_artifact_hashes") or {}).get(TRAINING_ARTIFACT_PATH.as_posix()),
            "==",
        ),
        (
            "upstream_adapter_binding",
            UPSTREAM_PATH,
            f"source_artifact_hashes.{sealed.ADAPTER_PATH.as_posix()}",
            EXPECTED_ADAPTER_SHA256,
            (upstream.get("source_artifact_hashes") or {}).get(sealed.ADAPTER_PATH.as_posix()),
            "==",
        ),
    )
    for check, path, field, expected, observed, operator in expected_rows:
        checks.append(
            precondition_row(
                check,
                path.as_posix(),
                field,
                expected,
                observed,
                operator=operator,
            )
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.extend(
        (
            precondition_row(
                "driving_requirement",
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-CL-7399",
                "REQ-CL-7399" if "REQ-CL-7399" in spec_text else None,
            ),
            precondition_row(
                "current_task_not_quarantined",
                "ops/exclusion_manifest.yaml",
                EXPERIMENT_ID,
                False,
                "experiment_id: 7399" in exclusion_text or EXPERIMENT_ID in exclusion_text,
            ),
            precondition_row(
                "cpu_jax_backend",
                "current_process",
                "jax.default_backend",
                "cpu",
                jax.default_backend(),
            ),
            precondition_row(
                "live_execution_guard",
                "current_process",
                "CARNOT_FORCE_LIVE",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
            ),
            precondition_row(
                "sealed_online_protocol",
                UPSTREAM_PATH.as_posix(),
                "online_protocol.source_sha256",
                EXPECTED_PROTOCOL_SHA256,
                (upstream.get("online_protocol") or {}).get("source_sha256"),
            ),
            precondition_row(
                "sealed_five_arms",
                UPSTREAM_PATH.as_posix(),
                "online_protocol.arm_definitions",
                sorted(ARMS),
                sorted((upstream.get("online_protocol") or {}).get("arm_definitions") or {}),
            ),
            precondition_row(
                "sealed_five_seeds",
                UPSTREAM_PATH.as_posix(),
                "online_protocol.training_seeds",
                list(TRAINING_SEEDS),
                (upstream.get("online_protocol") or {}).get("training_seeds"),
            ),
        )
    )
    final_rows = [
        row
        for row in protocol.get("feature_rows", [])
        if isinstance(row, Mapping) and row.get("partition") == "final_test"
    ]
    checks.append(
        precondition_row(
            "final_test_labels_sealed",
            PROTOCOL_PATH.as_posix(),
            "feature_rows.final_test.label_absent",
            True,
            bool(final_rows) and all("label" not in row for row in final_rows),
        )
    )
    return checks, hashes, upstream, protocol


def _gate(
    check: str, category: str, expected: Any, observed: Any, operator: str = "=="
) -> JsonDict:
    if operator == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    elif operator == "<=":
        passed = isinstance(observed, (int, float)) and observed <= expected
    else:
        passed = observed == expected
    return {
        "category": category,
        "check": check,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failed if row.get("category") != "scientific_efficacy"]
    scientific = [row for row in failed if row.get("category") == "scientific_efficacy"]
    return {
        "all_required_passed": not required,
        "failed_required_count": len(required),
        "first_required_failure": required[0] if required else None,
        "failed_scientific_gate_count": len(scientific),
        "first_scientific_failure": scientific[0] if scientific else None,
    }


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    names = set(validation_scope.REQUIRED_CHECK_NAMES) | {
        "declared_entrypoint_e2e",
        "independent_reducer",
        "independent_cold_replay",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    passing = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True
        and row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is False
    }
    return names <= passing


def _manifest_valid(rows: object) -> bool:
    if not isinstance(rows, list) or len(rows) != len(TRAINING_SEEDS):
        return False
    return {int(row.get("seed", -1)) for row in rows if isinstance(row, Mapping)} == set(
        TRAINING_SEEDS
    ) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
        and isinstance(row.get("checkpoint_hash"), str)
        and len(str(row.get("checkpoint_hash"))) == 71
        and str(row.get("checkpoint_hash")).startswith("sha256:")
        and isinstance(row.get("file_sha256"), str)
        and len(str(row.get("file_sha256"))) == 71
        and int(row.get("future_stream_groups_used", -1)) == 0
        for row in rows
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute capture and value separately from raw terminal evidence."""

    budget = artifact.get("sample_size_budget") or {}
    rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    planned = int(budget.get("planned_prediction_rows") or 0)
    required_fields = {
        "ordering",
        "feedback_condition",
        "stream_index",
        "group_id",
        "arm",
        "seed",
        "label",
        "pre_update_probability",
        "typed_action",
        "label_availability_time",
        "feedback_disposition",
        "state_hash_before_update",
        "state_hash_after_update",
        "brier_loss",
        "log_loss",
        *LATENCY_FIELDS,
    }
    identities = {
        (
            row.get("ordering"),
            row.get("feedback_condition"),
            row.get("stream_index"),
            row.get("arm"),
            row.get("seed"),
        )
        for row in rows
    }
    row_complete = (
        planned > 0
        and len(rows) == planned
        and len(identities) == planned
        and all(required_fields <= set(row) for row in rows)
        and all(row.get("prediction_before_feedback") is True for row in rows)
    )
    checkpoints = _manifest_valid(artifact.get("numeric_checkpoint_manifest"))
    erasures = artifact.get("causal_erasure_rows") or []
    erasure_passed = (
        len(erasures) == len(TRAINING_SEEDS)
        and {int(row.get("seed", -1)) for row in erasures} == set(TRAINING_SEEDS)
        and all(row.get("passed") is True for row in erasures)
    )
    restarts = artifact.get("restart_receipts") or []
    restart_passed = bool(restarts) and all(
        row.get("performed") is True and row.get("prediction_parity") is True for row in restarts
    )
    ledger = artifact.get("event_ledger") or []
    ledger_complete = len(ledger) == int(budget.get("planned_feedback_units") or -1) and all(
        row.get("prediction_before_feedback") is True
        and row.get("label_authority") == LABEL_AUTHORITY
        for row in ledger
    )
    counted = (
        int(budget.get("failed_units", -1)) == 0
        and int(budget.get("unscoreable_units", -1)) == 0
        and int(budget.get("unstarted_prediction_rows", -1)) == 0
    )
    validation = _required_receipts_pass(artifact.get("validation_receipts") or [])
    safety = artifact.get("flagged_adversarial") is False
    capture = int(
        row_complete
        and checkpoints
        and erasure_passed
        and restart_passed
        and ledger_complete
        and counted
        and validation
        and safety
    )
    primary = artifact.get("primary_value_reduction") or {}
    value = int(capture == 1 and primary.get("passed") is True)
    return {
        "row_completeness_passed": row_complete,
        "numeric_checkpoints_passed": checkpoints,
        "causal_erasure_passed": erasure_passed,
        "cold_restart_passed": restart_passed,
        "feedback_ledger_passed": ledger_complete,
        "failed_and_unscoreable_counted": counted,
        "required_validation_passed": validation,
        "safety_passed": safety,
        "online_capture_complete_score": capture,
        "online_value_score": value,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, inputs, protocol, checkpoints, rows, and causal evidence."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "online_protocol",
        "numeric_checkpoint_manifest",
        "policy_rows",
        "rows",
        "event_ledger",
        "causal_erasure_rows",
        "moving_block_intervals",
        "condition_reports",
        "primary_value_reduction",
        "sample_size_budget",
    )
    return canonical_hash({field: artifact.get(field) for field in fields})


def _base_artifact() -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "Host CPU runs tiny JAX Gibbs fits, NumPy scoring, bounded scalar updates, exact risk reduction, and JSON durability checks; no LLM loads.",
        "inference_substrate_details": {
            "device": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax_backend": "cpu",
            "current_llm_work": False,
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "random_seed": {
            "training_seeds": list(TRAINING_SEEDS),
            "moving_block_seed": BOOTSTRAP_SEED,
        },
        "verifier_is_oracle": False,
        "promotion_score": 0,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    principles = {
        "schema": "Versioned schema; ordinary top-level identity becomes terminal only after measured work and required checks.",
        "run_date": "The run uses 20260918 and records actual UTC start and end timestamps.",
        "preconditions_checked": "Exact hashes, eligibility, runtime, device, and entrypoint checks precede dependent work.",
        "MODEL_SPECS": "No current LLM is used; tiny Gibbs work is recorded separately.",
        "model_invoked": "False means no current real LLM load or generation was attempted.",
        "invocation_counts": "Counts cover current owned LLM events only and remain zero.",
        "inference_substrate": "A string describes current CPU, JAX, NumPy, and exact-reducer work.",
        "inference_substrate_class": "The exact current class is no_model_load with no duration padding.",
        "execution_venue": "The closed execution venue value is host.",
        "duration_s": "Monotonic task duration is separate from scientific and validation phase spans.",
        "phase_spans": "Actual phase boundaries retain UTC checkpoints and monotonic offsets.",
        "random_seed": "Frozen fit and block-resampling seeds make the replay reproducible.",
        "reproducibility_checksum": "The checksum binds current code, configuration, protocol, checkpoints, and raw rows.",
        "source_artifact_hashes": "Exact byte hashes retain source identity and do not copy historical invocation counters.",
        "rows": "Every paired unit retains prediction, action, label timing, state lineage, losses, disposition, and full cost.",
        "sample_size_budget": "Planned, attempted, completed, censored, failed, unscoreable, and unstarted units follow one fixed stop rule.",
        "acceptance_gate_results": "Validation, safety, completion, and efficacy checks remain separate.",
        "gate_check_summary": "Every blocked result names upstream, path, check, field, expected, and observed values.",
        "verifier_is_oracle": "Independent evaluation of a learned risk score does not make that score the correctness oracle.",
        "honest_verdict": "Completed findings start complete_; unchanged missing external inputs start blocked_.",
        "verdict_class": "The closed class is positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Critical producer or verifier findings prevent readiness and value.",
        "validation_receipts": "Receipts retain exact argv, environment, exit, duration, and hashed logs, including failures.",
        "repository_health": "Unrelated broad health stays separate while affected failures remain disqualifying.",
        "field_principles": "Principles explain ordinary fields without wrapping their machine-readable values.",
        "promotion_score": "Always zero; the trial cannot roll out, update generator weights, publish, or submit.",
        "online_capture_complete_score": "Complete valid stream evidence scores one independently of benefit.",
        "online_value_score": "Only the full registered primary efficacy conjunction scores one.",
        "numeric_checkpoint_manifest": "Initial-only code-free checkpoints let a fresh NumPy process reproduce scores.",
        "causal_erasure_rows": "Matched queries attribute changes only to earlier verified feedback.",
    }
    return {
        str(key): principles.get(
            str(key), "This ordinary field retains direct machine-readable experiment evidence."
        )
        for key in keys
    }


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]], source_hashes: Mapping[str, str]
) -> JsonDict:
    """Publish an external structured-gate failure without dependent work."""

    failed = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else precondition_row("unknown_precondition", "unknown", "unknown", True, None)
    )
    artifact = {
        **_base_artifact(),
        "status": "blocked_online_trial_precondition",
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "completed_at_utc": "2026-09-18T00:00:00+00:00",
        "duration_s": 0.0,
        "phase_spans": [],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [],
        "rows": [],
        "event_ledger": [],
        "numeric_checkpoint_manifest": [],
        "causal_erasure_rows": [],
        "restart_receipts": [],
        "policy_rows": [],
        "moving_block_intervals": [],
        "condition_reports": [],
        "latency_measurements": latency_report([], serialized_bytes=0, update_operations=0),
        "sample_size_budget": {
            "planned_prediction_rows": 1321 * len(ARMS) * len(TRAINING_SEEDS) * 6,
            "attempted_prediction_rows": 0,
            "completed_prediction_rows": 0,
            "censored_prediction_rows": 0,
            "failed_units": 0,
            "unscoreable_units": 0,
            "unstarted_prediction_rows": 1321 * len(ARMS) * len(TRAINING_SEEDS) * 6,
            "planned_feedback_units": 1321 * len(TRAINING_SEEDS) * 6,
            "effective_independent_group_count": 0,
            "stopping_rule": "Stop before dependent work after any failed structured prerequisite.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_required_passed": False,
            "failed_required_count": len(failed) or 1,
            "first_required_failure": first,
            "failed_scientific_gate_count": 0,
            "first_scientific_failure": None,
        },
        "honest_verdict": "blocked_online_trial_structured_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_evaluated",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_findings": [],
        },
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "primary_value_reduction": {"passed": False, "checks": []},
        "online_protocol": {},
        "small_ebm_training": {"performed": False, "current_llm_calls": 0},
        "evidence_scope": {"class": "blocked_before_replay"},
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "independent_reduction": {},
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _fixture_receipts() -> list[JsonDict]:
    names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_e2e",
        "independent_reducer",
        "independent_cold_replay",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    )
    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.1,
            "command_argv": [name],
            "command_environment": {},
            "log_path": f"/tmp/{name}.log",
            "log_sha256": canonical_hash(name),
        }
        for name in names
    ]


def build_fixture_artifact() -> JsonDict:
    """Build a compact complete-null artifact for mutation tests."""

    rows = []
    for arm in ARMS:
        rows.append(
            {
                "ordering": "fixed_hash_order",
                "feedback_condition": "primary_delay_1",
                "stream_index": 0,
                "group_id": "fixture-group",
                "arm": arm,
                "seed": TRAINING_SEEDS[0],
                "label": 0,
                "probability": 0.25,
                "pre_update_probability": 0.25,
                "decision": "escalate",
                "typed_action": "escalate",
                "feedback_available_at": 1,
                "label_availability_time": 1,
                "feedback_disposition": "pending",
                "state_hash_before_update": canonical_hash([arm, "before"]),
                "state_hash_after_update": canonical_hash([arm, "before"]),
                "prediction_before_feedback": True,
                "brier_loss": 0.0625,
                "log_loss": -math.log(0.75),
                **{field: 0.001 for field in LATENCY_FIELDS},
            }
        )
    manifests = [
        {
            "seed": seed,
            "path": f"/tmp/seed-{seed}.json",
            "checkpoint_hash": canonical_hash({"seed": seed}),
            "file_sha256": canonical_hash({"file": seed}),
            "serialized_bytes": 100,
            "future_stream_groups_used": 0,
        }
        for seed in TRAINING_SEEDS
    ]
    erasures = [
        {
            "seed": seed,
            "passed": True,
            "erased_probability": 0.25,
            "no_feedback_probability": 0.25,
        }
        for seed in TRAINING_SEEDS
    ]
    report = {
        "ordering": "fixed_hash_order",
        "feedback_condition": "primary_delay_1",
        "arm_metrics": {
            arm: {
                "mean_log_loss": 0.3,
                "coverage": 0.0,
                "accept_risk": None,
                "reject_risk": None,
                "accept_count": 0,
                "reject_count": 0,
            }
            for arm in ARMS
        },
    }
    value = sealed.reduce_primary_value([], [report])
    gates = [
        _gate("prediction_rows_complete", "completion", len(rows), len(rows)),
        _gate("required_validation", "validation", True, True),
        _gate("registered_value", "scientific_efficacy", True, value["passed"]),
    ]
    artifact = {
        **_base_artifact(),
        "status": "complete_online_trial_null",
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "completed_at_utc": "2026-09-18T00:00:01+00:00",
        "duration_s": 1.0,
        "phase_spans": [],
        "preconditions_checked": [{"check": "fixture", "passed": True}],
        "source_artifact_hashes": {},
        "historical_inference_sidecars": [],
        "rows": rows,
        "event_ledger": [
            {
                "event_id": "fixture-event",
                "feedback_available_at": 1,
                "label_authority": LABEL_AUTHORITY,
                "prediction_before_feedback": True,
            }
        ],
        "numeric_checkpoint_manifest": manifests,
        "causal_erasure_rows": erasures,
        "restart_receipts": [{"performed": True, "prediction_parity": True}],
        "policy_rows": [],
        "moving_block_intervals": [],
        "condition_reports": [report],
        "latency_measurements": latency_report(rows, serialized_bytes=500, update_operations=0),
        "sample_size_budget": {
            "planned_prediction_rows": len(rows),
            "attempted_prediction_rows": len(rows),
            "completed_prediction_rows": len(rows),
            "censored_prediction_rows": 0,
            "failed_units": 0,
            "unscoreable_units": 0,
            "unstarted_prediction_rows": 0,
            "planned_feedback_units": 1,
            "effective_independent_group_count": 1,
            "stopping_rule": "Run every frozen fixture unit once.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": "complete_null_online_registered_benefit_not_demonstrated",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": _fixture_receipts(),
        "repository_health": {
            "status": "healthy_for_affected_scope",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_findings": [],
        },
        "online_capture_complete_score": 1,
        "online_value_score": 0,
        "primary_value_reduction": value,
        "online_protocol": {"orders_are_real_chronology": False, "final_test_labels_read": False},
        "small_ebm_training": {"performed": True, "current_llm_calls": 0},
        "evidence_scope": {
            "class": "controlled_archive_replay_not_real_world_chronology",
            "conformal_guarantee": False,
            "real_time_deployment_claim": False,
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "independent_reduction": {},
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["independent_reduction"] = independent_reduce(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, reductions, and evidence hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    identity = (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
        or not isinstance(artifact.get("inference_substrate"), str)
    ):
        errors.append("substrate_declaration_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") or artifact.get("validation_receipts"):
            errors.append("blocked_artifact_has_dependent_work")
        failure = (artifact.get("gate_check_summary") or {}).get("first_required_failure")
        if not isinstance(failure, Mapping) or not {
            "upstream",
            "check",
            "artifact_field",
            "expected",
            "observed",
        } <= set(failure):
            errors.append("blocked_gate_summary_missing")
        if any(
            artifact.get(name) != 0
            for name in ("online_capture_complete_score", "online_value_score", "promotion_score")
        ):
            errors.append("blocked_scores_nonzero")
    else:
        reduced = independent_reduce(artifact)
        if artifact.get("independent_reduction") != reduced:
            errors.append("independent_reduction_mismatch")
        if (
            artifact.get("online_capture_complete_score")
            != reduced["online_capture_complete_score"]
        ):
            if "independent_reduction_mismatch" not in errors:
                errors.append("independent_reduction_mismatch")
        if artifact.get("online_value_score") != reduced["online_value_score"]:
            if "independent_reduction_mismatch" not in errors:
                errors.append("independent_reduction_mismatch")
        if not _manifest_valid(artifact.get("numeric_checkpoint_manifest")):
            errors.append("numeric_checkpoint_manifest_invalid")
    if artifact.get("flagged_adversarial") is True and any(
        artifact.get(name) != 0
        for name in ("online_capture_complete_score", "online_value_score", "promotion_score")
    ):
        errors.append("adversarial_scores_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def validate_candidate_artifact(value: object) -> list[str]:  # pragma: no cover - subprocess.
    """Validate measured evidence before terminal receipts can exist."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors = []
    artifact = dict(value)
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("candidate_identity_mismatch")
    budget = artifact.get("sample_size_budget") or {}
    if len(artifact.get("rows") or []) != int(budget.get("planned_prediction_rows") or -1):
        errors.append("candidate_rows_incomplete")
    if not _manifest_valid(artifact.get("numeric_checkpoint_manifest")):
        errors.append("numeric_checkpoint_manifest_invalid")
    if not all(row.get("passed") is True for row in artifact.get("causal_erasure_rows") or []):
        errors.append("causal_erasure_failed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


_COLD_FIELDS = (
    "ordering",
    "feedback_condition",
    "delay",
    "missing_fraction",
    "stream_index",
    "group_id",
    "source_row_index",
    "arm",
    "seed",
    "label",
    "raw_energy",
    "probability",
    "pre_update_probability",
    "brier_loss",
    "log_loss",
    "decision",
    "typed_action",
    "action_harm",
    "churn_contribution",
    "prediction_before_feedback",
    "feedback_available_at",
    "label_availability_time",
    "label_authority",
    "state_hash",
    "state_hash_before_update",
    "state_hash_after_update",
    "gibbs_weights_hash",
    "update_count",
    "feedback_disposition",
    "disposition",
)


def cold_replay_errors(
    artifact: Mapping[str, Any], protocol: Mapping[str, Any], root: Path
) -> list[str]:  # pragma: no cover - fresh-process capability E2E.
    """Recompute every scientific row from code-free numeric checkpoints."""

    streams = sealed.build_streams(protocol)
    initial_labels = [int(row["label"]) for row in sealed.initialization_rows(protocol)]
    missing = sealed.deterministic_missing_mask(
        [str(row["group_id"]) for row in streams["fixed_hash_order"]]
    )
    policies = {int(row["seed"]): row["policy"] for row in artifact.get("policy_rows", [])}
    manifests = {int(row["seed"]): row for row in artifact.get("numeric_checkpoint_manifest", [])}
    checkpoints = {seed: _load_object(root / manifests[seed]["path"]) for seed in TRAINING_SEEDS}
    states = {
        seed: states_from_numeric_checkpoint(checkpoints[seed], initial_labels)
        for seed in TRAINING_SEEDS
    }
    recomputed: list[JsonDict] = []
    causal: list[JsonDict] = []
    private = Path(tempfile.mkdtemp(prefix="exp7399-cold-", dir="/tmp"))
    for seed in TRAINING_SEEDS:
        causal.append(
            causal_erasure_probe(
                states[seed], streams["fixed_hash_order"], seed=seed, prefix_count=8
            )
        )
    for ordering in ORDERINGS:
        for condition in sealed.feedback_conditions():
            unit_missing = missing if float(condition["missing_fraction"]) > 0 else set()
            for seed in TRAINING_SEEDS:
                measured = measure_replay_unit(
                    states[seed],
                    streams[ordering],
                    ordering=ordering,
                    condition=condition,
                    seed=seed,
                    thresholds=policies[seed],
                    missing_group_ids=unit_missing,
                    checkpoint=checkpoints[seed],
                    durable_path=private / f"{ordering}-{condition['name']}-{seed}.json",
                )
                recomputed.extend(measured["rows"])
    expected = [
        {field: row.get(field) for field in _COLD_FIELDS} for row in artifact.get("rows", [])
    ]
    observed = [{field: row.get(field) for field in _COLD_FIELDS} for row in recomputed]
    errors = (
        [] if canonical_hash(expected) == canonical_hash(observed) else ["cold_row_replay_mismatch"]
    )
    expected_causal = [
        {key: value for key, value in row.items() if key not in {"duration_s"}}
        for row in artifact.get("causal_erasure_rows", [])
    ]
    if canonical_hash(expected_causal) != canonical_hash(causal):
        errors.append("cold_causal_erasure_mismatch")
    return errors


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = str(REPO_ROOT / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7399_v649_online_trial import validate_candidate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_candidate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "candidate"
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
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_cold_replay",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7399_v649_online_trial",
                    "--cold-replay",
                    str(candidate),
                ),
                "capability_end_to_end",
                1800.0,
            ),
            "completion",
            True,
        ),
    ]


def _utc_now() -> str:  # pragma: no cover - wall-clock boundary.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7399] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(phase: str, phase_started: float, run_started: float) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "checkpoint_utc": _utc_now(),
    }


def _historical_sidecar(root: Path, upstream: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    sidecar = {
        "schema": "carnot.exp7399.upstream_context.v1",
        "source_path": UPSTREAM_PATH.as_posix(),
        "source_sha256": sha256_file(root / UPSTREAM_PATH),
        "status": upstream.get("status"),
        "verdict_class": upstream.get("verdict_class"),
        "flagged_adversarial": upstream.get("flagged_adversarial"),
        "delayed_adapter_ready_score": upstream.get("delayed_adapter_ready_score"),
        "historical_invocation_counters_copied": False,
        "historical_substrate_copied": False,
        "current_llm_invocations": 0,
    }
    path = root / RAW_DIR / "upstream_exp7397_context.json"
    atomic_json(path, sidecar)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "scope": "eligible_protocol_and_tested_adapter_context_no_current_calls",
    }


def _entrypoint_receipt(
    root: Path, started_at: str, duration_s: float
) -> JsonDict:  # pragma: no cover
    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.log"
    argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    record = {
        "event": "declared_entrypoint_reached_terminal_validation",
        "argv": argv,
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "duration_s": duration_s,
    }
    atomic_json(path, record)
    return {
        "name": "declared_entrypoint_e2e",
        "command": " ".join(argv),
        "command_argv": argv,
        "scope": "capability_end_to_end",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": sha256_file(path),
        "passed": True,
        "timed_out": False,
        "output_tail": "declared entrypoint reached terminal validation",
        "command_category": "completion",
        "required": True,
        "started_at_utc": started_at,
        "ended_at_utc": record["ended_at_utc"],
        "command_environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }


def _assemble_artifact(  # pragma: no cover - measured execution assembly.
    *,
    started: float,
    started_at_utc: str,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    upstream_sidecar: Mapping[str, Any],
    protocol: Mapping[str, Any],
    checkpoint_manifest: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    ledgers: Sequence[Mapping[str, Any]],
    causal_rows: Sequence[Mapping[str, Any]],
    restarts: Sequence[Mapping[str, Any]],
    unit_receipts: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    fit_duration_s: float,
    flagged_adversarial: bool,
) -> JsonDict:
    value = sealed.reduce_primary_value(intervals, reports)
    failed_units = sum(int(row.get("failed_units", 0)) for row in unit_receipts)
    unscoreable_units = sum(int(row.get("unscoreable_units", 0)) for row in unit_receipts)
    serialized_bytes = sum(int(row.get("durable_state_bytes", 0)) for row in unit_receipts) + sum(
        int(row.get("serialized_bytes", 0)) for row in checkpoint_manifest
    )
    update_operations = sum(int(row.get("update_operation_count", 0)) for row in unit_receipts)
    affected_pass = set(validation_scope.REQUIRED_CHECK_NAMES) <= {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    terminal_pass = _required_receipts_pass(receipts)
    expected_rows = (
        len(ORDERINGS)
        * len(sealed.feedback_conditions())
        * len(TRAINING_SEEDS)
        * len(ARMS)
        * len((protocol.get("online_replay") or {}).get("later_group_ids") or [])
    )
    expected_ledgers = expected_rows // len(ARMS)
    gates = [
        _gate(
            "structured_preconditions",
            "completion",
            True,
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
        ),
        _gate("prediction_rows_complete", "completion", expected_rows, len(rows)),
        _gate("feedback_ledgers_complete", "completion", expected_ledgers, len(ledgers)),
        _gate(
            "numeric_checkpoints_complete",
            "completion",
            len(TRAINING_SEEDS),
            len(checkpoint_manifest),
        ),
        _gate(
            "causal_erasure_complete",
            "safety",
            True,
            len(causal_rows) == len(TRAINING_SEEDS)
            and all(row.get("passed") is True for row in causal_rows),
        ),
        _gate(
            "cold_restart_complete",
            "safety",
            True,
            bool(restarts)
            and all(
                row.get("performed") is True and row.get("prediction_parity") is True
                for row in restarts
            ),
        ),
        _gate("failed_units", "completion", 0, failed_units),
        _gate("unscoreable_units", "completion", 0, unscoreable_units),
        _gate("affected_validation", "validation", True, affected_pass),
        _gate("terminal_validation", "validation", True, terminal_pass),
        _gate("adversarial_clear", "safety", False, flagged_adversarial),
        *[deepcopy(dict(row)) for row in value["checks"]],
    ]
    replay = protocol.get("online_replay") or {}
    pending = sum(int(row.get("pending_feedback_at_end", 0)) for row in unit_receipts)
    missing = sum(row.get("missing") is True for row in ledgers)
    artifact: JsonDict = {
        **_base_artifact(),
        "status": "complete_online_trial_pending_reduction",
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "duration_s": time.monotonic() - started,
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [deepcopy(dict(upstream_sidecar))],
        "rows": [deepcopy(dict(row)) for row in rows],
        "event_ledger": [deepcopy(dict(row)) for row in ledgers],
        "numeric_checkpoint_manifest": [deepcopy(dict(row)) for row in checkpoint_manifest],
        "causal_erasure_rows": [deepcopy(dict(row)) for row in causal_rows],
        "restart_receipts": [deepcopy(dict(row)) for row in restarts],
        "unit_checkpoint_receipts": [deepcopy(dict(row)) for row in unit_receipts],
        "policy_rows": [deepcopy(dict(row)) for row in policies],
        "moving_block_intervals": [deepcopy(dict(row)) for row in intervals],
        "condition_reports": [deepcopy(dict(row)) for row in reports],
        "latency_measurements": latency_report(
            rows, serialized_bytes=serialized_bytes, update_operations=update_operations
        ),
        "sample_size_budget": {
            "planned_prediction_rows": expected_rows,
            "attempted_prediction_rows": len(rows),
            "completed_prediction_rows": len(rows),
            "censored_prediction_rows": 0,
            "failed_units": failed_units,
            "unscoreable_units": unscoreable_units,
            "unstarted_prediction_rows": max(0, expected_rows - len(rows)),
            "planned_feedback_units": expected_ledgers,
            "attempted_feedback_units": len(ledgers),
            "completed_feedback_units": sum(row.get("feedback") is not None for row in ledgers),
            "censored_feedback_units": pending + missing,
            "effective_independent_group_count": len(replay.get("later_group_ids") or []),
            "maximum_initial_training_steps": MAX_STEPS,
            "moving_block_draws": BOOTSTRAP_DRAWS,
            "stopping_rule": "Run every frozen order, condition, seed, arm, and event once; never stop from efficacy.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": "complete_disqualified_pending_required_validation",
        "verdict_class": "disqualified",
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": {
            "status": "healthy_for_affected_scope" if affected_pass else "affected_checks_failed",
            "as_of": RUN_DATE,
            "affects_required_checks": not affected_pass,
            "unrelated_findings": [],
            "full_python_suite_launched_by_experiment": False,
        },
        "online_capture_complete_score": 0,
        "online_value_score": 0,
        "primary_value_reduction": value,
        "online_protocol": {
            "source": UPSTREAM_PATH.as_posix(),
            "source_sha256": EXPECTED_UPSTREAM_SHA256,
            "sealed_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "initialization_group_count": len(replay.get("initialization_group_ids") or []),
            "later_group_count": len(replay.get("later_group_ids") or []),
            "final_test_labels_read": False,
            "orders": list(ORDERINGS),
            "orders_are_real_chronology": False,
            "conditions": list(sealed.feedback_conditions()),
            "fixed_missing_mask": "every fourth later-group identity in original fixed order",
            "training_seeds": list(TRAINING_SEEDS),
            "arms": list(ARMS),
            "threshold_source": "disjoint policy_calibration partition",
            "thresholds_tuned_after_outcomes": False,
            "steps_tuned_after_outcomes": False,
            "same_event_and_feedback_mask_for_every_arm": True,
            "moving_block_draws": BOOTSTRAP_DRAWS,
            "moving_block_lengths": list(BLOCK_LENGTHS),
            "moving_block_seed": BOOTSTRAP_SEED,
            "seed_reduction": "average_within_event_before_resampling",
            "risk_and_coverage_scope": "empirical_controlled_archive_replay",
        },
        "small_ebm_training": {
            "performed": True,
            "kind": "initial 2-4-1 Gibbs fit and numeric affine initialization only",
            "training_seeds": list(TRAINING_SEEDS),
            "maximum_steps_per_fit": MAX_STEPS,
            "fit_duration_s": fit_duration_s,
            "initial_groups_only": True,
            "gibbs_frozen_during_stream": True,
            "generator_loaded": False,
            "generator_weights_changed": False,
            "current_llm_calls": 0,
        },
        "evidence_scope": {
            "class": "tier_1_continuous_self_learning_controlled_archive_replay",
            "real_world_chronology": False,
            "real_time_deployment_claim": False,
            "generator_update_claim": False,
            "conformal_guarantee": False,
            "measured_100x_claim": False,
            "scientific_benefit_claim": False,
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "independent_reduction": {},
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    reduced = independent_reduce(artifact)
    artifact["independent_reduction"] = reduced
    artifact["online_capture_complete_score"] = reduced["online_capture_complete_score"]
    artifact["online_value_score"] = reduced["online_value_score"]
    if reduced["online_capture_complete_score"] == 0:
        artifact["status"] = "complete_online_trial_disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_validation_or_safety_failure"
        artifact["verdict_class"] = "disqualified"
    elif reduced["online_value_score"] == 1:
        artifact["status"] = "complete_online_trial_positive"
        artifact["honest_verdict"] = "complete_positive_online_registered_gate_passed_no_promotion"
        artifact["verdict_class"] = "positive"
        artifact["evidence_scope"]["scientific_benefit_claim"] = True
    else:
        artifact["status"] = "complete_online_trial_null"
        artifact["honest_verdict"] = "complete_null_online_registered_benefit_not_demonstrated"
        artifact["verdict_class"] = "null"
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(  # pragma: no cover - exercised by declared entrypoint.
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run sealed fitting, paired replay, scoped checks, and cold validation."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = _utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    _progress(started, "preconditions", "start")
    preconditions, source_hashes, upstream, protocol = collect_preconditions(root)
    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    spans.append(_span("preconditions", phase_started, started))
    _progress(started, "preconditions", "end", passed=preconditions_passed)
    if not preconditions_passed:
        blocked = build_blocked_artifact(preconditions, source_hashes)
        blocked["started_at_utc"] = started_at
        blocked["completed_at_utc"] = _utc_now()
        blocked["duration_s"] = time.monotonic() - started
        blocked["phase_spans"] = spans
        blocked["field_principles"] = _field_principles(tuple(blocked))
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        _progress(started, "write", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        _progress(started, "write", "after_atomic_blocked", status=blocked["status"])
        return blocked

    raw_dir = root / RAW_DIR
    checkpoint_dir = root / CHECKPOINT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    upstream_sidecar = _historical_sidecar(root, upstream)

    phase_started = time.monotonic()
    _progress(started, "model_load", "before", operation="not_attempted")
    spans.append(_span("model_load", phase_started, started))
    _progress(started, "model_load", "after", model_invoked=False)
    phase_started = time.monotonic()
    _progress(started, "generation", "before", operation="not_attempted")
    spans.append(_span("generation", phase_started, started))
    _progress(started, "generation", "after", generation_calls=0)

    initial = sealed.initialization_rows(protocol)
    initial_labels = [int(row["label"]) for row in initial]
    policy_source = sealed.policy_calibration_rows(protocol)
    initialized: dict[int, dict[str, JsonDict]] = {}
    checkpoints: dict[int, JsonDict] = {}
    checkpoint_manifest: list[JsonDict] = []
    policies: list[JsonDict] = []
    phase_started = time.monotonic()
    fit_started = time.monotonic()
    _progress(started, "initial_training", "start", units=len(TRAINING_SEEDS))
    for unit, seed in enumerate(TRAINING_SEEDS, start=1):
        _progress(
            started,
            "initial_training",
            "before_fit",
            unit=f"{unit}/{len(TRAINING_SEEDS)}",
            seed=seed,
        )
        fitted = sealed.initialize_seed_states(initial, seed)
        checkpoint = make_numeric_checkpoint(fitted["training_record"])
        checkpoint_path = checkpoint_dir / f"initial-seed{seed}.json"
        atomic_json(checkpoint_path, checkpoint)
        restored = _load_object(checkpoint_path)
        if checkpoint_hash(restored) != checkpoint_hash(checkpoint):
            raise RuntimeError("numeric_checkpoint_round_trip_mismatch")
        states = states_from_numeric_checkpoint(restored, initial_labels)
        policy = sealed.select_fixed_thresholds(states, policy_source)
        initialized[seed] = states
        checkpoints[seed] = restored
        policies.append({"seed": seed, "policy": policy})
        checkpoint_manifest.append(
            {
                "seed": seed,
                "path": checkpoint_path.relative_to(root).as_posix(),
                "checkpoint_hash": checkpoint_hash(restored),
                "file_sha256": sha256_file(checkpoint_path),
                "serialized_bytes": checkpoint_path.stat().st_size,
                "training_authority": restored["training_authority"],
                "initial_group_ids_sha256": restored["initial_group_ids_sha256"],
                "future_stream_groups_used": restored["future_stream_groups_used"],
                "probe": {
                    "features": [
                        float(initial[0]["entity_uptake"]),
                        float(initial[0]["falsifiability_score"]),
                    ],
                    **score_numeric_checkpoint(
                        restored,
                        [
                            float(initial[0]["entity_uptake"]),
                            float(initial[0]["falsifiability_score"]),
                        ],
                    ),
                },
            }
        )
        _progress(
            started,
            "initial_training",
            "after_fit",
            unit=f"{unit}/{len(TRAINING_SEEDS)}",
            seed=seed,
        )
    fit_duration = time.monotonic() - fit_started
    spans.append(_span("initial_training", phase_started, started))
    _progress(started, "initial_training", "end", duration_s=f"{fit_duration:.3f}")

    streams = sealed.build_streams(protocol)
    missing = sealed.deterministic_missing_mask(
        [str(row["group_id"]) for row in streams["fixed_hash_order"]]
    )
    policy_by_seed = {int(row["seed"]): row["policy"] for row in policies}
    all_rows: list[JsonDict] = []
    ledgers: list[JsonDict] = []
    restarts: list[JsonDict] = []
    unit_receipts: list[JsonDict] = []
    total_units = len(ORDERINGS) * len(sealed.feedback_conditions()) * len(TRAINING_SEEDS)
    completed = 0
    phase_started = time.monotonic()
    _progress(started, "measurement", "start", units=total_units)
    for ordering in ORDERINGS:
        for condition in sealed.feedback_conditions():
            unit_missing = missing if float(condition["missing_fraction"]) > 0 else set()
            for seed in TRAINING_SEEDS:
                _progress(
                    started,
                    "measurement",
                    "before_replay",
                    ordering=ordering,
                    condition=condition["name"],
                    seed=seed,
                )
                path = checkpoint_dir / f"{ordering}-{condition['name']}-seed{seed}.json"
                result = measure_replay_unit(
                    initialized[seed],
                    streams[ordering],
                    ordering=ordering,
                    condition=condition,
                    seed=seed,
                    thresholds=policy_by_seed[seed],
                    missing_group_ids=unit_missing,
                    checkpoint=checkpoints[seed],
                    durable_path=path,
                )
                all_rows.extend(result.pop("rows"))
                ledgers.extend(result.pop("event_ledger"))
                restarts.append(result.pop("restart_receipt"))
                unit_receipts.append(result)
                completed += 1
                _progress(
                    started,
                    "measurement",
                    "after_replay",
                    completed=f"{completed}/{total_units}",
                    rows=len(all_rows),
                )

    causal_rows = []
    _progress(started, "measurement", "before_causal_erasure", units=len(TRAINING_SEEDS))
    for seed in TRAINING_SEEDS:
        causal_rows.append(
            causal_erasure_probe(
                initialized[seed], streams["fixed_hash_order"], seed=seed, prefix_count=8
            )
        )
    _progress(
        started,
        "measurement",
        "after_causal_erasure",
        passed=all(row["passed"] for row in causal_rows),
    )
    _progress(started, "measurement", "before_moving_block_bootstrap", draws=BOOTSTRAP_DRAWS)
    intervals = sealed.paired_moving_block_intervals(all_rows)
    _progress(started, "measurement", "after_moving_block_bootstrap", intervals=len(intervals))
    reports = sealed.condition_reports(all_rows)
    spans.append(_span("measurement", phase_started, started))
    _progress(started, "measurement", "end", rows=len(all_rows), ledgers=len(ledgers))

    private_root = Path(tempfile.mkdtemp(prefix="exp7399-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    commands = build_validation_commands(root, private_root)
    plan_errors = validate_command_plan(root, V649_MANIFEST, commands)
    _progress(started, "affected_validation", "before_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    spans.append(_span("affected_validation", phase_started, started))
    _progress(started, "affected_validation", "after_subprocesses", receipts=len(affected))

    candidate = _assemble_artifact(
        started=started,
        started_at_utc=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        upstream_sidecar=upstream_sidecar,
        protocol=protocol,
        checkpoint_manifest=checkpoint_manifest,
        policies=policies,
        rows=all_rows,
        ledgers=ledgers,
        causal_rows=causal_rows,
        restarts=restarts,
        unit_receipts=unit_receipts,
        intervals=intervals,
        reports=reports,
        receipts=affected,
        fit_duration_s=fit_duration,
        flagged_adversarial=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    phase_started = time.monotonic()
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    spans.append(_span("candidate_write", phase_started, started))
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root, _terminal_commands(candidate_path), log_dir=raw_dir / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, started))
    adversarial_failed = any(
        row.get("name") == "adversarial_verify" and row.get("passed") is not True
        for row in terminal
    )
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in terminal),
        adversarial_failed=adversarial_failed,
    )

    entrypoint = _entrypoint_receipt(root, started_at, time.monotonic() - started)
    final = _assemble_artifact(
        started=started,
        started_at_utc=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        upstream_sidecar=upstream_sidecar,
        protocol=protocol,
        checkpoint_manifest=checkpoint_manifest,
        policies=policies,
        rows=all_rows,
        ledgers=ledgers,
        causal_rows=causal_rows,
        restarts=restarts,
        unit_receipts=unit_receipts,
        intervals=intervals,
        reports=reports,
        receipts=[*affected, *terminal, entrypoint],
        fit_duration_s=fit_duration,
        flagged_adversarial=adversarial_failed,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse public execution and fresh-process cold replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Dispatch the thin entrypoint or independent replay reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        artifact = _load_object(args.cold_replay)
        candidate_errors = validate_candidate_artifact(artifact)
        protocol = _load_object(REPO_ROOT / PROTOCOL_PATH)
        replay_errors = (
            cold_replay_errors(artifact, protocol, REPO_ROOT) if not candidate_errors else []
        )
        errors = [*candidate_errors, *replay_errors]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

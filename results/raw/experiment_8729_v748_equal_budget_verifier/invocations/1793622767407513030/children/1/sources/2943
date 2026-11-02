"""Measure recovery from persisted autoresearch rejection tails.

The experiment uses private logs. It never reads or rewrites the live resume
cache. One scripted proposal trains the existing compact Gibbs head and crosses
the conductor's real subprocess recompute path without calling a synthesis LLM.

Spec refs: REQ-AUTO-026 and SCENARIO-AUTO-026-A through
SCENARIO-AUTO-026-D.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.orchestrator import (
    AutoresearchConfig,
    run_loop,
    run_loop_with_generator,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "experiment_7435_v652_round_breaker"
SCHEMA = "carnot.exp7435.v652.round_breaker.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7435_v652_round_breaker.json")
RAW_DIR = Path("results/raw/experiment_7435_v652_round_breaker")
MODULE_PATH = Path("python/carnot/experiment_7435_v652_round_breaker.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7435_v652_round_breaker.py")
TEST_PATH = Path("tests/python/test_experiment_7435_v652_round_breaker.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
SEED = 7435
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
REQUIRED_VALIDATION_NAMES = frozenset(
    {
        *validation_scope.REQUIRED_CHECK_NAMES,
        "fresh_process_replay",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
)
REQUIRED_CASES = frozenset(
    {
        "historical_tail_recovery",
        "fresh_rejection_limit",
        "success_breaks_streak",
        "execution_failure_limit",
        "empty_log",
        "successive_invocation_recovery",
        "conductor_calibrated_decision",
    }
)
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("python/carnot/autoresearch/experiment_log.py"),
    Path("python/carnot/autoresearch/orchestrator.py"),
    Path("scripts/autoresearch_conductor_round.py"),
    Path("tests/python/test_autoresearch_orchestrator.py"),
    Path("tests/python/test_autoresearch_skills_loop.py"),
    Path("tests/python/test_autoresearch_conductor_round.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
)
MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(
        TEST_PATH.as_posix(),
        "tests/python/test_autoresearch_orchestrator.py",
        "tests/python/test_autoresearch_skills_loop.py",
        "tests/python/test_autoresearch_conductor_round.py",
    ),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(
        WRAPPER_PATH.as_posix(),
        "python/carnot/autoresearch/experiment_log.py",
        "python/carnot/autoresearch/orchestrator.py",
        "scripts/autoresearch_conductor_round.py",
    ),
)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete record while excluding only the checksum itself."""

    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness from raw breaker rows and command exits."""

    rows = {
        str(row.get("case_id")): row
        for row in artifact.get("breaker_rows", [])
        if isinstance(row, Mapping)
    }
    case_passes = {
        "historical_tail_recovery": bool(
            rows.get("historical_tail_recovery", {}).get("historical_count") == 10
            and rows.get("historical_tail_recovery", {}).get("proposal_count", 0) >= 1
            and rows.get("historical_tail_recovery", {}).get("evaluation_count", 0) >= 1
            and rows.get("historical_tail_recovery", {}).get("passed") is True
        ),
        "fresh_rejection_limit": bool(
            rows.get("fresh_rejection_limit", {}).get("local_count") == 10
            and rows.get("fresh_rejection_limit", {}).get("evaluation_count") == 10
            and rows.get("fresh_rejection_limit", {}).get("stop_reason") == "circuit_breaker"
            and rows.get("fresh_rejection_limit", {}).get("passed") is True
        ),
        "success_breaks_streak": bool(
            rows.get("success_breaks_streak", {}).get("local_count") == 1
            and rows.get("success_breaks_streak", {}).get("passed") is True
        ),
        "execution_failure_limit": bool(
            rows.get("execution_failure_limit", {}).get("local_count") == 2
            and rows.get("execution_failure_limit", {}).get("stop_reason") == "circuit_breaker"
            and rows.get("execution_failure_limit", {}).get("passed") is True
        ),
        "empty_log": bool(
            rows.get("empty_log", {}).get("local_count") == 0
            and rows.get("empty_log", {}).get("passed") is True
        ),
        "successive_invocation_recovery": bool(
            rows.get("successive_invocation_recovery", {}).get("historical_count") == 12
            and rows.get("successive_invocation_recovery", {}).get("proposal_count", 0) >= 1
            and rows.get("successive_invocation_recovery", {}).get("evaluation_count", 0) >= 1
            and rows.get("successive_invocation_recovery", {}).get("passed") is True
        ),
        "conductor_calibrated_decision": bool(
            rows.get("conductor_calibrated_decision", {}).get("historical_count") == 10
            and rows.get("conductor_calibrated_decision", {}).get("proposal_count") == 1
            and rows.get("conductor_calibrated_decision", {}).get("evaluation_count") == 1
            and rows.get("conductor_calibrated_decision", {}).get("passed") is True
        ),
    }
    receipts = {
        str(row.get("name")): row
        for row in artifact.get("validation_receipts", [])
        if isinstance(row, Mapping)
    }
    validation_passed = all(
        name in receipts
        and receipts[name].get("passed") is True
        and receipts[name].get("exit_code") == 0
        and receipts[name].get("timed_out") is not True
        for name in REQUIRED_VALIDATION_NAMES
    )
    failed_cases = sorted(name for name in REQUIRED_CASES if not case_passes.get(name, False))
    safety_passed = bool(
        artifact.get("history_preserved") is True
        and artifact.get("rejected_id_deduplication_unchanged") is True
        and artifact.get("flagged_adversarial") is False
    )
    ready = int(not failed_cases and validation_passed and safety_passed)
    return {
        "breaker_recovery_ready_score": ready,
        "failed_breaker_cases": failed_cases,
        "required_validation_passed": validation_passed,
        "safety_passed": safety_passed,
    }


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, compute declarations, reduction, and byte binding."""

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
    ):
        errors.append("current_model_declaration_mismatch")
    if (
        artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_or_venue_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score_nonzero")
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
    reduction = independent_reduce(artifact)
    if artifact.get("breaker_recovery_ready_score") != reduction["breaker_recovery_ready_score"]:
        errors.append("stored_reduction_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _utc_now() -> str:  # pragma: no cover - exercised by the declared entrypoint.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7435] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
    }


def _canonical_hash(value: Any) -> str:  # pragma: no cover
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _entry(index: int, outcome: str = "rejected") -> ExperimentEntry:  # pragma: no cover
    return ExperimentEntry(
        id=f"historical-{index:02d}",
        timestamp=f"2026-09-19T00:00:{index:02d}+00:00",
        hypothesis_code="def run(data): return {}",
        hypothesis_description=f"persisted valid rejection {index}",
        sandbox_success=False,
        sandbox_error="scientific gate rejected the measured hypothesis",
        eval_verdict="FAIL" if outcome == "rejected" else "PASS",
        eval_reason="valid fixture disposition",
        outcome=outcome,
    )


def _historical_log(count: int = 10) -> ExperimentLog:  # pragma: no cover
    log = ExperimentLog()
    for index in range(count):
        log.append(_entry(index))
    return log


def _baselines() -> BaselineRecord:  # pragma: no cover
    return BaselineRecord(
        benchmarks={
            "fixture": BenchmarkMetrics(
                benchmark_name="fixture",
                final_energy=1.0,
                convergence_steps=1,
                wall_clock_seconds=0.01,
                peak_memory_mb=1.0,
            )
        }
    )


_REJECTED_CODE = """\
def run(benchmark_data):
    return {"fixture": {"final_energy": 2.0, "wall_clock_seconds": 0.01}}
"""
_ACCEPTED_CODE = """\
def run(benchmark_data):
    return {"fixture": {"final_energy": 0.5, "wall_clock_seconds": 0.01}}
"""
_FAILED_CODE = """\
def run(benchmark_data):
    raise RuntimeError("bounded fixture execution failure")
"""
_CALIBRATED_TRAINING_CODE = """\
def run(benchmark_data):
    import jax
    import jax.numpy as jnp

    GibbsConfig = benchmark_data["GibbsConfig"]
    GibbsModel = benchmark_data["GibbsModel"]
    nce_loss = benchmark_data["nce_loss"]
    correct = jnp.asarray(benchmark_data["calibrated_decision_train_correct"])
    incorrect = jnp.asarray(benchmark_data["calibrated_decision_train_incorrect"])
    model = GibbsModel(GibbsConfig(input_dim=2, hidden_dims=[4]), key=jax.random.PRNGKey(7435))
    model.output_weight = 0.1 * jax.random.normal(jax.random.PRNGKey(7436), (4,))
    params = (model.layers[0][0], model.layers[0][1], model.output_weight, model.output_bias)

    def loss_fn(current):
        model.layers[0] = (current[0], current[1])
        model.output_weight = current[2]
        model.output_bias = current[3]
        return nce_loss(model, correct, incorrect)

    first_loss = None
    last_loss = None
    for _step in range(32):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        if first_loss is None:
            first_loss = float(loss)
        last_loss = float(loss)
        params = tuple(value - 0.03 * grad for value, grad in zip(params, grads))
    return {
        "calibrated_decision": {
            "final_state": {
                "w1": params[0].tolist(),
                "b1": params[1].tolist(),
                "w_out": params[2].tolist(),
                "b_out": float(params[3]),
            },
            "training_steps": 32,
            "random_seed": 7435,
            "nce_loss_first": first_loss,
            "nce_loss_last": last_loss,
            "wall_clock_seconds": 0.0,
        }
    }
"""


def _generator(code: str, counter: list[int]):  # pragma: no cover
    def generate(*_args: Any) -> list[tuple[str, str]]:
        counter.append(1)
        return [("deterministic fixture proposal", code)]

    return generate


def _row(
    case_id: str,
    *,
    invocation_start: int,
    historical_count: int,
    local_count: int,
    proposals: int,
    evaluations: int,
    stop_reason: str,
    passed: bool,
    failures: Sequence[str] = (),
) -> JsonDict:  # pragma: no cover
    return {
        "case_id": case_id,
        "condition": "invocation_local_consecutive_rejection_budget",
        "source_group": "private_persisted_log_fixture",
        "seed": SEED,
        "invocation_boundary": invocation_start,
        "historical_count": historical_count,
        "local_count": local_count,
        "proposal_count_before": 0,
        "proposal_count_after": proposals,
        "proposal_count": proposals,
        "evaluation_count_before": 0,
        "evaluation_count_after": evaluations,
        "evaluation_count": evaluations,
        "stop_reason": stop_reason,
        "disposition": "complete" if passed else "failed",
        "failed": not passed,
        "censored": False,
        "unstarted": False,
        "failures": list(failures),
        "passed": passed,
    }


def _run_shared_fixtures() -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    rows: list[JsonDict] = []

    historical = _historical_log()
    old_rows = [asdict(entry) for entry in historical.entries]
    calls: list[int] = []
    result = run_loop_with_generator(
        _generator(_REJECTED_CODE, calls),
        _baselines(),
        {},
        AutoresearchConfig(max_iterations=1, max_consecutive_failures=10),
        historical,
    )
    preserved = [asdict(entry) for entry in historical.entries[:10]] == old_rows
    rows.append(
        _row(
            "historical_tail_recovery",
            invocation_start=10,
            historical_count=10,
            local_count=historical.consecutive_failures_since(10),
            proposals=len(calls),
            evaluations=result.iterations,
            stop_reason="max_iterations",
            passed=len(calls) == result.iterations == 1 and preserved,
        )
    )

    fresh = ExperimentLog()
    calls = []
    result = run_loop_with_generator(
        _generator(_REJECTED_CODE, calls),
        _baselines(),
        {},
        AutoresearchConfig(max_iterations=20, max_consecutive_failures=10),
        fresh,
    )
    rows.append(
        _row(
            "fresh_rejection_limit",
            invocation_start=0,
            historical_count=0,
            local_count=fresh.consecutive_failures_since(0),
            proposals=len(calls),
            evaluations=result.iterations,
            stop_reason="circuit_breaker" if result.circuit_breaker_tripped else "unexpected",
            passed=result.circuit_breaker_tripped and result.iterations == 10,
        )
    )

    streak = ExperimentLog()
    result = run_loop(
        [
            ("reject one", _REJECTED_CODE),
            ("reject two", _REJECTED_CODE),
            ("accepted break", _ACCEPTED_CODE),
            ("reject after break", _REJECTED_CODE),
        ],
        _baselines(),
        {},
        AutoresearchConfig(max_iterations=4, max_consecutive_failures=3),
        streak,
    )
    rows.append(
        _row(
            "success_breaks_streak",
            invocation_start=0,
            historical_count=0,
            local_count=streak.consecutive_failures_since(0),
            proposals=4,
            evaluations=result.iterations,
            stop_reason="max_iterations",
            passed=result.iterations == 4 and not result.circuit_breaker_tripped,
        )
    )

    failures = ExperimentLog()
    calls = []
    result = run_loop_with_generator(
        _generator(_FAILED_CODE, calls),
        _baselines(),
        {},
        AutoresearchConfig(max_iterations=5, max_consecutive_failures=2),
        failures,
    )
    rows.append(
        _row(
            "execution_failure_limit",
            invocation_start=0,
            historical_count=0,
            local_count=failures.consecutive_failures_since(0),
            proposals=len(calls),
            evaluations=result.iterations,
            stop_reason="circuit_breaker" if result.circuit_breaker_tripped else "unexpected",
            passed=result.iterations == 2
            and result.circuit_breaker_tripped
            and all(entry.sandbox_error for entry in failures.entries),
        )
    )

    empty = ExperimentLog()
    rows.append(
        _row(
            "empty_log",
            invocation_start=0,
            historical_count=0,
            local_count=empty.consecutive_failures_since(0),
            proposals=0,
            evaluations=0,
            stop_reason="empty",
            passed=empty.consecutive_failures_since(0) == 0,
        )
    )

    successive = _historical_log()
    calls = []
    first = run_loop_with_generator(
        _generator(_REJECTED_CODE, calls),
        _baselines(),
        {},
        AutoresearchConfig(max_iterations=5, max_consecutive_failures=2),
        successive,
    )
    second_start = len(successive)
    second_calls: list[int] = []
    second = run_loop_with_generator(
        _generator(_ACCEPTED_CODE, second_calls),
        _baselines(),
        {},
        AutoresearchConfig(max_iterations=1, max_consecutive_failures=2),
        successive,
    )
    rows.append(
        _row(
            "successive_invocation_recovery",
            invocation_start=second_start,
            historical_count=second_start,
            local_count=successive.consecutive_failures_since(second_start),
            proposals=len(second_calls),
            evaluations=second.iterations,
            stop_reason="max_iterations",
            passed=first.circuit_breaker_tripped
            and first.iterations == 2
            and second.accepted == 1
            and second.iterations == 1,
        )
    )
    rejected_ids_unchanged = len(successive.rejected_ids()) == 12
    return rows, {
        "history_preserved": preserved,
        "history_before_hash": _canonical_hash(old_rows),
        "history_after_hash": _canonical_hash([asdict(entry) for entry in historical.entries[:10]]),
        "rejected_id_deduplication_unchanged": rejected_ids_unchanged,
    }


def _run_conductor_fixture(private: Path) -> tuple[JsonDict, JsonDict]:  # pragma: no cover
    import scripts.autoresearch_conductor_round as conductor

    private.mkdir(parents=True, exist_ok=True)
    baseline_path = private / "baseline.json"
    log_path = private / "experiment_log.json"
    receipt_path = private / "receipt.md"
    baseline = conductor.seed_baselines()
    baseline.benchmarks["calibrated_decision"].final_energy = 0.0
    baseline.save(baseline_path)
    history = _historical_log()
    history.save(log_path)
    before_rows = [asdict(entry) for entry in history.entries]
    before_bytes = log_path.read_bytes()
    calls: list[int] = []

    def scripted_generator(*_args: Any, **_kwargs: Any) -> list[tuple[str, str]]:
        calls.append(1)
        return [("bounded deterministic calibrated-decision training", _CALIBRATED_TRAINING_CODE)]

    with (
        patch.object(conductor, "codex_available", return_value=True),
        patch.object(
            conductor,
            "generate_hypotheses_with_fallback",
            side_effect=scripted_generator,
        ),
    ):
        return_code = conductor.run_round(
            model="scripted-no-llm",
            max_iterations=1,
            project_root=private,
            baseline_cache=baseline_path,
            log_cache=log_path,
            receipt_path=receipt_path,
        )
    after = ExperimentLog.load(log_path)
    old_rows_after = [asdict(entry) for entry in after.entries[:10]]
    new_entries = after.entries[10:]
    new_entry = new_entries[0] if len(new_entries) == 1 else None
    metrics = (
        new_entry.sandbox_metrics.get("calibrated_decision", {}) if new_entry is not None else {}
    )
    recomputed = isinstance(metrics, Mapping) and isinstance(metrics.get("final_energy"), float)
    rejected = new_entry is not None and new_entry.outcome == "rejected"
    receipt = receipt_path.read_text(encoding="utf-8") if receipt_path.is_file() else ""
    passed = bool(
        return_code == 0
        and len(calls) == 1
        and len(new_entries) == 1
        and recomputed
        and rejected
        and old_rows_after == before_rows
        and "breaker_invocation_start_position: 10" in receipt
        and "breaker_invocation_local_tail_at_start: 0" in receipt
    )
    row = _row(
        "conductor_calibrated_decision",
        invocation_start=10,
        historical_count=10,
        local_count=after.consecutive_failures_since(10),
        proposals=len(calls),
        evaluations=len(new_entries),
        stop_reason="max_iterations",
        passed=passed,
        failures=() if passed else ("real_round_or_recompute_contract_failed",),
    )
    training = {
        "performed": True,
        "model_family": "fixed_2x4x1_Gibbs_energy_head",
        "framework": "jax",
        "training_steps": metrics.get("training_steps") if isinstance(metrics, Mapping) else None,
        "random_seed": metrics.get("random_seed") if isinstance(metrics, Mapping) else None,
        "nce_loss_first": metrics.get("nce_loss_first") if isinstance(metrics, Mapping) else None,
        "nce_loss_last": metrics.get("nce_loss_last") if isinstance(metrics, Mapping) else None,
        "subprocess_recompute_exercised": recomputed,
        "recomputed_final_energy": metrics.get("final_energy")
        if isinstance(metrics, Mapping)
        else None,
        "recomputed_brier": metrics.get("brier") if isinstance(metrics, Mapping) else None,
        "scientific_acceptance": False,
        "current_llm_invocation": False,
        "scripted_proposal_sha256": _canonical_hash(_CALIBRATED_TRAINING_CODE),
    }
    binding = {
        "private_log_path": str(log_path),
        "persisted_bytes_before_sha256": "sha256:" + hashlib.sha256(before_bytes).hexdigest(),
        "old_rows_before_sha256": _canonical_hash(before_rows),
        "old_rows_after_sha256": _canonical_hash(old_rows_after),
        "old_rows_byte_equivalent": before_rows == old_rows_after,
        "entry_count_before": 10,
        "entry_count_after": len(after.entries),
    }
    return row, {"small_ebm_training": training, "history_binding": binding}


def _preconditions(repo_root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = repo_root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "source_readable_nonempty",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "expected": "readable_nonempty",
                "observed": "readable_nonempty" if available else None,
                "passed": available,
            }
        )
        if available:
            hashes[relative.as_posix()] = {
                "artifact_type": "branch_local_source",
                "sha256": sha256_file(path),
                "original_flags": {"available": True, "external_prerequisite": False},
            }
    spec_text = (repo_root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        {
            "check": "driving_requirement_present",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "expected": "REQ-AUTO-026",
            "observed": "REQ-AUTO-026" if "REQ-AUTO-026" in spec_text else None,
            "passed": "REQ-AUTO-026" in spec_text,
        }
    )
    exclusion = (repo_root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7435" in exclusion
    checks.append(
        {
            "check": "task_not_excluded",
            "upstream": "ops/exclusion_manifest.yaml",
            "path": "ops/exclusion_manifest.yaml",
            "field": "experiment_id: 7435",
            "expected": False,
            "observed": excluded,
            "passed": not excluded,
        }
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checks.append(
        {
            "check": "branch_local_identity",
            "upstream": ".git",
            "path": ".git",
            "field": "branch_and_head",
            "expected": "nonempty",
            "observed": {"branch": branch, "head": commit},
            "passed": bool(branch and commit),
        }
    )
    return checks, hashes


def _gate(
    check: str, category: str, operator: str, expected: Any, observed: Any, principle: str
) -> JsonDict:  # pragma: no cover
    if operator == "eq":
        passed = observed == expected
    elif operator == "gte":
        passed = bool(observed >= expected)
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    failed = [dict(gate) for gate in gates if gate.get("passed") is not True]
    return {
        "all_passed": not failed,
        "passed_count": len(gates) - len(failed),
        "failed_count": len(failed),
        "first_failure": failed[0] if failed else None,
    }


def _principles(keys: Sequence[str]) -> dict[str, str]:  # pragma: no cover
    required = {
        "schema": "Use a versioned plain top-level schema with experiment identity and status.",
        "run_date": "Use 20260919 and retain actual UTC and monotonic timing.",
        "preconditions_checked": "Name every checked path, identity, field, expected value, and observation.",
        "MODEL_SPECS": "Use an empty list because this run invokes no current LLM.",
        "model_invoked": "Keep current attempted LLM use separate from scripted proposal data.",
        "invocation_counts": "Reconcile attempted, terminal, cancelled, and in-flight current LLM calls.",
        "inference_substrate": "Describe current compute in text and keep devices in a separate object.",
        "inference_substrate_class": "Declare no_model_load so the applicable duration rule is explicit.",
        "execution_venue": "Use host and record CPU, CUDA, and external device identity separately.",
        "duration_s": "Measure current work and retain its phase split.",
        "phase_spans": "Bind flushed phase boundaries to monotonic elapsed time.",
        "random_seed": "Freeze the compact-head training seed.",
        "reproducibility_checksum": "Bind code, protocol, inputs, rows, and exact validation scope.",
        "source_artifact_hashes": "Preserve source identity and original availability flags in typed records.",
        "rows": "Keep every planned fixture result, including failures and stop reasons.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
        "acceptance_gate_results": "Separate validity, safety, completion, and benefit checks.",
        "gate_check_summary": "Name exact failed evidence without merging missing, null, and zero.",
        "verifier_is_oracle": "False because this is an execution-control test, not an oracle-scored benefit claim.",
        "honest_verdict": "Use a complete_ verdict for finished work.",
        "verdict_class": "Use null because recovery is verified without a scientific benefit claim.",
        "flagged_adversarial": "Prevent flagged evidence from supplying readiness.",
        "validation_receipts": "Retain exact commands, exits, durations, and hashed logs.",
        "field_principles": "Explain field intent separately from plain gate values.",
        "promotion_score": "Keep zero because this milestone authorizes no rollout or publication.",
        "breaker_recovery_ready_score": "Require recovered execution and preserved within-round stopping.",
        "breaker_rows": "Show each invocation boundary, historical tail, local tail, counts, and stop reason.",
        "history_preserved": "Bind unchanged old records instead of resetting them.",
        "small_ebm_training": "Separate real compact-head fitting from scripted proposal delivery.",
    }
    return {
        key: required.get(key, f"Retain {key} as explicit experiment evidence.") for key in keys
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = str(REPO_ROOT / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7435_v652_round_breaker import validate_artifact;"
        "value=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "errors=validate_artifact(value);print(errors,flush=True);"
        "raise SystemExit(bool(errors))"
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "fresh_process_replay", (python, "-u", "-c", replay, str(candidate)), "candidate"
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


def _build_artifact(  # pragma: no cover
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    training: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    live_log: Mapping[str, Any],
) -> JsonDict:
    shell: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "pending_reduction",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host CPU orchestration, JAX compact-head fitting, and subprocess validation; no LLM load",
        "inference_substrate_details": {
            "cpu": {"machine": platform.machine(), "processor": platform.processor()},
            "cuda": {"used": False, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "external_device": {"used": False, "identity": None},
            "python": platform.python_version(),
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": duration_s,
        "duration_breakdown_s": {str(span["phase"]): span["duration_s"] for span in spans},
        "phase_spans": [dict(span) for span in spans],
        "random_seed": SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "typed_sidecars": [
            {
                "artifact_type": "scripted_proposal_no_model_event",
                "scope": "current_deterministic_fixture",
                "sha256": _canonical_hash(_CALIBRATED_TRAINING_CODE),
            },
            {
                "artifact_type": "private_persisted_log_history",
                "scope": "current_private_fixture",
                "sha256": history["history_binding"]["old_rows_before_sha256"],
            },
        ],
        "rows": [dict(row) for row in rows],
        "sample_size_budget": {
            "planned_independent_units": len(REQUIRED_CASES),
            "attempted_independent_units": len(rows),
            "completed_independent_units": sum(
                row.get("disposition") == "complete" for row in rows
            ),
            "failed_independent_units": sum(bool(row.get("failed")) for row in rows),
            "censored_independent_units": sum(bool(row.get("censored")) for row in rows),
            "unstarted_independent_units": len(REQUIRED_CASES) - len(rows),
            "stopping_rule": "Run every frozen fixture once; stop stuck invocations at their configured local threshold.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "honest_verdict": "pending_reduction",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [dict(row) for row in receipts],
        "field_principles": {},
        "promotion_score": 0,
        "breaker_recovery_ready_score": 0,
        "breaker_rows": [dict(row) for row in rows],
        "history_preserved": bool(history["history_preserved"]),
        "history_binding": deepcopy(dict(history["history_binding"])),
        "live_log_untouched": deepcopy(dict(live_log)),
        "rejected_id_deduplication_unchanged": bool(history["rejected_id_deduplication_unchanged"]),
        "small_ebm_training": deepcopy(dict(training)),
        "scientific_acceptance_created": False,
        "production_defaults_changed": False,
        "research_roadmap_changed": False,
        "applicable_numbered_e2e": [],
        "capability_e2e": "declared entrypoint plus fresh-process artifact replay",
        "affected_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
        },
    }
    reduction = independent_reduce(shell)
    gates = [
        _gate(
            "all_breaker_cases",
            "validity",
            "eq",
            [],
            reduction["failed_breaker_cases"],
            "Every frozen execution-control case must pass.",
        ),
        _gate(
            "required_validation",
            "validity",
            "eq",
            True,
            reduction["required_validation_passed"],
            "Every affected and terminal command must exit cleanly.",
        ),
        _gate(
            "history_preserved",
            "safety",
            "eq",
            True,
            reduction["safety_passed"],
            "Recovery must not delete or relabel scientific rejections.",
        ),
        _gate(
            "scientific_benefit",
            "benefit",
            "eq",
            None,
            None,
            "This infrastructure repair makes no scientific benefit claim.",
        ),
    ]
    ready = reduction["breaker_recovery_ready_score"]
    shell.update(
        {
            "status": "complete_breaker_recovery_ready"
            if ready
            else "disqualified_breaker_recovery_validation",
            "honest_verdict": "complete_breaker_recovery_verified"
            if ready
            else "complete_disqualified_breaker_recovery_evidence",
            "verdict_class": "null" if ready else "disqualified",
            "breaker_recovery_ready_score": ready,
            "acceptance_gate_results": gates,
            "gate_check_summary": _gate_summary(gates),
            "independent_reduction": reduction,
        }
    )
    shell["field_principles"] = _principles(tuple(shell))
    shell["reproducibility_checksum"] = reproducibility_checksum(shell)
    return shell


def _blocked_artifact(  # pragma: no cover
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    started_at: str,
    duration_s: float,
) -> JsonDict:
    failed = next(dict(row) for row in preconditions if row.get("passed") is not True)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "blocked_missing_source_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host precondition checks only; no model load",
        "inference_substrate_details": {
            "cpu": platform.machine(),
            "cuda": None,
            "external_device": None,
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [],
        "random_seed": SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": len(REQUIRED_CASES),
            "attempted_independent_units": 0,
            "completed_independent_units": 0,
            "failed_independent_units": 0,
            "censored_independent_units": len(REQUIRED_CASES),
            "unstarted_independent_units": len(REQUIRED_CASES),
            "stopping_rule": "Stop dependent work when a required local source is unavailable.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "first_failure": failed,
        },
        "verifier_is_oracle": False,
        "honest_verdict": "blocked_missing_source_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": {},
        "promotion_score": 0,
        "breaker_recovery_ready_score": 0,
        "breaker_rows": [],
        "history_preserved": False,
        "rejected_id_deduplication_unchanged": False,
        "small_ebm_training": {"performed": False},
    }
    artifact["field_principles"] = _principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(  # pragma: no cover - executed through the declared entrypoint.
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run private recovery fixtures, scoped checks, cold readers, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    _progress(started, "preconditions", "start")
    preconditions, hashes = _preconditions(root)
    spans.append(_span("preconditions", phase_started, started))
    _progress(
        started,
        "preconditions",
        "end",
        completed=len(preconditions),
        passed=all(row["passed"] for row in preconditions),
    )
    if not all(row["passed"] for row in preconditions):
        blocked = _blocked_artifact(preconditions, hashes, started_at, time.monotonic() - started)
        _progress(started, "publish", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        _progress(started, "publish", "after_atomic_blocked", path=output_path)
        return blocked

    live_path = root / "ops/.autoresearch_experiment_log.json"
    live_before = sha256_file(live_path) if live_path.is_file() else None
    phase_started = time.monotonic()
    _progress(started, "fixtures", "before_shared_cases", completed=0, total=6)
    rows, shared = _run_shared_fixtures()
    _progress(started, "fixtures", "after_shared_cases", completed=len(rows), total=6)
    private = Path(tempfile.mkdtemp(prefix="exp7435-round-breaker-", dir="/tmp"))
    _progress(started, "fixtures", "before_calibrated_training", completed=6, total=7)
    conductor_row, conductor = _run_conductor_fixture(private)
    rows.append(conductor_row)
    _progress(started, "fixtures", "after_calibrated_training", completed=7, total=7)
    spans.append(_span("fixtures", phase_started, started))
    live_after = sha256_file(live_path) if live_path.is_file() else None
    live_log = {
        "path": "ops/.autoresearch_experiment_log.json",
        "before_sha256": live_before,
        "after_sha256": live_after,
        "unchanged": live_before == live_after,
        "missing_before_and_after": live_before is None and live_after is None,
    }
    shared["history_preserved"] = bool(
        shared["history_preserved"]
        and conductor["history_binding"]["old_rows_byte_equivalent"]
        and live_log["unchanged"]
    )
    history = {**shared, "history_binding": conductor["history_binding"]}

    phase_started = time.monotonic()
    private_validation = Path(tempfile.mkdtemp(prefix="exp7435-validation-", dir="/tmp"))
    commands = build_command_plan(root, MANIFEST, private_validation)
    plan_errors = validate_command_plan(root, MANIFEST, commands)
    _progress(started, "validation", "before_affected_subprocesses", errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw / "validation/affected",
        )
    affected_reduction = reduce_affected_receipts(root, MANIFEST, affected)
    spans.append(_span("validation", phase_started, started))
    _progress(
        started,
        "validation",
        "after_affected_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )

    candidate = _build_artifact(
        started_at=started_at,
        completed_at=_utc_now(),
        duration_s=time.monotonic() - started,
        spans=spans,
        preconditions=preconditions,
        hashes=hashes,
        rows=rows,
        history=history,
        training=conductor["small_ebm_training"],
        receipts=affected,
        live_log=live_log,
    )
    candidate_path = raw / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses", completed=0, total=3)
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row["passed"] for row in terminal),
    )
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    final = _build_artifact(
        started_at=started_at,
        completed_at=_utc_now(),
        duration_s=time.monotonic() - started,
        spans=spans,
        preconditions=preconditions,
        hashes=hashes,
        rows=rows,
        history=history,
        training=conductor["small_ebm_training"],
        receipts=[*affected, *terminal],
        live_log=live_log,
    )
    if critical:
        final["flagged_adversarial"] = True
        final["breaker_recovery_ready_score"] = 0
        final["status"] = "disqualified_adversarial_finding"
        final["honest_verdict"] = "complete_disqualified_adversarial_finding"
        final["verdict_class"] = "disqualified"
        final["field_principles"] = _principles(tuple(final))
        final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "publish", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

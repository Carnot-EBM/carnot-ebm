"""Regression checks for the invocation-local autoresearch circuit breaker.

Spec refs: REQ-AUTO-026 and SCENARIO-AUTO-026-A through
SCENARIO-AUTO-026-D.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog
from carnot.autoresearch.orchestrator import (
    AutoresearchConfig,
    run_loop,
    run_loop_with_generator,
    run_loop_with_skills,
)
from carnot.experiment_7435_v652_round_breaker import (
    ZERO_INVOCATION_COUNTS,
    independent_reduce,
    reproducibility_checksum,
    validate_artifact,
)


REJECTED_CODE = """\
def run(benchmark_data):
    return {"fixture": {"final_energy": 2.0, "wall_clock_seconds": 0.01}}
"""
ACCEPTED_CODE = """\
def run(benchmark_data):
    return {"fixture": {"final_energy": 0.5, "wall_clock_seconds": 0.01}}
"""
FAILED_CODE = """\
def run(benchmark_data):
    raise RuntimeError("measured execution failure")
"""


def _entry(index: int, outcome: str = "rejected") -> ExperimentEntry:
    return ExperimentEntry(
        id=f"fixture-{index}",
        timestamp=f"2026-09-19T00:00:{index:02d}+00:00",
        hypothesis_code="def run(data): return {}",
        hypothesis_description=f"fixture {index}",
        sandbox_success=outcome != "rejected",
        eval_verdict="FAIL" if outcome == "rejected" else "PASS",
        outcome=outcome,
    )


def _historical_log(count: int = 10) -> ExperimentLog:
    log = ExperimentLog()
    for index in range(count):
        log.append(_entry(index))
    return log


def _baselines() -> BaselineRecord:
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


def _generator(code: str):
    def generate(
        baselines: BaselineRecord,
        recent_failures: list[dict[str, Any]],
        iteration: int,
    ) -> list[tuple[str, str]]:
        del baselines, recent_failures, iteration
        return [("fixture proposal", code)]

    return generate


def _run_entrypoint(kind: str, log: ExperimentLog, code: str, threshold: int = 10):
    config = AutoresearchConfig(
        max_iterations=20,
        max_consecutive_failures=threshold,
        enable_trajectory_analysis=False,
    )
    if kind == "static":
        return run_loop(
            [(f"proposal-{index}", code) for index in range(20)],
            _baselines(),
            {},
            config,
            log,
        )
    if kind == "generator":
        return run_loop_with_generator(_generator(code), _baselines(), {}, config, log)
    if kind == "skills":
        return run_loop_with_skills(_generator(code), _baselines(), {}, config, log)
    raise AssertionError(f"unknown entrypoint {kind}")


def test_empty_log_and_legacy_counter_contract() -> None:
    """REQ-AUTO-026: the new boundary is additive; the old query stays all-time."""
    log = ExperimentLog()
    assert log.consecutive_failures() == 0
    assert log.consecutive_failures_since(0) == 0

    log.append(_entry(0))
    log.append(_entry(1))
    assert log.consecutive_failures() == 2
    assert log.consecutive_failures_since(1) == 1
    assert log.consecutive_failures_since(2) == 0

    with pytest.raises(ValueError, match="start_position"):
        log.consecutive_failures_since(-1)
    with pytest.raises(ValueError, match="start_position"):
        log.consecutive_failures_since(3)


def test_success_and_review_break_the_local_streak() -> None:
    """SCENARIO-AUTO-026-C: any non-rejection breaks the trailing streak."""
    log = _historical_log(2)
    start = len(log)
    log.append(_entry(2))
    log.append(_entry(3, "accepted"))
    log.append(_entry(4))
    assert log.consecutive_failures_since(start) == 1

    log.append(_entry(5, "pending_review"))
    assert log.consecutive_failures_since(start) == 0
    assert log.rejected_ids() == {"fixture-0", "fixture-1", "fixture-2", "fixture-4"}


@pytest.mark.parametrize("kind", ["static", "generator", "skills"])
def test_ten_historical_rejections_do_not_lock_any_entrypoint(kind: str) -> None:
    """SCENARIO-AUTO-026-A: each real loop gets a fresh rejection budget."""
    log = _historical_log()
    old_rows = [json.dumps(asdict(row), sort_keys=True).encode() for row in log.entries]

    result = _run_entrypoint(kind, log, ACCEPTED_CODE)

    assert result.iterations >= 1
    assert result.accepted >= 1
    assert result.circuit_breaker_tripped is False
    assert [
        json.dumps(asdict(row), sort_keys=True).encode() for row in log.entries[:10]
    ] == old_rows


@pytest.mark.parametrize("kind", ["static", "generator", "skills"])
def test_ten_fresh_rejections_stop_each_entrypoint(kind: str) -> None:
    """SCENARIO-AUTO-026-B: the configured in-invocation limit stays effective."""
    result = _run_entrypoint(kind, ExperimentLog(), REJECTED_CODE)

    assert result.iterations == 10
    assert result.rejected == 10
    assert result.circuit_breaker_tripped is True
    assert result.experiment_log.consecutive_failures_since(0) == 10


def test_execution_failures_count_as_fresh_rejections() -> None:
    """REQ-AUTO-026: sandbox errors keep their existing rejected disposition."""
    result = _run_entrypoint("generator", ExperimentLog(), FAILED_CODE, threshold=2)

    assert result.iterations == 2
    assert result.rejected == 2
    assert result.circuit_breaker_tripped is True
    assert all(not row.sandbox_success for row in result.experiment_log.entries)
    assert all(row.sandbox_error for row in result.experiment_log.entries)


def test_two_successive_invocations_share_history_but_not_budget(tmp_path: Path) -> None:
    """SCENARIO-AUTO-026-D: a stopped invocation cannot lock the next one."""
    path = tmp_path / "persisted-log.json"
    log = _historical_log()
    first = _run_entrypoint("generator", log, REJECTED_CODE, threshold=2)
    assert first.iterations == 2
    assert first.circuit_breaker_tripped is True
    first.experiment_log.save(path)

    persisted = ExperimentLog.load(path)
    old_rows = [json.dumps(asdict(row), sort_keys=True).encode() for row in persisted.entries]
    second = _run_entrypoint("generator", persisted, ACCEPTED_CODE, threshold=2)

    assert second.iterations >= 1
    assert second.accepted >= 1
    assert second.circuit_breaker_tripped is False
    assert [
        json.dumps(asdict(row), sort_keys=True).encode() for row in persisted.entries[:12]
    ] == old_rows
    assert len(persisted.rejected_ids()) == 12


def _artifact_shell() -> dict[str, Any]:
    breaker_rows = [
        {
            "case_id": "historical_tail_recovery",
            "historical_count": 10,
            "local_count": 1,
            "proposal_count": 1,
            "evaluation_count": 1,
            "stop_reason": "max_iterations",
            "passed": True,
        },
        {
            "case_id": "fresh_rejection_limit",
            "historical_count": 0,
            "local_count": 10,
            "proposal_count": 10,
            "evaluation_count": 10,
            "stop_reason": "circuit_breaker",
            "passed": True,
        },
        {
            "case_id": "success_breaks_streak",
            "historical_count": 0,
            "local_count": 1,
            "proposal_count": 4,
            "evaluation_count": 4,
            "stop_reason": "max_iterations",
            "passed": True,
        },
        {
            "case_id": "execution_failure_limit",
            "historical_count": 0,
            "local_count": 2,
            "proposal_count": 2,
            "evaluation_count": 2,
            "stop_reason": "circuit_breaker",
            "passed": True,
        },
        {
            "case_id": "empty_log",
            "historical_count": 0,
            "local_count": 0,
            "proposal_count": 0,
            "evaluation_count": 0,
            "stop_reason": "empty",
            "passed": True,
        },
        {
            "case_id": "successive_invocation_recovery",
            "historical_count": 12,
            "local_count": 0,
            "proposal_count": 1,
            "evaluation_count": 1,
            "stop_reason": "max_iterations",
            "passed": True,
        },
        {
            "case_id": "conductor_calibrated_decision",
            "historical_count": 10,
            "local_count": 1,
            "proposal_count": 1,
            "evaluation_count": 1,
            "stop_reason": "max_iterations",
            "passed": True,
        },
    ]
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7435.v652.round_breaker.v1",
        "experiment_id": "experiment_7435_v652_round_breaker",
        "milestone": "2026.09.652",
        "status": "complete_breaker_recovery_ready",
        "run_date": "20260919",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "verdict_class": "null",
        "honest_verdict": "complete_breaker_recovery_verified",
        "flagged_adversarial": False,
        "promotion_score": 0,
        "history_preserved": True,
        "rejected_id_deduplication_unchanged": True,
        "breaker_rows": breaker_rows,
        "validation_receipts": [
            {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
            for name in (
                "worktree_imports",
                "focused_pytest",
                "changed_module_coverage",
                "changed_module_coverage_report",
                "ruff_check",
                "ruff_format",
                "changed_module_mypy",
                "scoped_spec_coverage",
                "fresh_process_replay",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            )
        ],
        "breaker_recovery_ready_score": 1,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = {key: f"principle for {key}" for key in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def test_independent_reducer_requires_recovery_stopping_and_validation() -> None:
    """REQ-AUTO-026: readiness needs both recovery and the local stop limit."""
    artifact = _artifact_shell()
    reduced = independent_reduce(artifact)
    assert reduced["breaker_recovery_ready_score"] == 1
    assert reduced["required_validation_passed"] is True

    artifact["breaker_rows"][1]["evaluation_count"] = 9
    reduced = independent_reduce(artifact)
    assert reduced["breaker_recovery_ready_score"] == 0
    assert "fresh_rejection_limit" in reduced["failed_breaker_cases"]


def test_artifact_validator_recomputes_reduction_and_checksum() -> None:
    """REQ-AUTO-026: cold replay rejects readiness or evidence drift."""
    artifact = _artifact_shell()
    assert validate_artifact(artifact) == []
    assert validate_artifact([]) == ["artifact_not_object"]

    artifact["breaker_recovery_ready_score"] = 0
    assert "stored_reduction_mismatch" in validate_artifact(artifact)

    artifact = _artifact_shell()
    artifact["MODEL_SPECS"] = [{"name": "invented"}]
    assert "current_model_declaration_mismatch" in validate_artifact(artifact)

    artifact = _artifact_shell()
    artifact["field_principles"].pop("status")
    assert "field_principles_incomplete" in validate_artifact(artifact)

    artifact = _artifact_shell()
    artifact["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("run_date", "20260918", "identity_mismatch"),
        ("execution_venue", "external", "substrate_or_venue_mismatch"),
        ("promotion_score", 1, "promotion_score_nonzero"),
        ("verdict_class", "success", "verdict_class_invalid"),
    ],
)
def test_artifact_validator_rejects_closed_contract_drift(
    field: str, value: Any, error: str
) -> None:
    """REQ-AUTO-026: closed identity and safety fields fail cold validation."""
    artifact = _artifact_shell()
    artifact[field] = value
    assert error in validate_artifact(artifact)

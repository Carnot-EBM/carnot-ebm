"""Tests for Exp 4168 defensive decisive verifier graft.

Spec refs: REQ-LEARN-4168, SCENARIO-LEARN-4168-DEFER,
SCENARIO-LEARN-4168-COPY-GRAFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4168_decisive_verifier_graft_defensive as exp4168


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"

PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _tokens(grid: list[list[int]]) -> list[int]:
    return exp4109.encode_grid(grid)


def _bad_solution() -> list[int]:
    bad = [row[:] for row in SOLUTION]
    bad[0][0] = 6
    bad[0][2] = 5
    return _tokens(bad)


def _pool(puzzle_id: str) -> exp4109.CandidatePool:
    puzzle = _tokens(PUZZLE)
    label = _tokens(SOLUTION)
    good = exp4109.CandidateSample(f"{puzzle_id}:good", label, trm_score=0.1)
    bad_a = exp4109.CandidateSample(f"{puzzle_id}:bad-a", _bad_solution(), trm_score=0.9)
    bad_b = exp4109.CandidateSample(f"{puzzle_id}:bad-b", _bad_solution(), trm_score=0.8)
    return exp4109.CandidatePool(puzzle_id, puzzle, label, [bad_a, bad_b, good])


def _monitor(
    *,
    root: Path,
    stable: Path,
    current_val: float | None,
    baseline_faithful: bool,
    alive: bool,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4167_outerloop_training_monitor",
        "schema": "carnot.experiment_4167_outerloop_training_monitor.v1",
        "honest_verdict": "complete: fixture",
        "baseline_faithful": baseline_faithful,
        "outerloop_train_alive": alive,
        "current_val_exact_accuracy": current_val,
        "checkpoint_path": str(stable),
        "checkpoint_mtime": "2026-06-13T04:41:29Z",
        "outerloop_pid": 1234,
        "latest_metrics_path": str(root / "results" / "metrics.csv"),
    }


def test_req_learn_4168_spec_declares_required_contract() -> None:
    """REQ-LEARN-4168: OpenSpec declares the defensive graft artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4168" in spec
    assert "SCENARIO-LEARN-4168-DEFER" in spec
    assert "SCENARIO-LEARN-4168-COPY-GRAFT" in spec
    assert exp4168.RESULT_FILENAME in spec
    assert "SHALL NOT train" in spec
    assert "SHALL copy the stable" in spec
    assert "task-local checkpoint path" in spec
    for field in exp4168.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4168.FIELD_PRINCIPLES
        assert exp4168.FIELD_PRINCIPLES[field] in spec


def test_scenario_learn_4168_defer_stops_before_copy_sampling_or_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4168-DEFER: unfaithful baseline produces honest deferral only."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"stable checkpoint")
    output = tmp_path / "results" / exp4168.RESULT_FILENAME

    artifact = exp4168.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        monitor_provider=lambda: _monitor(
            root=tmp_path,
            stable=stable,
            current_val=0.514843761921,
            baseline_faithful=False,
            alive=False,
        ),
        checkpoint_copier=lambda _source, _target: pytest.fail("must not copy while deferred"),
        candidate_pool_provider=lambda _copy: pytest.fail("must not sample while deferred"),
        rft_runner=lambda _copy, _corpora: pytest.fail("must not train while deferred"),
    )

    assert artifact["honest_verdict"] == "complete: graft_deferred_outerloop_training_val_0.5148"
    assert artifact["graft_deferred"] is True
    assert artifact["baseline_status"]["faithful_stable"] is False
    assert artifact["checkpoint_copy_path"] is None
    assert artifact["checkpoint_copy_performed"] is False
    assert artifact["read_only_actions"]["training_launched"] is False
    assert artifact["read_only_actions"]["train_process_stop_attempted"] is False
    assert artifact["rerank_lift_vs_vote"]["status"] == "deferred_outerloop_training"
    assert artifact["rft_vs_ablation_delta"]["status"] == "deferred_outerloop_training"
    assert artifact["verifier_value_added"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert exp4168.artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4168_copy_graft_uses_task_local_checkpoint(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4168-COPY-GRAFT: faithful branch samples and trains from the copy."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"stable checkpoint")
    pools = [_pool(f"p{i}") for i in range(6)]

    def provider(copy_path: Path) -> list[exp4109.CandidatePool]:
        assert copy_path != stable
        assert copy_path.name == "last-exp4168-copy.ckpt"
        assert copy_path.read_bytes() == b"stable checkpoint"
        return pools

    def rft(copy_path: Path, corpora: dict[str, Any]) -> dict[str, Any]:
        assert copy_path != stable
        assert copy_path.read_bytes() == b"stable checkpoint"
        assert corpora["arm_a"] == "verifier_certified"
        assert corpora["arm_b"] == "vote_certified"
        assert corpora["n_matched"] == len(pools)
        return {
            "metric": "heldout_exact_accuracy",
            "training_mode": "default_cumulative_plus_budget_same_copy_init",
            "n_matched": corpora["n_matched"],
            "a_exact_accuracy": 0.9,
            "b_exact_accuracy": 0.6,
            "delta": 0.3,
            "ci95": [0.1, 0.5],
            "status": "ci95_excludes_zero",
        }

    artifact = exp4168.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4168.RESULT_FILENAME,
        monitor_provider=lambda: _monitor(
            root=tmp_path,
            stable=stable,
            current_val=0.872,
            baseline_faithful=True,
            alive=False,
        ),
        candidate_pool_provider=provider,
        rft_runner=rft,
        bootstrap_resamples=100,
    )

    assert artifact["honest_verdict"] == "success: verifier_value_added_rft_A_gt_B_copy_graft"
    assert artifact["graft_deferred"] is False
    assert artifact["checkpoint_copy_path"] != str(stable)
    assert artifact["checkpoint_copy_performed"] is True
    assert artifact["candidate_source"] == "provided_candidate_pool"
    assert artifact["rerank_lift_vs_vote"]["delta"] == pytest.approx(1.0)
    assert artifact["rerank_lift_vs_vote"]["delta_vs_oracle"] == pytest.approx(0.0)
    assert artifact["rerank_lift_vs_vote"]["ci95"][0] > 0.0
    assert artifact["rft_vs_ablation_delta"]["delta"] == pytest.approx(0.3)
    assert artifact["verifier_value_added"] is True
    assert artifact["acceptance_gate_passed"] is True
    assert exp4168.artifact_schema_errors(artifact) == []


def test_req_learn_4168_schema_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-4168: schema helpers reject malformed moat artifacts."""

    stable = tmp_path / "stable.ckpt"
    monitor = _monitor(root=tmp_path, stable=stable, current_val=None, baseline_faithful=True, alive=False)

    assert exp4168.baseline_is_faithful_stable(monitor) is False
    assert exp4168._float_or_none("0.5") == pytest.approx(0.5)
    assert exp4168._float_or_none(True) is None
    assert exp4168._float_or_none("bad") is None
    assert exp4168._float_or_none(object()) is None
    assert exp4168._jsonable(Path("x")) == "x"

    class Scalar:
        def item(self) -> int:
            return 7

    assert exp4168._jsonable(Scalar()) == 7
    assert exp4168._metric_has_ci([]) is False
    assert exp4168._format_val_tag(None) == "unknown"
    assert exp4168.verifier_value_added({"delta": 0.1, "ci95": [0.01, 0.2]}, graft_deferred=False) is True
    assert exp4168.verifier_value_added({"delta": 0.1, "ci95": [0.0, 0.2]}, graft_deferred=False) is False
    assert exp4168.verifier_value_added({"delta": object(), "ci95": [0.01, 0.2]}, graft_deferred=False) is False
    assert exp4168.verifier_value_added({"delta": 0.1, "ci95": [0.01, 0.2]}, graft_deferred=True) is False
    assert (
        exp4168._artifact_verdict(graft_deferred=False, value_added=False, monitor=monitor)
        == "complete: A~=B null"
    )
    assert exp4168._acceptance_gate({"honest_verdict": 7}) is False
    assert exp4168._acceptance_gate({"honest_verdict": "complete: x", "graft_deferred": "yes"}) is False
    assert (
        exp4168._acceptance_gate(
            {"honest_verdict": "complete: x", "graft_deferred": True, "verifier_value_added": "no"}
        )
        is False
    )
    assert (
        exp4168._acceptance_gate(
            {"honest_verdict": "complete: x", "graft_deferred": True, "verifier_value_added": False}
        )
        is False
    )

    blocked = exp4168.build_blocked_artifact(
        "blocked_stable_checkpoint",
        monitor=monitor,
        detail="missing",
        duration_s=0.0,
    )
    assert blocked["honest_verdict"] == "blocked_stable_checkpoint"
    assert blocked["acceptance_gate_passed"] is False
    assert exp4168.artifact_schema_errors(blocked) == []

    missing_stable = exp4168.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_stable.json",
        monitor_provider=lambda: _monitor(
            root=tmp_path,
            stable=tmp_path / "does-not-exist.ckpt",
            current_val=0.9,
            baseline_faithful=True,
            alive=False,
        ),
    )
    assert missing_stable["honest_verdict"] == "blocked_stable_checkpoint"

    real_stable = tmp_path / "real.ckpt"
    real_stable.write_bytes(b"stable")
    missing_runner = exp4168.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_runner.json",
        monitor_provider=lambda: _monitor(
            root=tmp_path,
            stable=real_stable,
            current_val=0.9,
            baseline_faithful=True,
            alive=False,
        ),
        candidate_pool_provider=lambda _copy: [_pool("blocked")],
    )
    assert missing_runner["honest_verdict"] == "blocked_native_rft_runner_missing"

    errors = exp4168.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "graft_deferred": "yes",
            "rerank_lift_vs_vote": {"ci95": [0.0]},
            "rft_vs_ablation_delta": {"delta": "bad", "ci95": [0.0, 0.0]},
            "verifier_value_added": "no",
            "field_principles": {"honest_verdict": "wrong"},
        }
    )

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "graft_deferred must be a bare bool" in errors
    assert "verifier_value_added must be a bare bool" in errors
    assert "rerank_lift_vs_vote.delta is required" in errors
    assert "rerank_lift_vs_vote.ci95 must have two numeric bounds" in errors
    assert "rft_vs_ablation_delta.delta must be numeric" in errors
    assert "field_principles.honest_verdict mismatch" in errors

    more_errors = exp4168.artifact_schema_errors(
        {
            "honest_verdict": 7,
            "graft_deferred": True,
            "rerank_lift_vs_vote": {"delta": 0.0, "ci95": [0.0, 0.0]},
            "rft_vs_ablation_delta": {"delta": 0.0, "ci95": [0.0, 0.0]},
            "verifier_value_added": True,
            "field_principles": exp4168.FIELD_PRINCIPLES,
            "acceptance_gate_passed": "yes",
            "reproducibility_checksum": "bad",
        }
    )
    assert "honest_verdict must be a string" in more_errors
    assert "verifier_value_added cannot be true when graft_deferred is true" in more_errors
    assert "acceptance_gate_passed must be a bare bool" in more_errors
    assert "reproducibility_checksum must be sha256-prefixed" in more_errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4168.validate_artifact({"honest_verdict": "bad"})

"""Tests for Exp 4168 v2 gate-082 defensive verifier graft.

Spec refs: REQ-LEARN-4168-V2, SCENARIO-LEARN-4168-V2-DEFER,
SCENARIO-LEARN-4168-V2-COPY-GRAFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4168_decisive_verifier_graft_v2_gate082 as exp4168v2


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


def _write_stable_root(tmp_path: Path, *, bestval: str = "0.822656", pid: str = "537783") -> Path:
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"stable checkpoint")
    stable.with_name("last.ckpt.bestval").write_text(bestval + "\n", encoding="utf-8")
    pid_path = tmp_path / "results" / "trm_runs" / "contiguous_run.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(pid + "\n", encoding="utf-8")
    return stable


def test_req_learn_4168_v2_spec_declares_gate082_contract() -> None:
    """REQ-LEARN-4168-V2: OpenSpec declares the fresh gate-082 artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4168-V2" in spec
    assert "SCENARIO-LEARN-4168-V2-DEFER" in spec
    assert "SCENARIO-LEARN-4168-V2-COPY-GRAFT" in spec
    assert "last.ckpt.bestval" in spec
    assert "0.82" in spec
    assert exp4168v2.RESULT_FILENAME in spec
    for field in exp4168v2.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4168v2.FIELD_PRINCIPLES
        assert exp4168v2.FIELD_PRINCIPLES[field] in spec


def test_scenario_learn_4168_v2_defer_stops_before_copy_sampling_or_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4168-V2-DEFER: failed fresh gate writes only deferral evidence."""

    _write_stable_root(tmp_path, bestval="0.8199")
    output = tmp_path / "results" / exp4168v2.RESULT_FILENAME

    artifact = exp4168v2.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        pid_alive_checker=lambda _pid: False,
        gpu_train_process_finder=lambda: {
            "available": True,
            "detail": "fixture no compute apps",
            "train_processes": [],
        },
        checkpoint_copier=lambda _source, _target: pytest.fail("must not copy while deferred"),
        candidate_pool_provider=lambda _copy: pytest.fail("must not sample while deferred"),
        native_rft_runner=lambda _copy, _corpora: pytest.fail("must not train while deferred"),
    )

    assert artifact["honest_verdict"] == "complete: graft_deferred_outerloop_training_val_0.8199"
    assert artifact["graft_deferred"] is True
    assert artifact["baseline_status"]["fresh_gate082_stable"] is False
    assert artifact["baseline_status"]["bestval_exact_accuracy"] == pytest.approx(0.8199)
    assert artifact["checkpoint_copy_path"] is None
    assert artifact["checkpoint_copy_performed"] is False
    assert artifact["read_only_actions"]["training_launched"] is False
    assert artifact["read_only_actions"]["train_process_stop_attempted"] is False
    assert artifact["rerank_lift_vs_vote"]["status"] == "deferred_outerloop_training"
    assert artifact["rft_vs_ablation_delta"]["status"] == "deferred_outerloop_training"
    assert artifact["verifier_value_added"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert exp4168v2.artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4168_v2_copy_graft_uses_task_local_checkpoint(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4168-V2-COPY-GRAFT: passing gate samples and scores from the copy."""

    stable = _write_stable_root(tmp_path)
    pools = [_pool(f"p{i}") for i in range(6)]

    def provider(copy_path: Path) -> list[exp4109.CandidatePool]:
        assert copy_path != stable
        assert copy_path.parent.name == "experiment_4168_decisive_verifier_graft_v2_gate082"
        assert copy_path.name == "last-exp4168-v2-gate082-copy.ckpt"
        assert copy_path.read_bytes() == b"stable checkpoint"
        return pools

    artifact = exp4168v2.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4168v2.RESULT_FILENAME,
        pid_alive_checker=lambda _pid: False,
        gpu_train_process_finder=lambda: {
            "available": True,
            "detail": "fixture no compute apps",
            "train_processes": [],
        },
        candidate_pool_provider=provider,
        bootstrap_resamples=100,
    )

    assert stable.read_bytes() == b"stable checkpoint"
    assert artifact["honest_verdict"] == "success: verifier_value_added_rft_A_gt_B_copy_graft"
    assert artifact["graft_deferred"] is False
    assert artifact["baseline_status"]["fresh_gate082_stable"] is True
    assert artifact["checkpoint_copy_path"] != str(stable)
    assert artifact["checkpoint_copy_performed"] is True
    assert artifact["candidate_source"] == "provided_candidate_pool"
    assert artifact["rerank_lift_vs_vote"]["delta"] == pytest.approx(1.0)
    assert artifact["rerank_lift_vs_vote"]["delta_vs_oracle"] == pytest.approx(0.0)
    assert artifact["rft_vs_ablation_delta"]["delta"] == pytest.approx(1.0)
    assert artifact["rft_vs_ablation_delta"]["ci95"] == [1.0, 1.0]
    assert artifact["rft_native_training_launched"] is False
    assert artifact["rft_vs_ablation_delta"]["training_mode"] == "matched_label_deconfound_no_native_training"
    assert artifact["read_only_actions"]["training_launched"] is False
    assert artifact["verifier_value_added"] is True
    assert artifact["acceptance_gate_passed"] is True
    assert exp4168v2.artifact_schema_errors(artifact) == []


def test_req_learn_4168_v2_helpers_and_schema_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-4168-V2: helper and schema failures are explicit."""

    stable = _write_stable_root(tmp_path, bestval="not-a-number", pid="pid: 42")
    state = exp4168v2.probe_gate082_preconditions(
        repo_root=tmp_path,
        pid_alive_checker=lambda pid: pid == 42,
        gpu_train_process_finder=lambda: {
            "available": True,
            "detail": "fixture train busy",
            "train_processes": [{"pid": 100, "args": "python nano-trm/src/nn/train.py"}],
        },
    )
    assert state["bestval_exact_accuracy"] is None
    assert state["bestval_passed"] is False
    assert state["outerloop_pid"] == 42
    assert state["outerloop_pid_alive"] is True
    assert state["gpu_train_processes"][0]["pid"] == 100
    assert state["fresh_gate082_stable"] is False

    assert exp4168v2._float_or_none("0.82") == pytest.approx(0.82)
    assert exp4168v2._float_or_none(True) is None
    assert exp4168v2._float_or_none("bad") is None
    assert exp4168v2._float_or_none(object()) is None
    assert exp4168v2._jsonable(Path("x")) == "x"
    assert exp4168v2._parse_pid_text("pid: 1234\n") == 1234
    assert exp4168v2._parse_pid_text("none") is None
    assert exp4168v2._format_val_tag(None) == "unknown"
    assert exp4168v2._copy_target_for(stable, repo_root=tmp_path).name == "last-exp4168-v2-gate082-copy.ckpt"
    assert exp4168v2.read_bestval(tmp_path / "missing.bestval") is None
    assert exp4168v2.read_pid(tmp_path / "missing.pid") is None

    class Scalar:
        def item(self) -> int:
            return 7

    assert exp4168v2._jsonable(Scalar()) == 7
    assert exp4168v2._metric_has_ci([]) is False
    assert exp4168v2.verifier_value_added({"delta": 0.1, "ci95": [0.01, 0.2]}, graft_deferred=False) is True
    assert exp4168v2.verifier_value_added({"delta": 0.1, "ci95": [0.0, 0.2]}, graft_deferred=False) is False
    assert exp4168v2.verifier_value_added({"delta": object(), "ci95": [0.01, 0.2]}, graft_deferred=False) is False
    assert exp4168v2.verifier_value_added({"delta": 0.1, "ci95": [0.01, 0.2]}, graft_deferred=True) is False
    assert (
        exp4168v2._artifact_verdict(graft_deferred=False, value_added=False, state=state)
        == "complete: A~=B null"
    )
    assert exp4168v2._acceptance_gate({"honest_verdict": 7}) is False
    assert exp4168v2._acceptance_gate({"honest_verdict": "complete: x", "graft_deferred": "yes"}) is False
    assert (
        exp4168v2._acceptance_gate(
            {"honest_verdict": "complete: x", "graft_deferred": True, "verifier_value_added": "no"}
        )
        is False
    )
    assert (
        exp4168v2._acceptance_gate(
            {"honest_verdict": "complete: x", "graft_deferred": True, "verifier_value_added": False}
        )
        is False
    )

    label_delta = exp4168v2.evaluate_matched_label_ablation(
        exp4109.build_matched_corpora([_pool("edge")]),
        bootstrap_resamples=10,
    )
    assert label_delta["training_mode"] == "matched_label_deconfound_no_native_training"

    native_root = tmp_path / "native"
    _write_stable_root(native_root)

    def native_rft(copy_path: Path, corpora: dict[str, Any]) -> dict[str, Any]:
        assert copy_path.name == "last-exp4168-v2-gate082-copy.ckpt"
        return {
            "metric": "heldout_exact_accuracy",
            "n_matched": corpora["n_matched"],
            "a_exact_accuracy": 0.5,
            "b_exact_accuracy": 0.5,
            "delta": 0.0,
            "ci95": [-0.1, 0.1],
            "status": "fixture_null",
        }

    native_artifact = exp4168v2.run_experiment(
        repo_root=native_root,
        output_path=native_root / "results" / exp4168v2.RESULT_FILENAME,
        pid_alive_checker=lambda _pid: False,
        gpu_train_process_finder=lambda: {
            "available": True,
            "detail": "fixture no compute apps",
            "train_processes": [],
        },
        candidate_pool_provider=lambda _copy: [_pool("native")],
        native_rft_runner=native_rft,
        bootstrap_resamples=10,
    )
    assert native_artifact["honest_verdict"] == "complete: A~=B null"
    assert native_artifact["rft_native_training_launched"] is True
    assert native_artifact["read_only_actions"]["training_launched"] is True
    assert native_artifact["rft_vs_ablation_delta"]["training_mode"] == "default_cumulative_plus_budget_same_copy_init"

    malformed_errors = exp4168v2.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "graft_deferred": "yes",
            "rerank_lift_vs_vote": {"ci95": [0.0]},
            "rft_vs_ablation_delta": {"delta": "bad", "ci95": [0.0, 0.0]},
            "verifier_value_added": "no",
            "field_principles": {"honest_verdict": "wrong"},
        }
    )
    assert "honest_verdict must be terminal-prefixed" in malformed_errors
    assert "graft_deferred must be a bare bool" in malformed_errors
    assert "verifier_value_added must be a bare bool" in malformed_errors
    assert "rerank_lift_vs_vote.delta is required" in malformed_errors
    assert "rerank_lift_vs_vote.ci95 must have two numeric bounds" in malformed_errors
    assert "rft_vs_ablation_delta.delta must be numeric" in malformed_errors
    assert "field_principles.honest_verdict mismatch" in malformed_errors

    more_errors = exp4168v2.artifact_schema_errors(
        {
            "honest_verdict": 7,
            "graft_deferred": True,
            "rerank_lift_vs_vote": {"delta": 0.0, "ci95": [0.0, 0.0]},
            "rft_vs_ablation_delta": {"delta": 0.0, "ci95": [0.0, 0.0]},
            "verifier_value_added": True,
            "field_principles": exp4168v2.FIELD_PRINCIPLES,
            "acceptance_gate_passed": "yes",
            "reproducibility_checksum": "bad",
        }
    )
    assert "honest_verdict must be a string" in more_errors
    assert "verifier_value_added cannot be true when graft_deferred is true" in more_errors
    assert "acceptance_gate_passed must be a bare bool" in more_errors
    assert "reproducibility_checksum must be sha256-prefixed" in more_errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4168v2.validate_artifact({"honest_verdict": "bad"})

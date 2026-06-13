"""Tests for Exp 4159 decisive verifier-as-reward graft.

Spec refs: REQ-LEARN-4159, SCENARIO-LEARN-4159-DEFER,
SCENARIO-LEARN-4159-PHASE0, SCENARIO-LEARN-4159-RFT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4159_decisive_verifier_reward_graft as exp4159


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


def _alternate_valid_solution() -> list[int]:
    swapped = [[2 if value == 1 else 1 if value == 2 else value for value in row] for row in SOLUTION]
    return _tokens(swapped)


def _pool(puzzle_id: str, *, verifier_exact: bool = True) -> exp4109.CandidatePool:
    puzzle = _tokens(PUZZLE if verifier_exact else [[0] * 9 for _ in range(9)])
    label = _tokens(SOLUTION)
    verifier_tokens = label if verifier_exact else _alternate_valid_solution()
    verifier = exp4109.CandidateSample(f"{puzzle_id}:verifier", verifier_tokens, trm_score=0.4)
    bad_a = exp4109.CandidateSample(f"{puzzle_id}:bad-a", _bad_solution(), trm_score=0.9)
    bad_b = exp4109.CandidateSample(f"{puzzle_id}:bad-b", _bad_solution(), trm_score=0.8)
    return exp4109.CandidatePool(puzzle_id, puzzle, label, [bad_a, bad_b, verifier])


def _write_exp4157(path: Path, *, stable: Path, current_val: float | None = 0.501041650772) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": "experiment_4157_baseline_harvest_contiguous_continue",
                "honest_verdict": "complete: fixture",
                "current_val": current_val,
                "max_val": current_val,
                "stable_checkpoint_path": str(stable),
                "estimated_passes_to_085": {
                    "basis": "fixture_rate",
                    "estimated_additional_val_intervals": 48,
                    "target_val": 0.85,
                },
                "preconditions_checked": [{"resource": "cuda_available", "available": True, "detail": "fixture"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_learn_4159_spec_declares_required_contract() -> None:
    """REQ-LEARN-4159: OpenSpec declares the reward-graft artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4159" in spec
    assert "SCENARIO-LEARN-4159-DEFER" in spec
    assert "SCENARIO-LEARN-4159-PHASE0" in spec
    assert "SCENARIO-LEARN-4159-RFT" in spec
    assert exp4159.RESULT_FILENAME in spec
    for field in exp4159.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4159.FIELD_PRINCIPLES


def test_req_learn_4159_baseline_preconditions_and_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-4159: Exp 4157 baseline, CUDA, and helpers gate the graft."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    artifact_path = _write_exp4157(tmp_path / "results" / exp4159.DEFAULT_EXP4157_ARTIFACT.name, stable=stable)
    baseline = exp4159.load_baseline_context(artifact_path)

    assert baseline.artifact_path == artifact_path
    assert baseline.stable_checkpoint_path == stable
    assert baseline.current_val == pytest.approx(0.501041650772)
    assert exp4159.baseline_is_faithful(baseline) is False
    estimate = exp4159.estimate_passes_to_converge_for_386(baseline)
    assert estimate["destination"] == ".386"
    assert estimate["estimated_additional_val_intervals"] == 48
    assert estimate["basis"] == "fixture_rate"
    assert exp4159.estimate_passes_to_converge_for_386(None)["basis"] == "missing_exp4157_current_val"

    faithful = exp4159.BaselineContext(
        artifact_path=Path("exp4157.json"),
        stable_checkpoint_path=Path("last.ckpt"),
        current_val=0.86,
        max_val=0.86,
        raw={},
    )
    assert exp4159.estimate_passes_to_converge_for_386(faithful)["estimated_additional_val_intervals"] == 0

    stalled = exp4159.BaselineContext(
        artifact_path=Path("exp4157.json"),
        stable_checkpoint_path=Path("last.ckpt"),
        current_val=0.5,
        max_val=0.5,
        raw={"val_trajectory": [{"val_exact_accuracy": 0.5}, {"val_exact_accuracy": 0.49}]},
    )
    assert exp4159.estimate_passes_to_converge_for_386(stalled)["basis"] == "no_positive_exp4157_convergence_rate"
    improving = exp4159.BaselineContext(
        artifact_path=Path("exp4157.json"),
        stable_checkpoint_path=Path("last.ckpt"),
        current_val=0.6,
        max_val=0.6,
        raw={"val_trajectory": [{"val_exact_accuracy": 0.4}, {"val_exact_accuracy": 0.5}, {"val_exact_accuracy": 0.6}]},
    )
    improving_estimate = exp4159.estimate_passes_to_converge_for_386(improving)
    assert improving_estimate["basis"] == "mean_positive_exp4157_delta"
    assert improving_estimate["estimated_additional_val_intervals"] == 3

    checks = exp4159.check_preconditions(baseline, cuda_checker=lambda: (True, "cuda fixture"))
    assert [check.resource for check in checks] == [
        "exp4157_artifact",
        "stable_checkpoint_path",
        "stable_checkpoint",
        "cuda_available",
    ]
    assert checks[2].available is False

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    checks = exp4159.check_preconditions(baseline, cuda_checker=lambda: (True, "cuda fixture"))
    assert all(check.available for check in checks)
    snapshot = exp4159.snapshot_checkpoint(stable)
    assert snapshot.name == "last-reward-snapshot.ckpt"
    assert snapshot.read_bytes() == b"checkpoint"

    failed = exp4159.check_preconditions(
        baseline,
        cuda_checker=lambda: (_ for _ in ()).throw(RuntimeError("cuda broke")),
    )
    assert failed[3].available is False
    assert "RuntimeError: cuda broke" in failed[3].detail

    corpora = {"rows": [{"a_exact": True}, {"a_exact": False}, {"a_exact": True}], "n_matched": 3}
    phase0 = exp4159.estimate_phase0_precision(corpora)
    assert phase0["precision"] == pytest.approx(2 / 3)
    assert phase0["status"] == "precision_gate_failed_label_noise"
    assert exp4159.estimate_phase0_precision({"rows": [], "n_matched": 0})["status"] == "no_verifier_certified_labels"
    assert exp4159.phase0_precision_passed({"precision": "bad"}) is False
    assert exp4159.verifier_value_added({"delta": 1.0, "ci95": [0.1, 1.0]}, graft_deferred=False) is True
    assert exp4159.verifier_value_added({"delta": "bad", "ci95": [0.1, 1.0]}, graft_deferred=False) is False
    assert exp4159.verifier_value_added({"delta": 1.0, "ci95": [0.1, 1.0]}, graft_deferred=True) is False
    assert exp4159._artifact_verdict(graft_deferred=False, value_added=False, defer_reason=None) == "complete: A~=B null"
    assert exp4159._acceptance_gate({"honest_verdict": "blocked_x"}) is False
    assert exp4159._acceptance_gate({"honest_verdict": "complete: x", "graft_deferred": "yes"}) is False
    assert exp4159._acceptance_gate(
        {"honest_verdict": "complete: x", "graft_deferred": False, "verifier_value_added": "no"}
    ) is False
    assert exp4159._acceptance_gate(
        {
            "honest_verdict": "complete: x",
            "graft_deferred": False,
            "verifier_value_added": False,
            "baseline_status": "bad",
        }
    ) is False

    class Scalar:
        def item(self) -> int:
            return 7

    assert exp4159._float_or_none("0.25") == pytest.approx(0.25)
    assert exp4159._float_or_none("bad") is None
    assert exp4159._float_or_none(False) is None
    assert exp4159._float_or_none(object()) is None
    assert exp4159._jsonable(Path("x")) == "x"
    assert exp4159._jsonable(Scalar()) == 7
    assert exp4159._metric_has_ci({"ci95": ["bad", 1.0]}) is False

    with pytest.raises(ValueError, match="JSON object"):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("[]\n", encoding="utf-8")
        exp4159.load_baseline_context(bad_json)


def test_scenario_learn_4159_defer_stops_before_sampling(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4159-DEFER: val<0.85 stops before reward-graft work."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    artifact_path = _write_exp4157(tmp_path / "results" / exp4159.DEFAULT_EXP4157_ARTIFACT.name, stable=stable)
    output = tmp_path / "results" / exp4159.RESULT_FILENAME

    artifact = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _snapshot: pytest.fail("must not sample candidates"),
        snapshotter=lambda _stable: pytest.fail("must not snapshot below threshold"),
    )

    assert artifact["honest_verdict"] == "complete: graft_deferred_baseline_below_0.85"
    assert artifact["graft_deferred"] is True
    assert artifact["verifier_value_added"] is False
    assert artifact["baseline_status"]["current_val"] == pytest.approx(0.501041650772)
    assert artifact["estimated_passes_to_converge_for_386"]["estimated_additional_val_intervals"] == 48
    assert artifact["phase0_precision"]["status"] == "deferred_baseline_below_0.85"
    assert artifact["rft_vs_ablation_delta"]["status"] == "deferred_baseline_below_0.85"
    assert artifact["acceptance_gate_passed"] is True
    assert exp4159.artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4159_phase0_failure_defers_before_rft(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4159-PHASE0: low demo-perfect precision blocks RFT attribution."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    artifact_path = _write_exp4157(
        tmp_path / "results" / exp4159.DEFAULT_EXP4157_ARTIFACT.name,
        stable=stable,
        current_val=0.86,
    )

    def provider(snapshot: Path) -> list[exp4109.CandidatePool]:
        assert snapshot.name == "last-reward-snapshot.ckpt"
        assert snapshot.read_bytes() == b"checkpoint"
        return [_pool(f"p{i}", verifier_exact=False) for i in range(3)]

    artifact = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "phase0.json",
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=provider,
        rft_runner=lambda *_args: pytest.fail("must not run RFT when phase0 fails"),
        bootstrap_resamples=100,
    )

    assert artifact["honest_verdict"] == "complete: graft_deferred_phase0_precision_below_0.85"
    assert artifact["graft_deferred"] is True
    assert artifact["phase0_precision"]["precision"] == pytest.approx(0.0)
    assert artifact["phase0_precision"]["status"] == "precision_gate_failed_label_noise"
    assert artifact["rft_vs_ablation_delta"]["status"] == "deferred_phase0_precision_below_0.85"
    assert artifact["acceptance_gate_passed"] is True
    assert exp4159.artifact_schema_errors(artifact) == []


def test_scenario_learn_4159_rft_reports_value_added_with_ci(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4159-RFT: A-vs-B reward training sets the headline bool."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    artifact_path = _write_exp4157(
        tmp_path / "results" / exp4159.DEFAULT_EXP4157_ARTIFACT.name,
        stable=stable,
        current_val=0.87,
    )
    pools = [_pool(f"p{i}", verifier_exact=True) for i in range(6)]
    rft_calls = []

    def rft_runner(
        baseline: exp4159.BaselineContext,
        snapshot_path: Path,
        corpora: dict[str, object],
    ) -> dict[str, object]:
        rft_calls.append((baseline, snapshot_path, corpora))
        assert snapshot_path.read_bytes() == b"checkpoint"
        assert corpora["arm_a"] == "verifier_certified"
        assert corpora["arm_b"] == "vote_certified"
        return {
            "metric": "heldout_exact_accuracy",
            "n_matched": corpora["n_matched"],
            "a_exact_accuracy": 1.0,
            "b_exact_accuracy": 0.0,
            "delta": 1.0,
            "ci95": [1.0, 1.0],
            "status": "ci95_excludes_zero",
            "training_mode": "bounded_contiguous_resume_same_init",
        }

    artifact = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "rft.json",
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _snapshot: pools,
        rft_runner=rft_runner,
        bootstrap_resamples=100,
    )

    assert artifact["honest_verdict"] == "success: verifier_value_added_rft_A_gt_B"
    assert artifact["graft_deferred"] is False
    assert artifact["phase0_precision"]["precision"] == pytest.approx(1.0)
    assert artifact["phase0_precision"]["status"] == "precision_gate_passed"
    assert artifact["rft_vs_ablation_delta"]["delta"] == pytest.approx(1.0)
    assert artifact["rft_vs_ablation_delta"]["ci95"] == [1.0, 1.0]
    assert artifact["verifier_value_added"] is True
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["rft_training_mode"] == "bounded_contiguous_resume_same_init"
    assert rft_calls and rft_calls[0][2]["n_matched"] == len(pools)
    assert exp4159.validate_artifact(artifact) is None


def test_req_learn_4159_blocked_and_schema_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-4159: blocked and malformed artifacts fail closed."""

    output = tmp_path / "results" / "blocked.json"
    missing = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4157_artifact_path=tmp_path / "missing.json",
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert missing["honest_verdict"] == "blocked_exp4157_artifact"
    assert missing["acceptance_gate_passed"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["honest_verdict"].startswith("blocked_")

    stable = tmp_path / "last.ckpt"
    artifact_path = _write_exp4157(tmp_path / "results" / "exp4157.json", stable=stable, current_val=0.9)
    precondition_blocked = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "precondition.json",
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (False, "no cuda"),
    )
    assert precondition_blocked["honest_verdict"] == "blocked_stable_checkpoint"
    assert precondition_blocked["acceptance_gate_passed"] is False

    stable.write_bytes(b"checkpoint")
    snapshot_blocked = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "snapshot.json",
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        snapshotter=lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot broke")),
    )
    assert snapshot_blocked["honest_verdict"] == "blocked_checkpoint_snapshot"
    assert "RuntimeError: snapshot broke" in snapshot_blocked["blocked_detail"]

    sampling_blocked = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "sampling.json",
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _snapshot: (_ for _ in ()).throw(RuntimeError("sampling broke")),
    )
    assert sampling_blocked["honest_verdict"] == "blocked_candidate_sampling_failed"

    runner_missing = exp4159.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "runner_missing.json",
        exp4157_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _snapshot: [_pool("p0", verifier_exact=True)],
    )
    assert runner_missing["honest_verdict"] == "blocked_native_rft_runner_missing"

    blocked = exp4159.build_blocked_artifact(
        "blocked_fixture",
        baseline=None,
        preconditions_checked=[{"resource": "x", "available": False, "detail": "fixture"}],
        duration_s=0.5,
        random_seed=4159,
        detail="fixture detail",
    )
    assert exp4159.artifact_schema_errors(blocked) == []

    bad = dict(blocked)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "graft_deferred": "yes",
            "phase0_precision": {"ci95": [0.0, 0.0]},
            "rft_vs_ablation_delta": {"delta": 1.0, "ci95": "bad"},
            "verifier_value_added": "no",
            "preconditions_checked": [{"resource": "x"}],
            "field_principles": {},
            "baseline_status": "bad",
            "duration_s": False,
            "acceptance_gate_passed": "yes",
            "reproducibility_checksum": "bad",
            "random_seed": False,
        }
    )
    errors = exp4159.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "graft_deferred must be a bare bool" in errors
    assert "phase0_precision.precision is required" in errors
    assert "rft_vs_ablation_delta.ci95 must have two numeric bounds" in errors
    assert "verifier_value_added must be a bare bool" in errors
    assert "preconditions_checked entries must include resource and available" in errors
    assert "field_principles.honest_verdict mismatch" in errors
    assert "baseline_status must be an object or null" in errors
    assert "duration_s must be numeric" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "random_seed must be a bare int" in errors
    assert "missing required field honest_verdict" in exp4159.artifact_schema_errors({})

    nonnumeric_phase0 = dict(blocked)
    nonnumeric_phase0["phase0_precision"] = {"precision": "bad"}
    assert "phase0_precision.precision must be numeric" in exp4159.artifact_schema_errors(nonnumeric_phase0)
    missing_rft_delta = dict(blocked)
    missing_rft_delta["rft_vs_ablation_delta"] = {"ci95": [0.0, 0.0]}
    assert "rft_vs_ablation_delta.delta is required" in exp4159.artifact_schema_errors(missing_rft_delta)

    bad_deferred_value = dict(blocked)
    bad_deferred_value["verifier_value_added"] = True
    assert "verifier_value_added is only meaningful when graft_deferred is false" in exp4159.artifact_schema_errors(
        bad_deferred_value
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp4159.validate_artifact(bad)

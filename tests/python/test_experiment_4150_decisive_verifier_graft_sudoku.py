"""Tests for Exp 4150 decisive Sudoku verifier graft gate.

Spec refs: REQ-LEARN-4150, SCENARIO-LEARN-4150-DEFER,
SCENARIO-LEARN-4150-GRAFT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4139_decisive_verifier_graft_sudoku as exp4139
from carnot import experiment_4150_decisive_verifier_graft_sudoku as exp4150


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


def _pool(puzzle_id: str) -> exp4139.CandidatePool:
    puzzle = _tokens(PUZZLE)
    label = _tokens(SOLUTION)
    good = exp4139.CandidateSample(f"{puzzle_id}:good", label, trm_score=0.4)
    bad_a = exp4139.CandidateSample(f"{puzzle_id}:bad-a", _bad_solution(), trm_score=0.9)
    bad_b = exp4139.CandidateSample(f"{puzzle_id}:bad-b", _bad_solution(), trm_score=0.8)
    return exp4139.CandidatePool(puzzle_id, puzzle, label, [bad_a, bad_b, good])


def _write_exp4149(path: Path, *, stable: Path, val: float | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "experiment_4149_sudoku_accumulate_pass4_convergence",
        "honest_verdict": "complete: fixture",
        "stable_checkpoint_path": str(stable),
        "val_exact_accuracy": val,
        "val_trajectory_v384": [
            {"pass_label": "v384_start", "effective_val_exact_accuracy": 0.278},
            {"pass_label": "pass4", "effective_val_exact_accuracy": val},
        ],
        "preconditions_checked": [{"resource": "cuda_available", "available": True, "detail": "fixture"}],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _checks(stable: Path) -> list[dict[str, object]]:
    return [
        {"resource": "exp4149_artifact", "available": True, "detail": "fixture"},
        {"resource": "stable_checkpoint_path", "available": True, "detail": str(stable)},
        {"resource": "baseline_checkpoint", "available": True, "detail": "loadable fixture"},
        {"resource": "cuda_available", "available": True, "detail": "cuda fixture"},
    ]


def test_req_learn_4150_spec_declares_required_contract() -> None:
    """REQ-LEARN-4150: OpenSpec declares the 4150 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4150" in spec
    assert "SCENARIO-LEARN-4150-DEFER" in spec
    assert "SCENARIO-LEARN-4150-GRAFT" in spec
    assert exp4150.RESULT_FILENAME in spec
    for field in exp4150.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4150.FIELD_PRINCIPLES


def test_req_learn_4150_loads_baseline_and_checks_preconditions(tmp_path: Path) -> None:
    """REQ-LEARN-4150: Exp 4149 val and runtime checks gate Exp 4150."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    artifact_path = _write_exp4149(tmp_path / "results" / exp4150.DEFAULT_EXP4149_ARTIFACT.name, stable=stable, val=0.278)
    baseline = exp4150.load_baseline_context(artifact_path)

    assert baseline.artifact_path == artifact_path
    assert baseline.stable_checkpoint_path == stable
    assert baseline.val_exact_accuracy == pytest.approx(0.278)
    assert exp4150.baseline_is_faithful(baseline) is False
    estimate = exp4150.estimate_passes_to_converge_for_385(baseline)
    assert estimate["target_val_exact_accuracy"] == pytest.approx(0.85)
    assert estimate["estimated_additional_passes"] is None
    assert estimate["basis"] == "no_positive_v384_convergence_rate"

    checks = exp4150.check_preconditions(
        baseline,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert [check.resource for check in checks] == [
        "exp4149_artifact",
        "stable_checkpoint_path",
        "baseline_checkpoint",
        "cuda_available",
    ]
    assert checks[2].available is False

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    checks = exp4150.check_preconditions(
        baseline,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert all(check.available for check in checks)

    failed = exp4150.check_preconditions(
        baseline,
        cuda_checker=lambda: (_ for _ in ()).throw(RuntimeError("cuda broke")),
        checkpoint_loader=lambda _path: (_ for _ in ()).throw(RuntimeError("load broke")),
    )
    assert failed[2].available is False
    assert "RuntimeError: load broke" in failed[2].detail
    assert failed[3].available is False
    assert "RuntimeError: cuda broke" in failed[3].detail

    trajectory_only = tmp_path / "results" / "trajectory_only.json"
    trajectory_only.write_text(
        json.dumps(
            {
                "stable_checkpoint_path": str(stable),
                "val_exact_accuracy": None,
                "val_trajectory_v384": [{"effective_val_exact_accuracy": "0.86"}],
            }
        ),
        encoding="utf-8",
    )
    assert exp4150.load_baseline_context(trajectory_only).val_exact_accuracy == pytest.approx(0.86)
    fallback_trajectory = tmp_path / "results" / "fallback_trajectory.json"
    fallback_trajectory.write_text(
        json.dumps(
            {
                "stable_checkpoint_path": str(stable),
                "val_trajectory_v384": [{"val_exact_accuracy": "0.87"}],
            }
        ),
        encoding="utf-8",
    )
    assert exp4150.load_baseline_context(fallback_trajectory).val_exact_accuracy == pytest.approx(0.87)
    with pytest.raises(ValueError, match="JSON object"):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("[]\n", encoding="utf-8")
        exp4150.load_baseline_context(bad_json)


def test_req_learn_4150_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-4150: helper edge cases do not fabricate gate evidence."""

    assert exp4150._float_or_none("bad") is None
    assert exp4150._float_or_none(object()) is None
    assert exp4150._latest_trajectory_val({"val_trajectory_v384": "bad"}) is None
    assert exp4150._latest_trajectory_val({"val_trajectory_v384": [{"val_exact_accuracy": "bad"}]}) is None
    assert exp4150.estimate_passes_to_converge_for_385(None)["basis"] == "missing_exp4149_val_exact_accuracy"

    faithful = exp4150.BaselineContext(
        artifact_path=Path("exp4149.json"),
        stable_checkpoint_path=Path("last.ckpt"),
        val_exact_accuracy=0.86,
        raw={},
    )
    assert exp4150.estimate_passes_to_converge_for_385(faithful)["estimated_additional_passes"] == 0

    improving = exp4150.BaselineContext(
        artifact_path=Path("exp4149.json"),
        stable_checkpoint_path=Path("last.ckpt"),
        val_exact_accuracy=0.6,
        raw={"val_trajectory_v384": [{"val_exact_accuracy": 0.4}, {"val_exact_accuracy": 0.5}, {"val_exact_accuracy": 0.6}]},
    )
    estimate = exp4150.estimate_passes_to_converge_for_385(improving)
    assert estimate["basis"] == "mean_positive_v384_delta"
    assert estimate["estimated_additional_passes"] == 3

    class Scalar:
        def item(self) -> int:
            return 7

    assert exp4150._jsonable(Path("x")) == "x"
    assert exp4150._jsonable(Scalar()) == 7
    assert exp4150.verifier_value_added({"delta": "bad", "ci95": [1.0, 1.0]}, graft_deferred=False) is False
    assert exp4150._artifact_verdict(graft_deferred=False, value_added=False) == "complete: A~=B null"
    assert exp4150._acceptance_gate({"honest_verdict": "blocked_x"}) is False
    assert exp4150._acceptance_gate({"honest_verdict": "complete: x", "graft_deferred": "yes"}) is False
    assert exp4150._acceptance_gate(
        {"honest_verdict": "complete: x", "graft_deferred": False, "verifier_value_added": "no"}
    ) is False
    assert exp4150._acceptance_gate(
        {
            "honest_verdict": "complete: x",
            "graft_deferred": False,
            "verifier_value_added": False,
            "baseline_status": "bad",
        }
    ) is False


def test_scenario_learn_4150_defer_writes_acceptance_grade_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4150-DEFER: val<0.85 stops before candidate sampling."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    artifact_path = _write_exp4149(tmp_path / "results" / exp4150.DEFAULT_EXP4149_ARTIFACT.name, stable=stable, val=0.278)
    output = tmp_path / "results" / exp4150.RESULT_FILENAME

    artifact = exp4150.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4149_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pytest.fail("must not sample candidates"),
    )

    assert artifact["honest_verdict"].startswith("complete: graft_deferred")
    assert artifact["graft_deferred"] is True
    assert artifact["verifier_value_added"] is False
    assert artifact["baseline_status"]["val_exact_accuracy"] == pytest.approx(0.278)
    assert artifact["estimated_passes_to_converge_for_385"]["estimated_additional_passes"] is None
    assert artifact["rerank_lift_vs_vote"]["status"] == "deferred_baseline_below_0.85"
    assert artifact["rft_vs_ablation_delta"]["status"] == "deferred_baseline_below_0.85"
    assert artifact["acceptance_gate_passed"] is True
    assert exp4150.artifact_schema_errors(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4150_graft_reports_rerank_and_rft_delta(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4150-GRAFT: faithful fixture reports both measurement arms."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    artifact_path = _write_exp4149(tmp_path / "results" / exp4150.DEFAULT_EXP4149_ARTIFACT.name, stable=stable, val=0.86)
    pools = [_pool(f"p{i}") for i in range(6)]
    rft_calls = []

    def rft_runner(_baseline: exp4150.BaselineContext, corpora: dict[str, object]) -> dict[str, object]:
        rft_calls.append(corpora)
        return {
            "metric": "heldout_exact_accuracy",
            "n_matched": corpora["n_matched"],
            "a_exact_accuracy": 1.0,
            "b_exact_accuracy": 0.0,
            "delta": 1.0,
            "ci95": [1.0, 1.0],
            "status": "ci95_excludes_zero",
        }

    artifact = exp4150.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "grafted.json",
        exp4149_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pools,
        rft_runner=rft_runner,
        bootstrap_resamples=100,
    )

    assert artifact["graft_deferred"] is False
    assert artifact["rerank_lift_vs_vote"]["delta"] == pytest.approx(1.0)
    assert artifact["rerank_lift_vs_vote"]["oracle_pass_at_k"] == pytest.approx(1.0)
    assert artifact["rft_vs_ablation_delta"]["delta"] == pytest.approx(1.0)
    assert artifact["verifier_value_added"] is True
    assert artifact["acceptance_gate_passed"] is True
    assert rft_calls and rft_calls[0]["n_matched"] == len(pools)
    assert exp4150.artifact_schema_errors(artifact) == []

    proxy_rft = exp4150.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "proxy_rft.json",
        exp4149_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pools,
        bootstrap_resamples=100,
    )
    assert proxy_rft["rft_vs_ablation_delta"]["status"] == "ci95_excludes_zero"
    assert proxy_rft["verifier_value_added"] is True


def test_req_learn_4150_schema_and_blocked_paths(tmp_path: Path) -> None:
    """REQ-LEARN-4150: schema rejects ambiguous or fake 4150 artifacts."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    baseline = exp4150.BaselineContext(
        artifact_path=tmp_path / "results" / "exp4149.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.9,
        raw={},
    )
    blocked = exp4150.build_blocked_artifact(
        "blocked_baseline_checkpoint_missing",
        baseline=baseline,
        preconditions_checked=[{"resource": "baseline_checkpoint", "available": False, "detail": "missing"}],
        duration_s=0.5,
    )
    assert blocked["acceptance_gate_passed"] is False
    assert exp4150.artifact_schema_errors(blocked) == []

    bad = dict(blocked)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "graft_deferred": "yes",
            "verifier_value_added": "no",
            "preconditions_checked": [{"resource": "x"}],
            "field_principles": {},
            "duration_s": False,
            "acceptance_gate_passed": "yes",
            "baseline_status": "bad",
            "rerank_lift_vs_vote": {"delta": 1.0, "ci95": "bad"},
            "rft_vs_ablation_delta": [],
            "reproducibility_checksum": "bad",
        }
    )
    errors = exp4150.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "graft_deferred must be a bare bool" in errors
    assert "verifier_value_added must be a bare bool" in errors
    assert "preconditions_checked entries must include resource and available" in errors
    assert "field_principles.honest_verdict mismatch" in errors
    assert "duration_s must be numeric" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "baseline_status must be an object" in errors
    assert "rerank_lift_vs_vote.ci95 must have two numeric bounds" in errors
    assert "rft_vs_ablation_delta must be an object" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "missing required field honest_verdict" in exp4150.artifact_schema_errors({})

    bad_deferred_value = dict(blocked)
    bad_deferred_value["verifier_value_added"] = True
    assert "verifier_value_added is only meaningful when graft_deferred is false" in exp4150.artifact_schema_errors(
        bad_deferred_value
    )
    missing_delta = dict(blocked)
    missing_delta["rerank_lift_vs_vote"] = {"ci95": [0.0, 0.0]}
    assert "rerank_lift_vs_vote.delta is required" in exp4150.artifact_schema_errors(missing_delta)

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4150.validate_artifact(bad)

    written = exp4150.write_artifact(tmp_path / "blocked.json", blocked)
    assert written == blocked

    missing_baseline = exp4150.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "missing.json",
        exp4149_artifact_path=tmp_path / "missing_exp4149.json",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert missing_baseline["honest_verdict"] == "blocked_exp4149_baseline_missing"

    artifact_path = _write_exp4149(tmp_path / "results" / exp4150.DEFAULT_EXP4149_ARTIFACT.name, stable=stable, val=0.86)
    precondition_blocked = exp4150.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "precondition_blocked.json",
        exp4149_artifact_path=artifact_path,
        cuda_checker=lambda: (False, "no cuda"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert precondition_blocked["honest_verdict"] == "blocked_exp4150_preconditions_missing"

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    sampling_blocked = exp4150.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "sampling_blocked.json",
        exp4149_artifact_path=artifact_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: (_ for _ in ()).throw(RuntimeError("sampling broke")),
    )
    assert sampling_blocked["honest_verdict"] == "blocked_candidate_sampling_failed"

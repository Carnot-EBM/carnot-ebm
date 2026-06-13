"""Tests for Exp 4139 decisive Sudoku verifier graft.

Spec refs: REQ-LEARN-4139, SCENARIO-LEARN-4139-NO-HEADROOM,
SCENARIO-LEARN-4139-RERANK, SCENARIO-LEARN-4139-RFT-OR-DEFER.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4139_decisive_verifier_graft_sudoku as exp4139


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


def _pool(puzzle_id: str, *, vote_wrong: bool) -> exp4139.CandidatePool:
    puzzle = _tokens(PUZZLE)
    label = _tokens(SOLUTION)
    good = exp4139.CandidateSample(f"{puzzle_id}:good", label, trm_score=0.4)
    bad_a = exp4139.CandidateSample(f"{puzzle_id}:bad-a", _bad_solution(), trm_score=0.9)
    bad_b = exp4139.CandidateSample(f"{puzzle_id}:bad-b", _bad_solution(), trm_score=0.8)
    candidates = [bad_a, bad_b, good] if vote_wrong else [good, good, bad_a]
    return exp4139.CandidatePool(puzzle_id, puzzle, label, candidates)


def _write_exp4138(
    path: Path,
    *,
    stable: Path,
    val: float | None = 0.75,
    matches: bool = False,
    near: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "experiment_4138_sudoku_accumulate_pass4_convergence_check",
        "honest_verdict": "complete: fixture",
        "matches_published_087": matches,
        "near_faithful_080": near,
        "stable_checkpoint_path": str(stable),
        "val_exact_accuracy": val,
        "estimated_passes_to_converge": 2 if val is not None and val < 0.8 else None,
        "val_trajectory_383": [
            {"label": ".382_anchor", "val_exact_accuracy": 0.5},
            {"label": ".383_pass4", "val_exact_accuracy": val},
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _checks(stable: Path) -> list[dict[str, object]]:
    return [
        {"resource": "exp4138_artifact", "available": True, "detail": "fixture"},
        {"resource": "stable_checkpoint_path", "available": True, "detail": str(stable)},
        {"resource": "baseline_checkpoint", "available": True, "detail": "loadable fixture"},
        {"resource": "cuda_available", "available": True, "detail": "cuda fixture"},
    ]


def test_req_learn_4139_spec_declares_oracle_separated_contract() -> None:
    """REQ-LEARN-4139: OpenSpec declares the decisive artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4139" in spec
    assert "SCENARIO-LEARN-4139-NO-HEADROOM" in spec
    assert "SCENARIO-LEARN-4139-RERANK" in spec
    assert "SCENARIO-LEARN-4139-RFT-OR-DEFER" in spec
    assert exp4139.RESULT_FILENAME in spec
    for field in exp4139.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4139.FIELD_PRINCIPLES


def test_req_learn_4139_loads_exp4138_and_checks_checkpoint(tmp_path: Path) -> None:
    """REQ-LEARN-4139: Exp 4138 fields and runtime preconditions gate the graft."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    artifact_path = _write_exp4138(tmp_path / "results" / exp4139.DEFAULT_EXP4138_ARTIFACT.name, stable=stable)
    baseline = exp4139.load_baseline_context(artifact_path)

    assert baseline.artifact_path == artifact_path
    assert baseline.stable_checkpoint_path == stable
    assert baseline.val_exact_accuracy == pytest.approx(0.75)
    assert baseline.matches_published_087 is False
    assert baseline.near_faithful_080 is False
    assert exp4139.baseline_runs_rft_arm(baseline) is False
    assert exp4139.build_384_estimate(baseline)["estimated_additional_passes"] == 2

    checks = exp4139.check_preconditions(
        baseline,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert [check.resource for check in checks] == [
        "exp4138_artifact",
        "stable_checkpoint_path",
        "baseline_checkpoint",
        "cuda_available",
    ]
    assert checks[2].available is False

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    checks = exp4139.check_preconditions(
        baseline,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert all(check.available for check in checks)

    failed = exp4139.check_preconditions(
        baseline,
        cuda_checker=lambda: (_ for _ in ()).throw(RuntimeError("cuda broke")),
        checkpoint_loader=lambda _path: (_ for _ in ()).throw(RuntimeError("load broke")),
    )
    assert failed[2].available is False
    assert "RuntimeError: load broke" in failed[2].detail
    assert failed[3].available is False
    assert "RuntimeError: cuda broke" in failed[3].detail

    bad_json = tmp_path / "results" / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp4139.load_baseline_context(bad_json)

    trajectory_only = tmp_path / "results" / "trajectory_only.json"
    trajectory_only.write_text(
        json.dumps(
            {
                "stable_checkpoint_path": str(stable),
                "val_exact_accuracy": None,
                "val_trajectory_383": [{"val_exact_accuracy": "0.82"}],
                "estimated_passes_to_converge": 2.0,
            }
        ),
        encoding="utf-8",
    )
    trajectory_baseline = exp4139.load_baseline_context(trajectory_only)
    assert trajectory_baseline.val_exact_accuracy == pytest.approx(0.82)
    assert trajectory_baseline.estimated_passes_to_converge == 2
    assert exp4139.baseline_runs_rft_arm(trajectory_baseline) is True


def test_req_learn_4139_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-4139: numeric, CI, and scoring helpers handle edge cases."""

    assert exp4139._float_or_none("0.25") == pytest.approx(0.25)
    assert exp4139._float_or_none("nan") is None
    assert exp4139._float_or_none(object()) is None
    assert exp4139._int_or_none(3.0) == 3
    assert exp4139._int_or_none(-1) is None
    assert exp4139._int_or_none(1.5) is None
    assert exp4139._bootstrap_ci([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert exp4139._last_trajectory_val({"val_trajectory_383": "bad"}) is None
    assert exp4139._last_trajectory_val({"val_trajectory_383": [{"val_exact_accuracy": "bad"}]}) is None
    assert exp4139.build_384_estimate(
        exp4139.BaselineContext(
            artifact_path=Path("exp4138.json"),
            stable_checkpoint_path=Path("results/trm_runs/sudoku_extreme_baseline/last.ckpt"),
            val_exact_accuracy=None,
            matches_published_087=False,
            near_faithful_080=False,
            estimated_passes_to_converge=None,
            raw={},
        )
    )["basis"] == "exp4138_missing_or_config_blocked_estimate"

    class Scalar:
        def item(self) -> int:
            return 9

    assert exp4139._jsonable(Scalar()) == 9
    assert exp4139.verifier_value_added(
        headroom_present=True,
        ensemble_rerank_lift_vs_vote={"delta": "bad", "ci95": [0.1, 0.2]},
        rft_vs_ablation_delta={"delta": 0.3, "ci95": [0.1, 0.4]},
    ) == (True, ["rft_vs_ablation_delta"])
    assert exp4139.verifier_value_added(
        headroom_present=True,
        ensemble_rerank_lift_vs_vote={"delta": 0.3, "ci95": [0.1, 0.4]},
        rft_vs_ablation_delta={"delta": "bad", "ci95": [0.1, 0.4]},
    ) == (True, ["ensemble_rerank_lift_vs_vote"])
    assert exp4139._artifact_verdict(
        headroom_present=True,
        graft_deferred=True,
        value_added=False,
        basis=[],
    ) == "complete: graft_deferred_baseline_below_0.80"
    assert exp4139._artifact_verdict(
        headroom_present=True,
        graft_deferred=False,
        value_added=False,
        basis=[],
    ) == "complete: A~=B null with headroom present"
    assert exp4139._acceptance_gate({"honest_verdict": "blocked_x"}) is False
    assert exp4139._acceptance_gate({"honest_verdict": "complete: x", "headroom_present": "yes"}) is False
    assert exp4139._acceptance_gate(
        {"honest_verdict": "complete: x", "headroom_present": True, "executable_verifier_is_oracle": False}
    ) is False
    assert exp4139._acceptance_gate(
        {"honest_verdict": "complete: x", "headroom_present": True, "executable_verifier_is_oracle": True}
    ) is False

    invalid_tokens = [99] * 81
    score = exp4139.non_oracle_ensemble_score(_tokens(PUZZLE), invalid_tokens)
    assert score["uses_exact_validity_check"] is False
    assert score["ensemble_score"] < 1.0
    empty_pool = exp4139.CandidatePool("empty", _tokens(PUZZLE), _tokens(SOLUTION), [])
    with pytest.raises(ValueError, match="empty"):
        exp4139.select_ensemble_candidate(empty_pool)


def test_scenario_learn_4139_rerank_separates_oracle_upper_bound() -> None:
    """SCENARIO-LEARN-4139-RERANK: executable oracle is not the headline verifier."""

    pools = [_pool(f"p{i}", vote_wrong=True) for i in range(6)]
    metrics = exp4139.evaluate_reranks(pools, random_seed=13, bootstrap_resamples=200)

    assert metrics["headroom_present"] is True
    assert metrics["false_negative_risk"] is False
    assert metrics["oracle_vs_vote_gap"] == pytest.approx(1.0)
    assert metrics["executable_verifier_is_oracle"] is True
    assert metrics["executable_oracle_upper_bound"]["delta"] == pytest.approx(
        metrics["oracle_vs_vote_gap"]
    )
    assert metrics["executable_oracle_upper_bound"]["interpretation"] == "oracle_upper_bound_not_verifier_value"
    assert metrics["ensemble_rerank_lift_vs_vote"]["delta"] == pytest.approx(1.0)
    assert metrics["ensemble_rerank_lift_vs_vote"]["score_components"] == [
        "continuous_sudoku_energy",
        "digit_text_statistics",
    ]
    assert len(metrics["ensemble_rerank_lift_vs_vote"]["ci95"]) == 2
    assert all(row["ensemble_sample_id"].endswith(":good") for row in metrics["per_puzzle"])


def test_scenario_learn_4139_no_headroom_sets_false_negative_risk() -> None:
    """SCENARIO-LEARN-4139-NO-HEADROOM: no oracle gap makes rerank uninformative."""

    pools = [_pool(f"p{i}", vote_wrong=False) for i in range(4)]
    metrics = exp4139.evaluate_reranks(pools, random_seed=19, bootstrap_resamples=100)

    assert metrics["headroom_present"] is False
    assert metrics["false_negative_risk"] is True
    assert metrics["oracle_vs_vote_gap"] == pytest.approx(0.0)
    assert metrics["executable_oracle_upper_bound"]["delta"] == pytest.approx(0.0)

    artifact = exp4139.build_result_artifact(
        baseline=exp4139.BaselineContext(
            artifact_path=Path("exp4138.json"),
            stable_checkpoint_path=Path("results/trm_runs/sudoku_extreme_baseline/last.ckpt"),
            val_exact_accuracy=0.86,
            matches_published_087=True,
            near_faithful_080=True,
            estimated_passes_to_converge=None,
            raw={},
        ),
        rerank_metrics=metrics,
        rft_vs_ablation_delta={"metric": "heldout_exact_accuracy", "delta": 1.0, "ci95": [1.0, 1.0]},
        preconditions_checked=_checks(Path("results/trm_runs/sudoku_extreme_baseline/last.ckpt")),
        duration_s=1.0,
        candidate_source="fixture",
        k_candidates=3,
        n_candidate_pools=len(pools),
    )
    assert artifact["honest_verdict"] == "complete: uninformative_no_headroom_false_negative_risk"
    assert artifact["verifier_value_added"] is False
    assert artifact["false_negative_risk"] is True


def test_scenario_learn_4139_rft_or_defer_controls_value_added(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4139-RFT-OR-DEFER: RFT runs only for near-faithful baselines."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    low = exp4139.BaselineContext(
        artifact_path=tmp_path / "results" / "experiment_4138.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.75,
        matches_published_087=False,
        near_faithful_080=False,
        estimated_passes_to_converge=3,
        raw={},
    )
    near = exp4139.BaselineContext(
        artifact_path=tmp_path / "results" / "experiment_4138.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.81,
        matches_published_087=False,
        near_faithful_080=True,
        estimated_passes_to_converge=None,
        raw={},
    )
    metrics = exp4139.evaluate_reranks(
        [_pool(f"p{i}", vote_wrong=True) for i in range(5)],
        random_seed=23,
        bootstrap_resamples=100,
    )

    deferred = exp4139.build_result_artifact(
        baseline=low,
        rerank_metrics=metrics,
        rft_vs_ablation_delta=exp4139.deferred_rft_delta(low),
        preconditions_checked=_checks(stable),
        duration_s=2.0,
        candidate_source="fixture",
        k_candidates=3,
        n_candidate_pools=5,
    )
    assert deferred["graft_deferred"] is True
    assert deferred["rft_vs_ablation_delta"]["status"] == "deferred_baseline_below_0.80"
    assert deferred["verifier_value_added"] is True
    assert deferred["verifier_value_added_basis"] == ["ensemble_rerank_lift_vs_vote"]
    assert deferred["estimated_passes_to_converge_for_384"]["estimated_additional_passes"] == 3
    assert exp4139.artifact_schema_errors(deferred) == []

    rft_delta = {"metric": "heldout_exact_accuracy", "n_matched": 5, "delta": 0.4, "ci95": [0.2, 0.6]}
    grafted = exp4139.build_result_artifact(
        baseline=near,
        rerank_metrics=metrics,
        rft_vs_ablation_delta=rft_delta,
        preconditions_checked=_checks(stable),
        duration_s=2.0,
        candidate_source="fixture",
        k_candidates=3,
        n_candidate_pools=5,
    )
    assert grafted["graft_deferred"] is False
    assert grafted["rft_vs_ablation_delta"] == rft_delta
    assert grafted["verifier_value_added"] is True
    assert grafted["verifier_value_added_basis"] == [
        "ensemble_rerank_lift_vs_vote",
        "rft_vs_ablation_delta",
    ]


def test_req_learn_4139_schema_write_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-4139: schema rejects ambiguous oracle/verifier artifacts."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    blocked = exp4139.build_blocked_artifact(
        "blocked_baseline_checkpoint_missing",
        baseline=None,
        preconditions_checked=[{"resource": "baseline_checkpoint", "available": False, "detail": "missing"}],
        duration_s=0.5,
    )
    assert blocked["honest_verdict"] == "blocked_baseline_checkpoint_missing"
    assert blocked["executable_verifier_is_oracle"] is True
    assert blocked["ensemble_rerank_lift_vs_vote"]["status"] == "not_run_preconditions_failed"
    assert exp4139.artifact_schema_errors(blocked) == []

    bad = dict(blocked)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "headroom_present": "false",
            "oracle_vs_vote_gap": None,
            "executable_verifier_is_oracle": False,
            "graft_deferred": "yes",
            "verifier_value_added": "no",
            "preconditions_checked": [{"resource": "x"}],
            "field_principles": {},
            "duration_s": False,
            "acceptance_gate_passed": "yes",
        }
    )
    errors = exp4139.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "headroom_present must be a bare bool" in errors
    assert "oracle_vs_vote_gap must be numeric" in errors
    assert "executable_verifier_is_oracle must be true for unique-solution Sudoku" in errors
    assert "graft_deferred must be a bare bool" in errors
    assert "verifier_value_added must be a bare bool" in errors
    assert "preconditions_checked entries must include resource and available" in errors
    assert "field_principles.honest_verdict mismatch" in errors
    assert "duration_s must be numeric" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "missing required field honest_verdict" in exp4139.artifact_schema_errors({})

    bad_metrics = dict(blocked)
    bad_metrics.update(
        {
            "headroom_present": False,
            "verifier_value_added": True,
            "oracle_vs_vote_gap": 0.25,
            "executable_oracle_upper_bound": {"delta": 0.0, "ci95": [0.0, 0.0]},
            "ensemble_rerank_lift_vs_vote": {
                "delta": 0.0,
                "ci95": "bad",
                "uses_exact_validity_check": True,
            },
            "rft_vs_ablation_delta": [],
            "verifier_value_added_basis": ["executable_oracle_upper_bound"],
            "reproducibility_checksum": "bad",
        }
    )
    metric_errors = exp4139.artifact_schema_errors(bad_metrics)
    assert "verifier_value_added requires headroom_present" in metric_errors
    assert "rft_vs_ablation_delta must be an object" in metric_errors
    assert "ensemble_rerank_lift_vs_vote.ci95 must have two numeric bounds" in metric_errors
    assert "executable_oracle_upper_bound.delta must equal oracle_vs_vote_gap" in metric_errors
    assert "ensemble_rerank_lift_vs_vote must not use exact validity" in metric_errors
    assert "verifier_value_added_basis must not include executable oracle" in metric_errors
    assert "reproducibility_checksum must be sha256-prefixed" in metric_errors
    bad_basis = dict(blocked)
    bad_basis["verifier_value_added_basis"] = "bad"
    assert "verifier_value_added_basis must be a list" in exp4139.artifact_schema_errors(bad_basis)
    missing_delta = dict(blocked)
    missing_delta["ensemble_rerank_lift_vs_vote"] = {"ci95": [0.0, 0.0]}
    assert "ensemble_rerank_lift_vs_vote.delta is required" in exp4139.artifact_schema_errors(missing_delta)

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4139.validate_artifact(bad)
    output = tmp_path / exp4139.RESULT_FILENAME
    written = exp4139.write_artifact(output, blocked)
    assert written == blocked
    assert json.loads(output.read_text(encoding="utf-8")) == blocked

    assert exp4139._jsonable(stable) == str(stable)
    assert exp4139._metric_has_ci({"ci95": [0.0, 1.0]}) is True
    assert exp4139._metric_has_ci({"ci95": ["bad", 1.0]}) is False


def test_req_learn_4139_run_experiment_writes_blocked_deferred_and_grafted(tmp_path: Path) -> None:
    """REQ-LEARN-4139: run_experiment writes the selected terminal artifact."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    low_path = _write_exp4138(tmp_path / "results" / exp4139.DEFAULT_EXP4138_ARTIFACT.name, stable=stable)
    output = tmp_path / "results" / exp4139.RESULT_FILENAME

    blocked = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4138_artifact_path=low_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pytest.fail("must stop before sampling"),
    )
    assert blocked["honest_verdict"] == "blocked_baseline_checkpoint_missing"
    assert json.loads(output.read_text(encoding="utf-8")) == blocked

    missing_baseline = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_baseline.json",
        exp4138_artifact_path=tmp_path / "results" / "missing_exp4138.json",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
    )
    assert missing_baseline["honest_verdict"] == "blocked_exp4138_baseline_missing"

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    cuda_blocked = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "cuda_blocked.json",
        exp4138_artifact_path=low_path,
        cuda_checker=lambda: (False, "no cuda"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pytest.fail("must stop before sampling"),
    )
    assert cuda_blocked["honest_verdict"] == "blocked_exp4139_preconditions_missing"

    sampling_blocked = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "sampling_blocked.json",
        exp4138_artifact_path=low_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: (_ for _ in ()).throw(RuntimeError("sampling broke")),
    )
    assert sampling_blocked["honest_verdict"] == "blocked_candidate_sampling_failed"

    pools = [_pool(f"p{i}", vote_wrong=True) for i in range(5)]
    deferred = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4138_artifact_path=low_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pools,
        bootstrap_resamples=100,
    )
    assert deferred["graft_deferred"] is True
    assert deferred["candidate_source"] == "provided_candidate_pool"
    assert deferred["headroom_present"] is True

    near_path = _write_exp4138(
        tmp_path / "results" / "experiment_4138_near.json",
        stable=stable,
        val=0.81,
        matches=False,
        near=True,
    )
    rft_calls = []

    def rft_runner(_baseline: exp4139.BaselineContext, corpora: dict[str, object]) -> dict[str, object]:
        rft_calls.append(corpora)
        return {"metric": "heldout_exact_accuracy", "n_matched": corpora["n_matched"], "delta": 0.2, "ci95": [0.1, 0.3]}

    grafted = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "grafted.json",
        exp4138_artifact_path=near_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pools,
        rft_runner=rft_runner,
        bootstrap_resamples=100,
    )
    assert grafted["graft_deferred"] is False
    assert grafted["rft_vs_ablation_delta"]["delta"] == pytest.approx(0.2)
    assert len(rft_calls) == 1

    proxy_rft = exp4139.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "proxy_rft.json",
        exp4138_artifact_path=near_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        candidate_pool_provider=lambda _baseline: pools,
        bootstrap_resamples=100,
    )
    assert proxy_rft["rft_vs_ablation_delta"]["status"] == "ci95_excludes_zero"

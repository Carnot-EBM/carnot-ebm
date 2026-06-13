"""Tests for Exp 4158 Sudoku verifier rerank recovery moat.

Spec refs: REQ-LEARN-4158, SCENARIO-LEARN-4158-NO-HEADROOM,
SCENARIO-LEARN-4158-RERANK-RECOVERY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4158_verifier_rerank_recovery_moat as exp4158


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


def _pool(puzzle_id: str, *, vote_wrong: bool) -> exp4109.CandidatePool:
    puzzle = _tokens(PUZZLE)
    label = _tokens(SOLUTION)
    good = exp4109.CandidateSample(f"{puzzle_id}:good", label, trm_score=0.2)
    bad_a = exp4109.CandidateSample(f"{puzzle_id}:bad-a", _bad_solution(), trm_score=0.9)
    bad_b = exp4109.CandidateSample(f"{puzzle_id}:bad-b", _bad_solution(), trm_score=0.8)
    candidates = [bad_a, bad_b, good] if vote_wrong else [good, good, bad_a]
    return exp4109.CandidatePool(puzzle_id, puzzle, label, candidates)


def _write_exp4157(path: Path, *, stable: Path, current_val: float = 0.501041650772) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "blocked_noop_step_unchanged",
                "current_val": current_val,
                "max_val": current_val,
                "stable_checkpoint_path": str(stable),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _cost_fixture() -> dict[str, object]:
    return {
        "verifier_mean_per_candidate_us": 2.0,
        "llm_judge_estimate_per_candidate_us": 1_000_000.0,
        "ratio": 500_000.0,
        "estimate": "Assumes 1.0s per remote LLM judge call.",
    }


def test_req_learn_4158_spec_declares_required_contract() -> None:
    """REQ-LEARN-4158: OpenSpec declares the rerank recovery artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4158" in spec
    assert "SCENARIO-LEARN-4158-NO-HEADROOM" in spec
    assert "SCENARIO-LEARN-4158-RERANK-RECOVERY" in spec
    assert exp4158.RESULT_FILENAME in spec
    for field in exp4158.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4158.FIELD_PRINCIPLES


def test_scenario_learn_4158_rerank_recovery_counts_outvoted_answers() -> None:
    """SCENARIO-LEARN-4158-RERANK-RECOVERY: verifier recovers present-but-out-voted answers."""

    pools = [_pool(f"p{i}", vote_wrong=True) for i in range(5)]
    metrics = exp4158.evaluate_recovery_moat(pools, random_seed=41, bootstrap_resamples=100)

    assert metrics["headroom_present"] is True
    assert metrics["oracle_at_k"] == pytest.approx(1.0)
    assert metrics["vote_at_1"] == pytest.approx(0.0)
    assert metrics["rerank_lift_vs_vote"]["delta"] == pytest.approx(1.0)
    assert metrics["rerank_lift_vs_vote"]["ci95"][0] > 0.0
    assert metrics["verifier_recovers_outvoted"] == 5


def test_scenario_learn_4158_no_headroom_reports_honest_guard(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4158-NO-HEADROOM: oracle<=vote writes no-headroom verdict."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"stable checkpoint")
    exp4157_path = _write_exp4157(tmp_path / "results" / exp4158.DEFAULT_EXP4157_ARTIFACT.name, stable=stable)
    output = tmp_path / "results" / exp4158.RESULT_FILENAME

    def provider(snapshot: Path) -> list[exp4109.CandidatePool]:
        assert snapshot.name == "last-rerank-snapshot.ckpt"
        assert snapshot.read_bytes() == b"stable checkpoint"
        return [_pool(f"p{i}", vote_wrong=False) for i in range(4)]

    artifact = exp4158.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4157_artifact_path=exp4157_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=provider,
        verifier_cost_provider=lambda _pools: _cost_fixture(),
        bootstrap_resamples=100,
    )

    assert artifact["honest_verdict"] == "complete: no_headroom_rerank_uninformative_at_val_0.50"
    assert artifact["headroom_present"] is False
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["vote_at_1"] == pytest.approx(1.0)
    assert artifact["rerank_lift_vs_vote"]["status"] == "no_headroom_oracle_at_k_lte_vote_at_1"
    assert artifact["acceptance_gate_passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp4158.artifact_schema_errors(artifact) == []


def test_req_learn_4158_run_experiment_reports_headroom_backed_moat(tmp_path: Path) -> None:
    """REQ-LEARN-4158: headroom-backed rerank artifact includes CI, recovery, and cost."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"stable checkpoint")
    exp4157_path = _write_exp4157(tmp_path / "results" / exp4158.DEFAULT_EXP4157_ARTIFACT.name, stable=stable)
    pools = [_pool(f"p{i}", vote_wrong=True) for i in range(6)]

    artifact = exp4158.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "moat.json",
        exp4157_artifact_path=exp4157_path,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _snapshot: pools,
        verifier_cost_provider=lambda _pools: _cost_fixture(),
        bootstrap_resamples=100,
    )

    assert artifact["honest_verdict"] == "complete: verifier_rerank_moat_ci95_excludes_zero_at_val_0.50"
    assert artifact["headroom_present"] is True
    assert artifact["rerank_lift_vs_vote"]["verifier_pass_at_1"] == pytest.approx(1.0)
    assert artifact["rerank_lift_vs_vote"]["vote_at_1"] == pytest.approx(0.0)
    assert artifact["rerank_lift_vs_vote"]["ci95"][0] > 0.0
    assert artifact["verifier_recovers_outvoted"] == len(pools)
    assert artifact["cost_ratio_vs_llm_judge"]["ratio"] == pytest.approx(500_000.0)
    assert artifact["acceptance_gate_passed"] is True
    assert exp4158.validate_artifact(artifact) is None


def test_req_learn_4158_blocked_and_schema_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-4158: blocked artifacts and malformed schema errors are explicit."""

    output = tmp_path / "results" / "blocked.json"
    blocked = exp4158.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        exp4157_artifact_path=tmp_path / "missing.json",
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _snapshot: pytest.fail("must not sample when blocked"),
    )

    assert blocked["honest_verdict"] == "blocked_exp4157_artifact"
    assert blocked["acceptance_gate_passed"] is False
    assert blocked["rerank_lift_vs_vote"]["status"] == "blocked_exp4157_artifact"
    assert json.loads(output.read_text(encoding="utf-8"))["honest_verdict"].startswith("blocked_")
    assert exp4158.artifact_schema_errors(blocked) == []

    errors = exp4158.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "headroom_present": "yes",
            "oracle_at_k": "1.0",
            "vote_at_1": None,
            "rerank_lift_vs_vote": {"ci95": [0.0]},
            "verifier_recovers_outvoted": True,
            "cost_ratio_vs_llm_judge": [],
            "random_seed": False,
            "field_principles": {"honest_verdict": "wrong"},
        }
    )

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "headroom_present must be a bare bool" in errors
    assert "oracle_at_k must be a number" in errors
    assert "vote_at_1 must be a number" in errors
    assert "rerank_lift_vs_vote.delta is required" in errors
    assert "rerank_lift_vs_vote.ci95 must have two bounds" in errors
    assert "verifier_recovers_outvoted must be a bare int" in errors
    assert "cost_ratio_vs_llm_judge must be an object" in errors
    assert "random_seed must be a bare int" in errors
    assert "field_principles.honest_verdict mismatch" in errors

    with pytest.raises(ValueError, match="headroom_present"):
        exp4158.validate_artifact({"honest_verdict": "bad"})

    errors = exp4158.artifact_schema_errors(
        {
            "honest_verdict": 7,
            "headroom_present": False,
            "oracle_at_k": 0.0,
            "vote_at_1": 0.0,
            "rerank_lift_vs_vote": exp4158._empty_rerank("fixture"),
            "verifier_recovers_outvoted": 0,
            "cost_ratio_vs_llm_judge": {},
            "random_seed": 4158,
            "field_principles": exp4158.FIELD_PRINCIPLES,
        }
    )
    assert "honest_verdict must be a string" in errors


def test_req_learn_4158_helper_edges_do_not_fabricate_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LEARN-4158: helper branches keep resource and metric failures honest."""

    with pytest.raises(ValueError, match="Exp 4157 artifact must be a JSON object"):
        bad = tmp_path / "bad.json"
        bad.write_text("[]", encoding="utf-8")
        exp4158.load_baseline_context(bad)

    bad_output = tmp_path / "results" / "bad_exp4157.json"
    bad_run = exp4158.run_experiment(
        output_path=bad_output,
        exp4157_artifact_path=bad,
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert bad_run["honest_verdict"] == "blocked_exp4157_artifact"
    assert "ValueError" in bad_run["blocked_detail"]

    stable = tmp_path / "last.ckpt"
    exp4157_path = _write_exp4157(tmp_path / "exp4157.json", stable=stable, current_val=0.6)
    baseline = exp4158.load_baseline_context(exp4157_path)
    assert baseline.current_val == pytest.approx(0.6)
    assert baseline.max_val == pytest.approx(0.6)
    assert exp4158._format_val_tag(None) == "unknown"
    assert exp4158._float_or_none("0.25") == pytest.approx(0.25)
    assert exp4158._float_or_none("bad") is None
    assert exp4158._float_or_none(False) is None
    assert exp4158._float_or_none(object()) is None
    assert exp4158._metric_has_ci({"ci95": ["bad", 1.0]}) is False
    assert exp4158._is_json_number("1.0") is False

    class Scalar:
        def item(self) -> int:
            return 7

    assert exp4158._jsonable(tmp_path / "x") == str(tmp_path / "x")
    assert exp4158._jsonable(Scalar()) == 7

    checks = exp4158.check_preconditions(baseline, cuda_checker=lambda: (False, "no cuda"))
    assert [check.resource for check in checks] == [
        "exp4157_artifact",
        "stable_checkpoint_path",
        "stable_checkpoint",
        "cuda_available",
    ]
    assert checks[2].available is False
    assert checks[3].detail == "no cuda"

    precondition_run = exp4158.run_experiment(
        output_path=tmp_path / "results" / "precondition_blocked.json",
        exp4157_artifact_path=exp4157_path,
        cuda_checker=lambda: (False, "no cuda"),
    )
    assert precondition_run["honest_verdict"] == "blocked_stable_checkpoint"
    assert precondition_run["blocked_detail"] == str(stable)

    stable.write_bytes(b"source")
    snapshot = exp4158.snapshot_checkpoint(stable)
    assert snapshot.read_bytes() == b"source"
    assert exp4158.measure_verifier_cost_us([], perf_counter=lambda: 1.0)["status"] == "no_candidates"

    measured = exp4158.measure_verifier_cost_us(
        [_pool("cost", vote_wrong=True)],
        perf_counter=iter([1.0, 1.000012]).__next__,
    )
    assert measured["status"] == "measured_local_constraint_check"
    assert measured["verifier_mean_per_candidate_us"] == pytest.approx(4.0)
    assert measured["ratio"] == pytest.approx(250_000.0)

    monkeypatch.setattr(
        exp4158.exp4109,
        "evaluate_rerank",
        lambda *_args, **_kwargs: {
            "n_puzzles": 2,
            "vote_pass_at_1": 0.5,
            "verifier_pass_at_1": 0.5,
            "oracle_ceiling_pass_at_1": 1.0,
            "delta": 0.0,
            "ci95": [-0.5, 0.5],
            "per_puzzle": [],
        },
    )
    null_metrics = exp4158.evaluate_recovery_moat([_pool("null", vote_wrong=True)], bootstrap_resamples=10)
    assert null_metrics["headroom_present"] is True
    assert null_metrics["rerank_lift_vs_vote"]["status"] == "headroom_backed_null_ci95_includes_zero"
    assert exp4158._verdict(null_metrics, 0.6) == "complete: headroom_backed_A_approx_vote_null_at_val_0.60"

    assert exp4158._acceptance_gate({"honest_verdict": "blocked_x"}) is False
    assert exp4158._acceptance_gate({"honest_verdict": "complete: x", "headroom_present": "yes"}) is False

    def fail_snapshot(_path: Path) -> Path:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(exp4158, "snapshot_checkpoint", fail_snapshot)
    snapshot_failed = exp4158.run_experiment(
        output_path=tmp_path / "results" / "snapshot_failed.json",
        exp4157_artifact_path=exp4157_path,
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert snapshot_failed["honest_verdict"] == "blocked_checkpoint_snapshot"
    assert "snapshot failed" in snapshot_failed["blocked_detail"]

"""Tests for Exp 4099 TRM-pool verifier discrimination probe.

Spec refs: REQ-LEARN-4099, SCENARIO-LEARN-4099,
SCENARIO-LEARN-4099-UNDERPOWERED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import arc_exp4099_trm_pool_verifier_discrimination_probe as exp4099


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _candidate(
    candidate_id: str,
    grid: list[list[int]],
    *,
    votes: int,
    correct: bool,
    avg_q: float = 0.5,
) -> exp4099.CandidateGrid:
    return exp4099.CandidateGrid(
        candidate_id=candidate_id,
        grid=grid,
        vote_count=votes,
        avg_q=avg_q,
        correct=correct,
    )


def _task(
    task_id: str,
    *,
    good_votes: int,
    bad_votes: int,
    good_grid: list[list[int]] | None = None,
) -> exp4099.TaskCandidatePool:
    good = good_grid or [[1, 0], [0, 1]]
    bad = [[9, 9], [9, 9]]
    other_bad = [[8, 8], [8, 8]]
    return exp4099.TaskCandidatePool(
        task_id=task_id,
        train_pairs=[{"input": [[1, 0], [0, 1]], "output": good}],
        test_input=[[1, 0], [0, 1]],
        candidates=[
            _candidate(f"{task_id}:bad", bad, votes=bad_votes, avg_q=0.9, correct=False),
            _candidate(f"{task_id}:bad2", other_bad, votes=max(bad_votes - 1, 0), avg_q=0.8, correct=False),
            _candidate(f"{task_id}:good", good, votes=good_votes, avg_q=0.1, correct=True),
        ],
        source="fixture",
    )


def test_req_learn_4099_spec_declares_contract() -> None:
    """REQ-LEARN-4099: OpenSpec declares the strict gate and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4099" in spec
    assert "SCENARIO-LEARN-4099" in spec
    assert "SCENARIO-LEARN-4099-UNDERPOWERED" in spec
    for field in exp4099.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "bootstrap 95% confidence interval" in spec


def test_scenario_learn_4099_strict_reranker_win_sets_bare_gate() -> None:
    """SCENARIO-LEARN-4099: CI-separated pass@2 win sets verifier_beats_trm_vote."""

    pool = [_task(f"win-{idx}", good_votes=1, bad_votes=4) for idx in range(6)]

    artifact = exp4099.build_probe_artifact(pool, n_boot=1200, random_seed=7)

    assert artifact["verifier_beats_trm_vote"] is True
    assert isinstance(artifact["verifier_beats_trm_vote"], bool)
    assert artifact["captured_pp_directional"] > 0.0
    assert artifact["per_reranker"]["DEMO_FIT"]["captured_pp_ci95"][0] > 0.0
    assert artifact["per_reranker"]["TRM_VOTE"]["captured_pp"] == 0.0
    assert artifact["oracle_ceiling"]["pass@2"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert exp4099.artifact_schema_errors(artifact) == []


def test_scenario_learn_4099_underpowered_no_ci_win_preserves_direction() -> None:
    """SCENARIO-LEARN-4099-UNDERPOWERED: no CI win is complete and directional."""

    pool = [
        _task("same-a", good_votes=5, bad_votes=1),
        _task("same-b", good_votes=5, bad_votes=1),
        _task("same-c", good_votes=5, bad_votes=1),
    ]

    artifact = exp4099.build_probe_artifact(pool, n_boot=1200, random_seed=9)

    assert artifact["pool_n_tasks"] == 3
    assert artifact["underpowered"] is True
    assert artifact["verifier_beats_trm_vote"] is False
    assert artifact["captured_pp_directional"] == 0.0
    assert artifact["honest_verdict"].startswith("complete:")
    assert exp4099.artifact_schema_errors(artifact) == []


def test_req_learn_4099_signals_and_ranking_are_deterministic() -> None:
    """REQ-LEARN-4099: model-free signals rank without oracle access."""

    task = _task("signals", good_votes=2, bad_votes=5)
    scored = exp4099.score_task_candidates(task)
    by_id = {row.candidate_id: row for row in scored}

    assert by_id["signals:good"].signals["demo_fit_energy"] == 0.0
    assert by_id["signals:good"].signals["agreement_count"] == 2.0
    assert by_id["signals:bad"].signals["demo_fit_energy"] > 0.0
    assert exp4099.rank_candidates(scored, "DEMO_FIT")[0].candidate_id == "signals:good"
    assert exp4099.rank_candidates(scored, "TRM_VOTE")[0].candidate_id == "signals:bad"


def test_req_learn_4099_blocked_artifact_when_cached_pool_missing(tmp_path: Path) -> None:
    """REQ-LEARN-4099: missing cached pool writes honest blocked artifact."""

    output_path = tmp_path / "result.json"

    artifact = exp4099.run_experiment(repo_root=tmp_path, output_path=output_path)

    assert artifact["honest_verdict"] == "blocked_trm_pool_missing"
    assert artifact["verifier_beats_trm_vote"] is False
    assert artifact["pool_n_tasks"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert exp4099.artifact_schema_errors(artifact) == []


def test_req_learn_4099_schema_errors_are_explicit() -> None:
    """REQ-LEARN-4099: schema guard rejects wrapped gates and bad CI shapes."""

    artifact = exp4099.build_probe_artifact([_task("schema", good_votes=1, bad_votes=4)], n_boot=1200)
    bad = dict(artifact)
    bad["honest_verdict"] = "maybe"
    bad["verifier_beats_trm_vote"] = {"value": False}
    bad["underpowered"] = "true"
    bad["per_reranker"] = {"X": {"captured_pp_ci95": [0.0]}}

    errors = exp4099.artifact_schema_errors(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "verifier_beats_trm_vote must be a bare bool" in errors
    assert "underpowered must be a bare bool" in errors
    assert "per_reranker entries must include pass@1, pass@2, captured_pp, captured_pp_ci95" in errors


def test_req_learn_4099_defensive_helpers_and_schema_branches(tmp_path: Path) -> None:
    """REQ-LEARN-4099: defensive branches fail closed and stay deterministic."""

    assert exp4099._normalized_hamming([1, 2, 3], [[1]]) == 1.0
    assert exp4099._normalized_hamming([[1, 2]], [[1]]) == 1.0
    assert exp4099._safe_grid_hash([1, 2, 3]).startswith("invalid:")
    assert exp4099._demo_fit_energy([[1]], []) == 1.0
    assert exp4099._augmentation_signature_energy([[1]], []) == 1.0
    assert exp4099._augmentation_signature_energy([1, 2, 3], [{"output": [[1]]}]) == 1.0
    assert exp4099._consensus_grid([]) is None
    assert exp4099._bootstrap_ci([], n_boot=10, seed=1) == (0.0, 0.0)

    with pytest.raises(ValueError, match="unknown reranker"):
        exp4099.rank_candidates(exp4099.score_task_candidates(_task("bad-rank", good_votes=1, bad_votes=2)), "NOPE")

    results = tmp_path / "results"
    results.mkdir()
    (results / "trm_verifier_rerank_opportunity.json").write_text("{}", encoding="utf-8")
    (results / "arc3_trm_verifier_rerank_gap1.json").write_text("{", encoding="utf-8")
    checks, blocker = exp4099.check_preconditions(repo_root=tmp_path)

    assert blocker == "blocked_trm_pool_missing"
    assert checks[0].available is True
    assert checks[1].available is False
    assert "JSONDecodeError" in checks[1].detail

    bad_schema = exp4099.build_blocked_artifact("blocked_trm_pool_missing", preconditions_checked=[])
    del bad_schema["pool_n_tasks"]
    bad_schema["honest_verdict"] = 123
    bad_schema["captured_pp_directional"] = False
    bad_schema["n_tasks_scored"] = True
    bad_schema["reproducibility_checksum"] = 123
    bad_schema["oracle_ceiling"] = {}
    bad_schema["per_reranker"] = []
    bad_schema["inference_substrate"] = "live"

    errors = exp4099.artifact_schema_errors(bad_schema)

    assert "missing required field pool_n_tasks" in errors
    assert "honest_verdict must be a string" in errors
    assert "captured_pp_directional must be numeric" in errors
    assert "n_tasks_scored must be a bare int" in errors
    assert "reproducibility_checksum must be a string" in errors
    assert "oracle_ceiling must include numeric pass@2" in errors
    assert "per_reranker must be a dict" in errors
    assert "inference_substrate must declare offline saved TRM candidate rerank" in errors


def test_req_learn_4099_run_experiment_success_empty_and_exception(tmp_path: Path) -> None:
    """REQ-LEARN-4099: runner writes success and fail-closed pool-loader outcomes."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "trm_verifier_rerank_opportunity.json").write_text("{}", encoding="utf-8")
    (results / "arc3_trm_verifier_rerank_gap1.json").write_text("{}", encoding="utf-8")

    success_path = tmp_path / "success.json"
    success = exp4099.run_experiment(
        repo_root=tmp_path,
        output_path=success_path,
        pool_loader=lambda: [_task("runner", good_votes=1, bad_votes=4)],
    )
    assert success["pool_n_tasks"] == 1
    assert success["preconditions_checked"][-1]["available"] is True
    assert json.loads(success_path.read_text(encoding="utf-8")) == success

    empty = exp4099.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "empty.json",
        pool_loader=lambda: [],
    )
    assert empty["honest_verdict"] == "blocked_trm_pool_missing"
    assert empty["preconditions_checked"][-1]["detail"] == "no task candidates loaded"

    def raises() -> list[exp4099.TaskCandidatePool]:
        raise RuntimeError("pool broke")

    failed = exp4099.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "failed.json",
        pool_loader=raises,
    )
    assert failed["honest_verdict"] == "blocked_trm_pool_missing"
    assert "RuntimeError: pool broke" in failed["preconditions_checked"][-1]["detail"]


def test_req_learn_4099_builders_raise_on_internal_schema_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LEARN-4099: builder-time schema regressions are hard errors."""

    monkeypatch.setattr(exp4099, "artifact_schema_errors", lambda _artifact: ["boom"])

    with pytest.raises(ValueError, match="boom"):
        exp4099.build_probe_artifact([_task("raise", good_votes=1, bad_votes=4)])
    with pytest.raises(ValueError, match="boom"):
        exp4099.build_blocked_artifact("blocked_trm_pool_missing", preconditions_checked=[])

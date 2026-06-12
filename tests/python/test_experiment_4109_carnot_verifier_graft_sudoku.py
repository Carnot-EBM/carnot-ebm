"""Tests for Exp 4109 Sudoku verifier graft over nano-trm.

Spec refs: REQ-LEARN-4109, SCENARIO-LEARN-4109-RERANK,
SCENARIO-LEARN-4109-RFT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109


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


def _bad_solution(value: int = 6) -> list[int]:
    bad = [row[:] for row in SOLUTION]
    bad[0][0] = value
    return _tokens(bad)


def _pool(puzzle_id: str, *, vote_wrong: bool) -> exp4109.CandidatePool:
    puzzle = _tokens(PUZZLE)
    label = _tokens(SOLUTION)
    good = exp4109.CandidateSample(f"{puzzle_id}:good", label, trm_score=0.4)
    bad_a = exp4109.CandidateSample(f"{puzzle_id}:bad-a", _bad_solution(6), trm_score=0.9)
    bad_b = exp4109.CandidateSample(f"{puzzle_id}:bad-b", _bad_solution(6), trm_score=0.8)
    candidates = [bad_a, bad_b, good] if vote_wrong else [good, good, bad_a]
    return exp4109.CandidatePool(puzzle_id, puzzle, label, candidates)


def test_req_learn_4109_spec_declares_required_contract() -> None:
    """REQ-LEARN-4109: OpenSpec declares the artifact and scenario contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4109" in spec
    assert "SCENARIO-LEARN-4109-RERANK" in spec
    assert "SCENARIO-LEARN-4109-RFT" in spec
    assert exp4109.RESULT_FILENAME in spec
    for field in exp4109.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4109.FIELD_PRINCIPLES


def test_req_learn_4109_executable_sudoku_verifier_scores_exact_validity() -> None:
    """REQ-LEARN-4109: executable verifier checks range, units, and clues."""

    puzzle = _tokens(PUZZLE)
    valid = exp4109.score_sudoku_candidate(puzzle, _tokens(SOLUTION))
    clue_bad = exp4109.score_sudoku_candidate(puzzle, _bad_solution(6))
    range_bad = exp4109.score_sudoku_candidate(puzzle, [99] * 81)

    assert valid.exact_valid is True
    assert valid.satisfied_constraints == valid.total_constraints
    assert valid.normalized_score == pytest.approx(1.0)

    assert clue_bad.exact_valid is False
    assert clue_bad.satisfied_constraints < clue_bad.total_constraints
    assert "clue" in " ".join(clue_bad.failure_reasons)
    assert range_bad.exact_valid is False
    assert range_bad.normalized_score < clue_bad.normalized_score
    assert "range" in " ".join(range_bad.failure_reasons)


def test_scenario_learn_4109_rerank_reports_pass1_lift_and_oracle() -> None:
    """SCENARIO-LEARN-4109-RERANK: verifier rerank beats TRM vote on paired pools."""

    pools = [_pool("p0", vote_wrong=True), _pool("p1", vote_wrong=False)]

    metrics = exp4109.evaluate_rerank(pools, random_seed=7, bootstrap_resamples=200)

    assert metrics["vote_pass_at_1"] == pytest.approx(0.5)
    assert metrics["verifier_pass_at_1"] == pytest.approx(1.0)
    assert metrics["oracle_ceiling_pass_at_1"] == pytest.approx(1.0)
    assert metrics["delta"] == pytest.approx(0.5)
    assert len(metrics["ci95"]) == 2
    assert metrics["ci95"][0] <= metrics["delta"] <= metrics["ci95"][1]
    assert metrics["per_puzzle"][0]["vote_correct"] is False
    assert metrics["per_puzzle"][0]["verifier_correct"] is True


def test_scenario_learn_4109_matched_corpora_and_a_vs_b_delta() -> None:
    """SCENARIO-LEARN-4109-RFT: A-vs-B isolates verifier label from vote label."""

    pools = [_pool(f"p{i}", vote_wrong=True) for i in range(5)]
    corpora = exp4109.build_matched_corpora(pools)
    delta = exp4109.evaluate_label_arms(corpora, random_seed=11, bootstrap_resamples=200)

    assert corpora["n_matched"] == 5
    assert all(row["a_exact"] for row in corpora["rows"])
    assert not any(row["b_exact"] for row in corpora["rows"])
    assert delta["metric"] == "heldout_exact_accuracy"
    assert delta["delta"] == pytest.approx(1.0)
    assert delta["ci95"][0] > 0.0
    assert delta["status"] == "ci95_excludes_zero"
    assert exp4109.verifier_value_added(delta) is True


def test_req_learn_4109_preconditions_choose_4108_then_4107_fallback(tmp_path: Path) -> None:
    """REQ-LEARN-4109: checkpoint preconditions prefer 4108 and fall back honestly."""

    ckpt4108 = tmp_path / "exp4108.ckpt"
    ckpt4108.write_bytes(b"4108")
    ckpt4107 = tmp_path / "exp4107.ckpt"
    ckpt4107.write_bytes(b"4107")
    exp4108_path = tmp_path / "experiment_4108.json"
    exp4107_path = tmp_path / "experiment_4107.json"
    exp4108_path.write_text(json.dumps({"checkpoint_path": str(ckpt4108), "matches_published_087": False}))
    exp4107_path.write_text(json.dumps({"checkpoint_path": str(ckpt4107), "nanotrm_trainer_checkpoint_ok": True}))

    choice = exp4109.resolve_checkpoint_choice(exp4108_path, exp4107_path)
    assert choice.source_experiment == "exp4108"
    assert choice.checkpoint_path == ckpt4108
    assert choice.limitation == "exp4108_partial_baseline_matches_published_087_false"

    ckpt4108.unlink()
    fallback = exp4109.resolve_checkpoint_choice(exp4108_path, exp4107_path)
    assert fallback.source_experiment == "exp4107"
    assert fallback.checkpoint_path == ckpt4107
    assert fallback.limitation == "exp4108_checkpoint_missing_fell_back_to_exp4107_smoke"

    checks, checked_choice = exp4109.check_preconditions(
        exp4108_artifact_path=exp4108_path,
        exp4107_artifact_path=exp4107_path,
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert checked_choice == fallback
    assert [check.resource for check in checks] == ["checkpoint_path", "cuda_available"]
    assert all(check.available for check in checks)


def test_req_learn_4109_artifact_schema_and_checksum(tmp_path: Path) -> None:
    """REQ-LEARN-4109: artifact contains required fields and drift checksum."""

    pools = [_pool(f"p{i}", vote_wrong=True) for i in range(5)]
    rerank = exp4109.evaluate_rerank(pools, random_seed=17, bootstrap_resamples=200)
    corpora = exp4109.build_matched_corpora(pools)
    rft_delta = exp4109.evaluate_label_arms(corpora, random_seed=19, bootstrap_resamples=200)
    checksum = exp4109.compute_reproducibility_checksum(corpora, heldout_ids=["p0", "p1"])
    changed = exp4109.compute_reproducibility_checksum(corpora, heldout_ids=["p0", "p2"])

    assert checksum.startswith("sha256:")
    assert changed != checksum

    artifact = exp4109.build_result_artifact(
        rerank_metrics=rerank,
        rft_delta=rft_delta,
        corpus_summary={"n_matched": corpora["n_matched"]},
        preconditions_checked=[
            exp4109.PreconditionCheck("checkpoint_path", True, "/tmp/model.ckpt").to_dict(),
            exp4109.PreconditionCheck("cuda_available", True, "cuda fixture").to_dict(),
        ],
        checkpoint_choice=exp4109.CheckpointChoice(
            checkpoint_path=tmp_path / "model.ckpt",
            source_experiment="exp4108",
            limitation="exp4108_partial_baseline_matches_published_087_false",
        ),
        random_seed=4109,
        reproducibility_checksum=checksum,
        duration_s=3.25,
        native_training_launched=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["rerank_lift_vs_vote"]["delta"] == pytest.approx(1.0)
    assert artifact["rft_vs_ablation_delta"]["delta"] == pytest.approx(1.0)
    assert artifact["verifier_value_added"] is True
    assert artifact["native_training_launched"] is False
    assert artifact["field_principles"]["honest_verdict"].startswith("Terminal-prefixed")
    assert exp4109.artifact_schema_errors(artifact) == []

    output_path = tmp_path / exp4109.RESULT_FILENAME
    written = exp4109.write_artifact(output_path, artifact)
    assert written == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact

    assert "missing required field honest_verdict" in exp4109.artifact_schema_errors({})


def test_req_learn_4109_defensive_branches_and_null_deltas(tmp_path: Path) -> None:
    """REQ-LEARN-4109: helper branches fail closed and report honest nulls."""

    with pytest.raises(ValueError, match="expected 81 tokens"):
        exp4109.decode_tokens([2, 3])
    assert exp4109._box_dims(4) == (2, 2)
    assert exp4109._box_dims(6) == (2, 3)
    assert exp4109._box_dims(16) == (4, 4)
    with pytest.raises(ValueError, match="unsupported"):
        exp4109._box_dims(5)

    with pytest.raises(ValueError, match="empty"):
        exp4109.select_vote_candidate([])
    empty_pool = exp4109.CandidatePool("empty", _tokens(PUZZLE), _tokens(SOLUTION), [])
    with pytest.raises(ValueError, match="empty"):
        exp4109.select_verifier_candidate(empty_pool)
    assert exp4109._bootstrap_ci([], random_seed=1, resamples=10) == [0.0, 0.0]

    invalid_only = exp4109.CandidatePool(
        "invalid",
        _tokens(PUZZLE),
        _tokens(SOLUTION),
        [exp4109.CandidateSample("bad", _bad_solution(6))],
    )
    corpora = exp4109.build_matched_corpora([invalid_only])
    assert corpora["rows"] == []
    assert corpora["skipped_no_verifier_valid"] == ["invalid"]
    no_labels = exp4109.evaluate_label_arms(corpora)
    assert no_labels["status"] == "no_matched_verifier_valid_labels"

    negative = exp4109.evaluate_label_arms(
        {"rows": [{"a_exact": False, "b_exact": True} for _ in range(4)]},
        random_seed=2,
        bootstrap_resamples=50,
    )
    assert negative["status"] == "negative_ci95_excludes_zero"
    null = exp4109.evaluate_label_arms(
        {"rows": [{"a_exact": True, "b_exact": True}, {"a_exact": False, "b_exact": False}]},
        random_seed=3,
        bootstrap_resamples=50,
    )
    assert null["status"] == "honest_null_ci95_includes_zero"
    assert exp4109.verifier_value_added({"delta": 1.0, "ci95": [0.0, 1.0]}) is False

    class Scalar:
        def item(self) -> int:
            return 7

    assert exp4109._jsonable(tmp_path / "x") == str(tmp_path / "x")
    assert exp4109._jsonable(Scalar()) == 7

    assert exp4109._load_json_object(tmp_path / "missing.json") is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp4109._load_json_object(bad_json) is None
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    assert exp4109._load_json_object(not_object) is None
    assert exp4109.resolve_checkpoint_choice(tmp_path / "none4108.json", tmp_path / "none4107.json").source_experiment == "none"

    checks, _choice = exp4109.check_preconditions(
        exp4108_artifact_path=tmp_path / "none4108.json",
        exp4107_artifact_path=tmp_path / "none4107.json",
        cuda_checker=lambda: (_ for _ in ()).throw(RuntimeError("cuda check broke")),
    )
    assert checks[1].available is False
    assert "RuntimeError: cuda check broke" in checks[1].detail
    assert exp4109._summarize_corpora({"arm_a": "A", "arm_b": "B", "rows": [{"a_exact": True, "b_exact": False}]}) == {
        "arm_a": "A",
        "arm_b": "B",
        "n_matched": 0,
        "skipped_no_verifier_valid": 0,
        "a_exact_count": 1,
        "b_exact_count": 0,
    }


def test_req_learn_4109_artifact_schema_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-4109: schema validation names malformed deliverable fields."""

    errors = exp4109.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "rerank_lift_vs_vote": {"ci95": [0.0]},
            "rft_vs_ablation_delta": [],
            "verifier_value_added": "yes",
            "preconditions_checked": [{}],
            "random_seed": False,
            "reproducibility_checksum": "bad",
            "field_principles": {"honest_verdict": "wrong"},
        }
    )

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "rerank_lift_vs_vote.delta is required" in errors
    assert "rerank_lift_vs_vote.ci95 must have two bounds" in errors
    assert "rft_vs_ablation_delta must be an object" in errors
    assert "verifier_value_added must be a bare bool" in errors
    assert "random_seed must be a bare int" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "preconditions_checked entries must include resource and available" in errors
    assert "field_principles.honest_verdict mismatch" in errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4109.validate_artifact({"honest_verdict": "bad"})

    base_rerank = {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": 0,
        "vote_pass_at_1": 0.0,
        "verifier_pass_at_1": 0.0,
        "oracle_ceiling_pass_at_1": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "per_puzzle": [],
    }
    no_matched = {
        "metric": "heldout_exact_accuracy",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": "no_matched_verifier_valid_labels",
    }
    artifact = exp4109.build_result_artifact(
        rerank_metrics=base_rerank,
        rft_delta=no_matched,
        corpus_summary={},
        preconditions_checked=[exp4109.PreconditionCheck("checkpoint_path", True, "ckpt").to_dict()],
        checkpoint_choice=exp4109.CheckpointChoice(tmp_path / "ckpt", "exp4108", "fixture"),
        random_seed=4109,
        reproducibility_checksum="sha256:" + ("0" * 64),
        duration_s=0.1,
        native_training_launched=True,
        extra={"extra_field": tmp_path / "extra"},
    )
    assert artifact["honest_verdict"] == "complete: honest_null_no_verifier_valid_training_labels"
    assert artifact["native_training_limitation"] is None
    assert artifact["extra_field"] == str(tmp_path / "extra")


def test_scenario_learn_4109_run_experiment_with_provider_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4109-RFT: run_experiment writes fixture and blocked artifacts."""

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"checkpoint")
    exp4108_path = tmp_path / "experiment_4108.json"
    exp4108_path.write_text(json.dumps({"checkpoint_path": str(ckpt), "matches_published_087": True}))
    exp4107_path = tmp_path / "experiment_4107.json"
    exp4107_path.write_text(json.dumps({}))

    output = tmp_path / exp4109.RESULT_FILENAME
    artifact = exp4109.run_experiment(
        output_path=output,
        exp4108_artifact_path=exp4108_path,
        exp4107_artifact_path=exp4107_path,
        max_puzzles=3,
        k_candidates=3,
        bootstrap_resamples=100,
        cuda_checker=lambda: (True, "cuda fixture"),
        candidate_pool_provider=lambda _choice: [_pool(f"p{i}", vote_wrong=True) for i in range(3)],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["candidate_source"] == "provided_candidate_pool"
    assert artifact["n_candidate_pools"] == 3
    assert artifact["verifier_value_added"] is True

    blocked_output = tmp_path / "blocked.json"
    blocked = exp4109.run_experiment(
        output_path=blocked_output,
        exp4108_artifact_path=tmp_path / "missing4108.json",
        exp4107_artifact_path=tmp_path / "missing4107.json",
        cuda_checker=lambda: (False, "no cuda"),
        candidate_pool_provider=lambda _choice: pytest.fail("provider must not run when blocked"),
    )
    assert blocked["honest_verdict"] == "blocked_exp4109_preconditions_missing"
    assert blocked["candidate_source"] == "none_preconditions_missing"
    assert json.loads(blocked_output.read_text(encoding="utf-8"))["honest_verdict"].startswith("blocked_")

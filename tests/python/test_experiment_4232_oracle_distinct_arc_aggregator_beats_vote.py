"""Tests for Exp 4232 held-out oracle-distinct ARC aggregator rerank.

Spec refs: REQ-VERIFY-4232, SCENARIO-VERIFY-4232,
SCENARIO-VERIFY-4232-NO-HEADROOM, SCENARIO-VERIFY-4232-DEFERRED.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import oracle_distinct_arc_aggregator_4232 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _stub_exp4208_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests on the lightweight Exp 4231 loader path."""

    class FakeDetector:
        @staticmethod
        def load_arc_rows(pool_path: Path, programs_path: Path) -> list[dict]:
            with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
                entries = json.load(handle).get("entries", [])
            json.loads(programs_path.read_text(encoding="utf-8"))
            return [
                {"domain": "arc", "output": f"{entry.get('task')}:{candidate_index}"}
                for entry in entries
                if isinstance(entry, dict)
                for candidate_index, _candidate in enumerate(entry.get("candidates", []))
            ]

    monkeypatch.setattr(mod.exp4231, "_import_detector_module", lambda: FakeDetector)


def _adversarial_clean(_path: Path) -> dict:
    return {
        "returncode": 0,
        "flagged_count": 0,
        "reports": [
            {
                "flags": [],
                "flag_count": 0,
                "max_severity": 0,
            }
        ],
    }


def _write_arc_gate_fixture(
    root: Path,
    *,
    correct_indices: list[int],
    vote_weights: list[list[int]],
    learned_scores: list[list[float]],
) -> Path:
    entries = []
    programs = []
    raw_task_ids = [f"arc-task-{index}" for index in range(len(correct_indices))]
    task_ids = [f"gap3_stage2:{task_id}" for task_id in raw_task_ids]
    for task_index, (raw_task_id, correct_index) in enumerate(
        zip(raw_task_ids, correct_indices, strict=True)
    ):
        pred_grid = [[9, 9], [9, 9]]
        candidates = []
        for candidate_index, votes in enumerate(vote_weights[task_index]):
            candidates.append(
                {
                    "votes": votes,
                    "q_mean": 0.8 if candidate_index == correct_index else 0.2,
                    "correct": candidate_index != correct_index,
                    "grid": pred_grid if candidate_index == correct_index else [[0, 0], [0, 0]],
                }
            )
        entries.append(
            {"task": raw_task_id, "demos": [], "test_input": [[0]], "candidates": candidates}
        )
        programs.append(
            {
                "entry_i": task_index,
                "task": raw_task_id,
                "demo_fit": 1.0,
                "pred_grid": pred_grid,
                "code": f"def transform(grid):\n    return {task_index}\n",
            }
        )

    pool_path = root / "results" / "arc3_gap3_stage2_eval_pool.json.gz"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)
    _write_json(root / "results" / "arc3_gap4_induced_programs.json", {"programs": programs})

    oof_rows = []
    for task_id, scores in zip(task_ids, learned_scores, strict=True):
        for candidate_index, score in enumerate(scores):
            oof_rows.append(
                {
                    "candidate_id": f"{task_id}::candidate{candidate_index}",
                    "correct": candidate_index == correct_indices[task_ids.index(task_id)],
                    "fold": 0,
                    "score": score,
                    "task_id": task_id,
                    "train_task_ids": [other for other in task_ids if other != task_id],
                }
            )
    aggregator_path = root / "results" / "experiment_4231_oracle_distinct_arc_aggregator_model.json"
    _write_json(
        aggregator_path,
        {
            "model_type": "constant_score",
            "feature_names": list(mod.exp4231.FEATURE_NAMES),
            "constant_score": 0.5,
            "fold_task_ids": [task_ids],
            "held_out_task_n": len(task_ids),
            "oof_rows": oof_rows,
            "random_seed": 4231,
            "reproducibility_checksum": "fixture-a1",
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4231_oracle_distinct_arc_aggregator_build.json",
        {
            "honest_verdict": "complete: fixture",
            "aggregator_trained": True,
            "learned_verifier_path": str(aggregator_path),
            "verifier_is_oracle": False,
            "wrong_majority_n": sum(
                int(weights.index(max(weights)) != correct)
                for correct, weights in zip(correct_indices, vote_weights, strict=True)
            ),
        },
    )
    return root


def test_req_4232_spec_declares_heldout_aggregator_gate_contract() -> None:
    """REQ-VERIFY-4232: OpenSpec declares the powered held-out gate."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4232",
        "SCENARIO-VERIFY-4232",
        "SCENARIO-VERIFY-4232-NO-HEADROOM",
        "SCENARIO-VERIFY-4232-DEFERRED",
        "python/carnot/reporting/oracle_distinct_arc_aggregator_4232.py",
        "results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.py",
        "oracle_distinct_beats_vote",
        "aggregator_minus_vote_delta",
        "aggregator_minus_vote_ci95",
        "margin_override_minus_vote",
        "matched_control_delta",
        "held_out_task_n",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_heldout_aggregator_beats_vote_with_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4232: learned OOF scores beat vote with headroom."""

    _write_arc_gate_fixture(
        tmp_path,
        correct_indices=[1, 0, 2, 1],
        vote_weights=[[9, 1, 0], [9, 1, 0], [9, 0, 1], [9, 1, 0]],
        learned_scores=[[0.1, 0.9, 0.2], [0.8, 0.2, 0.1], [0.1, 0.2, 0.9], [0.1, 0.9, 0.2]],
    )

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "oracle_distinct_aggregator_beats_vote"
    assert artifact["honest_verdict"] == "complete: oracle_distinct_aggregator_beats_vote"
    assert artifact["oracle_distinct_beats_vote"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 4
    assert artifact["pass_rates"]["vote_at_1"] == pytest.approx(0.25)
    assert artifact["pass_rates"]["aggregator_at_1"] == pytest.approx(1.0)
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["pass_rates"]["matched_control_at_1"] == pytest.approx(0.25)
    assert artifact["aggregator_minus_vote_delta"] == pytest.approx(0.75)
    assert artifact["aggregator_minus_vote_ci95"][0] > 0.0
    assert artifact["matched_control_delta"] == pytest.approx(0.75)
    assert artifact["margin_override_minus_vote"] == pytest.approx(0.75)
    assert artifact["bootstrap_resamples"] >= 2000
    assert artifact["margin_trigger_threshold"] == pytest.approx(mod.MARGIN_TRIGGER_THRESHOLD)
    assert artifact["headroom_exists"] is True
    assert artifact["clt_floor_caveat"] is True
    assert artifact["acceptance_gate"] is True
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert artifact["task_rows"][0]["aggregator_train_task_excluded"] is True


def test_no_headroom_positive_control_is_uninformative_not_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4232-NO-HEADROOM: oracle@K ~= vote blocks a false null."""

    _write_arc_gate_fixture(
        tmp_path,
        correct_indices=[0, 0, 0],
        vote_weights=[[9, 1], [8, 2], [7, 3]],
        learned_scores=[[0.1, 0.9], [0.8, 0.2], [0.7, 0.1]],
    )

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "oracle_distinct_arc_no_headroom_uninformative"
    assert artifact["honest_verdict"] == "complete_oracle_distinct_arc_no_headroom_uninformative"
    assert artifact["oracle_distinct_beats_vote"] is False
    assert artifact["oracle_at_k"] == artifact["pass_rates"]["vote_at_1"] == pytest.approx(1.0)
    assert artifact["headroom_exists"] is False
    assert artifact["acceptance_gate"] is True
    assert "aggregator_fails" not in artifact["honest_verdict"]

    tied_root = tmp_path / "tied"
    _write_arc_gate_fixture(
        tied_root,
        correct_indices=[1, 0],
        vote_weights=[[9, 1], [9, 1]],
        learned_scores=[[0.9, 0.1], [0.9, 0.1]],
    )
    tied = mod.run(tied_root, adversarial_runner=_adversarial_clean)
    assert tied["headline_outcome"] == (
        "oracle_distinct_aggregator_ties_vote_with_headroom_at_power"
    )
    assert tied["headroom_exists"] is True
    assert tied["oracle_distinct_beats_vote"] is False
    assert tied["aggregator_minus_vote_ci95"] == [0.0, 0.0]


def test_missing_or_unreadable_a1_aggregator_defers_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4232-DEFERRED: missing built aggregator stops before scoring."""

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete_oracle_distinct_arc_gate_deferred_no_built_aggregator"
    )
    assert artifact["headline_outcome"] == "oracle_distinct_arc_gate_deferred_no_built_aggregator"
    assert artifact["oracle_distinct_beats_vote"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 0
    assert artifact["acceptance_gate"] is True

    _write_json(
        tmp_path / "results" / "experiment_4231_oracle_distinct_arc_aggregator_build.json",
        {
            "aggregator_trained": True,
            "verifier_is_oracle": False,
            "learned_verifier_path": str(tmp_path / "results" / "missing.json"),
        },
    )
    unreadable = mod.run(tmp_path, adversarial_runner=_adversarial_clean)
    assert unreadable["honest_verdict"] == artifact["honest_verdict"]

    bad_cases = [
        {"aggregator_trained": False, "verifier_is_oracle": False, "learned_verifier_path": ""},
        {"aggregator_trained": True, "verifier_is_oracle": True, "learned_verifier_path": ""},
        {"aggregator_trained": True, "verifier_is_oracle": False, "learned_verifier_path": ""},
    ]
    for index, a1_payload in enumerate(bad_cases):
        case_root = tmp_path / f"bad-a1-{index}"
        _write_json(
            case_root / "results" / "experiment_4231_oracle_distinct_arc_aggregator_build.json",
            a1_payload,
        )
        assert mod.run(case_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
            artifact["honest_verdict"]
        )

    malformed_root = tmp_path / "malformed-a1"
    malformed_path = (
        malformed_root
        / "results"
        / "experiment_4231_oracle_distinct_arc_aggregator_build.json"
    )
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("[]", encoding="utf-8")
    assert mod.run(malformed_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        artifact["honest_verdict"]
    )

    list_aggregator_root = tmp_path / "list-aggregator"
    list_aggregator = list_aggregator_root / "results" / "aggregator.json"
    list_aggregator.parent.mkdir(parents=True, exist_ok=True)
    list_aggregator.write_text("[]", encoding="utf-8")
    _write_json(
        list_aggregator_root
        / "results"
        / "experiment_4231_oracle_distinct_arc_aggregator_build.json",
        {
            "aggregator_trained": True,
            "verifier_is_oracle": False,
            "learned_verifier_path": str(list_aggregator),
        },
    )
    assert mod.run(list_aggregator_root, adversarial_runner=_adversarial_clean)[
        "honest_verdict"
    ] == artifact["honest_verdict"]

    oracle_aggregator_root = tmp_path / "oracle-aggregator"
    oracle_aggregator = oracle_aggregator_root / "results" / "aggregator.json"
    _write_json(oracle_aggregator, {"verifier_is_oracle": True})
    _write_json(
        oracle_aggregator_root
        / "results"
        / "experiment_4231_oracle_distinct_arc_aggregator_build.json",
        {
            "aggregator_trained": True,
            "verifier_is_oracle": False,
            "learned_verifier_path": str(oracle_aggregator),
        },
    )
    assert mod.run(oracle_aggregator_root, adversarial_runner=_adversarial_clean)[
        "honest_verdict"
    ] == artifact["honest_verdict"]


def test_validation_and_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4232: schema, bootstrap, and adversarial parsing are deterministic."""

    base = mod._deferred_artifact(
        "complete_oracle_distinct_arc_gate_deferred_no_built_aggregator",
        random_seed=mod.RANDOM_SEED,
        checksum="abc",
        duration_s=0.1,
    )
    invalid_cases = [
        ({k: v for k, v in base.items() if k != "oracle_at_k"}, "missing required"),
        ({**base, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**base, "oracle_distinct_beats_vote": {"value": False}}, "bare bool"),
        ({**base, "aggregator_minus_vote_delta": None}, "bare float"),
        ({**base, "aggregator_minus_vote_ci95": [0.0]}, "ci95"),
        ({**base, "held_out_task_n": 4232.0}, "bare int"),
        ({**base, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**base, "random_seed": 4232.0}, "bare int"),
        ({**base, "field_principles": {}}, "field_principles"),
        ({**base, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    assert mod._rate([]) == 0.0
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.2, -0.1]) is True
    assert mod._ci_excludes_zero([-0.1, 0.1]) is False
    assert mod._clean_adversarial_report({"reports": [{"flags": []}]})["status"] == "clean"
    flagged = mod._clean_adversarial_report(
        {"reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}]}]}
    )
    assert flagged["status"] == "flagged"
    assert flagged["circular_moat_overclaim_clean"] is False
    assert mod._oof_score_map(
        {"oof_rows": [None, {"candidate_id": 7}, {"candidate_id": "c", "task_id": "t"}]}
    ) == {"c": (0.0, True)}

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_json_artifact"):
        mod._read_json_object(list_json)

    fallback_root = _write_arc_gate_fixture(
        tmp_path / "fallback",
        correct_indices=[0],
        vote_weights=[[1, 0]],
        learned_scores=[[0.8, 0.1]],
    )
    aggregator_path = fallback_root / "results" / "experiment_4231_oracle_distinct_arc_aggregator_model.json"
    aggregator = {
        "model_type": "constant_score",
        "feature_names": list(mod.exp4231.FEATURE_NAMES),
        "constant_score": 0.25,
        "verifier_is_oracle": False,
    }
    with pytest.raises(mod.BlockedRun, match="no_heldout_oof_scores"):
        mod.load_heldout_pool(fallback_root, aggregator, aggregator_path)

    class EmptyCorpus:
        rows: list = []
        source_paths = []

    original_loader = mod.exp4231.load_labeled_arc_pool
    mod.exp4231.load_labeled_arc_pool = lambda _root: EmptyCorpus()
    try:
        with pytest.raises(mod.BlockedRun, match="no_heldout_oof_scores"):
            mod.load_heldout_pool(tmp_path, aggregator, aggregator_path)
    finally:
        mod.exp4231.load_labeled_arc_pool = original_loader


def test_module_does_not_rank_with_execution_or_correctness() -> None:
    """REQ-VERIFY-4232: learned inference remains oracle-distinct."""

    source = inspect.getsource(mod)
    assert "arc_gap4_execution_verifier" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "extract_dsl_rules" not in source
    assert "apply_rule" not in source
    assert "get_consistency_energy" not in source
    assert "key=lambda candidate: (candidate.correct" not in source

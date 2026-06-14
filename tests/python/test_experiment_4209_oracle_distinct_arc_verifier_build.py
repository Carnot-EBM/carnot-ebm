"""Tests for Exp 4209 oracle-distinct ARC verifier build.

Spec refs: REQ-VERIFY-4209, SCENARIO-VERIFY-4209, SCENARIO-VERIFY-4209-LABELED.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import oracle_distinct_arc_verifier_4209 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _unlabeled_repo(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "results" / "arc3_trm_verifier_rerank.json",
        {
            "honest_verdict": "complete: task summary only",
            "per_task": [
                {
                    "task": "50a16a69",
                    "n_candidates": 5,
                    "TRM_VOTE_pass@1": True,
                    "base_top1_correct": True,
                }
            ],
        },
    )
    return tmp_path


def _labeled_repo(tmp_path: Path) -> Path:
    rows = []
    for task_index in range(6):
        task_id = f"task-{task_index}"
        rows.append(
            {
                "task": task_id,
                "candidates": [
                    {
                        "candidate_id": f"{task_id}-wrong",
                        "candidate_index": 0,
                        "program": "fill zeros",
                        "output": [[0, 0], [0, 0]],
                        "vote_weight": 0.65,
                        "self_consistency_margin": -0.35,
                        "region_confidence": [0.2, 0.3],
                        "is_correct": False,
                    },
                    {
                        "candidate_id": f"{task_id}-right",
                        "candidate_index": 1,
                        "program": "paint salient cells nine",
                        "output": [[9, 9], [9, 9]],
                        "vote_weight": 0.35,
                        "self_consistency_margin": 0.35,
                        "region_confidence": [0.8, 0.9],
                        "is_correct": True,
                    },
                ],
            }
        )
    _write_json(
        tmp_path / "results" / "arc3_trm_verifier_rerank.json",
        {"honest_verdict": "complete: labeled fixture", "per_task": rows},
    )
    return tmp_path


def test_req_4209_spec_declares_arc_verifier_contract() -> None:
    """REQ-VERIFY-4209: OpenSpec declares the ARC verifier schema and blocker."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4209",
        "SCENARIO-VERIFY-4209",
        "SCENARIO-VERIFY-4209-LABELED",
        "python/carnot/reporting/oracle_distinct_arc_verifier_4209.py",
        "results/experiment_4209_oracle_distinct_arc_verifier_build.py",
        "blocked_arc_pool_no_candidate_labels",
        "selector_trained",
        "oracle_distinct_auroc",
        "verifier_is_oracle=false",
        "learned_verifier_path",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_arc_pool_without_candidate_labels_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4209: task-level ARC summaries are an honest blocker."""

    artifact = mod.run(_unlabeled_repo(tmp_path))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_arc_pool_no_candidate_labels"
    assert artifact["selector_trained"] is False
    assert artifact["oracle_distinct_auroc"] == 0.0
    assert artifact["oracle_distinct_auroc_ci95"] == [0.0, 0.0]
    assert artifact["verifier_is_oracle"] is False
    assert artifact["learned_verifier_path"] == ""
    assert artifact["acceptance_gate"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["candidate_pool_source"].endswith("results/arc3_trm_verifier_rerank.json")
    assert artifact["model_specs"]["status"] == "blocked_no_candidate_labels"


def test_labeled_arc_candidates_featurize_without_oracle_label(tmp_path: Path) -> None:
    """REQ-VERIFY-4209: features use candidate content, not correctness labels."""

    corpus = mod.load_arc_candidate_pool(_labeled_repo(tmp_path))
    assert mod.accepted_rejected_counts(corpus.rows) == {
        "accepted": 6,
        "rejected": 6,
        "total": 12,
    }

    wrong, right = corpus.rows[:2]
    assert wrong.correct is False
    assert right.correct is True
    assert "is_correct" not in right.features
    assert right.features["output_mean"] > wrong.features["output_mean"]
    assert right.features["region_confidence_mean"] > wrong.features["region_confidence_mean"]
    assert right.features["program_length"] > wrong.features["program_length"]


def test_oof_training_persists_oracle_distinct_verifier(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4209-LABELED: labeled candidates train task-held-out."""

    root = _labeled_repo(tmp_path)

    artifact = mod.run(root, random_seed=mod.RANDOM_SEED, n_folds=3)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["selector_trained"] is True
    assert artifact["oracle_distinct_auroc"] > 0.5
    assert artifact["oracle_distinct_auroc_ci95"][0] >= 0.5
    assert artifact["verifier_is_oracle"] is False
    assert artifact["accepted_rejected_n"] == {"accepted": 6, "rejected": 6, "total": 12}
    assert artifact["spec_refs"] == [
        "REQ-VERIFY-4209",
        "SCENARIO-VERIFY-4209",
        "SCENARIO-VERIFY-4209-LABELED",
    ]
    assert artifact["model_specs"]["base"] == "Exp4176 V-STaR logistic selector"

    verifier_path = Path(artifact["learned_verifier_path"])
    assert verifier_path.exists()
    verifier = mod.load_verifier(verifier_path)
    assert verifier["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert verifier["model_specs"] == artifact["model_specs"]
    assert len(verifier["oof_rows"]) == 12
    for row in verifier["oof_rows"]:
        assert row["task_id"] not in row["train_task_ids"]

    low_score = mod.score_with_verifier(
        verifier,
        {name: 0.0 for name in verifier["feature_names"]},
    )
    high_features = {name: 0.0 for name in verifier["feature_names"]}
    high_features["output_mean"] = 9.0
    high_features["region_confidence_mean"] = 0.85
    high_features["self_consistency_margin"] = 0.35
    assert mod.score_with_verifier(verifier, high_features) > low_score


def test_schema_and_precondition_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4209: malformed pools and artifacts fail with clear blockers."""

    assert mod._as_float(True) == 0.0
    assert mod._as_float("not-a-number") == 0.0
    assert mod._flatten_grid("not-grid") == []
    assert mod._flatten_grid([1, [2]]) == [1.0, 2.0]
    assert mod._grid_shape("not-grid") == (0, 0)
    assert mod._grid_stats("not-grid")["output_cells"] == 0.0
    assert mod._confidence_values({}) == []

    missing = tmp_path / "missing"
    with pytest.raises(mod.BlockedRun, match="blocked_arc_pool_missing"):
        mod.load_arc_candidate_pool(missing)

    list_json = tmp_path / "results" / "arc3_trm_verifier_rerank.json"
    list_json.parent.mkdir(parents=True, exist_ok=True)
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_arc_pool"):
        mod.load_arc_candidate_pool(tmp_path)

    _write_json(tmp_path / "results" / "arc3_trm_verifier_rerank.json", {"not_per_task": []})
    with pytest.raises(mod.BlockedRun, match="blocked_arc_pool_no_candidate_labels"):
        mod.load_arc_candidate_pool(tmp_path)

    _write_json(
        tmp_path / "results" / "arc3_trm_verifier_rerank.json",
        {"per_task": [None, {"task": "one", "candidates": [None]}]},
    )
    with pytest.raises(mod.BlockedRun, match="blocked_arc_pool_no_candidate_labels"):
        mod.load_arc_candidate_pool(tmp_path)

    _write_json(
        tmp_path / "results" / "arc3_trm_verifier_rerank.json",
        {"per_task": [{"task": "one", "candidates": [{"is_correct": True}]}]},
    )
    with pytest.raises(mod.BlockedRun, match="blocked_arc_pool_no_candidate_labels"):
        mod.load_arc_candidate_pool(tmp_path)

    with pytest.raises(mod.BlockedRun, match="blocked_arc_pool_lacks_accepted_rejected"):
        mod.train_oof_verifier(
            [mod.ArcCandidateRow("t", "c", 0, True, dict.fromkeys(mod.FEATURE_NAMES, 0.0))]
        )
    with pytest.raises(mod.BlockedRun, match="blocked_arc_pool_needs_two_tasks"):
        mod.train_oof_verifier(
            [
                mod.ArcCandidateRow("t", "c0", 0, True, dict.fromkeys(mod.FEATURE_NAMES, 1.0)),
                mod.ArcCandidateRow("t", "c1", 1, False, dict.fromkeys(mod.FEATURE_NAMES, 0.0)),
            ]
        )
    with pytest.raises(mod.BlockedRun, match="blocked_arc_fold_lacks_label_contrast"):
        mod.train_oof_verifier(
            [
                mod.ArcCandidateRow("a", "a0", 0, True, dict.fromkeys(mod.FEATURE_NAMES, 1.0)),
                mod.ArcCandidateRow("b", "b0", 0, False, dict.fromkeys(mod.FEATURE_NAMES, 0.0)),
            ],
            n_folds=2,
        )

    valid_blocked = mod._blocked_artifact(
        "blocked_fixture",
        candidate_pool_source="fixture.json",
        random_seed=mod.RANDOM_SEED,
        checksum="abc",
    )
    invalid_cases = [
        ({k: v for k, v in valid_blocked.items() if k != "model_specs"}, "missing required"),
        ({**valid_blocked, "honest_verdict": "not-terminal"}, "terminal-prefixed"),
        ({**valid_blocked, "selector_trained": {"value": False}}, "bare bool"),
        ({**valid_blocked, "oracle_distinct_auroc": {"value": 0.0}}, "bare float"),
        ({**valid_blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**valid_blocked, "field_principles": {}}, "field_principles"),
        (
            {
                **valid_blocked,
                "honest_verdict": "complete: missing_model",
                "selector_trained": True,
                "learned_verifier_path": str(tmp_path / "missing.json"),
                "accepted_rejected_n": {"accepted": 1, "rejected": 1, "total": 2},
            },
            "persisted verifier",
        ),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    bad_verifier = tmp_path / "bad_verifier.json"
    bad_verifier.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod.load_verifier(bad_verifier)


def test_bootstrap_ci_falls_back_when_resamples_have_one_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4209: AUROC CI has a deterministic point fallback."""

    class AlwaysFirstRandom:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def randrange(self, n: int) -> int:
            return 0

    monkeypatch.setattr(mod.random, "Random", AlwaysFirstRandom)

    assert mod._bootstrap_auroc_ci95([True, False], [0.9, 0.1], mod.RANDOM_SEED) == (1.0, 1.0)


def test_module_does_not_import_execution_verifier() -> None:
    """REQ-VERIFY-4209: the scorer is oracle-distinct at inference."""

    source = inspect.getsource(mod)
    assert "arc_gap4_execution_verifier" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "get_consistency_energy" not in source

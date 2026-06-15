"""Tests for Exp 4220 labeled oracle-distinct ARC verifier build.

Spec refs: REQ-VERIFY-4220, SCENARIO-VERIFY-4220, SCENARIO-VERIFY-4220-BLOCKED.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import oracle_distinct_arc_verifier_4220 as mod


@pytest.fixture(autouse=True)
def _stub_exp4208_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests off the heavy torch-backed Exp 4208 import path."""

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
                if isinstance(entry.get("candidates"), list)
            ]

    monkeypatch.setattr(mod, "_import_detector_module", lambda: FakeDetector)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_gap_fixture(root: Path, *, task_count: int = 6, wrong_majority: bool = True) -> Path:
    entries = []
    programs = []
    for task_index in range(task_count):
        task_id = f"arc-task-{task_index}"
        pred_grid = [[9, 9], [9, 9]]
        wrong_votes = 9 if wrong_majority else 1
        right_votes = 1 if wrong_majority else 9
        entries.append(
            {
                "task": task_id,
                "demos": [],
                "test_input": [[0, 0], [0, 0]],
                "candidates": [
                    {
                        "votes": wrong_votes,
                        "q_mean": 0.2,
                        "correct": True,
                        "grid": [[0, 0], [0, 0]],
                    },
                    {
                        "votes": right_votes,
                        "q_mean": 0.9,
                        "correct": False,
                        "grid": pred_grid,
                    },
                ],
            }
        )
        programs.append(
            {
                "entry_i": task_index,
                "task": task_id,
                "demo_fit": 1.0,
                "pred_grid": pred_grid,
                "code": f"def transform(grid):\n    return grid + {task_index}\n",
            }
        )
    pool_path = root / "results" / "arc3_gap3_stage2_eval_pool.json.gz"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)
    _write_json(root / "results" / "arc3_gap4_induced_programs.json", {"programs": programs})
    return root


def test_req_4220_spec_declares_labeled_gap4_contract() -> None:
    """REQ-VERIFY-4220: OpenSpec declares the labeled GAP-4 ARC build."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4220",
        "SCENARIO-VERIFY-4220",
        "SCENARIO-VERIFY-4220-BLOCKED",
        "python/carnot/reporting/oracle_distinct_arc_verifier_4220.py",
        "results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.py",
        "blocked_arc_gap4_pools_missing",
        "wrong_majority_n",
        "selector_trained",
        "oracle_distinct_auroc",
        "verifier_is_oracle=false",
        "learned_verifier_path",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_labeled_pool_uses_pred_grid_target_not_candidate_correct_flag(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4220: labels come from GAP-4 pred_grid equality."""

    root = _write_gap_fixture(tmp_path, task_count=3)

    corpus = mod.load_labeled_arc_pool(root)

    assert mod.accepted_rejected_counts(corpus.rows) == {
        "accepted": 3,
        "rejected": 3,
        "total": 6,
    }
    wrong, right = corpus.rows[:2]
    assert wrong.correct is False
    assert right.correct is True
    assert wrong.raw_candidate_correct_flag is True
    assert right.raw_candidate_correct_flag is False
    assert "is_correct" not in right.features
    assert right.features["vote_weight"] < wrong.features["vote_weight"]
    assert right.features["self_consistency_margin"] < 0.0
    assert right.features["cell_confidence_mean"] > wrong.features["cell_confidence_mean"]
    assert right.features["grid_nonzero_frac"] > wrong.features["grid_nonzero_frac"]
    assert right.features["program_length"] == wrong.features["program_length"]


def test_oof_training_persists_labeled_oracle_distinct_verifier(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4220: accepted/rejected candidates train task-held-out."""

    root = _write_gap_fixture(tmp_path, task_count=6, wrong_majority=True)

    artifact = mod.run(root, random_seed=mod.RANDOM_SEED, n_folds=3)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["selector_trained"] is True
    assert artifact["oracle_distinct_auroc"] > 0.5
    assert artifact["oracle_distinct_auroc_ci95"][0] >= 0.5
    assert artifact["wrong_majority_n"] == 6
    assert artifact["positive_candidate_n"] == 6
    assert artifact["positive_sparsity_flag"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["accepted_rejected_n"] == {"accepted": 6, "rejected": 6, "total": 12}
    assert artifact["spec_refs"] == [
        "REQ-VERIFY-4220",
        "SCENARIO-VERIFY-4220",
        "SCENARIO-VERIFY-4220-BLOCKED",
    ]
    assert artifact["model_specs"]["base"] == "Exp4176 V-STaR/AggLM logistic selector"

    verifier_path = Path(artifact["learned_verifier_path"])
    assert verifier_path.exists()
    verifier = mod.load_verifier(verifier_path)
    assert verifier["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert verifier["model_specs"] == artifact["model_specs"]
    assert len(verifier["oof_rows"]) == 12
    for row in verifier["oof_rows"]:
        assert row["task_id"] not in row["train_task_ids"]

    low_features = {name: 0.0 for name in verifier["feature_names"]}
    high_features = {name: 0.0 for name in verifier["feature_names"]}
    high_features["cell_confidence_mean"] = 0.9
    high_features["grid_nonzero_frac"] = 1.0
    high_features["grid_entropy"] = 0.1
    assert mod.score_with_verifier(verifier, high_features) > mod.score_with_verifier(
        verifier, low_features
    )


def test_missing_or_malformed_gap_pools_block_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4220-BLOCKED: precondition failures stop before training."""

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"
    assert artifact["selector_trained"] is False
    assert artifact["oracle_distinct_auroc"] == 0.0
    assert artifact["oracle_distinct_auroc_ci95"] == [0.0, 0.0]
    assert artifact["wrong_majority_n"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["learned_verifier_path"] == ""
    assert artifact["acceptance_gate"] is True

    malformed = tmp_path / "malformed"
    (malformed / "results").mkdir(parents=True)
    (malformed / "results" / "arc3_gap3_stage2_eval_pool.json.gz").write_text(
        "not gzip",
        encoding="utf-8",
    )
    _write_json(malformed / "results" / "arc3_gap4_induced_programs.json", {"programs": []})
    malformed_artifact = mod.run(malformed)
    assert malformed_artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"


def test_validation_and_score_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4220: schema and scorer failures are clear."""

    blocked = mod._blocked_artifact(
        "blocked_arc_gap4_pools_missing",
        random_seed=mod.RANDOM_SEED,
        checksum="abc",
        duration_s=0.1,
    )
    invalid_cases = [
        ({k: v for k, v in blocked.items() if k != "wrong_majority_n"}, "missing required"),
        ({**blocked, "honest_verdict": "not-terminal"}, "terminal-prefixed"),
        ({**blocked, "selector_trained": {"value": False}}, "bare bool"),
        ({**blocked, "oracle_distinct_auroc": {"value": 0.0}}, "bare float"),
        ({**blocked, "wrong_majority_n": {"value": 0}}, "bare int"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        (
            {
                **blocked,
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
    with pytest.raises(ValueError, match="unknown verifier model_type"):
        mod.score_with_verifier({"model_type": "unknown", "feature_names": []}, {})


def test_helper_edges_and_no_signal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-4220: sparse or malformed rows produce deterministic nulls."""

    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._flatten_grid("not-grid") == []
    assert mod._flatten_grid([1, [2]]) == [1.0, 2.0]
    assert mod._grid_shape("not-grid") == (0, 0)
    assert mod._grid_stats("not-grid")["grid_cells"] == 0.0
    assert mod._rank_fraction(1.0, [1.0]) == 1.0
    assert mod._program_stats({"code": {"op": 7}})["program_digit_fraction"] > 0.0
    assert mod._auroc([True], [0.9]) == 0.0
    assert mod._bootstrap_auroc_ci95([], [], mod.RANDOM_SEED) == (0.0, 0.0)

    malformed = tmp_path / "schema"
    pool_path = malformed / "results" / "arc3_gap3_stage2_eval_pool.json.gz"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {"not": "a-list"}}, handle)
    _write_json(malformed / "results" / "arc3_gap4_induced_programs.json", {"programs": []})
    with pytest.raises(mod.BlockedRun, match="blocked_arc_gap4_pools_missing"):
        mod.load_labeled_arc_pool(malformed)

    skipped = tmp_path / "skipped"
    pool_path = skipped / "results" / "arc3_gap3_stage2_eval_pool.json.gz"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "entries": [
                    None,
                    {"task": "bad-candidates", "candidates": "not-a-list"},
                    {
                        "task": "non-dict-candidate",
                        "candidates": [None, {"votes": 0, "q_mean": 0.1, "grid": [[1]]}],
                    },
                    {"task": "no-positive", "candidates": [{"votes": 1, "q_mean": 0.1, "grid": [[0]]}]},
                ]
            },
            handle,
        )
    _write_json(
        skipped / "results" / "arc3_gap4_induced_programs.json",
        {"programs": [{"entry_i": 2, "pred_grid": [[9]], "code": ""}]},
    )
    corpus = mod.load_labeled_arc_pool(skipped)
    assert corpus.rows == []
    assert corpus.raw_candidate_n == 3

    sparse = _write_gap_fixture(tmp_path / "sparse", task_count=1, wrong_majority=True)
    sparse_artifact = mod.run(sparse, n_folds=3)
    assert sparse_artifact["selector_trained"] is True
    assert sparse_artifact["honest_verdict"].startswith(
        "complete: oracle_distinct_arc_verifier_no_learnable_signal"
    )
    sparse_verifier = mod.load_verifier(sparse_artifact["learned_verifier_path"])
    assert mod.score_with_verifier(sparse_verifier, {}) == pytest.approx(0.5)

    split_rows = [
        mod.ArcCandidateRow("a", "a0", 0, 0.6, True, dict.fromkeys(mod.FEATURE_NAMES, 1.0), None),
        mod.ArcCandidateRow("a", "a1", 1, 0.4, True, dict.fromkeys(mod.FEATURE_NAMES, 1.0), None),
        mod.ArcCandidateRow("b", "b0", 0, 0.6, False, dict.fromkeys(mod.FEATURE_NAMES, 0.0), None),
        mod.ArcCandidateRow("b", "b1", 1, 0.4, False, dict.fromkeys(mod.FEATURE_NAMES, 0.0), None),
    ]
    fold_report = mod.train_oof_verifier(split_rows, n_folds=2)
    assert len(fold_report.oof_rows) == 4

    class AlwaysFirstRandom:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def randrange(self, n: int) -> int:
            return 0

    monkeypatch.setattr(mod.random, "Random", AlwaysFirstRandom)
    assert mod._bootstrap_auroc_ci95([True, False], [0.9, 0.1], mod.RANDOM_SEED) == (1.0, 1.0)


def test_module_does_not_import_execution_verifier() -> None:
    """REQ-VERIFY-4220: the learned scorer is oracle-distinct at inference."""

    source = inspect.getsource(mod)
    assert "arc_gap4_execution_verifier" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "extract_dsl_rules" not in source
    assert "apply_rule" not in source
    assert "get_consistency_energy" not in source

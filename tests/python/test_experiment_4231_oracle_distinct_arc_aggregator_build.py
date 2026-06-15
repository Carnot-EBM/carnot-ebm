"""Tests for Exp 4231 cross-candidate oracle-distinct ARC aggregator.

Spec refs: REQ-VERIFY-4231, SCENARIO-VERIFY-4231,
SCENARIO-VERIFY-4231-NO-GAIN, SCENARIO-VERIFY-4231-BLOCKED.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import oracle_distinct_arc_aggregator_4231 as mod


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


def _write_gap_pair(
    root: Path,
    *,
    pool_name: str,
    programs_name: str,
    source_label: str,
    task_count: int,
    wrong_majority: bool = True,
    no_positive: bool = False,
) -> None:
    entries = []
    programs = []
    for task_index in range(task_count):
        task_id = f"arc-task-{task_index}"
        pred_grid = [[9, task_index % 3], [9, 9]]
        target_grid = [[7, 7], [7, 7]] if no_positive else pred_grid
        wrong_votes = 11 if wrong_majority else 1
        right_votes = 1 if wrong_majority else 11
        entries.append(
            {
                "task": task_id,
                "demos": [],
                "test_input": [[0, 0], [0, 0]],
                "candidates": [
                    {
                        "votes": wrong_votes,
                        "q_mean": 0.1,
                        "correct": True,
                        "grid": [[0, 0], [0, 0]],
                    },
                    {
                        "votes": right_votes,
                        "q_mean": 0.95,
                        "correct": False,
                        "grid": target_grid,
                    },
                    {
                        "votes": 2,
                        "q_mean": 0.2,
                        "correct": False,
                        "grid": [[0, 1], [0, 0]],
                    },
                ],
            }
        )
        programs.append(
            {
                "entry_i": task_index,
                "task": task_id,
                "demo_fit": 1.0,
                "n_calls": 2,
                "pred_grid": pred_grid,
                "code": f"def transform(grid):\n    return grid + {task_index}\n",
            }
        )
    pool_path = root / "results" / pool_name
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"experiment": source_label, "entries": entries}, handle)
    _write_json(root / "results" / programs_name, {"programs": programs})


def _write_primary_and_extra(root: Path, *, primary_tasks: int = 6, extra_tasks: int = 3) -> Path:
    _write_gap_pair(
        root,
        pool_name="arc3_gap3_stage2_eval_pool.json.gz",
        programs_name="arc3_gap4_induced_programs.json",
        source_label="primary",
        task_count=primary_tasks,
        wrong_majority=True,
    )
    _write_gap_pair(
        root,
        pool_name="arc3_gap4_arc2_eval_pool.json.gz",
        programs_name="arc3_gap4_arc2_induced_programs.json",
        source_label="arc2",
        task_count=extra_tasks,
        wrong_majority=False,
    )
    return root


def test_req_4231_spec_declares_cross_candidate_contract() -> None:
    """REQ-VERIFY-4231: OpenSpec declares the strengthened aggregator build."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4231",
        "SCENARIO-VERIFY-4231",
        "SCENARIO-VERIFY-4231-NO-GAIN",
        "SCENARIO-VERIFY-4231-BLOCKED",
        "python/carnot/reporting/oracle_distinct_arc_aggregator_4231.py",
        "results/experiment_4231_oracle_distinct_arc_aggregator_build.py",
        "blocked_arc_gap4_pools_missing",
        "aggregator_trained",
        "held_out_task_n",
        "wrong_majority_n",
        "oracle_distinct_auroc",
        "verifier_is_oracle=false",
        "calibrated imbalance-aware",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_labeled_pool_grows_and_uses_pred_grid_target(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4231: labels and task groups come from GAP-4 pred_grid."""

    root = _write_primary_and_extra(tmp_path, primary_tasks=2, extra_tasks=1)

    corpus = mod.load_labeled_arc_pool(root)

    assert corpus.held_out_task_n == 3
    assert corpus.wrong_majority_n == 2
    assert mod.accepted_rejected_counts(corpus.rows) == {
        "accepted": 3,
        "rejected": 6,
        "total": 9,
    }
    assert len({row.candidate_id for row in corpus.rows}) == len(corpus.rows)
    wrong, right = corpus.rows[:2]
    assert wrong.correct is False
    assert right.correct is True
    assert wrong.raw_candidate_correct_flag is True
    assert right.raw_candidate_correct_flag is False
    assert "is_correct" not in right.features
    assert right.features["set_candidate_count"] == 3.0
    assert right.features["vote_weight_rank_fraction"] < wrong.features["vote_weight_rank_fraction"]
    assert right.features["cell_confidence_rank_fraction"] > wrong.features["cell_confidence_rank_fraction"]
    assert right.features["grid_nonzero_frac"] > wrong.features["grid_nonzero_frac"]
    assert "modal_cell_agreement_frac" in right.features
    assert "shape_family_frac" in right.features
    assert "palette_family_frac" in right.features
    assert "program_length" in right.features


def test_oof_training_persists_calibrated_cross_candidate_aggregator(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4231: cross-candidate rows train task-held-out."""

    root = _write_primary_and_extra(tmp_path, primary_tasks=32, extra_tasks=4)

    artifact = mod.run(root, random_seed=mod.RANDOM_SEED, n_folds=4, bootstrap_n=64)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["aggregator_trained"] is True
    assert artifact["oracle_distinct_auroc"] > 0.5
    assert artifact["held_out_task_n"] == 36
    assert artifact["wrong_majority_n"] == 32
    assert artifact["verifier_is_oracle"] is False
    assert artifact["accepted_rejected_n"] == {"accepted": 36, "rejected": 72, "total": 108}
    assert artifact["spec_refs"] == [
        "REQ-VERIFY-4231",
        "SCENARIO-VERIFY-4231",
        "SCENARIO-VERIFY-4231-NO-GAIN",
        "SCENARIO-VERIFY-4231-BLOCKED",
    ]
    assert artifact["model_specs"]["architecture"] == (
        "cross_candidate_augmented_calibrated_logistic_aggregator"
    )
    assert artifact["model_specs"]["calibration"] == "train_fold_isotonic_on_raw_probabilities"
    assert artifact["model_specs"]["imbalance_loss"] == "class_weight_balanced_logistic_loss"

    aggregator_path = Path(artifact["learned_verifier_path"])
    assert aggregator_path.exists()
    aggregator = mod.load_aggregator(aggregator_path)
    assert aggregator["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert aggregator["model_specs"] == artifact["model_specs"]
    assert len(aggregator["oof_rows"]) == 108
    for row in aggregator["oof_rows"]:
        assert row["task_id"] not in row["train_task_ids"]

    corpus = mod.load_labeled_arc_pool(root)
    correct_row = next(row for row in corpus.rows if row.correct)
    wrong_row = next(row for row in corpus.rows if not row.correct)
    assert mod.score_with_aggregator(
        aggregator, correct_row.features
    ) > mod.score_with_aggregator(aggregator, wrong_row.features)


def test_no_gain_verdict_still_persists_aggregator(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4231-NO-GAIN: non-improvement is complete and persisted."""

    root = _write_primary_and_extra(tmp_path, primary_tasks=5, extra_tasks=2)

    artifact = mod.run(
        root,
        random_seed=mod.RANDOM_SEED,
        n_folds=3,
        bootstrap_n=32,
        baseline_auroc=1.1,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(
        "complete_oracle_distinct_arc_aggregator_no_learnable_gain_auroc"
    )
    assert artifact["aggregator_trained"] is True
    assert Path(artifact["learned_verifier_path"]).exists()
    assert artifact["verifier_is_oracle"] is False


def test_missing_or_malformed_gap_pools_block_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4231-BLOCKED: primary precondition failures stop training."""

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"
    assert artifact["aggregator_trained"] is False
    assert artifact["oracle_distinct_auroc"] == 0.0
    assert artifact["oracle_distinct_auroc_ci95"] == [0.0, 0.0]
    assert artifact["held_out_task_n"] == 0
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

    malformed_schema = tmp_path / "malformed-schema"
    (malformed_schema / "results").mkdir(parents=True)
    with gzip.open(
        malformed_schema / "results" / "arc3_gap3_stage2_eval_pool.json.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump({"entries": {"not": "a-list"}}, handle)
    _write_json(
        malformed_schema / "results" / "arc3_gap4_induced_programs.json",
        {"programs": []},
    )
    malformed_schema_artifact = mod.run(malformed_schema)
    assert malformed_schema_artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"


def test_validation_and_score_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4231: schema and scorer failures are clear."""

    blocked = mod._blocked_artifact(
        "blocked_arc_gap4_pools_missing",
        random_seed=mod.RANDOM_SEED,
        checksum="abc",
        duration_s=0.1,
    )
    invalid_cases = [
        ({key: value for key, value in blocked.items() if key != "wrong_majority_n"}, "missing required"),
        ({**blocked, "honest_verdict": "not-terminal"}, "terminal-prefixed"),
        ({**blocked, "aggregator_trained": {"value": False}}, "bare bool"),
        ({**blocked, "oracle_distinct_auroc": {"value": 0.0}}, "bare float"),
        ({**blocked, "held_out_task_n": {"value": 0}}, "bare int"),
        ({**blocked, "wrong_majority_n": {"value": 0}}, "bare int"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        (
            {
                **blocked,
                "honest_verdict": "complete: missing_model",
                "aggregator_trained": True,
                "learned_verifier_path": str(tmp_path / "missing.json"),
                "accepted_rejected_n": {"accepted": 1, "rejected": 1, "total": 2},
            },
            "persisted aggregator",
        ),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    bad_aggregator = tmp_path / "bad_aggregator.json"
    bad_aggregator.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod.load_aggregator(bad_aggregator)
    with pytest.raises(ValueError, match="unknown aggregator model_type"):
        mod.score_with_aggregator({"model_type": "unknown", "feature_names": []}, {})


def test_helper_edges_and_no_signal_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4231: sparse or malformed rows produce deterministic nulls."""

    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._flatten_grid("not-grid") == []
    assert mod._flatten_grid([1, [2]]) == [1.0, 2.0]
    assert mod._grid_shape("not-grid") == (0, 0)
    assert mod._grid_stats("not-grid")["grid_cells"] == 0.0
    assert mod._rank_fraction(1.0, [1.0]) == 1.0
    assert mod._rank_fraction(2.0, [1.0, 3.0]) == 0.0
    assert mod._program_stats({"code": {"op": 7}})["program_digit_fraction"] > 0.0
    assert mod._auroc([True], [0.9]) == 0.0
    assert mod._bootstrap_auroc_ci95([], [], mod.RANDOM_SEED, 8) == (0.0, 0.0)
    assert mod._mean([]) == 0.0
    assert mod._std([], 0.0) == 0.0
    assert mod._modal_grids_by_shape([[1]]) == {"1x1": [[0.0]]}
    assert mod._modal_cell_agreement([[1]], []) == 0.0
    assert mod._fit_isotonic([0.2, 0.2], [True, False]) == {"x": [0.0, 1.0], "y": [0.5, 0.5]}
    assert mod._apply_isotonic(0.3, {"x": [], "y": []}) == 0.3

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
    assert corpus.raw_candidate_n == 1

    optional_bad = _write_primary_and_extra(tmp_path / "optional-bad", primary_tasks=1, extra_tasks=1)
    (optional_bad / "results" / "arc3_gap4_arc2_eval_pool.json.gz").write_text(
        "not gzip",
        encoding="utf-8",
    )
    optional_bad_corpus = mod.load_labeled_arc_pool(optional_bad)
    assert "gap4_arc2" in optional_bad_corpus.skipped_optional_pools

    optional_schema = _write_primary_and_extra(tmp_path / "optional-schema", primary_tasks=1, extra_tasks=1)
    with gzip.open(
        optional_schema / "results" / "arc3_gap4_arc2_eval_pool.json.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump({"entries": {"not": "a-list"}}, handle)
    optional_schema_corpus = mod.load_labeled_arc_pool(optional_schema)
    assert "gap4_arc2" in optional_schema_corpus.skipped_optional_pools

    sparse = tmp_path / "sparse"
    _write_gap_pair(
        sparse,
        pool_name="arc3_gap3_stage2_eval_pool.json.gz",
        programs_name="arc3_gap4_induced_programs.json",
        source_label="primary",
        task_count=1,
    )
    sparse_artifact = mod.run(sparse, n_folds=3, bootstrap_n=8)
    assert sparse_artifact["aggregator_trained"] is True
    sparse_aggregator = mod.load_aggregator(sparse_artifact["learned_verifier_path"])
    assert mod.score_with_aggregator(sparse_aggregator, {}) == pytest.approx(1.0 / 3.0)

    reason_corpus = mod.ArcAggregatorCorpus([], [], {}, 40, 10, 0, 0, [])
    assert (
        mod._no_gain_reason({"accepted": 30, "rejected": 30, "total": 60}, reason_corpus, 1.0, 0.7)
        == "too_few_wrong_majority_tasks_after_growth"
    )
    reason_corpus = mod.ArcAggregatorCorpus([], [], {}, 40, 40, 0, 0, [])
    assert (
        mod._no_gain_reason({"accepted": 30, "rejected": 30, "total": 60}, reason_corpus, 0.7, 0.7)
        == "no_gain_over_391_logistic_baseline"
    )

    split_rows = [
        mod.ArcAggregatorRow("p", "a", "a0", 0, 0.6, True, dict.fromkeys(mod.FEATURE_NAMES, 1.0), None),
        mod.ArcAggregatorRow("p", "a", "a1", 1, 0.4, True, dict.fromkeys(mod.FEATURE_NAMES, 1.0), None),
        mod.ArcAggregatorRow("p", "b", "b0", 0, 0.6, False, dict.fromkeys(mod.FEATURE_NAMES, 0.0), None),
        mod.ArcAggregatorRow("p", "b", "b1", 1, 0.4, False, dict.fromkeys(mod.FEATURE_NAMES, 0.0), None),
    ]
    fold_report = mod.train_oof_aggregator(split_rows, n_folds=2, bootstrap_n=8)
    assert len(fold_report.oof_rows) == 4

    class AlwaysFirstRandom:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def randrange(self, n: int) -> int:
            return 0

    monkeypatch.setattr(mod.random, "Random", AlwaysFirstRandom)
    assert mod._bootstrap_auroc_ci95([True, False], [0.9, 0.1], mod.RANDOM_SEED, 8) == (
        1.0,
        1.0,
    )


def test_module_does_not_import_execution_verifier() -> None:
    """REQ-VERIFY-4231: the learned scorer is oracle-distinct at inference."""

    source = inspect.getsource(mod)
    assert "arc_gap4_execution_verifier" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "extract_dsl_rules" not in source
    assert "apply_rule" not in source
    assert "get_consistency_energy" not in source

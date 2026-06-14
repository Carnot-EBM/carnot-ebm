import json
import gzip
import math

import pytest
import torch

from scripts import exp_verifier_detector_auroc as det


def test_metric_report_has_ci_controls_and_abstention_curve():
    """REQ-VERIFY-4208 / SCENARIO-VERIFY-4208: detector AUROC carries controls."""
    labels = [1] * 50 + [0] * 50
    scores = [0.9] * 50 + [0.1] * 50

    report = det.score_rows_to_report(scores, labels, seed=4208, bootstrap_n=80)

    assert report["n"] == 100
    assert report["auroc"] == pytest.approx(1.0)
    assert report["ci95"] == [1.0, 1.0]
    assert report["base_rate"] == pytest.approx(0.5)
    assert report["brier"] == pytest.approx(0.01)
    assert report["precision_at_recall_0.9"]["precision"] == pytest.approx(1.0)
    assert len(report["abstention_curve"]) == 5
    assert 0.35 <= report["random_auroc"] <= 0.65
    assert len(report["random_auroc_ci95"]) == 2


def test_build_artifact_keeps_required_domain_maps_bare(tmp_path):
    """REQ-VERIFY-4208: required headline fields stay bare machine-readable maps."""
    domain_reports = {
        "sudoku": {
            "n": 10,
            "auroc": 0.8,
            "ci95": [0.7, 0.9],
            "base_rate": 0.6,
            "random_auroc": 0.51,
            "random_auroc_ci95": [0.4, 0.6],
            "brier": 0.2,
            "precision_at_recall_0.9": {"precision": 0.75},
            "abstention_curve": [],
            "valid_but_wrong_auroc": 0.5,
            "valid_but_wrong_auroc_ci95": [0.5, 0.5],
            "valid_but_wrong_n": 4,
        },
        "code": {
            "n": 2,
            "auroc": 1.0,
            "ci95": [1.0, 1.0],
            "base_rate": 0.5,
            "random_auroc": 0.5,
            "random_auroc_ci95": [0.5, 0.5],
            "brier": 0.0,
            "precision_at_recall_0.9": {"precision": 1.0},
            "abstention_curve": [],
        },
        "math": det.unavailable_report("no_candidate_rows"),
        "arc": {
            "n": 4,
            "auroc": 0.75,
            "ci95": [0.5, 1.0],
            "base_rate": 0.25,
            "random_auroc": 0.49,
            "random_auroc_ci95": [0.25, 0.75],
            "brier": 0.3,
            "precision_at_recall_0.9": {"precision": 0.5},
            "abstention_curve": [],
        },
    }
    selector_headroom = {"sudoku": 0.0007, "code": 0.18, "math": 0.0, "arc": 0.129}
    verifier_is_oracle = {"sudoku": True, "code": True, "math": True, "arc": False}

    artifact = det.build_artifact(
        domain_reports=domain_reports,
        selector_headroom=selector_headroom,
        verifier_is_oracle=verifier_is_oracle,
        decode_sanity={"sudoku": {"checked": True}},
        source_paths=[tmp_path],
        duration_s=1.25,
        seed=4208,
    )

    assert artifact["spec_refs"] == ["REQ-VERIFY-4208", "SCENARIO-VERIFY-4208"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["detection_auroc_by_domain"] == {
        "sudoku": 0.8,
        "code": 1.0,
        "math": None,
        "arc": 0.75,
    }
    assert artifact["selector_headroom_by_domain"] == selector_headroom
    assert artifact["verifier_is_oracle_by_domain"] == verifier_is_oracle
    assert artifact["controls"]["sudoku"]["valid_but_wrong_auroc"] == 0.5
    assert "principle" not in artifact["detection_auroc_by_domain"]["sudoku"].__class__.__name__
    assert "detection_auroc_by_domain" in artifact["field_principles"]


def test_sudoku_valid_but_wrong_split_restricts_to_valid_outputs():
    """SCENARIO-VERIFY-4208: Sudoku hard split ignores invalid-grid negatives."""
    scores = [1.0, 1.0, 0.25]
    labels = [1, 0, 0]
    valid_flags = [True, True, False]

    report = det.valid_but_wrong_report(scores, labels, valid_flags, seed=1, bootstrap_n=20)

    assert report["valid_but_wrong_n"] == 2
    assert report["valid_but_wrong_auroc"] == pytest.approx(0.5)
    assert report["valid_but_wrong_auroc_ci95"] == [0.5, 0.5]


def test_arc_gap4_demo_fit_execution_rows_score_candidate_outputs():
    """REQ-VERIFY-4208: ARC rows use cached GAP-4 demo-fit execution consistency."""
    entries = [
        {
            "task": "toy",
            "candidates": [
                {"correct": True, "grid": [[1, 2], [3, 4]], "votes": 3, "q_mean": 0.9},
                {"correct": False, "grid": [[1, 2], [3, 5]], "votes": 1, "q_mean": 0.8},
                {"correct": False, "grid": [[9]], "votes": 1, "q_mean": 0.1},
            ],
        }
    ]
    programs = [
        {"task": "toy", "entry_i": 0, "demo_fit": 1.0, "pred_grid": [[1, 2], [3, 4]]}
    ]

    rows = det.arc_rows_from_entries(entries, programs)

    assert [row["is_correct"] for row in rows] == [1, 0, 0]
    assert rows[0]["score"] == pytest.approx(1.0)
    assert 0.0 < rows[1]["score"] < rows[0]["score"]
    assert rows[2]["score"] == pytest.approx(0.0)


def test_code_and_math_loaders_use_cached_executable_labels(tmp_path):
    """REQ-VERIFY-4208: code/math cached rows expose verifier scores and labels."""
    code_path = tmp_path / "code.json"
    code_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "HumanEval/0",
                        "baseline_passed": False,
                        "repair_passed": True,
                    },
                    {
                        "task_id": "HumanEval/1",
                        "baseline_passed": True,
                    }
                ]
            }
        )
    )
    math_path = tmp_path / "math.jsonl"
    math_path.write_text(
        json.dumps(
            {
                "problem_id": "m0",
                "greedy": {"correct": False},
                "samples": [{"correct": True}, {"correct": False}],
            }
        )
        + "\n"
        + "\n"
        + json.dumps({"problem_id": "m1", "samples": [{"text": "no label"}]})
        + "\n"
    )

    code_rows = det.load_code_rows(code_path)
    math_rows = det.load_math_rows(math_path)

    assert [(r["score"], r["is_correct"]) for r in code_rows] == [
        (0.0, 0),
        (1.0, 1),
        (1.0, 1),
    ]
    assert [(r["score"], r["is_correct"]) for r in math_rows] == [(0.0, 0), (1.0, 1), (0.0, 0)]


def test_defensive_metric_paths_and_sudoku_constraint_helper():
    """REQ-VERIFY-4208: helper edge cases are deterministic and non-fabricating."""
    valid = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )
    invalid = valid.clone()
    invalid[0, 0] = 0

    assert det.constraint_sat_fraction(valid) == pytest.approx(1.0)
    assert det.constraint_sat_fraction(invalid) < 1.0
    assert math.isnan(det.auroc([0.1, 0.2], [1, 1]))
    assert det._round_or_none(None) is None
    assert det._round_or_none(float("nan")) is None
    assert det.bootstrap_ci95([0.1], [1], bootstrap_n=4) == [None, None]
    assert det.bootstrap_ci95([0.1, 0.9], [0, 1], bootstrap_n=0) == [None, None]
    assert det.brier_score([], []) is None
    assert det.precision_at_recall([], [], target_recall=0.9)["precision"] is None
    assert det.precision_at_recall([0.9], [1], target_recall=1.1)["precision"] is None
    assert det.abstention_curve([], []) == []
    assert det.score_rows_to_report([], [], bootstrap_n=2)["auroc"] is None
    assert det.valid_but_wrong_report([1.0], [1], [True], bootstrap_n=2)[
        "valid_but_wrong_auroc"
    ] is None
    assert det._scores_and_labels([{"score": 0.7, "is_correct": 1}]) == ([0.7], [1])


def test_cached_loaders_checksums_and_artifact_verdict_branches(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-4208: cached-source helpers report nulls and verdicts honestly."""
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    with gzip.open(pool_path, "wt") as f:
        json.dump(
            {
                "entries": [
                    {
                        "task": "toy",
                        "candidates": [{"correct": False, "grid": [[0]], "votes": 1, "q_mean": 0.0}],
                    }
                ]
            },
            f,
        )
    programs_path.write_text(json.dumps({"programs": [{"entry_i": 0, "demo_fit": 1.0}]}))

    assert det.load_arc_rows(pool_path, programs_path)[0]["score"] == 0.0

    census_path = tmp_path / "census.json"
    arc_path = tmp_path / "arc.json"
    sudoku_path = tmp_path / "sudoku_headroom.json"
    census_path.write_text(
        json.dumps(
            {
                "per_domain_headroom": {
                    "code": {"selectable_headroom": 0.18},
                    "math": {"selectable_headroom": 0.0},
                }
            }
        )
    )
    arc_path.write_text(json.dumps({"present_but_misvoted_headroom": 0.129}))
    sudoku_path.write_text(json.dumps({"best_headroom": 0.0007}))
    monkeypatch.setattr(det, "REQUIRED_CACHED_POOLS", [arc_path, census_path])
    monkeypatch.setattr(det, "SUDOKU_HEADROOM", sudoku_path)

    assert det.load_selector_headroom() == {
        "sudoku": 0.0007,
        "code": 0.18,
        "math": 0.0,
        "arc": 0.129,
    }

    checksum = det.hash_source_paths([tmp_path / "missing", tmp_path, census_path])
    assert checksum.startswith("sha256:")
    blocked = det.blocked_artifact("blocked_detector_cached_pools_missing", 0.1)
    assert blocked["honest_verdict"] == "blocked_detector_cached_pools_missing"

    no_divergence = det.build_artifact(
        domain_reports={"code": {**det.unavailable_report("x"), "auroc": 0.5}},
        selector_headroom={"sudoku": 0.2, "code": 0.18, "math": 0.0, "arc": 0.129},
        verifier_is_oracle={"sudoku": True, "code": True, "math": True, "arc": False},
        decode_sanity={},
        source_paths=[census_path],
        duration_s=0.0,
    )
    assert no_divergence["honest_verdict"].startswith("complete: detector_axis_measured")

    no_rows = det.build_artifact(
        domain_reports={},
        selector_headroom={"sudoku": None, "code": None, "math": None, "arc": None},
        verifier_is_oracle={"sudoku": True, "code": True, "math": True, "arc": False},
        decode_sanity={},
        source_paths=[census_path],
        duration_s=0.0,
    )
    assert no_rows["honest_verdict"] == "complete: detector_axis_no_scored_rows_available"

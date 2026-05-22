"""Tests for Exp 2865 clean .271 cross-corpus matrix generation.

Spec refs: REQ-REPORT-2865, SCENARIO-REPORT-2865.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import cross_corpus_matrix_v5_2865 as exp2865


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fover_payload(
    *,
    verdict: str = "complete: FoVer dual-condition integrity rerun measured dataset-only production-vs-architecture contribution",
    production: float = 0.9131336,
    architecture: float = 0.8946624,
    n_examples: int = 1000,
    n_seeds: int = 5,
) -> dict[str, object]:
    return {
        "honest_verdict": verdict,
        "condition_a_production_auroc_mean": production,
        "condition_b_architecture_only_auroc_mean": architecture,
        "learning_contribution": production - architecture,
        "n_examples": n_examples,
        "n_seeds": n_seeds,
        "adversarial_verify_passed": True,
        "adversarial_verify_flags": [],
    }


def _halueval_fever_payload(
    *,
    verdict: str = "complete: HaluEval/FEVER local calibration ready",
    halueval_auroc: float | None = 0.553072,
    fever_auroc: float | None = 0.33114331723027374,
    halueval_n: int = 500,
    fever_n: int = 500,
) -> dict[str, object]:
    return {
        "honest_verdict": verdict,
        "halueval_fever_ready": True,
        "full_benchmark_ready": True,
        "live_model_invoked": False,
        "halueval_auroc": halueval_auroc,
        "fever_auroc": fever_auroc,
        "halueval_n_examples": halueval_n,
        "fever_n_examples": fever_n,
        "auroc_ci95_by_dataset": {
            "halueval": [0.50, 0.60],
            "fever": [0.28, 0.38],
        },
        "label_counts_by_dataset": {
            "halueval": {"0": 250, "1": 250},
            "fever": {"0": 270, "1": 230},
        },
        "adversarial_verify_passed": True,
        "adversarial_verify_flags": [],
        "run_date": "20260522",
    }


def test_scenario_report_2865_halueval_fever_builds_first_clean_non_fover_row(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2865: FoVer plus clean HaluEval/FEVER builds the matrix."""

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _fover_payload(),
    )
    _write_json(
        tmp_path,
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        _halueval_fever_payload(),
    )

    artifact = exp2865.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    required = {
        "honest_verdict",
        "cross_corpus_matrix_built",
        "verifier_corpus_dual_matrix",
        "row_status_by_corpus",
        "paper_eligible_rows",
        "clean_corpus_count",
        "blocked_corpus_count",
        "flagged_corpus_count",
        "missing_corpus_count",
        "source_artifacts",
        "excluded_from_headline",
        "claim_boundary_notes",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete: cross-corpus matrix built")
    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["row_status_by_corpus"] == {
        "FoVer": "clean",
        "HaluEval/FEVER": "clean",
        "MBPP": "missing",
        "HumanEval": "missing",
        "TruthfulQA": "missing",
    }
    assert artifact["clean_corpus_count"] == 2
    assert artifact["blocked_corpus_count"] == 0
    assert artifact["flagged_corpus_count"] == 0
    assert artifact["missing_corpus_count"] == 3
    assert artifact["paper_eligible_rows"] == ["FoVer", "HaluEval/FEVER"]
    assert artifact["source_artifacts"] == [
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
    ]

    matrix = artifact["verifier_corpus_dual_matrix"]
    assert set(matrix) == {"FoVer", "HaluEval/FEVER"}
    assert matrix["FoVer"]["production_auroc"] == pytest.approx(0.9131336)
    assert matrix["FoVer"]["architecture_only_auroc"] == pytest.approx(0.8946624)
    assert matrix["FoVer"]["learning_contribution"] == pytest.approx(0.0184712)
    assert matrix["HaluEval/FEVER"]["measured_auroc_by_dataset"] == {
        "halueval": pytest.approx(0.553072),
        "fever": pytest.approx(0.33114331723027374),
    }
    assert matrix["HaluEval/FEVER"]["n_examples_by_dataset"] == {
        "halueval": 500,
        "fever": 500,
    }
    assert matrix["HaluEval/FEVER"]["n_examples"] == 1000
    assert artifact["excluded_from_headline"] == {
        "MBPP": "source_artifact_missing",
        "HumanEval": "source_artifact_missing",
        "TruthfulQA": "source_artifact_missing",
    }
    assert "Cross-corpus matrix built from clean rows: FoVer, HaluEval/FEVER." in artifact[
        "claim_boundary_notes"
    ]


def test_req_report_2865_non_clean_rows_are_excluded_not_imputed(tmp_path: Path) -> None:
    """REQ-REPORT-2865: blocked, flagged, and missing rows stay out of the matrix."""

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _fover_payload(),
    )
    _write_json(
        tmp_path,
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        {
            **_halueval_fever_payload(),
            "honest_verdict": "blocked_adversarial_verify",
            "adversarial_verify_passed": False,
            "adversarial_verify_flags": [{"kind": "UNIT_TEST", "severity": "critical"}],
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2851_mbpp_dual_condition_v4.json",
        {
            **_fover_payload(verdict="complete: MBPP measured"),
            "adversarial_verify_flags": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )

    artifact = exp2865.build_artifact(tmp_path)

    assert artifact["cross_corpus_matrix_built"] is False
    assert artifact["verifier_corpus_dual_matrix"].keys() == {"FoVer"}
    assert artifact["row_status_by_corpus"] == {
        "FoVer": "clean",
        "HaluEval/FEVER": "blocked",
        "MBPP": "flagged",
        "HumanEval": "missing",
        "TruthfulQA": "missing",
    }
    assert artifact["blocked_corpus_count"] == 1
    assert artifact["flagged_corpus_count"] == 1
    assert artifact["missing_corpus_count"] == 2
    assert artifact["excluded_from_headline"]["HaluEval/FEVER"] == "blocked_adversarial_verify"
    assert artifact["excluded_from_headline"]["MBPP"] == "adversarial_flag_or_required_metric_missing"
    assert "HaluEval/FEVER is blocked; no metric values were inferred." in artifact[
        "claim_boundary_notes"
    ]
    assert "MBPP is flagged; no metric values were inferred." in artifact["claim_boundary_notes"]


def test_req_report_2865_helper_branches_and_write_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-2865: malformed inputs, status helpers, and persistence are explicit."""

    assert exp2865.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert exp2865.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2865.read_json(array) == {}

    assert exp2865.classify_row_status("FoVer", {}) == "missing"
    assert exp2865.classify_row_status("FoVer", {"honest_verdict": "blocked_cache"}) == "blocked"
    assert exp2865.classify_row_status("FoVer", {"flagged_adversarial": True}) == "flagged"
    assert exp2865.classify_row_status("FoVer", {"corrigendum_pending": ["x"]}) == "flagged"
    assert exp2865.classify_row_status("FoVer", {"adversarial_verify_passed": False}) == "flagged"
    assert exp2865.classify_row_status(
        "FoVer",
        {"adversarial_verify_summary": {"flag_count": 1}},
    ) == "flagged"
    assert (
        exp2865.classify_row_status(
            "FoVer",
            {**_fover_payload(), "honest_verdict": "running"},
        )
        == "flagged"
    )
    assert exp2865.classify_row_status(
        "FoVer",
        {**_fover_payload(), "condition_a_production_auroc_mean": float("nan")},
    ) == "flagged"
    assert exp2865.classify_row_status(
        "FoVer",
        {**_fover_payload(), "condition_a_production_auroc_mean": True},
    ) == "flagged"
    assert exp2865.classify_row_status(
        "FoVer",
        {**_fover_payload(), "n_examples": 0},
    ) == "flagged"
    assert exp2865.classify_row_status(
        "HaluEval/FEVER",
        {**_halueval_fever_payload(), "full_benchmark_ready": False},
    ) == "flagged"
    assert exp2865.classify_row_status(
        "HaluEval/FEVER",
        {**_halueval_fever_payload(), "halueval_auroc": None},
    ) == "flagged"

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _fover_payload(),
    )
    _write_json(
        tmp_path,
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        _halueval_fever_payload(),
    )

    out = exp2865.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2865_cross_corpus_matrix_v5.json"
    assert payload["duration_s"] == pytest.approx(0.25)
    assert payload["cross_corpus_matrix_built"] is True

"""Tests for Exp 2855 clean .270 cross-corpus matrix generation.

Spec refs: REQ-REPORT-2855, SCENARIO-REPORT-2855.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import cross_corpus_matrix_v4_2855 as exp2855


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _clean_payload(
    *,
    verdict: str = "complete: measured clean row",
    production: float = 0.81,
    architecture: float = 0.74,
    n_examples: int = 100,
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


def test_req_report_2855_required_schema_and_current_row_statuses(tmp_path: Path) -> None:
    """REQ-REPORT-2855: artifact emits required fields and row-status counts."""

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _clean_payload(production=0.9131336, architecture=0.8946624, n_examples=1000),
    )
    _write_json(
        tmp_path,
        "results/experiment_2854_halueval_fever_full_calibration_v2.json",
        {
            "honest_verdict": "blocked_missing_eval_manifests",
            "adversarial_verify_passed": False,
            "adversarial_verify_flags": [
                {"kind": "precondition", "severity": "blocking", "detail": "missing manifests"}
            ],
            "halueval_n_examples": 0,
            "fever_n_examples": 0,
        },
    )

    artifact = exp2855.build_artifact(tmp_path, started_s=10.0, now_s=10.75)

    required = {
        "honest_verdict",
        "cross_corpus_matrix_built",
        "verifier_corpus_dual_matrix",
        "row_status_by_corpus",
        "clean_corpus_count",
        "blocked_corpus_count",
        "flagged_corpus_count",
        "missing_corpus_count",
        "paper_eligible_rows",
        "claim_boundary_notes",
        "source_artifacts",
        "duration_s",
        "run_date",
    }
    assert required <= artifact.keys()
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(0.75)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["cross_corpus_matrix_built"] is False
    assert artifact["row_status_by_corpus"] == {
        "FoVer": "clean",
        "MBPP": "missing",
        "HumanEval": "missing",
        "TruthfulQA": "missing",
        "HaluEval/FEVER": "blocked",
    }
    assert artifact["clean_corpus_count"] == 1
    assert artifact["blocked_corpus_count"] == 1
    assert artifact["flagged_corpus_count"] == 0
    assert artifact["missing_corpus_count"] == 3
    assert artifact["paper_eligible_rows"] == ["FoVer"]
    assert "Paper-eligible rows: FoVer." in artifact["claim_boundary_notes"]
    assert "requires clean FoVer plus at least one clean non-FoVer row" in " ".join(
        artifact["claim_boundary_notes"]
    )
    assert artifact["source_artifacts"] == [
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        "results/experiment_2854_halueval_fever_full_calibration_v2.json",
    ]


def test_scenario_report_2855_no_imputation_for_blocked_flagged_or_missing_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2855: non-clean rows stay visible with null metrics."""

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _clean_payload(production=0.91, architecture=0.89, n_examples=1000),
    )
    _write_json(
        tmp_path,
        "results/experiment_2851_mbpp_dual_condition_v4.json",
        {
            **_clean_payload(production=0.83, architecture=0.77),
            "adversarial_verify_flags": [{"kind": "TAUTOLOGY", "severity": "critical"}],
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2852_humaneval_dual_condition_v4.json",
        {
            "honest_verdict": "success: terminal but incomplete",
            "condition_a_production_auroc_mean": 0.72,
            "adversarial_verify_passed": True,
            "adversarial_verify_flags": [],
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2854_halueval_fever_full_calibration_v2.json",
        {"honest_verdict": "blocked_missing_eval_manifests"},
    )

    artifact = exp2855.build_artifact(tmp_path)
    matrix = artifact["verifier_corpus_dual_matrix"]

    assert matrix["FoVer"]["row_status"] == "clean"
    assert matrix["FoVer"]["production_auroc"] == pytest.approx(0.91)
    assert matrix["FoVer"]["architecture_only_auroc"] == pytest.approx(0.89)
    assert matrix["FoVer"]["learning_contribution"] == pytest.approx(0.02)
    assert matrix["FoVer"]["n_examples"] == 1000
    assert matrix["FoVer"]["n_seeds"] == 5
    for corpus in ("MBPP", "HumanEval", "TruthfulQA", "HaluEval/FEVER"):
        assert matrix[corpus]["row_status"] in {"blocked", "flagged", "missing"}
        assert matrix[corpus]["production_auroc"] is None
        assert matrix[corpus]["architecture_only_auroc"] is None
        assert matrix[corpus]["learning_contribution"] is None
        assert matrix[corpus]["n_examples"] is None
        assert matrix[corpus]["n_seeds"] is None

    assert artifact["row_status_by_corpus"]["MBPP"] == "flagged"
    assert artifact["row_status_by_corpus"]["HumanEval"] == "flagged"
    assert artifact["row_status_by_corpus"]["TruthfulQA"] == "missing"
    assert artifact["row_status_by_corpus"]["HaluEval/FEVER"] == "blocked"


def test_req_report_2855_cross_corpus_gate_requires_clean_non_fover(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2855: matrix is built only from FoVer plus clean non-FoVer rows."""

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _clean_payload(production=0.91, architecture=0.89, n_examples=1000),
    )
    _write_json(
        tmp_path,
        "results/experiment_2851_mbpp_dual_condition_v4.json",
        _clean_payload(verdict="success: MBPP measured", production=0.83, architecture=0.77),
    )

    artifact = exp2855.build_artifact(tmp_path)

    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["honest_verdict"].startswith("complete: cross-corpus matrix built")
    assert artifact["paper_eligible_rows"] == ["FoVer", "MBPP"]
    assert artifact["clean_corpus_count"] == 2
    assert artifact["verifier_corpus_dual_matrix"]["MBPP"]["learning_contribution"] == (
        pytest.approx(0.06)
    )
    assert "Cross-corpus matrix built from clean rows: FoVer, MBPP." in artifact[
        "claim_boundary_notes"
    ]


def test_req_report_2855_helper_branches_and_write_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-2855: helper branches classify malformed inputs and persist JSON."""

    assert exp2855.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert exp2855.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2855.read_json(array) == {}

    assert exp2855.classify_row_status({}) == "missing"
    assert exp2855.classify_row_status({"honest_verdict": "blocked_cache"}) == "blocked"
    assert exp2855.classify_row_status({"flagged_adversarial": True}) == "flagged"
    assert exp2855.classify_row_status({"corrigendum_pending": [{"kind": "x"}]}) == "flagged"
    assert exp2855.classify_row_status({"adversarial_verify_passed": False}) == "flagged"
    assert exp2855.classify_row_status(
        {"adversarial_verify_summary": {"flag_count": 1}}
    ) == "flagged"
    assert exp2855.classify_row_status({**_clean_payload(), "honest_verdict": "running"}) == (
        "flagged"
    )
    assert exp2855.classify_row_status(
        {
            **_clean_payload(),
            "condition_a_production_auroc_mean": float("nan"),
        }
    ) == "flagged"
    assert exp2855.classify_row_status(
        {
            **_clean_payload(),
            "condition_a_production_auroc_mean": True,
        }
    ) == "flagged"
    assert exp2855.classify_row_status({**_clean_payload(), "n_examples": 0}) == "flagged"

    _write_json(
        tmp_path,
        "results/experiment_2850_fover_dual_condition_integrity_v4.json",
        _clean_payload(),
    )
    out = exp2855.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2855_cross_corpus_matrix_v4.json"
    assert payload["duration_s"] == pytest.approx(0.25)
    assert payload["row_status_by_corpus"]["FoVer"] == "clean"

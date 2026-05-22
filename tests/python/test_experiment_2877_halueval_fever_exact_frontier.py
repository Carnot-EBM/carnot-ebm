"""Tests for Exp 2877 HaluEval/FEVER exact-frontier expansion.

Spec: REQ-VERIFY-2877, SCENARIO-VERIFY-2877.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import halueval_fever_exact_frontier_expansion as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _supported_halueval_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "2006",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": (
                'Context: House of Anubis is based on "Het Huis Anubis". '
                "It first aired in September 2006 and the last episode was broadcast "
                "on December 4, 2009.\nQuestion: The Dutch-Belgian television series "
                'that "House of Anubis" was based on first aired in what year?'
            ),
            "reference": "2006",
            "stable_id": "halueval-8-right",
        },
        {
            "candidate": 'The inspiration for "House of Anubis" first aired in 2003.',
            "dataset": "HaluEval",
            "label": 1,
            "prompt": (
                'Context: House of Anubis is based on "Het Huis Anubis". '
                "It first aired in September 2006 and the last episode was broadcast "
                "on December 4, 2009.\nQuestion: The Dutch-Belgian television series "
                'that "House of Anubis" was based on first aired in what year?'
            ),
            "reference": "2006",
            "stable_id": "halueval-8-hallucinated",
        },
        {
            "candidate": "The Wolfhounds",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": (
                "Context: Hole, which Courtney Love formed in 1989. The Wolfhounds "
                "are an indie pop/noise pop band formed in Romford, UK in 1985.\n"
                "Question: Which band was founded first, Hole or The Wolfhounds?"
            ),
            "reference": "The Wolfhounds",
            "stable_id": "halueval-22-right",
        },
        {
            "candidate": (
                "Hole, the rock band that Courtney Love was a frontwoman of was founded first."
            ),
            "dataset": "HaluEval",
            "label": 1,
            "prompt": (
                "Context: Hole, which Courtney Love formed in 1989. The Wolfhounds "
                "are an indie pop/noise pop band formed in Romford, UK in 1985.\n"
                "Question: Which band was founded first, Hole or The Wolfhounds?"
            ),
            "reference": "The Wolfhounds",
            "stable_id": "halueval-22-hallucinated",
        },
        {
            "candidate": "Aleksander Ford",
            "dataset": "HaluEval",
            "label": 0,
            "prompt": (
                "Context: Pablo Trapero (Born 4 October 1971) is an Argentine film "
                "director. Aleksander Ford (born Mosze Lifszyc; 24 November 1908) "
                "was a Polish film director.\nQuestion: Who was born first, Pablo "
                "Trapero or Aleksander Ford?"
            ),
            "reference": "Aleksander Ford",
            "stable_id": "halueval-31-right",
        },
        {
            "candidate": "Pablo Trapero was born first.",
            "dataset": "HaluEval",
            "label": 1,
            "prompt": (
                "Context: Pablo Trapero (Born 4 October 1971) is an Argentine film "
                "director. Aleksander Ford (born Mosze Lifszyc; 24 November 1908) "
                "was a Polish film director.\nQuestion: Who was born first, Pablo "
                "Trapero or Aleksander Ford?"
            ),
            "reference": "Aleksander Ford",
            "stable_id": "halueval-31-hallucinated",
        },
    ]


def _supported_fever_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim": "Steam is the gaseous state of water, also known as water vapor.",
            "dataset": "FEVER",
            "label": 0,
            "label_text": "SUPPORTS",
            "prompt": (
                "Water strictly refers to the liquid state of that substance; but it "
                "often refers also to its solid state (ice) or its gaseous state "
                "(steam or water vapor)."
            ),
            "stable_id": "fever-84514",
        },
        {
            "claim": "Dog Day Afternoon stars at least one actor.",
            "dataset": "FEVER",
            "label": 0,
            "label_text": "SUPPORTS",
            "prompt": (
                "Dog Day Afternoon is a 1975 American crime drama film. The film stars "
                "Al Pacino, John Cazale, Charles Durning, Chris Sarandon, and others."
            ),
            "stable_id": "fever-182889",
        },
    ]


def _unsupported_row() -> dict[str, Any]:
    return {
        "candidate": "Paris",
        "dataset": "HaluEval",
        "label": 0,
        "prompt": "Question: What city is named?",
        "reference": "Paris",
        "stable_id": "halueval-unsupported",
    }


def _write_inputs(
    tmp_path: Path,
    *,
    halueval_rows: list[dict[str, Any]] | None = None,
    fever_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    halueval_path = tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl"
    fever_path = tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl"
    if halueval_rows is None:
        halueval_rows = [*_supported_halueval_rows(), _unsupported_row()]
    if fever_rows is None:
        fever_rows = _supported_fever_rows()
    _write_jsonl(halueval_path, halueval_rows)
    _write_jsonl(fever_path, fever_rows)

    exact_path = tmp_path / "results" / "experiment_2866_beaver_exact_tiny_frontier_v1.json"
    calibration_path = (
        tmp_path / "results" / "experiment_2864_halueval_fever_full_calibration_v3.json"
    )
    _write_json(
        exact_path,
        {
            "exact_frontier_available": True,
            "honest_verdict": "complete: tiny exact Z3 arithmetic frontier available",
            "solver_used": "z3-solver 4.16.0",
        },
    )
    _write_json(
        calibration_path,
        {
            "halueval_fever_ready": True,
            "honest_verdict": "complete: HaluEval/FEVER local calibration ready",
            "manifest_paths_used": {
                "fever": str(fever_path),
                "halueval": str(halueval_path),
            },
        },
    )
    return halueval_path, fever_path, exact_path, calibration_path


def _config(tmp_path: Path, exact_path: Path, calibration_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        exact_frontier_artifact_path=exact_path,
        calibration_artifact_path=calibration_path,
        output_path=tmp_path / "custom_results" / exp.OUTPUT_FILENAME,
        tests_run=("focused-pytest",),
        started_at=10.0,
        clock=lambda: 12.5,
    )


def test_scenario_verify_2877_writes_required_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2877: manual constraints certify HaluEval and FEVER rows."""

    _, _, exact_path, calibration_path = _write_inputs(tmp_path)

    artifact = exp.run_experiment(_config(tmp_path, exact_path, calibration_path))
    saved = json.loads((tmp_path / "custom_results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["frontier_expansion_ready"] is True
    assert artifact["n_candidate_rows"] == 9
    assert artifact["n_exact_supported_rows"] == 8
    assert artifact["n_unsupported_rows"] == 1
    assert artifact["unsupported_reasons"] == {"unsupported_no_manual_exact_constraint": 1}
    assert artifact["tests_run"] == ["focused-pytest"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert str(artifact["exact_solver_backend"]).startswith("z3-solver ")
    assert artifact["source_artifacts"] == [
        "results/experiment_2866_beaver_exact_tiny_frontier_v1.json",
        "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        "data/eval_manifests/halueval_20260522.jsonl",
        "data/eval_manifests/fever_20260522.jsonl",
    ]
    assert len(artifact["source_artifact_sha256"]) == 4
    assert artifact["field_principles"]["selection_rule"].startswith("Manual")

    by_id = {certificate["stable_id"]: certificate for certificate in artifact["certificates"]}
    assert set(by_id) == {
        "halueval-8-right",
        "halueval-8-hallucinated",
        "halueval-22-right",
        "halueval-22-hallucinated",
        "halueval-31-right",
        "halueval-31-hallucinated",
        "fever-84514",
        "fever-182889",
    }
    assert by_id["halueval-8-right"]["solver_status"] == "sat"
    assert by_id["halueval-8-hallucinated"]["solver_status"] == "unsat"
    assert by_id["halueval-22-hallucinated"]["exact_verdict"] == "contradiction_verified"
    assert by_id["fever-84514"]["constraint_type"] == "anchored_entailment"
    assert by_id["fever-182889"]["dataset"] == "FEVER"
    assert all(len(certificate["certificate_sha256"]) == 64 for certificate in by_id.values())


def test_req_verify_2877_unsupported_rows_stay_outside_exact_frontier(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2877: unsupported and anchor-mismatched rows are not promoted."""

    mismatched_registered_row = _supported_halueval_rows()[0] | {"candidate": "2005"}
    _, _, exact_path, calibration_path = _write_inputs(
        tmp_path,
        halueval_rows=[mismatched_registered_row, _unsupported_row()],
        fever_rows=[],
    )

    artifact = exp.run_experiment(_config(tmp_path, exact_path, calibration_path), write=False)

    assert artifact["frontier_expansion_ready"] is False
    assert artifact["n_candidate_rows"] == 2
    assert artifact["n_exact_supported_rows"] == 0
    assert artifact["n_unsupported_rows"] == 2
    assert artifact["certificates"] == []
    assert artifact["unsupported_reasons"] == {
        "unsupported_manual_constraint_failed": 1,
        "unsupported_no_manual_exact_constraint": 1,
    }

    wrong_dataset_row = _supported_halueval_rows()[0] | {"_dataset_key": "fever"}
    assert (
        exp._build_certificate(wrong_dataset_row, exp.MANUAL_CONSTRAINTS["halueval-8-right"])
        is None
    )
    wrong_expected_status = exp.MANUAL_CONSTRAINTS["halueval-8-right"] | {
        "expected_status": "unsat"
    }
    assert (
        exp._build_certificate(
            _supported_halueval_rows()[0] | {"_dataset_key": "halueval"},
            wrong_expected_status,
        )
        is None
    )


def test_req_verify_2877_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-2877: schema validation enforces count and field boundaries."""

    _, _, exact_path, calibration_path = _write_inputs(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path, exact_path, calibration_path), write=False)

    exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: no"})
    with pytest.raises(ValueError, match="run_date"):
        exp.validate_artifact(artifact | {"run_date": "20260101"})
    with pytest.raises(ValueError, match="source_artifacts"):
        exp.validate_artifact(artifact | {"source_artifacts": "not-a-list"})
    with pytest.raises(ValueError, match="unsupported count"):
        exp.validate_artifact(artifact | {"n_unsupported_rows": 0})
    with pytest.raises(ValueError, match="unsupported count"):
        exp.validate_artifact(artifact | {"unsupported_reasons": {}})
    with pytest.raises(ValueError, match="certificate count"):
        exp.validate_artifact(artifact | {"certificates": artifact["certificates"][:-1]})
    with pytest.raises(ValueError, match="solver backend"):
        exp.validate_artifact(artifact | {"exact_solver_backend": ""})

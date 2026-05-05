"""Tests for Exp 1391 full-scale pipeline failure diagnosis.

Spec: REQ-VERIFY-1391, SCENARIO-VERIFY-1391.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fullscale_pipeline_failure_diagnosis as mod


def _semantic(
    case_id: str,
    *,
    expected_state: str,
    certificate_state: str,
    semantic_result: str,
    fover_label: str,
    passed: bool,
    failure_reason: str | None = None,
    probability: float = 0.7,
    evaluated: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "claim_route": "dvi_updated_fover_semantic_validator",
        "expected_state": expected_state,
        "certificate_state": certificate_state,
        "semantic_result": semantic_result,
        "constraint_passed": passed,
        "constraint_evaluated": evaluated,
        "dvi_incorrect_probability": probability,
        "dvi_incorrect_threshold": 0.72,
        "semantic_margin": round(abs(probability - 0.72), 6),
        "fover_label": fover_label,
        "failure_reason": failure_reason,
    }


def _certificate(
    case_id: str,
    *,
    state: str = "SAT",
    parseable: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "parseable": parseable,
        "tag_state": state,
        "dispatched_state": state,
        "truthful": parseable,
        "errors": [] if parseable else ["missing_tag_state"],
    }


def _exp1382_fixture() -> dict[str, Any]:
    return {
        "status": "complete",
        "total_fover_cases": 6,
        "certificate_parse_rate": 1.0,
        "semantic_validation_pass_rate": 1 / 6,
        "semantic_validation_rows": [
            _semantic(
                "pass_correct",
                expected_state="SAT",
                certificate_state="SAT",
                semantic_result="SAT",
                fover_label="correct",
                passed=True,
                probability=0.1,
            ),
            _semantic(
                "bad_math_1",
                expected_state="REPAIR_HINT",
                certificate_state="REPAIR_HINT",
                semantic_result="SAT",
                fover_label="incorrect",
                passed=False,
                failure_reason="dvi_disagrees_with_fover_label",
                probability=0.65,
            ),
            _semantic(
                "bad_math_2",
                expected_state="REPAIR_HINT",
                certificate_state="REPAIR_HINT",
                semantic_result="SAT",
                fover_label="incorrect",
                passed=False,
                failure_reason="dvi_disagrees_with_fover_label",
                probability=0.71,
            ),
            _semantic(
                "valid_sat_rejected",
                expected_state="SAT",
                certificate_state="SAT",
                semantic_result="REPAIR_HINT",
                fover_label="correct",
                passed=False,
                failure_reason="dvi_disagrees_with_fover_label",
                probability=0.74,
            ),
            _semantic(
                "missing_cert",
                expected_state="SAT",
                certificate_state="",
                semantic_result="SAT",
                fover_label="correct",
                passed=False,
                failure_reason="certificate_parse_failed",
                probability=0.2,
                evaluated=False,
            ),
            _semantic(
                "solver_unsat",
                expected_state="SAT",
                certificate_state="SAT",
                semantic_result="SAT",
                fover_label="correct",
                passed=False,
                failure_reason="z3_solver_unsat",
                probability=0.2,
            ),
        ],
        "certificate_rows": [
            _certificate("pass_correct"),
            _certificate("bad_math_1", state="REPAIR_HINT"),
            _certificate("bad_math_2", state="REPAIR_HINT"),
            _certificate("valid_sat_rejected"),
            _certificate("missing_cert", state="", parseable=False),
            _certificate("solver_unsat"),
        ],
    }


def test_req1391_classifies_failures_and_estimates_recovery() -> None:
    """REQ-VERIFY-1391: all failed semantic rows receive a ranked taxonomy."""

    artifact = mod.build_failure_diagnosis_artifact(
        exp1382_artifact=_exp1382_fixture(),
        fover_case_lookup={
            "bad_math_1": {"source": "math_z3"},
            "bad_math_2": {"source": "math_z3_v3"},
            "valid_sat_rejected": {"source": "fover_v4"},
        },
        project_root="/repo",
        run_date="20260505",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["total_cases_analyzed"] == 6
    assert artifact["parse_rate_confirmed"] is True
    assert artifact["semantic_validation_failures_classified"] == 5
    assert artifact["failure_category_counts"] == {
        "Z3_CONSTRAINT_MISMATCH": 1,
        "MISSING_CERTIFICATE_FIELD": 1,
        "SEMANTIC_CONTRADICTION": 0,
        "CORPUS_SPECIFIC": 2,
        "VALIDATOR_BUG": 1,
        "OTHER": 0,
    }
    assert artifact["top_failure_category"] == "CORPUS_SPECIFIC"
    assert artifact["fixable_failure_pct"] == pytest.approx(0.8)
    assert artifact["estimated_semantic_validation_pass_rate_after_fixes"] == pytest.approx(5 / 6)
    assert artifact["failure_analysis_complete"] is True
    assert artifact["failure_categories"][0]["category"] == "CORPUS_SPECIFIC"
    assert artifact["recommended_fixes"][0]["category"] == "CORPUS_SPECIFIC"


def test_scenario1391_runner_writes_progress_and_final_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1391: runner persists bootstrap and complete diagnosis JSON."""

    exp1382_path = tmp_path / "exp1382.json"
    fover_path = tmp_path / "fover.jsonl"
    output_path = tmp_path / "exp1391.json"
    exp1382_path.write_text(json.dumps(_exp1382_fixture()), encoding="utf-8")
    fover_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "bad_math_1",
                        "step_text": "2 + 2 = 5",
                        "label": "incorrect",
                        "source": "math_z3",
                    }
                ),
                json.dumps(
                    {
                        "question_id": "bad_math_2",
                        "step_text": "3 + 3 = 7",
                        "label": "incorrect",
                        "source": "math_z3_v3",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1382_path=exp1382_path,
        fover_path=fover_path,
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["top_failure_category"] == "CORPUS_SPECIFIC"
    assert artifact["failure_analysis_complete"] is True

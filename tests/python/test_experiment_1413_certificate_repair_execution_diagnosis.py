"""Tests for Exp 1413 certificate repair execution diagnosis.

Spec: REQ-VERIFY-1413, SCENARIO-VERIFY-1413
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import certificate_repair_execution_diagnosis as mod


def _exp1397_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "cases_evaluated": 10,
        "certificate_parse_rate": 1.0,
        "semantic_validation_pass_rate": 1.0,
        "full_pipeline_pass_rate": 0.3,
        "repair_localization_rows": [
            {
                "case_id": "needs_step_rewrite",
                "localized_constraint": "fover_incorrect_reasoning_step",
                "minimal_local_change": "repair_or_remove_incorrect_arithmetic_step",
                "repair_hint": "Repair the localized FoVer reasoning step before accepting.",
            },
            {
                "case_id": "needs_constraint_rewrite",
                "localized_constraint": "upper_bound_premise",
                "minimal_local_change": "add_missing_capacity_bound",
                "repair_hint": "Add the missing bound constraint before accepting.",
            },
        ],
        "scheduler_rows": [
            {"case_id": "pass_0", "full_pipeline_pass": True, "repair_required": False},
            {"case_id": "pass_1", "full_pipeline_pass": True, "repair_required": False},
            {"case_id": "pass_2", "full_pipeline_pass": True, "repair_required": False},
            {
                "case_id": "needs_step_rewrite",
                "full_pipeline_pass": False,
                "repair_required": True,
            },
            {
                "case_id": "needs_constraint_rewrite",
                "full_pipeline_pass": False,
                "repair_required": True,
            },
            {"case_id": "low_margin_0", "full_pipeline_pass": False, "repair_required": False},
            {"case_id": "low_margin_1", "full_pipeline_pass": False, "repair_required": False},
            {"case_id": "low_margin_2", "full_pipeline_pass": False, "repair_required": False},
            {"case_id": "low_margin_3", "full_pipeline_pass": False, "repair_required": False},
            {"case_id": "low_margin_4", "full_pipeline_pass": False, "repair_required": False},
        ],
    }


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {
                "localized_constraint": "certificate_state_mismatch",
                "minimal_local_change": "set_certificate_state_to_unsat",
                "repair_hint": "Change only the certificate state field.",
            },
            "FIELD_REWRITE",
        ),
        (
            {
                "localized_constraint": "fover_incorrect_reasoning_step",
                "minimal_local_change": "repair_or_remove_incorrect_arithmetic_step",
                "repair_hint": "Repair the localized reasoning step.",
            },
            "STEP_REWRITE",
        ),
        (
            {
                "localized_constraint": "cnf_unit_conflict",
                "minimal_local_change": "relax_one_minimal_conflicting_cnf_clause",
                "repair_hint": "Relax the conflicting formula constraint.",
            },
            "CONSTRAINT_REWRITE",
        ),
        (
            {
                "localized_constraint": "certificate_parse_failure",
                "minimal_local_change": "regenerate_tag_first_certificate",
                "repair_hint": "Regenerate the tag-first certificate before validation.",
            },
            "CERTIFICATE_REGENERATE",
        ),
        (
            {
                "localized_constraint": "opaque",
                "minimal_local_change": "manual_review",
                "repair_hint": "Needs expert inspection.",
            },
            "UNKNOWN",
        ),
    ],
)
def test_req1413_classifies_repair_hint_taxonomy(row: dict[str, str], expected: str) -> None:
    """REQ-VERIFY-1413: every repair hint maps into the bounded taxonomy."""

    assert mod.classify_hint_category(row) == expected


def test_req1413_diagnosis_uses_repair_specific_denominator() -> None:
    """REQ-VERIFY-1413: 50 percent repaired uses repair rows when available."""

    artifact = mod.build_certificate_repair_execution_diagnosis(
        exp1397_artifact=_exp1397_artifact(),
        run_date="20260506",
        project_root="/repo",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["total_cases_analyzed"] == 10
    assert artifact["repair_hint_cases_total"] == 2
    assert artifact["no_repair_cases_total"] == 8
    assert artifact["repair_execution_diagnosis_complete"] is True
    assert artifact["hint_category_counts"] == {
        "FIELD_REWRITE": 0,
        "STEP_REWRITE": 1,
        "CONSTRAINT_REWRITE": 1,
        "CERTIFICATE_REGENERATE": 0,
        "UNKNOWN": 0,
    }
    assert artifact["executable_hint_pct"] == pytest.approx(1.0)
    assert artifact["expected_full_pipeline_pass_rate_if_50pct_repaired"] == pytest.approx(0.4)
    assert "missing_repair_execution" in artifact["honest_verdict"]
    assert artifact["expected_rate_basis"]["used_repair_specific_denominator"] is True

    contract = artifact["recommended_executor_contract"]
    assert {"inputs", "outputs", "validation_call", "timeout", "fallback_behavior"} <= set(
        contract
    )
    assert "calibrated_fover_semantic_validation_row" in contract["validation_call"]
    assert contract["timeout"]["per_case_seconds"] == 30
    assert contract["fallback_behavior"]["preserve_original_repair_hint"] is True


def test_scenario1413_runner_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1413: runner persists bootstrap and terminal JSON."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1413.json"
    source.write_text(json.dumps(_exp1397_artifact()), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        exp1397_path=source,
        output_path=output,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["artifact_metadata"]["spec"] == [
        "REQ-VERIFY-1413",
        "SCENARIO-VERIFY-1413",
    ]


def test_req1413_fallback_denominator_is_used_without_repair_rows() -> None:
    """REQ-VERIFY-1413: fallback estimate is used when no repair denominator exists."""

    artifact = mod.build_certificate_repair_execution_diagnosis(
        exp1397_artifact={
            "status": "complete",
            "certificate_rows": [{"case_id": "a"}, {"case_id": "b"}],
            "repair_localization_rows": "not-a-list",
            "scheduler_rows": "not-a-list",
            "full_pipeline_pass_rate": 0.25,
        },
        run_date="20260506",
        project_root="/repo",
    )

    assert artifact["total_cases_analyzed"] == 2
    assert artifact["repair_hint_cases_total"] == 0
    assert artifact["expected_full_pipeline_pass_rate_if_50pct_repaired"] == pytest.approx(0.625)
    assert artifact["expected_rate_basis"]["used_repair_specific_denominator"] is False
    assert artifact["diagnostic_evidence"]["scheduler_nonrepair_failures"] == 0
    assert artifact["honest_verdict"] == "repair_execution_diagnosis_complete_no_repair_hints_found"


def test_req1413_unknown_only_hints_are_not_executable() -> None:
    """REQ-VERIFY-1413: UNKNOWN hints do not count as bounded local execution."""

    artifact = mod.build_certificate_repair_execution_diagnosis(
        exp1397_artifact={
            "status": "complete",
            "cases_evaluated": 4,
            "full_pipeline_pass_rate": 0.25,
            "repair_localization_rows": [
                {
                    "case_id": "manual",
                    "localized_constraint": "opaque",
                    "minimal_local_change": "manual_review",
                    "repair_hint": "Needs expert inspection.",
                }
            ],
        },
        run_date="20260506",
        project_root="/repo",
    )

    assert artifact["hint_category_counts"]["UNKNOWN"] == 1
    assert artifact["executable_hint_pct"] == pytest.approx(0.0)
    assert artifact["honest_verdict"] == (
        "repair_execution_diagnosis_complete_missing_repair_execution_but_hints_not_executable"
    )


def test_req1413_empty_source_artifact_reports_zero_cases() -> None:
    """REQ-VERIFY-1413: malformed empty sources produce a complete zero-case diagnosis."""

    artifact = mod.build_certificate_repair_execution_diagnosis(
        exp1397_artifact={"repair_localization_rows": [], "scheduler_rows": "bad"},
        run_date="20260506",
        project_root="/repo",
    )

    assert artifact["total_cases_analyzed"] == 0
    assert artifact["repair_execution_diagnosis_complete"] is True
    assert artifact["honest_verdict"] == "repair_execution_diagnosis_complete_no_repair_hints_found"


def test_req1413_cli_main_runs_with_explicit_paths(tmp_path: Path, capsys: Any) -> None:
    """REQ-VERIFY-1413: CLI entrypoint writes the requested artifact."""

    source = tmp_path / "exp1397.json"
    output = tmp_path / "exp1413.json"
    source.write_text(json.dumps(_exp1397_artifact()), encoding="utf-8")

    assert (
        mod.main(
            [
                "--project-root",
                str(tmp_path),
                "--run-date",
                "20260506",
                "--exp1397-path",
                str(source),
                "--output-path",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete"
    assert summary["repair_hint_cases_total"] == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"

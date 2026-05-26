"""Tests for Exp 3114 fragment-level verification pilot.

Spec refs: REQ-VERIFY-3114, SCENARIO-VERIFY-3114.
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fragment_verification_pilot_3114 as exp


REQUIRED_FIELDS = {
    "fragment_verification_pilot_ready",
    "exact_fixture_count",
    "fragment_count",
    "failing_fragment_count",
    "unknown_fragment_count",
    "repair_target_manifest_path",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _manifest_row(
    fixture_id: str,
    *,
    family: str,
    perturbation: str,
    expected_answer: str,
    solver_label: str,
    payload: dict[str, Any],
    label_source: str,
    repairable: bool,
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "task_family": family,
        "task_axis": "repairing" if repairable else "verifying",
        "perturbation_type": perturbation,
        "exact_label_kind": "repairability" if repairable else "arithmetic_assertion",
        "expected_answer": expected_answer,
        "solver_label": solver_label,
        "label_source": label_source,
        "leakage_safe_prompt_payload": payload,
        "source_prompt_payload_sha256": fixture_id * 2,
        "repair_target": {"applicable": repairable, "repairable": repairable},
        "verifier_target": {"expected_action": "reject" if "fail" in solver_label else "accept"},
    }


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        _manifest_row(
            "arith-valid",
            family="arithmetic_code_assertions",
            perturbation="arithmetic_true_verification",
            expected_answer="VALID",
            solver_label="assertion_passes",
            label_source="python_ast_runtime_execution",
            repairable=False,
            payload={
                "candidate_assertion": "assert (((3 + 2) * 1) - 0) == 5",
                "expression": "((3 + 2) * 1) - 0",
            },
        ),
        _manifest_row(
            "arith-invalid",
            family="arithmetic_code_assertions",
            perturbation="arithmetic_false_verification",
            expected_answer="INVALID",
            solver_label="assertion_fails",
            label_source="python_ast_runtime_execution",
            repairable=False,
            payload={
                "candidate_assertion": "assert (((4 + 3) * 2) - 0) == 16",
                "expression": "((4 + 3) * 2) - 0",
            },
        ),
        _manifest_row(
            "json-repair",
            family="repairable_invalid_candidates",
            perturbation="json_syntax_repair",
            expected_answer="REPAIRABLE",
            solver_label="repairable",
            label_source="json_parser",
            repairable=True,
            payload={
                "candidate": '{"mode": "bounded" "limit": 2}',
                "required_fields": ["mode", "limit"],
            },
        ),
        _manifest_row(
            "numeric-repair",
            family="repairable_invalid_candidates",
            perturbation="numeric_bound_repair",
            expected_answer="REPAIRABLE",
            solver_label="repairable",
            label_source="z3_solver",
            repairable=True,
            payload={
                "variables": ["rx_1", "ry_1"],
                "candidate_assignment": {"rx_1": 2, "ry_1": 1},
                "constraints": ["rx_1 >= 0", "ry_1 >= 0", "rx_1 + ry_1 == 11"],
            },
        ),
        _manifest_row(
            "py-repair",
            family="repairable_invalid_candidates",
            perturbation="python_assertion_repair",
            expected_answer="REPAIRABLE",
            solver_label="repairable",
            label_source="python_ast_runtime_execution",
            repairable=True,
            payload={
                "candidate_assertion": "assert ((7 * 2) - 2) == 13",
                "expression": "(7 * 2) - 2",
            },
        ),
    ]


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_sources(root: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(root, exp.MANIFEST_REL_PATH, rows)
    _write_json(
        root,
        exp.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "stratified_eval_manifest_path": exp.MANIFEST_REL_PATH.as_posix(),
        },
    )
    _write_json(
        root,
        exp.EXP3100_REL_PATH,
        {
            "artifact": "experiment_3100_z3_oracle_feedback_v2",
            "z3_available": True,
            "fixture_results": [],
            "honest_verdict": "complete_blocked_headline: cached_sota_pair_unavailable",
        },
    )
    _write_json(
        root,
        exp.EXP3111_REL_PATH,
        {
            "artifact": "experiment_3111_certified_coherence_z3_mcs_feedback_v3",
            "certified_coherence_feedback_v3_ready": True,
            "certificate_count": len(rows),
        },
    )


def test_req_verify_3114_spec_anchor_and_module_contract() -> None:
    """REQ-VERIFY-3114: OpenSpec declares artifact and manifest requirements."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3114" in spec
    assert "SCENARIO-VERIFY-3114" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert exp.REPAIR_TARGET_MANIFEST_REL_PATH.as_posix() in spec


def test_scenario_verify_3114_fragment_checks_localize_failures() -> None:
    """SCENARIO-VERIFY-3114: fixture fragments get exact localized statuses."""

    rows = _fixture_rows()
    fragments = [fragment for row in rows for fragment in exp.fragment_checks_for_row(row)]
    by_fixture = {}
    for fragment in fragments:
        by_fixture.setdefault(fragment["fixture_id"], []).append(fragment)
    targets = exp.repair_targets_from_fragments(fragments)

    assert {fragment["status"] for fragment in fragments} == {"fail", "non-applicable", "pass"}
    assert len(by_fixture["arith-valid"]) == 2
    assert by_fixture["arith-invalid"][-1]["status"] == "fail"
    assert by_fixture["arith-invalid"][-1]["expected_direction"] == "replace claimed value 16 with 14"
    assert by_fixture["json-repair"][0]["status"] == "fail"
    assert by_fixture["json-repair"][1]["status"] == "non-applicable"
    assert by_fixture["numeric-repair"][-1]["failing_constraint"] == "rx_1 + ry_1 == 11"
    assert by_fixture["numeric-repair"][-1]["expected_direction"] == (
        "increase sum by 8 across ['rx_1', 'ry_1']"
    )
    assert by_fixture["py-repair"][-1]["expected_direction"] == "replace claimed value 13 with 12"
    assert len(targets) == 4
    assert all(
        {"fixture_id", "fragment_id", "failing_constraint", "expected_direction", "solver_evidence"}
        <= set(target)
        for target in targets
    )


def test_req_verify_3114_unknown_fragment_is_counted_honestly() -> None:
    """REQ-VERIFY-3114: unsupported rows degrade to an unknown fragment label."""

    fragments = exp.fragment_checks_for_row(
        {
            "source_fixture_id": "unsupported",
            "task_family": "repairable_invalid_candidates",
            "perturbation_type": "opaque_repair",
            "leakage_safe_prompt_payload": {"candidate": "not structured here"},
        }
    )

    assert fragments == [
        {
            "fixture_id": "unsupported",
            "fragment_id": "unsupported:opaque_repair",
            "status": "unknown",
            "failing_constraint": "unsupported_fragment_parser",
            "expected_direction": "manual parser/checker extension required",
            "solver_evidence": {
                "authority": "fragment_verification_pilot",
                "reason": "unsupported task_family or perturbation_type",
            },
        }
    ]


def test_req_verify_3114_helper_branches_are_deterministic() -> None:
    """REQ-VERIFY-3114: helper paths cover valid JSON and bound directions."""

    valid_json_fragments = exp.fragment_checks_for_row(
        _manifest_row(
            "json-valid",
            family="repairable_invalid_candidates",
            perturbation="json_syntax_repair",
            expected_answer="REPAIRABLE",
            solver_label="repairable",
            label_source="json_parser",
            repairable=True,
            payload={"candidate": '{"mode": "bounded"}', "required_fields": ["mode", "limit"]},
        )
    )
    ge_passed, ge_evidence, ge_direction = exp._evaluate_constraint("x >= 5", {"x": 3})
    le_passed, le_evidence, le_direction = exp._evaluate_constraint("x <= 1", {"x": 4})
    unary_value = exp._eval_int(ast.parse("-3", mode="eval").body)
    blocked_verdict = exp._honest_verdict(
        {
            "fragment_verification_pilot_ready": False,
            "exact_fixture_count": 0,
            "failing_fragment_count": 0,
        }
    )

    assert [fragment["status"] for fragment in valid_json_fragments] == ["pass", "pass", "fail"]
    assert valid_json_fragments[-1]["expected_direction"] == "add required field limit"
    assert ge_passed is False
    assert ge_evidence["lhs_value"] == 3
    assert ge_direction == "increase x by 2"
    assert le_passed is False
    assert le_evidence["rhs_value"] == 1
    assert le_direction == "decrease x by 3"
    assert unary_value == -3
    assert blocked_verdict == "blocked_fragment_verification_pilot_missing_required_evidence"


def test_scenario_verify_3114_artifact_and_repair_manifest_are_written(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3114: terminal artifact points to concrete repair targets."""

    rows = _fixture_rows()
    _write_sources(tmp_path, rows)

    artifact_path = exp.write_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.25,
        tests_run=["focused pytest"],
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / artifact["repair_target_manifest_path"]
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fragment_verification_pilot_ready"] is True
    assert artifact["exact_fixture_count"] == 5
    assert artifact["fragment_count"] == 12
    assert artifact["failing_fragment_count"] == 4
    assert artifact["unknown_fragment_count"] == 0
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["inference_substrate"]["live_llm_calls"] == 0
    assert artifact["inference_substrate"]["uses_checked_in_artifacts_only"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == artifact["failing_fragment_count"]
    assert manifest_rows == artifact["repair_target_manifest"]
    assert {row["fixture_id"] for row in manifest_rows} == {
        "arith-invalid",
        "json-repair",
        "numeric-repair",
        "py-repair",
    }
    assert all(row["present"] for row in artifact["source_artifacts"])

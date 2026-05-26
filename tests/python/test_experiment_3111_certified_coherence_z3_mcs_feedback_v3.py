"""Tests for Exp 3111 certified coherence feedback v3.

Spec refs: REQ-REPORT-3111, SCENARIO-REPORT-3111.
"""

from __future__ import annotations

import ast
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import certified_coherence_feedback_v3_3111 as mod


REQUIRED_FIELDS = {
    "certified_coherence_feedback_v3_ready",
    "z3_available",
    "exact_ground_truth_count",
    "certificate_count",
    "unsat_core_count",
    "minimal_repair_distance_summary",
    "solver_only_success_count",
    "guided_success_count",
    "formal_feedback_delta",
    "vacuity_guard_passed",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


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


def _manifest_row(
    fixture_id: str,
    *,
    family: str,
    perturbation: str,
    expected_answer: str,
    solver_label: str,
    payload: dict[str, Any],
    label_source: str,
    expected_action: str,
    repairable: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "task_family": family,
        "task_axis": "repairing" if repairable else "verifying",
        "perturbation_type": perturbation,
        "exact_label_kind": "repairability" if repairable else "formal_label",
        "expected_answer": expected_answer,
        "solver_label": solver_label,
        "label_source": label_source,
        "leakage_safe_prompt_payload": payload,
        "source_prompt_payload_sha256": fixture_id * 2,
        "repair_target": {
            "applicable": repairable,
            "repairable": repairable,
            "repair_validation": "passed" if repairable else None,
        },
        "verifier_target": {
            "expected_action": expected_action,
            "expected_reject": expected_action == "reject",
        },
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
            expected_action="accept",
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
            expected_action="reject",
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
            expected_action="reject",
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
            expected_action="reject",
            repairable=True,
            payload={
                "variables": ["rx_1", "ry_1"],
                "candidate_assignment": {"rx_1": 2, "ry_1": 1},
                "constraints": ["rx_1 >= 0", "ry_1 >= 0", "rx_1 + ry_1 == 11"],
            },
        ),
        _manifest_row(
            "smt-unsat",
            family="smt_constraints",
            perturbation="smt_unsat_abstention",
            expected_answer="UNSAT",
            solver_label="unsat",
            label_source="z3_solver",
            expected_action="reject",
            payload={
                "variables": ["x_0", "y_0"],
                "constraints": ["x_0 >= 2", "x_0 <= 1", "y_0 >= 0"],
            },
        ),
        _manifest_row(
            "smt-sat",
            family="smt_constraints",
            perturbation="smt_sat_solving",
            expected_answer="SAT",
            solver_label="sat",
            label_source="z3_solver",
            expected_action="accept",
            payload={
                "variables": ["x_1", "y_1"],
                "constraints": [
                    "x_1 >= 1",
                    "x_1 <= 7",
                    "y_1 >= 0",
                    "y_1 <= 7",
                    "x_1 + y_1 == 4",
                ],
            },
        ),
    ]


def _write_common_sources(root: Path) -> None:
    rows = _fixture_rows()
    _write_jsonl(root, mod.MANIFEST_REL_PATH, rows)
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "usable_fixture_count": len(rows),
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3098_REL_PATH,
        {
            "artifact": "experiment_3098_maxsat_abstention_routing_policy_v1",
            "maxsat_policy_ready": True,
            "hard_constraints": [{"id": "HC_EXACT_LABEL_AGREEMENT"}],
            "soft_constraints": [{"id": "SC_LOCALIZE_CONTRADICTION", "weight": 40}],
            "objective_terms": {"weights": {"localize_contradiction": 40}},
            "honest_verdict": "complete: maxsat_policy_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3100_REL_PATH,
        {
            "artifact": "experiment_3100_z3_oracle_feedback_v2",
            "z3_available": True,
            "exact_ground_truth_count": len(rows),
            "solver_only_success_count": 2,
            "guided_success_count": 0,
            "formal_feedback_delta": -0.333333,
            "vacuity_guard_passed": True,
            "formal_feedback_v2_ready": False,
            "headline_blocked_reason": "cached_sota_pair_unavailable",
            "fixture_results": [],
            "honest_verdict": "complete_blocked_headline: cached_sota_pair_unavailable",
        },
    )
    _write_json(
        root,
        mod.EXP3110_REL_PATH,
        {
            "artifact": "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1",
            "sota_model_manifest_ready": True,
            "cached_sota_pair_available": False,
            "downstream_usage": {
                "solver_only_tasks": {"allowed_without_cached_sota_pair": True}
            },
            "honest_verdict": "complete: solver-only tasks may proceed",
        },
    )


def test_req_report_3111_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3111: OpenSpec declares the certificate contract before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3111" in spec
    assert "SCENARIO-REPORT-3111" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3111_builds_solver_certificates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3111: exact fixtures become localized certificates."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        tests_run=["unit-test"],
    )
    by_id = {row["fixture_id"]: row for row in artifact["certificates"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["certified_coherence_feedback_v3_ready"] is True
    assert artifact["z3_available"] is True
    assert artifact["exact_ground_truth_count"] == len(_fixture_rows())
    assert artifact["certificate_count"] == len(_fixture_rows())
    assert artifact["unsat_core_count"] >= 3
    assert artifact["solver_only_success_count"] == 2
    assert artifact["guided_success_count"] == 0
    assert artifact["formal_feedback_delta"] == pytest.approx(-0.333333)
    assert artifact["vacuity_guard_passed"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert by_id["arith-valid"]["coherence_status"] == "coherent"
    assert by_id["arith-invalid"]["coherence_status"] == "incoherent"
    assert by_id["arith-invalid"]["repair_distance"] == 2
    assert by_id["arith-invalid"]["minimal_correction_set"]["kind"] == "replace_claimed_value"
    assert by_id["json-repair"]["minimal_correction_set"]["kind"] == "json_token_edit"
    assert by_id["json-repair"]["repair_distance"] == 1
    assert by_id["numeric-repair"]["solver_authority"] == "z3_solver"
    assert by_id["numeric-repair"]["repair_distance"] == 8
    assert by_id["numeric-repair"]["minimal_correction_set"]["kind"] == "relax_assignment"
    assert by_id["smt-unsat"]["unsat_core"]
    assert by_id["smt-unsat"]["coherence_gap"] == 1
    assert by_id["smt-sat"]["model"]

    summary = artifact["minimal_repair_distance_summary"]
    assert summary["count"] >= 4
    assert summary["max"] == 8
    assert summary["mean"] > 0
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["cached_sota_pair_required_for_readiness"] is False


def test_req_report_3111_write_and_validate_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-3111: write helper persists a schema-valid terminal artifact."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=1.75,
        tests_run=["write-test"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["certified_coherence_feedback_v3_ready"] is True
    assert artifact["tests_run"] == ["write-test"]
    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: no"})
    with pytest.raises(ValueError, match="certificate count"):
        mod.validate_artifact(artifact | {"certificate_count": 0})


def test_req_report_3111_blocks_missing_sources_and_z3(tmp_path: Path) -> None:
    """REQ-REPORT-3111: missing authorities or Z3 fail closed with diagnostics."""

    missing = mod.build_artifact(tmp_path, z3_module=None)

    assert missing["certified_coherence_feedback_v3_ready"] is False
    assert missing["z3_available"] is False
    assert missing["honest_verdict"].startswith("blocked_certified_coherence_feedback_v3")
    assert missing["missing_source_artifacts"]

    _write_common_sources(tmp_path)
    no_z3 = mod.build_artifact(tmp_path, z3_module=None)

    assert no_z3["certificate_count"] == len(_fixture_rows())
    assert no_z3["certified_coherence_feedback_v3_ready"] is False
    assert no_z3["readiness_checks"]["z3_available"] is False
    assert mod.read_json_object(tmp_path / "does-not-exist.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}


def test_req_report_3111_defensive_certificate_edges() -> None:
    """REQ-REPORT-3111: malformed rows produce explicit diagnostic certificates."""

    malformed_jsonl = '\nnot-json\n{"source_fixture_id": "ok"}\n'
    rows = mod.read_jsonl_rows_from_text(malformed_jsonl)
    cert = mod.certificate_for_row({"source_fixture_id": "unknown", "task_family": "mystery"})

    assert rows == [{"source_fixture_id": "ok"}]
    assert cert["coherence_status"] == "unsupported"
    assert cert["minimal_correction_set"]["kind"] == "unsupported_fixture_family"


def test_req_report_3111_full_branch_coverage_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3111: rare parser and solver edges remain deterministic."""

    _write_common_sources(tmp_path)
    relative_output = mod.write_artifact(tmp_path, output_path=mod.OUTPUT_REL_PATH)
    artifact = json.loads(relative_output.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="at least one unsat core"):
        mod.validate_artifact(artifact | {"unsat_core_count": 0})
    with pytest.raises(ValueError, match="repair-distance summary"):
        mod.validate_artifact(
            artifact
            | {
                "minimal_repair_distance_summary": {
                    "count": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                }
            }
        )

    json_missing = mod.certificate_for_row(
        _manifest_row(
            "json-missing",
            family="repairable_invalid_candidates",
            perturbation="json_syntax_repair",
            expected_answer="REPAIRABLE",
            solver_label="repairable",
            label_source="json_parser",
            expected_action="reject",
            repairable=True,
            payload={"candidate": '{"mode": "bounded"}', "required_fields": ["mode", "limit"]},
        )
    )
    numeric_valid = mod.certificate_for_row(
        _manifest_row(
            "numeric-valid",
            family="repairable_invalid_candidates",
            perturbation="numeric_bound_repair",
            expected_answer="REPAIRABLE",
            solver_label="repairable",
            label_source="z3_solver",
            expected_action="reject",
            repairable=True,
            payload={
                "variables": ["rx_2", "ry_2"],
                "candidate_assignment": {"rx_2": 10, "ry_2": 1},
                "constraints": ["rx_2 >= 0", "ry_2 >= 0", "rx_2 + ry_2 == 11"],
            },
        )
    )
    smt_equal = mod.certificate_for_row(
        _manifest_row(
            "smt-equal",
            family="smt_constraints",
            perturbation="smt_sat_solving",
            expected_answer="SAT",
            solver_label="sat",
            label_source="z3_solver",
            expected_action="accept",
            payload={"variables": ["x"], "constraints": ["x == 1"]},
        )
    )

    assert json_missing["minimal_correction_set"] == {
        "kind": "add_missing_fields",
        "fields": ["limit"],
    }
    assert numeric_valid["coherence_status"] == "coherent"
    assert smt_equal["model"]["x"] == 1

    with pytest.raises(ValueError, match="assert statement"):
        mod._assertion_values("1 == 1")
    with pytest.raises(ValueError, match="compare two expressions"):
        mod._assertion_values("assert 1")
    assert mod._eval_numeric_ast(ast.parse("-4", mode="eval").body) == Fraction(-4)
    assert mod._eval_numeric_ast(ast.parse("8 / 2", mode="eval").body) == Fraction(4)
    with pytest.raises(ValueError, match="unsupported arithmetic AST node"):
        mod._eval_numeric_ast(ast.parse("name", mode="eval").body)
    assert mod._arithmetic_unsat_core({}, Fraction(1), Fraction(1), mod._z3) == []
    with pytest.raises(ValueError, match="unsupported constraint"):
        mod._constraint_to_z3("x != 1", {"x": mod._z3.Int("x")}, mod._z3)

    class _UnknownSolver:
        def assert_and_track(self, _constraint: object, _name: str) -> None:
            return None

        def check(self) -> str:
            return "unknown"

    class _UnknownZ3:
        sat = "sat"
        unsat = "unsat"

        @staticmethod
        def Solver() -> _UnknownSolver:
            return _UnknownSolver()

    assert mod._check_named_constraints([("c", object())], _UnknownZ3)["status"] == "unknown"
    monkeypatch.setattr(mod, "_sat", lambda _constraints, _z3_module: False)
    assert mod._minimal_correction_set([("a", object())], object()) == ["a"]
    assert mod._assignment_repair({"constraints": []}, {"x": 1}) == (0, {"x": 1})

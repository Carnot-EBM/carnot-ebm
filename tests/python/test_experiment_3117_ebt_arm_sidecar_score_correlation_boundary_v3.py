"""Tests for Exp 3117 EBT/ARM sidecar score correlation boundary v3.

Spec refs: REQ-VERIFY-3117, SCENARIO-VERIFY-3117.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import ebt_arm_sidecar_score_correlation_boundary_v3 as mod


REQUIRED_FIELDS = {
    "sidecar_score_correlation_boundary_v3_ready",
    "exact_fixture_count",
    "score_correlation_summary",
    "calibration_summary",
    "failure_cases",
    "no_live_model_integration_claim",
    "no_weight_update_claim",
    "no_speedup_claim",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_row(
    fixture_id: str,
    answer: str,
    action: str,
    payload: dict[str, Any],
    *,
    family: str,
    perturbation_type: str,
    repairable: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"{fixture_id}-hash",
        "task_family": family,
        "task_axis": "repairing" if repairable else "verifying",
        "perturbation_type": perturbation_type,
        "expected_answer": answer,
        "solver_label": answer.lower(),
        "label_source": "unit_exact_authority",
        "exact_label_kind": "repairability" if repairable else "unit",
        "leakage_safe_prompt_payload": payload,
        "verifier_target": {"expected_action": action, "expected_reject": action == "reject"},
        "repair_target": {"applicable": repairable, "repairable": repairable},
        "evaluation_tasks": ["sidecar_score_correlation_boundary_v3"],
        "stratum_key": f"{family}|{perturbation_type}|{answer}",
    }


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        _manifest_row(
            "case-accept",
            "VALID",
            "accept",
            {
                "candidate_assertion": "assert (2 + 3) == 5",
                "expression": "2 + 3",
                "task": "Classify the candidate arithmetic assertion.",
            },
            family="arithmetic_code_assertions",
            perturbation_type="arithmetic_true_verification",
        ),
        _manifest_row(
            "case-reject",
            "INVALID",
            "reject",
            {
                "candidate_assertion": "assert (2 + 3) == 6",
                "expression": "2 + 3",
                "task": "Classify the candidate arithmetic assertion.",
            },
            family="arithmetic_code_assertions",
            perturbation_type="arithmetic_false_verification",
        ),
        _manifest_row(
            "case-repair",
            "REPAIRABLE",
            "reject",
            {
                "candidate": '{"mode": "bounded" "limit": 2}',
                "required_fields": ["mode", "limit"],
                "task": "Repair the candidate so the object parses and preserves fields.",
            },
            family="repairable_invalid_candidates",
            perturbation_type="json_syntax_repair",
            repairable=True,
        ),
        _manifest_row(
            "case-sat",
            "SAT",
            "accept",
            {
                "constraints": ["x >= 0", "x <= 5", "y >= 0", "x + y == 4"],
                "task": "Classify the integer constraints as SAT or UNSAT.",
                "variables": ["x", "y"],
            },
            family="smt_constraints",
            perturbation_type="smt_sat_solving",
        ),
    ]


def _write_sources(root: Path, rows: list[dict[str, Any]] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake verifier recovery\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "EBT/ARM sidecar citation watch\n", encoding="utf-8"
    )
    _write_jsonl(root, mod.MANIFEST_REL_PATH, rows or _fixture_rows())
    _write_json(
        root,
        mod.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "usable_fixture_count": len(rows or _fixture_rows()),
            "stratified_eval_manifest_path": mod.MANIFEST_REL_PATH.as_posix(),
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3104_REL_PATH,
        {
            "artifact": "experiment_3104_ebt_arm_sidecar_pipeline_boundary_v2",
            "sidecar_boundary_v2_ready": True,
            "no_live_model_integration_claim": True,
            "honest_verdict": "complete: sidecar_boundary_v2_ready=true",
        },
    )
    (root / mod.SIDECAR_SCHEMA_REL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.SIDECAR_SCHEMA_REL_PATH).write_text('{"schema": "unit"}\n', encoding="utf-8")
    (root / mod.SIDECAR_SCORER_REL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.SIDECAR_SCORER_REL_PATH).write_text("# unit scorer path\n", encoding="utf-8")


def test_req_verify_3117_spec_anchor_exists() -> None:
    """REQ-VERIFY-3117: OpenSpec declares the diagnostic before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3117" in spec
    assert "SCENARIO-VERIFY-3117" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "score_correlation_summary" in spec
    assert "no_speedup_claim" in spec


def test_scenario_verify_3117_builds_diagnostic_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3117: sidecar scores are diagnostic-only exact-fixture evidence."""

    _write_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        min_exact_count=3,
        started_s=10.0,
        now_s=11.5,
        tests_run=["focused-unit"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["sidecar_score_correlation_boundary_v3_ready"] is True
    assert artifact["exact_fixture_count"] == 4
    assert artifact["outcome_counts"] == {"accepted": 2, "rejected": 1, "repairable": 1}
    assert (
        artifact["score_correlation_summary"]["label_blind_feature_energy"][
            "spearman_reject_or_repair"
        ]
        > 0.0
    )
    assert artifact["score_correlation_summary"]["label_blind_feature_energy"][
        "separability_vs_accept"
    ] == pytest.approx(1.0)
    assert (
        artifact["score_correlation_summary"]["replay_total_energy_label_aware"][
            "uses_exact_label_reference"
        ]
        is True
    )
    assert sum(row["count"] for row in artifact["calibration_summary"]["bins"]) == 4
    assert artifact["calibration_summary"]["score_name"] == "label_blind_feature_energy"
    assert artifact["failure_cases"] == []
    assert artifact["no_live_model_integration_claim"] is True
    assert artifact["no_weight_update_claim"] is True
    assert artifact["no_speedup_claim"] is True
    assert artifact["tests_run"] == ["focused-unit"]
    assert artifact["inference_substrate"]["live_model_inference"] is False
    assert artifact["inference_substrate"]["model_weights_loaded"] is False
    assert artifact["inference_substrate"]["generation_performed"] is False
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        min_exact_count=3,
        started_s=20.0,
        now_s=21.0,
        tests_run=["relative-output"],
    )
    assert relative_output == tmp_path / mod.OUTPUT_REL_PATH


def test_req_verify_3117_fail_closed_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-3117: missing or undercovered inputs cannot claim readiness."""

    missing = mod.build_artifact(tmp_path, min_exact_count=3, tests_run=["missing"])
    assert missing["sidecar_score_correlation_boundary_v3_ready"] is False
    assert missing["honest_verdict"].startswith("blocked_missing_inputs")
    mod.validate_artifact(missing)

    _write_sources(tmp_path, rows=_fixture_rows()[:2])
    tiny = mod.build_artifact(tmp_path, min_exact_count=3, tests_run=["tiny"])
    assert tiny["sidecar_score_correlation_boundary_v3_ready"] is False
    assert tiny["honest_verdict"].startswith("blocked_insufficient_exact_fixtures")
    mod.validate_artifact(tiny)

    ready = mod.build_artifact(tmp_path, min_exact_count=2, tests_run=["ready"])
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="no_live_model_integration_claim"):
        mod.validate_artifact(ready | {"no_live_model_integration_claim": False})
    with pytest.raises(ValueError, match="no_weight_update_claim"):
        mod.validate_artifact(ready | {"no_weight_update_claim": False})
    with pytest.raises(ValueError, match="no_speedup_claim"):
        mod.validate_artifact(ready | {"no_speedup_claim": False})
    with pytest.raises(ValueError, match="live_model_inference"):
        mod.validate_artifact(
            ready
            | {"inference_substrate": ready["inference_substrate"] | {"live_model_inference": True}}
        )
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(ready | {"honest_verdict": "ready"})


def test_req_verify_3117_helpers_and_failure_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-3117: parsing, replay rows, and metrics stay deterministic."""

    malformed = '\nnot-json\n{"source_fixture_id": "ok"}\n[]\n'
    assert mod.read_jsonl_rows_from_text(malformed) == [{"source_fixture_id": "ok"}]
    assert mod.read_jsonl_rows(tmp_path / "missing.jsonl") == []
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert (
        mod.relative_path(tmp_path, tmp_path / "nested" / "artifact.json") == "nested/artifact.json"
    )
    assert mod.relative_path(tmp_path, Path("/outside/artifact.json")) == "/outside/artifact.json"
    assert mod.rate(1, 0) == 0.0
    assert mod.pearson([1.0, 2.0], [5.0, 5.0]) == 0.0
    assert mod.spearman([1.0, 2.0, 3.0], [0.0, 1.0, 1.0]) > 0.0

    scored = mod.score_fixture_rows(_fixture_rows(), root=tmp_path)
    assert [row["exact_outcome"] for row in scored] == [
        "accepted",
        "rejected",
        "repairable",
        "accepted",
    ]
    assert all(
        row["synthetic_sidecar_record"]["record_id"].startswith("exp3117-") for row in scored
    )
    assert all(row["label_blind_feature_energy"] <= row["replay_total_energy"] for row in scored)
    assert mod.calibration_bins(scored, "label_blind_feature_energy", bin_count=2)[0]["count"] == 2
    assert mod.separability(scored, "label_blind_feature_energy") == pytest.approx(1.0)

    broken = scored[0] | {
        "sidecar_action": "reject",
        "label_blind_feature_energy": 100.0,
        "replay_total_energy": 100.0,
    }
    failures = mod.failure_cases([broken, *scored[1:]], "label_blind_feature_energy", limit=3)
    assert failures[0]["fixture_id"] == "case-accept"
    assert failures[0]["reason"] == "sidecar_action_mismatch"


def test_req_verify_3117_defensive_parser_edges() -> None:
    """REQ-VERIFY-3117: defensive parser branches remain explicit and covered."""

    assert mod.label_blind_violation(
        {"perturbation_type": "numeric_bound_repair", "leakage_safe_prompt_payload": {}}
    ) == (1.0, "numeric_assignment_missing")
    assert mod.label_blind_violation(
        {"leakage_safe_prompt_payload": {"candidate_assertion": "assert 1 == 1"}}
    ) == (0.0, "arithmetic_assertion_passed")
    assert mod.label_blind_violation(
        {"leakage_safe_prompt_payload": {"candidate": '{"ok": true}'}}
    ) == (0.0, "json_candidate_valid")
    assert mod.label_blind_violation({"leakage_safe_prompt_payload": {}}) == (
        0.0,
        "no_label_blind_violation_detected",
    )

    assert mod.arithmetic_violation({"candidate_assertion": "x = 1"}) == (
        1.0,
        "unsupported_arithmetic_assertion_shape",
    )
    assert mod.arithmetic_violation({"candidate_assertion": "assert 1 < 2"}) == (
        1.0,
        "unsupported_arithmetic_comparison",
    )
    assert mod.arithmetic_violation({"candidate_assertion": "assert (1 / 0) == 0"}) == (
        1.0,
        "arithmetic_assertion_parse_failed",
    )
    assert mod.arithmetic_violation(
        {"candidate_assertion": "assert (((-2 * 3) / 2) - 1) == -4"}
    ) == (
        0.0,
        "arithmetic_assertion_passed",
    )
    with pytest.raises(ValueError, match="unsupported arithmetic node"):
        mod._eval_arithmetic_ast(ast.parse("2 ** 3", mode="eval").body)

    assert mod.smt_satisfiability_violation({"constraints": ["x >= 3", "x <= 1"]}) == (
        3.0,
        "smt_bounds_contradiction_detected",
    )
    assert mod.json_repair_violation({}) == (1.0, "json_candidate_missing")
    assert mod.json_repair_violation(
        {"candidate": '{"mode": "bounded"}', "required_fields": ["limit"]}
    ) == (
        1.0,
        "json_required_fields_missing",
    )

    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {"x": 0}, "constraints": ["x >= 2"]}
    ) == (
        1.02,
        "numeric_assignment_violates_constraints",
    )
    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {"x": 3}, "constraints": ["x <= 2"]}
    ) == (
        1.01,
        "numeric_assignment_violates_constraints",
    )
    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {"x": 3}, "constraints": ["x == 2"]}
    ) == (
        1.01,
        "numeric_assignment_violates_constraints",
    )
    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {}, "constraints": ["x >= 2"]}
    ) == (
        1.01,
        "numeric_assignment_violates_constraints",
    )
    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {"x": 3}, "constraints": ["x + y == 4"]}
    ) == (1.01, "numeric_assignment_violates_constraints")
    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {"x": 3, "y": 1}, "constraints": ["x + y == 5"]}
    ) == (1.01, "numeric_assignment_violates_constraints")
    assert mod.numeric_assignment_violation(
        {"candidate_assignment": {"x": 3, "y": 1}, "constraints": ["x + y == 4"]}
    ) == (0.0, "numeric_assignment_satisfies_constraints")

    assert mod.expected_action({"expected_answer": "SAT"}) == "accept"
    assert mod.expected_action({"expected_answer": "UNSAT"}) == "reject"
    assert mod.expected_action({"expected_answer": "UNKNOWN"}) == "abstain"
    assert mod.finite_correlation_summary({"bad": []}) is False
    assert (
        mod.finite_correlation_summary(
            {
                "bad": {
                    "spearman_reject_or_repair": float("nan"),
                    "spearman_outcome_ordinal": 0.0,
                    "pearson_reject_or_repair": 0.0,
                    "separability_vs_accept": 0.0,
                }
            }
        )
        is False
    )

    tie_rows = [
        {"fixture_id": "a", "exact_outcome": "accepted", "score": 1.0},
        {"fixture_id": "b", "exact_outcome": "rejected", "score": 1.0},
    ]
    assert mod.separability(tie_rows, "score") == pytest.approx(0.5)
    assert mod.candidate_text({"other": "value"}) == '{"other": "value"}'
    assert mod.honest_verdict({}) == "blocked_incomplete_diagnostics: readiness checks missing"
    assert (
        mod.honest_verdict(
            {
                "sidecar_score_correlation_boundary_v3_ready": False,
                "readiness_checks": {
                    "exp3097_protocol_ready": True,
                    "required_sources_present": True,
                    "minimum_exact_fixture_count_met": True,
                    "accepted_cases_present": True,
                    "rejected_cases_present": True,
                    "repairable_cases_present": True,
                },
            }
        )
        == "blocked_incomplete_diagnostics: finite metrics or calibration accounting missing"
    )

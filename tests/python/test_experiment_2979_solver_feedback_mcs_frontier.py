"""Tests for Exp 2979 solver-feedback MCS/MUS frontier.

Spec refs: REQ-VERIFY-2979, SCENARIO-VERIFY-2979.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import logic_frontier_materializer as exp2966
from carnot.reporting import solver_feedback_mcs_frontier_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_FIELDS = {
    "honest_verdict",
    "mcs_feedback_schema_ready",
    "frontier_upgrade_ready",
    "reference_z3_execution_rate",
    "reference_solver_verified_accuracy",
    "feedback_schema",
    "frontier_items",
    "failure_categories_from_exp2967",
    "mcs_mus_examples",
    "exp2980_input_path",
    "inference_substrate",
    "duration_s",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sources(root: Path) -> tuple[Path, Path]:
    manifest_path = root / "data" / "research" / exp2966.MANIFEST_FILENAME
    records = [item.to_manifest_record() for item in exp2966.build_logic_frontier_items()]
    _write_json(
        manifest_path,
        {
            "schema": "carnot.logic_frontier.v1",
            "items": records,
            "model_specs_for_downstream_live_use": list(exp2966.MODEL_SPECS),
        },
    )

    exp2966_path = root / "results" / exp.EXP2966_FILENAME
    _write_json(
        exp2966_path,
        {
            "honest_verdict": "complete: source fixture",
            "logic_frontier_materialized": True,
            "manifest_path": str(manifest_path),
            "n_items": len(records),
            "skill_labels": list(exp2966.SKILL_LABELS),
            "reference_z3_execution_rate": 1.0,
            "reference_solver_accuracy": 1.0,
        },
    )

    exp2967_path = root / "results" / exp.EXP2967_FILENAME
    _write_json(
        exp2967_path,
        {
            "honest_verdict": "complete: source fixture",
            "failure_categories": {
                "solver_verified_correct": 1,
                "unparseable": 1,
                "wrong_answer": 1,
                "wrong_formula": 1,
                "z3_exception": 1,
            },
            "per_item_results": [
                {
                    "item_id": "lf-2966-001",
                    "failure_category": "solver_verified_correct",
                    "parseable": True,
                    "z3_executed": True,
                    "parse_error": None,
                    "skill_labels": ["symbolization", "quantifier handling", "validity"],
                    "structured_proposal": {"assertions": ["(assert true)"]},
                    "z3_result": {"actual_solver_status": "unsat", "z3_error": None},
                },
                {
                    "item_id": "lf-2966-003",
                    "failure_category": "unparseable",
                    "parseable": False,
                    "z3_executed": False,
                    "parse_error": "no_json_object",
                    "skill_labels": ["symbolization", "quantifier handling", "satisfiability"],
                    "structured_proposal": None,
                    "z3_result": {"z3_error": "no_json_object"},
                },
                {
                    "item_id": "lf-2966-007",
                    "failure_category": "wrong_formula",
                    "parseable": True,
                    "z3_executed": True,
                    "parse_error": None,
                    "skill_labels": ["symbolization", "satisfiability"],
                    "structured_proposal": {"assertions": ["(assert true)"]},
                    "z3_result": {"actual_solver_status": "sat", "z3_error": None},
                },
                {
                    "item_id": "lf-2966-013",
                    "failure_category": "wrong_answer",
                    "parseable": True,
                    "z3_executed": True,
                    "parse_error": None,
                    "skill_labels": ["answer extraction"],
                    "structured_proposal": {"answer_extraction": {"symbols": ["answer"]}},
                    "z3_result": {"actual_solver_status": "sat", "actual_answer_values": {"answer": "8"}},
                },
                {
                    "item_id": "lf-2966-014",
                    "failure_category": "z3_exception",
                    "parseable": True,
                    "z3_executed": False,
                    "parse_error": None,
                    "skill_labels": ["answer extraction", "satisfiability"],
                    "structured_proposal": {"assertions": ["(assert"]},
                    "z3_result": {"z3_error": "Z3Exception: b'parser error'"},
                },
            ],
        },
    )
    return exp2966_path, exp2967_path


def _config(tmp_path: Path) -> exp.SolverFeedbackConfig:
    return exp.SolverFeedbackConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 16.0,
    )


def test_req_verify_2979_spec_anchor_exists() -> None:
    """REQ-VERIFY-2979: the solver-feedback frontier is OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-2979" in spec
    assert "SCENARIO-VERIFY-2979" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="deterministic_z3_and_artifact_generation"' in spec
    assert "solver_feedback" in spec


def test_scenario_verify_2979_builds_consumable_feedback_frontier(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2979: Exp 2967 failures become structured feedback rows."""
    _write_sources(tmp_path)

    artifact = exp.build_artifact(_config(tmp_path))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["mcs_feedback_schema_ready"] is True
    assert artifact["frontier_upgrade_ready"] is True
    assert artifact["reference_z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["reference_solver_verified_accuracy"] == pytest.approx(1.0)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(6.0)

    schema = artifact["feedback_schema"]
    assert schema["object"] == "solver_feedback"
    assert set(exp.REQUIRED_FEEDBACK_FIELDS) <= set(schema["fields"])
    assert schema["ready_for_exp2980"] is True

    frontier_items = artifact["frontier_items"]
    assert {item["skill_label"] for item in frontier_items} == set(exp2966.SKILL_LABELS)
    assert all(set(exp.REQUIRED_FEEDBACK_FIELDS) <= set(item["solver_feedback"]) for item in frontier_items)
    assert all(item["reference_z3_result"]["z3_executed"] is True for item in frontier_items)
    assert all(item["accepted_reference_formalization"]["format"] == "smt2" for item in frontier_items)
    assert any(item["solver_feedback"]["parse_error"] for item in frontier_items)
    assert any(item["solver_feedback"]["z3_exception"] for item in frontier_items)
    assert any(item["solver_feedback"]["model_counterexample"] for item in frontier_items)
    assert any(item["solver_feedback"]["unsat_core_or_mus"] for item in frontier_items)

    assert artifact["failure_categories_from_exp2967"]["parse_errors"]["unparseable"] == 1
    assert artifact["failure_categories_from_exp2967"]["execution_errors"]["z3_exception"] == 1
    assert artifact["failure_categories_from_exp2967"]["solver_wrong"]["wrong_answer"] == 1
    assert artifact["failure_categories_from_exp2967"]["solver_wrong"]["wrong_formula"] == 1
    assert len(artifact["mcs_mus_examples"]) >= 2
    assert all(example["minimal_correction_hint"] for example in artifact["mcs_mus_examples"])
    assert artifact["exp2980_input_path"] == str(tmp_path / "results" / exp.OUTPUT_FILENAME)


def test_req_verify_2979_write_and_validate_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-2979: the terminal artifact is stable and validates."""
    _write_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: x"})
    with pytest.raises(ValueError, match="deterministic_z3_and_artifact_generation"):
        exp.validate_artifact(artifact | {"inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="schema ready requires all feedback fields"):
        broken_schema = dict(artifact["feedback_schema"])
        broken_schema["fields"] = {"parse_error": {"type": "string"}}
        exp.validate_artifact(artifact | {"feedback_schema": broken_schema})
    with pytest.raises(ValueError, match="full reference Z3 execution"):
        exp.validate_artifact(artifact | {"reference_z3_execution_rate": 0.5})
    with pytest.raises(ValueError, match="perfect reference solver accuracy"):
        exp.validate_artifact(artifact | {"reference_solver_verified_accuracy": 0.5})


def test_req_verify_2979_defensive_edges_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-2979: empty and no-model diagnostic edges fail closed."""
    _write_sources(tmp_path)
    import z3

    manifest = json.loads(
        (tmp_path / "data" / "research" / exp2966.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    unsat_item = next(
        item for item in manifest["items"] if item["expected_solver_status"] == "unsat"
    )

    assert exp._failure_row_for_skill([], "symbolization") is None
    assert exp._reference_rates([]) == (0.0, 0.0)
    assert exp._model_counterexample(unsat_item, z3) is None
    assert exp._minimal_unsat_subsets([("a", z3.Bool("a"))], z3) == []
    assert exp._z3_error(None) is None


def test_req_verify_2979_blocks_with_environment_diagnostics_when_z3_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2979: missing Z3 produces an exact blocked diagnostic artifact."""
    artifact = exp.build_artifact(_config(tmp_path), z3_module=None)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == "blocked_dependency: z3 import failed"
    assert artifact["mcs_feedback_schema_ready"] is False
    assert artifact["frontier_upgrade_ready"] is False
    assert artifact["reference_z3_execution_rate"] == 0.0
    assert artifact["reference_solver_verified_accuracy"] == 0.0
    assert artifact["frontier_items"] == []
    assert artifact["mcs_mus_examples"] == []
    assert artifact["environment_diagnostics"]["z3_import_ok"] is False
    assert artifact["environment_diagnostics"]["python_executable"]
    assert artifact["preconditions_checked"][0]["name"] == "z3_import"
    assert artifact["preconditions_checked"][0]["ok"] is False

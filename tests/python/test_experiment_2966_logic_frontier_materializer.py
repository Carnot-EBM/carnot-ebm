"""Tests for Exp 2966 skill-labeled exact logic frontier materializer.

Spec: REQ-BENCH-2966, SCENARIO-BENCH-2966.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import logic_frontier_materializer as exp


def test_req_bench_2966_spec_anchor_exists() -> None:
    """REQ-BENCH-2966: the exact logic frontier is OpenSpec anchored."""

    spec = Path("openspec/capabilities/benchmarks/spec.md").read_text(encoding="utf-8")

    assert "REQ-BENCH-2966" in spec
    assert "SCENARIO-BENCH-2966" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="deterministic_wiring"' in spec
    assert "MUST NOT invoke a live model" in spec


def test_req_bench_2966_items_cover_all_skills_and_execute_reference_z3() -> None:
    """REQ-BENCH-2966: all frontier labels are explicit and Z3-checkable."""

    items = exp.build_logic_frontier_items()
    results = exp.execute_reference_formalizations(items)
    skill_counts = exp.skill_label_counts(items)

    assert 20 <= len(items) <= 30
    assert len({item.item_id for item in items}) == len(items)
    assert set(skill_counts) == set(exp.SKILL_LABELS)
    assert all(count > 0 for count in skill_counts.values())
    assert all(item.expected_solver_status in {"sat", "unsat"} for item in items)
    assert all("reference_z3" in item.to_manifest_record() for item in items)
    assert all(result["z3_executed"] is True for result in results)
    assert all(result["solver_status_matches_expected"] is True for result in results)
    assert all(result["answer_extraction_matches_expected"] is True for result in results)
    assert {result["actual_solver_status"] for result in results} == {"sat", "unsat"}
    assert any("quantifier handling" in item.skill_labels for item in items)
    assert any("countermodel construction" in item.skill_labels for item in items)
    assert any(result["expected_answer_values"] for result in results)


def test_scenario_bench_2966_runner_writes_manifest_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-2966: successful materialization writes stable evidence."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / "data" / "research" / exp.MANIFEST_FILENAME,
            started_at=10.0,
            clock=lambda: 13.25,
        )
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    manifest_path = Path(artifact["manifest_path"])
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "complete: exact skill-labeled logic frontier materialized"
    assert artifact["z3_import_ok"] is True
    assert artifact["logic_frontier_materialized"] is True
    assert artifact["n_items"] == len(manifest["items"]) == 24
    assert artifact["skill_labels"] == list(exp.SKILL_LABELS)
    assert artifact["reference_formalizations_executed"] == 24
    assert artifact["reference_z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["reference_solver_accuracy"] == pytest.approx(1.0)
    assert artifact["manifest_sha256"] == hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    assert artifact["model_specs_for_downstream_live_use"] == list(exp.MODEL_SPECS)
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert any(
        row["name"] == "z3_import" and row["ok"] for row in artifact["preconditions_checked"]
    )
    assert manifest["schema"] == "carnot.logic_frontier.v1"
    assert manifest["items"][0]["reference_execution"]["z3_executed"] is True
    assert manifest["items"][0]["reference_z3"]["format"] == "smt2"
    assert manifest["model_specs_for_downstream_live_use"] == list(exp.MODEL_SPECS)


def test_req_bench_2966_blocks_honestly_when_z3_missing(tmp_path: Path) -> None:
    """REQ-BENCH-2966: missing Z3 writes a blocked artifact with preconditions."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "blocked.json",
            manifest_path=tmp_path / "manifest.json",
            started_at=2.0,
            clock=lambda: 2.5,
        ),
        z3_module=None,
    )

    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_dependency: z3 import failed"
    assert artifact["preconditions_checked"][0]["name"] == "z3_import"
    assert artifact["preconditions_checked"][0]["ok"] is False
    assert artifact["z3_import_ok"] is False
    assert artifact["logic_frontier_materialized"] is False
    assert artifact["n_items"] == 0
    assert artifact["reference_formalizations_executed"] == 0
    assert artifact["reference_z3_execution_rate"] == 0.0
    assert artifact["reference_solver_accuracy"] == 0.0
    assert artifact["manifest_path"] == ""
    assert artifact["manifest_sha256"] == ""
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert not (tmp_path / "manifest.json").exists()


def test_req_bench_2966_validation_and_error_edges(tmp_path: Path) -> None:
    """REQ-BENCH-2966: validation rejects malformed exact-frontier artifacts."""

    items = exp.build_logic_frontier_items()
    bad_item = exp.LogicFrontierItem(
        item_id="bad-parse",
        prompt="Malformed SMT-LIB should fail closed.",
        expected_label="unsatisfiable",
        check_kind="satisfiability",
        expected_solver_status="unsat",
        skill_labels=("symbolization", "satisfiability"),
        reference_smt2="(assert",
    )
    bad_result = exp.execute_reference_formalization(bad_item)
    zero_metrics = exp.aggregate_execution_metrics([])

    assert bad_result["z3_executed"] is False
    assert bad_result["solver_status_matches_expected"] is False
    assert bad_result["answer_extraction_matches_expected"] is False
    unavailable = exp.execute_reference_formalization(items[0], z3_module=None)
    assert unavailable["z3_error"] == "z3_unavailable"

    unsat_answer_item = exp.LogicFrontierItem(
        item_id="bad-answer",
        prompt="Unsat item cannot expose an answer value.",
        expected_label="answer=1",
        check_kind="answer_extraction",
        expected_solver_status="unsat",
        skill_labels=("symbolization", "satisfiability", "answer extraction"),
        reference_smt2="(declare-const answer Int) (assert (= answer 1)) (assert (= answer 2))",
        expected_answer_values={"answer": "1"},
    )
    unsat_answer = exp.execute_reference_formalization(unsat_answer_item)
    assert unsat_answer["z3_executed"] is True
    assert unsat_answer["answer_extraction_matches_expected"] is False

    assert zero_metrics == {
        "reference_formalizations_executed": 0,
        "reference_z3_execution_rate": 0.0,
        "reference_solver_accuracy": 0.0,
    }
    with pytest.raises(ValueError, match="unknown skill label"):
        exp.LogicFrontierItem(
            item_id="bad-skill",
            prompt="Bad skill.",
            expected_label="satisfiable",
            check_kind="satisfiability",
            expected_solver_status="sat",
            skill_labels=("not-a-skill",),
            reference_smt2="(assert true)",
        ).to_manifest_record()
    with pytest.raises(ValueError, match="expected_solver_status"):
        exp.LogicFrontierItem(
            item_id="bad-status",
            prompt="Bad status.",
            expected_label="satisfiable",
            check_kind="satisfiability",
            expected_solver_status="unknown",
            skill_labels=("symbolization",),
            reference_smt2="(assert true)",
        ).to_manifest_record()

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "result.json",
            manifest_path=tmp_path / "manifest.json",
            started_at=5.0,
            clock=lambda: 6.0,
        )
    )
    exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: x"})
    with pytest.raises(ValueError, match="must be deterministic_wiring"):
        exp.validate_artifact(artifact | {"inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="missing mandated downstream model specs"):
        exp.validate_artifact(artifact | {"model_specs_for_downstream_live_use": []})
    with pytest.raises(ValueError, match="materialized artifact requires all references"):
        exp.validate_artifact(artifact | {"reference_formalizations_executed": len(items) - 1})
    with pytest.raises(ValueError, match="20-30 items"):
        exp.validate_artifact(artifact | {"n_items": 19, "reference_formalizations_executed": 19})
    with pytest.raises(ValueError, match="full Z3 execution"):
        exp.validate_artifact(artifact | {"reference_z3_execution_rate": 0.5})
    with pytest.raises(ValueError, match="exact reference solver accuracy"):
        exp.validate_artifact(artifact | {"reference_solver_accuracy": 0.5})

"""Tests for Exp 3044 SMT/SAT validator-tree exactness upgrade.

Spec refs: REQ-VERIFY-3044, SCENARIO-VERIFY-3044.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import smt_sat_validator_tree_exactness_upgrade_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1.py"
)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        evidence_path=tmp_path / exp.EVIDENCE_REL_PATH,
        transcript_path=tmp_path / exp.TRANSCRIPT_REL_PATH,
        started_at=10.0,
        clock=lambda: 12.0,
        tests_run=("pytest focused",),
    )


def test_req_verify_3044_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3044: the exactness upgrade is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3044" in spec
    assert "SCENARIO-VERIFY-3044" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "validator_tree_exactness_ready" in spec
    assert "correction_sets" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3044_fixture_rows_are_distinguishable() -> None:
    """SCENARIO-VERIFY-3044: exact, unresolved, irrelevant, and fallback rows separate."""

    rows = exp.evaluate_fixtures(exp.build_validator_fixtures())
    by_id = {row["item_id"]: row for row in rows}

    assert by_id["sat-sum-ok"]["classification"] == "verified"
    assert by_id["sat-sum-ok"]["solver_status"] == "sat"
    assert by_id["sat-sum-bad"]["classification"] == "correction_set"
    assert by_id["sat-sum-bad"]["solver_status"] == "unsat"
    assert by_id["sat-sum-bad"]["correction_set"] == {
        "candidate_fields": ["total"],
        "minimal_assignment_ids": ["candidate.total"],
        "suggested_assignments": {"total": 5},
        "failing_constraint_ids": ["sum_relation"],
    }
    assert by_id["semantic-boundary"]["classification"] == "irrelevant"
    assert by_id["quantifier-text"]["classification"] == "unresolved"
    assert by_id["enumerator-fallback"]["classification"] == "fallback_only"


def test_scenario_verify_3044_runner_writes_artifact_and_evidence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3044: run writes terminal artifact and exact row evidence."""

    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    evidence_rows = exp.load_jsonl(tmp_path / artifact["exact_evidence_path"])

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["validator_tree_exactness_ready"] is True
    assert artifact["exact_validator_path"] == exp.EXACT_VALIDATOR_PATH.as_posix()
    assert artifact["exact_evidence_path"] == exp.EVIDENCE_REL_PATH.as_posix()
    assert artifact["verified_count"] == 1
    assert artifact["correction_set_count"] == 1
    assert artifact["irrelevant_count"] == 1
    assert artifact["unresolved_count"] == 1
    assert artifact["fallback_only_count"] == 1
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert artifact["model_specs"] == []
    assert artifact["inference_substrate"] == {
        "mode": "deterministic_z3_cpu_validator_tree",
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "z3_solver_used": True,
        "hardware_acceleration": False,
    }
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(evidence_rows) == len(exp.build_validator_fixtures())
    assert (tmp_path / exp.TRANSCRIPT_REL_PATH).is_file()

    exp.validate_artifact(artifact)


def test_req_verify_3044_validation_and_blocked_solver_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-3044: readiness fails closed without exact solver evidence."""

    artifact = exp.run_experiment(_config(tmp_path))
    exp.validate_artifact(artifact)

    blocked = exp.run_experiment(_config(tmp_path), z3_module=None)
    assert blocked["validator_tree_exactness_ready"] is False
    assert blocked["verified_count"] == 0
    assert blocked["correction_sets"] == []
    assert blocked["honest_verdict"].startswith("blocked_")
    exp.validate_artifact(blocked)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(artifact | {"model_specs": ["unexpected-live-model"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": {"live_llm_inference": True}})
    with pytest.raises(ValueError, match="blocked_ prefix"):
        exp.validate_artifact(blocked | {"honest_verdict": "waiting"})
    with pytest.raises(ValueError, match="verified_count"):
        exp.validate_artifact(artifact | {"verified_count": 0})
    with pytest.raises(ValueError, match="correction_sets"):
        exp.validate_artifact(artifact | {"correction_sets": []})
    with pytest.raises(ValueError, match="fallback_only_count"):
        exp.validate_artifact(artifact | {"fallback_only_count": 0})
    with pytest.raises(ValueError, match="unresolved_count"):
        exp.validate_artifact(artifact | {"unresolved_count": 0})
    with pytest.raises(ValueError, match="exact_validator_path"):
        exp.validate_artifact(artifact | {"exact_validator_present": False})
    with pytest.raises(ValueError, match="exact evidence file"):
        exp.validate_artifact(artifact | {"exact_evidence_present": False})
    with pytest.raises(ValueError, match="exact evidence"):
        exp.validate_artifact(artifact | {"exact_evidence_path": "missing.jsonl"})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong prefix"})
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")

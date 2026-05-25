"""Tests for Exp 3073 EBT/ARM-EBM adapter feasibility audit.

Spec refs: REQ-VERIFY-3073, SCENARIO-VERIFY-3073.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import ebt_arm_ebm_adapter_feasibility_audit_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


class FakeClock:
    def __init__(self) -> None:
        self.value = 40.0

    def __call__(self) -> float:
        self.value += 0.5
        return self.value


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=REPO_ROOT,
        output_path=tmp_path / exp.ARTIFACT_FILENAME,
        tests_run=("pytest focused",),
        clock=FakeClock(),
    )


def test_req_verify_3073_spec_anchor_exists() -> None:
    """REQ-VERIFY-3073: the audit is anchored in the verification spec."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3073" in spec
    assert "SCENARIO-VERIFY-3073" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "ebt_arm_adapter_feasibility_ready" in spec
    assert "adapter_implementation_claimed=false" in spec


def test_scenario_verify_3073_builds_consumable_feasibility_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3073: local surfaces and prerequisites are explicit."""

    artifact = exp.build_artifact(_config(tmp_path), duration_s=0.125)

    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["ebt_arm_adapter_feasibility_ready"] is True
    assert artifact["ebt_arm_adapter_feasible"] is True
    assert artifact["bounded_theory_context_only"] is False
    assert artifact["adapter_implementation_claimed"] is False
    assert artifact["inference_substrate"]["live_model_inference"] is False
    assert artifact["inference_substrate"]["model_weights_loaded"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_or_checks_run"] == ["pytest focused"]

    surface_names = {surface["surface"] for surface in artifact["adapter_surface"]}
    assert {
        "core_energy_protocol",
        "verify_repair_pipeline",
        "arm_logprob_energy_bridge",
        "ebt_reasoning_bridge",
    } <= surface_names
    assert all(surface["local_evidence_present"] for surface in artifact["adapter_surface"])
    assert all((REPO_ROOT / surface["path"]).exists() for surface in artifact["adapter_surface"])

    prerequisite_gates = {row["gate"] for row in artifact["required_prerequisites"]}
    assert prerequisite_gates == {
        "data_shape",
        "energy_objective",
        "verifier_interface",
        "sampling_path",
        "evaluation_metric",
        "rollback_claim_boundaries",
    }
    assert artifact["near_term_adapter_opportunities"]
    assert artifact["paper_context_only_references"]
    assert artifact["blockers"]
    assert "offline" in artifact["recommended_next_experiment"].lower()


def test_req_verify_3073_source_refs_and_claim_boundaries_are_traceable(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3073: literature references cannot masquerade as local code."""

    artifact = exp.build_artifact(_config(tmp_path), duration_s=0.25)

    ref_ids = {ref["id"] for ref in artifact["source_refs"]}
    assert {"arXiv:2507.02092", "arXiv:2512.15605"} <= ref_ids
    assert "python/carnot/inference/arm_ebm_bridge.py" in {
        ref.get("path") for ref in artifact["source_refs"]
    }
    assert all(
        ref["claim_boundary"] == "context_only_not_local_implementation"
        for ref in artifact["paper_context_only_references"]
    )
    assert any(
        opportunity["first_test"].startswith("fixture")
        for opportunity in artifact["near_term_adapter_opportunities"]
    )
    assert "no live model inference" in artifact["methodology_note"].lower()


def test_req_verify_3073_writer_persists_terminal_json(tmp_path: Path) -> None:
    """REQ-VERIFY-3073: run_experiment writes the requested artifact."""

    config = _config(tmp_path)
    artifact = exp.run_experiment(config, write=True)
    saved = json.loads((tmp_path / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert artifact["duration_s"] == pytest.approx(0.5)
    assert artifact["schema"] == exp.SCHEMA
    exp.validate_artifact(artifact)


def test_req_verify_3073_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3073: validation blocks implementation and inference overclaims."""

    artifact = exp.build_artifact(_config(tmp_path), duration_s=0.25)
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="adapter_implementation_claimed"):
        exp.validate_artifact(artifact | {"adapter_implementation_claimed": True})
    with pytest.raises(ValueError, match="live_model_inference"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_model_inference": True}
            }
        )
    with pytest.raises(ValueError, match="required_prerequisites"):
        exp.validate_artifact(artifact | {"required_prerequisites": []})
    with pytest.raises(ValueError, match="bounded_theory_context_only"):
        exp.validate_artifact(
            artifact | {"ebt_arm_adapter_feasible": False, "bounded_theory_context_only": False}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})

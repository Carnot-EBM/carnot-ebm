"""Tests for Exp 5292 CPU p-bit guidance for CDCL on factor fixtures.

Spec refs: REQ-VERIFY-5292, SCENARIO-VERIFY-5292.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_verify_5292_spec_declares_cpu_cdcl_guidance_contract() -> None:
    """REQ-VERIFY-5292: OpenSpec anchors the CPU-only CDCL guidance artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5292") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5292",
        "SCENARIO-VERIFY-5292",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "simulated sampler guidance",
        "hardware_speedup_claimed.value=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5292_generates_cpu_simulated_assumptions_from_factor_fixture() -> None:
    """REQ-VERIFY-5292: assumptions are CPU simulated and anchored to Exp 5278."""

    instances = mod.build_factor_guidance_instances()
    by_class = {instance.instance_class: instance for instance in instances}
    assumption_sets = {
        instance.instance_class: mod.generate_assumptions(instance)
        for instance in instances
    }

    assert set(by_class) == {
        "aligned_factor_sat",
        "misleading_factor_sat",
        "neutral_factor_sat",
    }
    assert all(instance.source_fixture_id == "small_pair_sum" for instance in instances)
    assert assumption_sets["aligned_factor_sat"].literals == (3, 8)
    assert assumption_sets["misleading_factor_sat"].literals == (4, 7)
    assert assumption_sets["neutral_factor_sat"].literals == ()
    for assumptions in assumption_sets.values():
        assert assumptions.simulated_guidance is True
        assert assumptions.hardware_execution is False
        assert assumptions.method.startswith("cpu_simulated")


def test_scenario_verify_5292_bad_assumptions_cannot_force_wrong_result() -> None:
    """SCENARIO-VERIFY-5292: misleading assumptions are advisory only."""

    instance = next(
        row for row in mod.build_factor_guidance_instances()
        if row.instance_class == "misleading_factor_sat"
    )
    assumptions = mod.generate_assumptions(instance)
    row = mod.run_guided_instance(instance, assumptions)

    assert row["pure"]["status"] == "sat"
    assert row["guided"]["primary_status"] == "unsat"
    assert row["guided"]["final_status"] == "sat"
    assert row["guided"]["fallback_used"] is True
    assert row["guided"]["overwrite_count"] > 0
    assert row["correctness_preserved"] is True
    assert mod.verify_model(instance.clauses, row["guided"]["final_model"])


def test_req_verify_5292_benchmark_classifies_help_harm_and_neutral() -> None:
    """REQ-VERIFY-5292: distribution sensitivity is measured by class."""

    benchmark = mod.run_benchmark()
    gate = benchmark["instance_class_gate"]
    by_class = benchmark["savings_by_class"]

    assert gate["helps"] == ["aligned_factor_sat"]
    assert gate["harms"] == ["misleading_factor_sat"]
    assert gate["neutral"] == ["neutral_factor_sat"]
    assert gate["distribution_sensitivity_expected"] is True
    assert benchmark["correctness_preserved"] is True
    assert benchmark["fallback_overwrite_count"] > 0
    assert by_class["aligned_factor_sat"]["conflicts_saved"] > 0
    assert by_class["misleading_factor_sat"]["conflicts_saved"] < 0
    assert by_class["neutral_factor_sat"]["conflicts_saved"] == 0
    for row in benchmark["per_instance_results"]:
        assert set(row["pure"]["metrics"]) == {
            "conflicts",
            "propagations",
            "decisions",
            "restarts",
            "wall_clock_s",
        }
        assert set(row["guided"]["metrics"]) == {
            "conflicts",
            "propagations",
            "decisions",
            "restarts",
            "wall_clock_s",
        }


def test_scenario_verify_5292_correctness_guard_rejects_label_drift() -> None:
    """SCENARIO-VERIFY-5292: guided labels must match the unassumed authority."""

    instance = mod.build_factor_guidance_instances()[0]
    metrics = {
        "conflicts": 0,
        "propagations": 0,
        "decisions": 0,
        "restarts": 0,
        "wall_clock_s": 0.0,
    }
    pure_label_drift = mod.CdclRun(status="unsat", model=(), metrics=metrics)
    real_pure = mod.run_cdcl(instance.clauses, n_vars=instance.n_vars)
    guided_label_drift = mod.CdclRun(status="unsat", model=(), metrics=metrics)

    assert mod._correctness_preserved(instance, pure_label_drift, pure_label_drift) is False
    assert mod._correctness_preserved(instance, real_pure, guided_label_drift) is False


def test_req_verify_5292_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5292: artifact exposes wrapped fields and the bare gate bool."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.5,
        tests_run=[{"command": "unit exp5292", "outcome": "passed"}],
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "p-bit/CDCL" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["pbit_cdcl_guidance_positive"] is True
    assert artifact["pbit_cdcl_guidance_positive_principle"] == mod.PBIT_GUIDANCE_POSITIVE_PRINCIPLE
    assert _value(artifact, "correctness_preserved") is True
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert _value(artifact, "instance_class_gate")["harms"] == ["misleading_factor_sat"]
    assert artifact["tests_run"] == [{"command": "unit exp5292", "outcome": "passed"}]
    assert "REQ-VERIFY-5292" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5292_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5292: invalid speedup or correctness claims are rejected."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5292", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["hardware_speedup_claimed"] = mod.wrap_field("hardware_speedup_claimed", True)
    with pytest.raises(AssertionError, match="hardware"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["correctness_preserved"] = mod.wrap_field("correctness_preserved", False)
    with pytest.raises(AssertionError, match="correctness"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["pbit_cdcl_guidance_positive"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_verify_5292() -> None:
    """SCENARIO-VERIFY-5292: committed deliverable satisfies the V483 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["pbit_cdcl_guidance_positive"] is True
    assert _value(artifact, "correctness_preserved") is True
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert _value(artifact, "instance_class_gate")["distribution_sensitivity_expected"] is True

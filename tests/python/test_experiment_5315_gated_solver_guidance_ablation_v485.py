"""Tests for Exp 5315 gated solver-guidance ablation.

Spec refs: REQ-VERIFY-5315, SCENARIO-VERIFY-5315.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5315_gated_solver_guidance_ablation_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _class_payload(benchmark: dict[str, object], instance_class: str) -> dict[str, object]:
    harm = benchmark["per_class_harm"]
    assert isinstance(harm, dict)
    classes = harm["classes"]
    assert isinstance(classes, dict)
    payload = classes[instance_class]
    assert isinstance(payload, dict)
    return payload


def test_req_verify_5315_spec_declares_gated_ablation_contract() -> None:
    """REQ-VERIFY-5315: OpenSpec anchors the gated solver-guidance ablation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5315") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5315",
        "SCENARIO-VERIFY-5315",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "solver-only",
        "p-bit/CDCL gated hints",
        "smooth-relaxation gated hints",
        "combined gated hints",
        "`cdcl_fallback_authoritative=true`",
        "`no_hardware_speedup_claim=true`",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5315_builds_same_bounded_instance_set() -> None:
    """REQ-VERIFY-5315: every method receives the same deterministic fixtures."""

    instances = mod.build_ablation_instances()

    assert [instance.instance_class for instance in instances] == list(
        mod.EXPECTED_INSTANCE_CLASSES,
    )
    assert {instance.source_experiment for instance in instances} == {"exp5292", "exp5299"}
    assert {instance.source_fixture_id for instance in instances} == {"small_pair_sum"}
    assert all(instance.n_vars == 8 for instance in instances)
    assert all(instance.hardware_execution is False for instance in instances)


def test_scenario_verify_5315_confirms_upstream_gates_before_ablation() -> None:
    """SCENARIO-VERIFY-5315: smooth, p-bit, and LNS gates must all pass first."""

    gates = mod.confirm_upstream_gates()

    assert gates["constraint_lns_fixture_ready"] is True
    assert gates["pbit_gate_ready"] is True
    assert gates["smooth_relaxation_ready"] is True
    assert gates["smooth_relaxation_gate_passed"] is True
    assert gates["all_required_gates_confirmed"] is True
    assert str(mod.smooth.RESULT_RELATIVE_PATH) in gates["source_artifacts"]


def test_req_verify_5315_method_matrix_uses_same_rows_for_every_method() -> None:
    """REQ-VERIFY-5315: method matrix compares methods without distribution drift."""

    benchmark = mod.run_benchmark()
    matrix = benchmark["method_matrix"]
    assert isinstance(matrix, dict)
    methods = matrix["methods"]
    assert isinstance(methods, dict)
    expected_ids = tuple(matrix["instance_set"]["instance_ids"])

    assert tuple(methods) == mod.EXPECTED_METHODS
    for method_name in mod.EXPECTED_METHODS:
        method = methods[method_name]
        assert tuple(row["instance_id"] for row in method["per_instance"]) == expected_ids
        assert tuple(method["per_class"]) == mod.EXPECTED_INSTANCE_CLASSES
        assert method["aggregate"]["conflicts"] >= 0
        assert "wall_clock_s" in method["aggregate"]

    assert methods["solver_only"]["aggregate"]["conflicts"] == 24
    assert methods["lns"]["aggregate"]["conflicts"] == 18
    assert methods["pbit_cdcl_gated"]["aggregate"]["conflicts"] == 15
    assert methods["smooth_relaxation"]["aggregate"]["conflicts"] == 15
    assert methods["combined_hints"]["aggregate"]["conflicts"] == 15


def test_scenario_verify_5315_blocks_misleading_pbit_and_smooth_harm() -> None:
    """SCENARIO-VERIFY-5315: raw misleading hint harm is reported and gated."""

    benchmark = mod.run_benchmark()

    assert benchmark["solver_guidance_ablation_complete"] is True
    assert benchmark["misleading_class_blocked"] is True
    assert benchmark["cdcl_fallback_authoritative"] is True
    assert benchmark["aggregate_conflict_delta"] == 9
    for instance_class in mod.MISLEADING_CLASSES:
        payload = _class_payload(benchmark, instance_class)
        raw = payload["raw_hint_added_conflicts"]
        final = payload["final_added_conflicts"]

        assert raw["pbit_cdcl_ungated"] > 0 or raw["smooth_relaxation_ungated"] > 0
        assert final["pbit_cdcl_gated"] == 0
        assert final["smooth_relaxation"] == 0
        assert final["combined_hints"] == 0
        assert payload["misleading_class"] is True
        assert payload["blocked_by_gate"]


def test_req_verify_5315_symbolic_cdcl_authority_covers_all_final_methods() -> None:
    """REQ-VERIFY-5315: final labels and SAT models remain CDCL-valid."""

    benchmark = mod.run_benchmark()
    methods = benchmark["method_matrix"]["methods"]

    for method_name in mod.EXPECTED_METHODS:
        for row in methods[method_name]["per_instance"]:
            assert row["final_status"] == row["solver_only_status"]
            assert row["cdcl_validated"] is True
            assert row["conflicts"] >= 0


def test_req_verify_5315_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5315: artifact exposes principle fields and bare gates."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5315", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.25,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "gates_confirmed")["smooth_relaxation_gate_passed"] is True
    assert artifact["solver_guidance_ablation_complete"] is True
    assert artifact["aggregate_conflict_delta"] == 9
    assert artifact["misleading_class_blocked"] is True
    assert artifact["cdcl_fallback_authoritative"] is True
    assert artifact["no_hardware_speedup_claim"] is True
    assert _value(artifact, "tests_run") == tests_run
    assert "REQ-VERIFY-5315" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5315_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5315: invalid gates, authority, or speedup claims fail."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5315", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["cdcl_fallback_authoritative"] = False
    with pytest.raises(AssertionError, match="CDCL"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["no_hardware_speedup_claim"] = False
    with pytest.raises(AssertionError, match="hardware"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["solver_guidance_ablation_complete"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["misleading_class_blocked"] = False
    with pytest.raises(AssertionError, match="misleading"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "wrong")
    with pytest.raises(AssertionError, match="inference"):
        mod.validate_artifact(broken)


def test_req_verify_5315_honest_verdict_prefixes_cover_non_complete_outcomes() -> None:
    """REQ-VERIFY-5315: verdicts keep terminal prefixes for null/harm/blocked."""

    benchmark = mod.run_benchmark()
    null_case = copy.deepcopy(benchmark)
    null_case["aggregate_conflict_delta"] = 0
    harm_case = copy.deepcopy(benchmark)
    harm_case["misleading_class_blocked"] = False
    blocked_case = copy.deepcopy(benchmark)
    blocked_case["solver_guidance_ablation_complete"] = False

    assert mod.honest_verdict(benchmark).startswith("complete:")
    assert mod.honest_verdict(null_case).startswith("null:")
    assert mod.honest_verdict(harm_case).startswith("harmful_")
    assert mod.honest_verdict(blocked_case).startswith("blocked_")


def test_deliverable_file_validates_for_scenario_verify_5315() -> None:
    """SCENARIO-VERIFY-5315: checked-in deliverable satisfies the V485 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["solver_guidance_ablation_complete"] is True
    assert artifact["misleading_class_blocked"] is True
    assert artifact["cdcl_fallback_authoritative"] is True
    assert artifact["no_hardware_speedup_claim"] is True

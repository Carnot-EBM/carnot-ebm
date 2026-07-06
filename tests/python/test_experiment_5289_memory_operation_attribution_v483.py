"""Tests for Exp 5289 memory operation attribution.

Spec refs: REQ-LEARN-5289, SCENARIO-LEARN-5289.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot.pipeline import memory_operation_attribution as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5289_spec_declares_operation_attribution_contract() -> None:
    """REQ-LEARN-5289: OpenSpec anchors stage attribution and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5289") :]

    for marker in (
        "REQ-LEARN-5289",
        "SCENARIO-LEARN-5289",
        "extraction",
        "update/write",
        "retrieval/routing",
        "maintenance/eviction",
        "use/action",
        "rollback",
        "stale memory",
        "conflicting memory",
        "shuffled memory",
        "missing provenance",
        "harmful memory",
        mod.RESULT_RELATIVE_PATH,
        "aggregation_from_upstream_artifacts",
        "offline_deterministic_fixture_no_llm",
        "memory_attribution_ready",
        "unsafe_propagations",
        "local_maintenance_cost",
        "decision_impact_summary",
    ):
        assert marker in section

    normalized_section = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5289_builds_bounded_stage_control_cases() -> None:
    """REQ-LEARN-5289-1/3: bounded controls map to primary operation stages."""

    cases = mod.build_attribution_cases(root=REPO)
    by_control = {case["control_kind"]: case for case in cases}

    assert mod.OPERATION_STAGE_LABELS == {
        "extraction": "extraction",
        "update": "update/write",
        "routing": "retrieval/routing",
        "maintenance": "maintenance/eviction",
        "use": "use/action",
        "rollback": "rollback",
    }
    assert set(by_control) == {
        "valid_promoted_memory",
        "missing_provenance",
        "conflicting_memory",
        "shuffled_memory",
        "stale_memory",
        "poisoning_like",
        "harmful_memory",
    }
    assert by_control["missing_provenance"]["primary_stage"] == "extraction"
    assert by_control["conflicting_memory"]["primary_stage"] == "update"
    assert by_control["shuffled_memory"]["primary_stage"] == "routing"
    assert by_control["stale_memory"]["primary_stage"] == "maintenance"
    assert by_control["poisoning_like"]["primary_stage"] == "use"
    assert by_control["harmful_memory"]["primary_stage"] == "rollback"

    for control, case in by_control.items():
        assert case["primary_stage"] in mod.STAGE_KEYS
        assert case["operation_stage_label"] == mod.OPERATION_STAGE_LABELS[case["primary_stage"]]
        if control == "valid_promoted_memory":
            assert case["error_detected"] is False
            assert case["unsafe_propagated"] is False
            assert case["propagation_blocked"] is False
        else:
            assert case["error_detected"] is True
            assert case["propagation_blocked"] is True
            assert case["unsafe_propagated"] is False


def test_req_learn_5289_counts_coverage_cost_and_decision_impact() -> None:
    """REQ-LEARN-5289-2/4/5: metrics expose stage errors and safe impact."""

    cases = mod.build_attribution_cases(root=REPO)
    summary = mod.attribute_memory_operations(cases, upstream_artifacts=mod.load_upstream_artifacts(REPO))

    assert summary["operation_stage_error_counts"] == {
        "extraction": 1,
        "update": 1,
        "routing": 1,
        "maintenance": 1,
        "use": 1,
        "rollback": 1,
    }
    assert summary["attribution_coverage"]["coverage_rate"] == 1.0
    assert summary["attribution_coverage"]["attributed_cases"] == 7
    assert summary["unsafe_propagations"]["count"] == 0
    assert summary["unsafe_propagations"]["blocked_control_count"] == 6
    assert summary["local_maintenance_cost"]["total_cost_units"] == 7
    assert summary["decision_impact_summary"]["memory_assisted_quality_rate"] == 1.0
    assert summary["decision_impact_summary"]["always_full_quality_rate"] == 1.0
    assert summary["decision_impact_summary"]["decision_quality_delta"] == 0.0
    assert summary["decision_impact_summary"]["calls_avoided_rate"] == 0.857143
    assert summary["decision_impact_summary"]["allocation_changed_by_memory_count"] == 3
    assert summary["decision_impact_summary"]["final_decision_regressions"] == 0
    assert summary["continuous_self_learning_evidence"]["usable_for_exp5290"] is True
    assert summary["memory_attribution_ready"] is True


def test_scenario_learn_5289_unsafe_control_propagation_blocks_readiness() -> None:
    """SCENARIO-LEARN-5289: propagated controlled faults make the gate false."""

    cases = mod.build_attribution_cases(root=REPO)
    mutated = deepcopy(cases)
    for case in mutated:
        if case["control_kind"] == "shuffled_memory":
            case["propagation_blocked"] = False
            case["selected_decision"] = "accept_hardware_speedup_from_smoke_only"
            case["expected_decision"] = "block_hardware_speedup_claim_until_transcript"
            case["unsafe_propagated"] = mod.is_unsafe_propagation(case)

    summary = mod.attribute_memory_operations(
        mutated,
        upstream_artifacts=mod.load_upstream_artifacts(REPO),
    )

    assert summary["unsafe_propagations"]["count"] == 1
    assert summary["unsafe_propagations"]["control_kinds"] == ["shuffled_memory"]
    assert summary["decision_impact_summary"]["final_decision_regressions"] == 1
    assert summary["memory_attribution_ready"] is False
    assert "unsafe propagation" in mod._honest_verdict(summary)

    incomplete = deepcopy(summary)
    incomplete["unsafe_propagations"]["count"] = 0
    incomplete["decision_impact_summary"]["final_decision_regressions"] = 0
    incomplete["attribution_coverage"]["coverage_rate"] = 0.5
    assert "operation attribution incomplete" in mod._honest_verdict(incomplete)

    regressed = deepcopy(incomplete)
    regressed["attribution_coverage"]["coverage_rate"] = 1.0
    regressed["decision_impact_summary"]["final_decision_regressions"] = 1
    assert "final decision regressions=1" in mod._honest_verdict(regressed)

    null_ready = deepcopy(regressed)
    null_ready["decision_impact_summary"]["final_decision_regressions"] = 0
    null_ready["memory_attribution_ready"] = False
    assert mod._honest_verdict(null_ready) == "null: operation attribution is not usable for Exp5290"


def test_req_learn_5289_artifact_schema_and_run_are_stable(tmp_path: Path) -> None:
    """REQ-LEARN-5289: run() writes the required principle-wrapped artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit attribution", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["memory_attribution_ready"] is True
    assert artifact["memory_attribution_ready_principle"]
    assert artifact["tests_run"] == tests_run
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "operation attribution is usable" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["operation_stage_error_counts"]["principle"] == (
        mod.FIELD_PRINCIPLES["operation_stage_error_counts"]
    )
    assert artifact["unsafe_propagations"]["value"]["count"] == 0
    assert artifact["attribution_coverage"]["value"]["coverage_rate"] == 1.0
    assert artifact["local_maintenance_cost"]["value"]["total_cost_units"] == 7
    assert artifact["decision_impact_summary"]["value"]["final_decision_regressions"] == 0
    assert artifact["continuous_self_learning_evidence"]["value"]["usable_for_exp5290"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod._sha256_file(tmp_path / "missing.json") is None

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_learn_5289_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5289: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert result["memory_attribution_ready"] is True
    assert result["operation_stage_error_counts"]["extraction"] == 1
    assert result["operation_stage_error_counts"]["update"] == 1
    assert result["operation_stage_error_counts"]["routing"] == 1
    assert result["operation_stage_error_counts"]["maintenance"] == 1
    assert result["operation_stage_error_counts"]["use"] == 1
    assert result["operation_stage_error_counts"]["rollback"] == 1
    assert result["unsafe_propagations"]["value"]["count"] == 0
    mod.validate_artifact(result)

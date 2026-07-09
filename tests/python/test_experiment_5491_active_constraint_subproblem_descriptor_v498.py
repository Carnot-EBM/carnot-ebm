"""Tests for Exp5491 active-constraint subproblem descriptors.

Spec refs: REQ-VERIFY-5491, SCENARIO-VERIFY-5491.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5491_active_constraint_subproblem_descriptor_v498 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5491_active_constraint_subproblem_descriptor_v498.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5491_active_constraint_subproblem_descriptor_v498.py "
    "-m pytest "
    "tests/python/test_experiment_5491_active_constraint_subproblem_descriptor_v498.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5491_active_constraint_subproblem_descriptor_v498.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5491 is deterministic descriptor emission; "
    "no live training, PyO3, KV260, board timing, or hardware execution e2e path applies"
)


def _tests_run() -> list[dict[str, object]]:
    return [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_REPORT_COMMAND, "outcome": "passed"},
        {"command": FULL_SUITE_COMMAND, "outcome": "passed"},
        {"command": SPEC_COVERAGE_COMMAND, "outcome": "passed"},
        {"command": E2E_COMMAND, "outcome": "not_applicable"},
    ]


def test_req_verify_5491_spec_declares_descriptor_contract() -> None:
    """REQ-VERIFY-5491: OpenSpec anchors the portable descriptor contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5491") : spec.index("### REQ-VERIFY-5462")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5491",
        "SCENARIO-VERIFY-5491",
        str(mod.RESULT_RELATIVE_PATH),
        "variables",
        "domains",
        "hard_constraints",
        "soft_preferences",
        "coupling_type",
        "update_schedule",
        "partition_id",
        "exact_fallback",
        "admissible_hardware_mapping",
        "Preference-MaxSAT",
        mod.INFERENCE_SUBSTRATE,
        "hardware_speedup_claim",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_verify_5491_descriptors_are_portable_and_authoritative() -> None:
    """SCENARIO-VERIFY-5491: descriptors round-trip and keep exact authority."""

    descriptors = mod.build_descriptors(repo_root=REPO)
    summary = mod.summarize_descriptors(descriptors)

    assert len(descriptors) == mod.EXPECTED_DESCRIPTOR_COUNT
    assert summary["descriptor_roundtrip_rate"] == pytest.approx(1.0)
    assert summary["exact_fallback_completeness"] == pytest.approx(1.0)
    assert summary["unsafe_false_accept_count"] == 0
    assert summary["advisory_improvement_delta"] > 0.0
    assert set(summary["update_schedule_types"]) == {
        "pbit_async_sweep",
        "pdit_block_gibbs",
        "preference_maxsat_batch",
    }

    for descriptor in descriptors:
        assert set(mod.REQUIRED_DESCRIPTOR_FIELDS) <= set(descriptor)
        assert descriptor["status"] == "solved"
        assert descriptor["exact_fallback"]["complete"] is True
        assert descriptor["exact_fallback"]["canonical_reference_agreement"] is True
        assert descriptor["admissible_hardware_mapping"]["advisory_only"] is True
        assert descriptor["admissible_hardware_mapping"]["board_timing_collected"] is False
        assert descriptor["admissible_hardware_mapping"]["speedup_claim_allowed"] is False
        assert descriptor["partition_id"]
        assert descriptor["update_schedule"]["update_count"] > 0
        mod.validate_descriptor(descriptor)


def test_req_verify_5491_loads_available_preference_maxsat_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5491: available Exp5485 hard/soft rows populate descriptors."""

    exp5485 = tmp_path / mod.EXP5485_RELATIVE_PATH
    exp5485.parent.mkdir(parents=True)
    exp5485.write_text(
        json.dumps(
            {
                "row_records": [
                    {
                        "row_id": "external_pref",
                        "partition_id": "external_partition",
                        "variables": [
                            {
                                "name": "decision",
                                "domain": ["accept", "reject", "abstain"],
                            }
                        ],
                        "hard_constraints": [
                            {
                                "id": "HC_NO_ACCEPT",
                                "type": "clause",
                                "literals": [
                                    {"variable": "decision", "equals": "reject"},
                                    {"variable": "decision", "equals": "abstain"},
                                ],
                            }
                        ],
                        "soft_preferences": [
                            {
                                "id": "SP_REJECT",
                                "type": "value_reward",
                                "variable": "decision",
                                "value": "reject",
                                "weight": 7,
                            }
                        ],
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    descriptors = mod.build_preference_maxsat_descriptors(repo_root=tmp_path)

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor["source_fixture_id"] == "external_pref"
    assert descriptor["partition_id"] == "external_partition"
    assert descriptor["exact_fallback"]["solution"] == {"decision": "reject"}
    assert descriptor["source_artifact"] == mod.EXP5485_RELATIVE_PATH.as_posix()
    mod.validate_descriptor(descriptor)


def test_req_verify_5491_validation_rejects_unsafe_solved_descriptors() -> None:
    """SCENARIO-VERIFY-5491: solved descriptors require exact reference agreement."""

    descriptor = mod.build_descriptors(repo_root=REPO)[0]
    mod.validate_descriptor(descriptor)

    missing_fallback = deepcopy(descriptor)
    missing_fallback["exact_fallback"]["complete"] = False
    with pytest.raises(ValueError, match="exact_fallback"):
        mod.validate_descriptor(missing_fallback)

    disagreeing_reference = deepcopy(descriptor)
    disagreeing_reference["canonical_reference"]["solution_hash"] = "bad"
    with pytest.raises(ValueError, match="canonical_reference"):
        mod.validate_descriptor(disagreeing_reference)

    advisory_hardware = deepcopy(descriptor)
    advisory_hardware["admissible_hardware_mapping"]["advisory_only"] = False
    with pytest.raises(ValueError, match="advisory_only"):
        mod.validate_descriptor(advisory_hardware)

    timing_claim = deepcopy(descriptor)
    timing_claim["admissible_hardware_mapping"]["board_timing_collected"] = True
    with pytest.raises(ValueError, match="board_timing_collected"):
        mod.validate_descriptor(timing_claim)

    missing_partition = deepcopy(descriptor)
    missing_partition["partition_id"] = ""
    with pytest.raises(ValueError, match="partition_id"):
        mod.validate_descriptor(missing_partition)

    unsupported_constraint = deepcopy(descriptor)
    unsupported_constraint["hard_constraints"] = [
        {"id": "HC_UNKNOWN", "type": "unknown", "variables": ["x1"]}
    ]
    with pytest.raises(ValueError, match="constraint_type"):
        mod.validate_descriptor(unsupported_constraint)

    unsupported_preference = deepcopy(descriptor)
    unsupported_preference["soft_preferences"] = [
        {"id": "SP_UNKNOWN", "type": "unknown", "weight": 1}
    ]
    with pytest.raises(ValueError, match="preference_type"):
        mod.validate_descriptor(unsupported_preference)


def test_req_verify_5491_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5491: deliverable JSON exposes all required bare fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, repo_root=REPO, tests_run=_tests_run())
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(saved)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(saved)
    assert saved["field_principles"] == mod.FIELD_PRINCIPLES
    assert saved["descriptor_count"] == mod.EXPECTED_DESCRIPTOR_COUNT
    assert saved["descriptor_roundtrip_rate"] == pytest.approx(1.0)
    assert saved["exact_fallback_completeness"] == pytest.approx(1.0)
    assert saved["unsafe_false_accept_count"] == 0
    assert saved["advisory_improvement_delta"] > 0.0
    assert saved["hardware_speedup_claim"] is False
    assert saved["subproblem_descriptor_ready"] is True
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["tests_run"] == _tests_run()
    assert saved["research_conductor_modified"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_verify_5491_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5491: checked-in JSON remains stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(repo_root=REPO, tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["subproblem_descriptor_ready"] is True
    assert checked_in["hardware_speedup_claim"] is False
    mod.validate_artifact(checked_in)


def test_req_verify_5491_artifact_validation_fails_closed() -> None:
    """REQ-VERIFY-5491: artifact validation rejects schema and authority drift."""

    artifact = mod.build_artifact(repo_root=REPO, tests_run=_tests_run())
    mod.validate_artifact(artifact)
    assert mod.honest_verdict(False, ["roundtrip_failed"]).startswith("blocked:")

    missing = deepcopy(artifact)
    missing.pop("descriptor_count")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(bad_hardware)

    bad_roundtrip = deepcopy(artifact)
    bad_roundtrip["descriptor_roundtrip_rate"] = 0.5
    with pytest.raises(ValueError, match="descriptor_roundtrip_rate"):
        mod.validate_artifact(bad_roundtrip)

    bad_exact = deepcopy(artifact)
    bad_exact["exact_fallback_completeness"] = 0.5
    with pytest.raises(ValueError, match="exact_fallback_completeness"):
        mod.validate_artifact(bad_exact)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accept_count"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accept_count"):
        mod.validate_artifact(bad_unsafe)

    bad_descriptor = deepcopy(artifact)
    bad_descriptor["descriptors"][0]["status"] = "advisory_only"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_descriptor)

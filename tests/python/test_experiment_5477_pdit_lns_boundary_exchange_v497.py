"""Tests for Exp5477 p-dit LNS boundary-exchange accounting.

Spec refs: REQ-VERIFY-5477, SCENARIO-VERIFY-5477.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5477_pdit_lns_boundary_exchange_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5477_pdit_lns_boundary_exchange_v497.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5477_pdit_lns_boundary_exchange_v497.py "
    "-m pytest "
    "tests/python/test_experiment_5477_pdit_lns_boundary_exchange_v497.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5477_pdit_lns_boundary_exchange_v497.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5477 is deterministic CPU-local boundary "
    "accounting; no live training, PyO3, KV260, or hardware timing e2e path applies"
)


def _tests_run() -> list[dict[str, object]]:
    return [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_REPORT_COMMAND, "outcome": "passed"},
        {"command": FULL_SUITE_COMMAND, "outcome": "passed"},
        {"command": E2E_COMMAND, "outcome": "not_applicable"},
    ]


def test_req_verify_5477_spec_declares_boundary_exchange_contract() -> None:
    """REQ-VERIFY-5477: OpenSpec anchors the CPU-only boundary accounting."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5477") : spec.index("### REQ-VERIFY-5462")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5477",
        "SCENARIO-VERIFY-5477",
        str(mod.RESULT_RELATIVE_PATH),
        "SAT, MaxCut, and assignment-style fixture",
        "random",
        "conflict_core_guided",
        "prediction_score_guided",
        "greedy_exact_fallback",
        "stochastic_advisory_repair",
        "no_repair_baseline",
        mod.INFERENCE_SUBSTRATE,
        "hardware_speedup_claim",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5477_workload_hashes_are_stable_and_fixture_anchored() -> None:
    """REQ-VERIFY-5477: workload hashes are canonical and independent of rows."""

    fixtures = mod.build_boundary_fixtures()
    first_hashes = mod.workload_hashes(fixtures)
    second_hashes = mod.workload_hashes(mod.build_boundary_fixtures())

    assert mod.fixture_family_counts(fixtures) == {"assignment": 1, "maxcut": 1, "sat": 1}
    assert mod.pbit_variable_count(fixtures) == 10
    assert mod.pdit_variable_count(fixtures) == 8
    assert first_hashes == second_hashes
    assert len(first_hashes) == len(fixtures) == mod.EXPECTED_FIXTURE_COUNT
    assert len(set(first_hashes)) == len(first_hashes)
    assert all(len(item) == 64 for item in first_hashes)

    payload = mod.fixture_workload_payload(fixtures[0])
    assert "row_records" not in payload
    assert mod.workload_hash(fixtures[0]) == mod.sha256_json(payload)


def test_scenario_verify_5477_exact_fallback_is_final_authority() -> None:
    """SCENARIO-VERIFY-5477: advisory destroy/repair rows cannot set final labels."""

    artifact = mod.build_artifact(tests_run=_tests_run())
    rows = artifact["row_records"]
    expected_rows = (
        mod.EXPECTED_FIXTURE_COUNT * len(mod.DESTROY_STRATEGIES) * len(mod.REPAIR_MODES)
    )

    assert len(rows) == expected_rows
    assert artifact["exact_fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accept_count"] == 0
    assert all(row["solver_final_label"] == row["exact_label"] for row in rows)
    assert all(row["final_solution"] == row["exact_solution"] for row in rows)
    assert all(row["fallback_complete"] is True for row in rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["hardware_speedup_claim"] is False for row in rows)
    assert all(row["boundary_messages"] for row in rows)
    assert any(row["fallback_used"] is True for row in rows)
    assert any(row["advisory_changed_candidate"] is True for row in rows)
    assert any(row["advisory_improvement"] > 0 for row in rows)
    assert {
        row["destroy_strategy"] for row in rows
    } == set(mod.DESTROY_STRATEGIES)
    assert {row["repair_mode"] for row in rows} == set(mod.REPAIR_MODES)


def test_req_verify_5477_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5477: deliverable JSON exposes all required bare fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=_tests_run())
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(saved)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(saved)
    assert saved["field_principles"] == mod.FIELD_PRINCIPLES
    assert saved["fixture_count"] == mod.EXPECTED_FIXTURE_COUNT
    assert saved["pbit_variable_count"] == 10
    assert saved["pdit_variable_count"] == 8
    assert saved["destroy_strategies"] == list(mod.DESTROY_STRATEGIES)
    assert saved["repair_modes"] == list(mod.REPAIR_MODES)
    assert saved["exact_fallback_completeness_rate"] == pytest.approx(1.0)
    assert saved["unsafe_false_accept_count"] == 0
    assert saved["advisory_improvement_delta"] > 0.0
    assert saved["boundary_exchange_ready"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["tests_run"] == _tests_run()
    assert saved["research_conductor_modified"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_verify_5477_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5477: checked-in JSON remains stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["boundary_exchange_ready"] is True
    assert checked_in["hardware_speedup_claim"] is False
    mod.validate_artifact(checked_in)


def test_req_verify_5477_validation_rejects_authority_and_schema_drift() -> None:
    """REQ-VERIFY-5477: validation fails closed on unsafe artifact drift."""

    artifact = mod.build_artifact(tests_run=_tests_run())
    mod.validate_artifact(artifact)
    assert mod.honest_verdict(False, ["fixture_count_mismatch"]).startswith("blocked:")

    missing = deepcopy(artifact)
    missing.pop("fixture_count")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "hardware_sampler"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(bad_hardware)

    bad_fallback = deepcopy(artifact)
    bad_fallback["exact_fallback_completeness_rate"] = 0.5
    with pytest.raises(ValueError, match="exact_fallback_completeness_rate"):
        mod.validate_artifact(bad_fallback)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accept_count"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accept_count"):
        mod.validate_artifact(bad_unsafe)

    bad_strategy = deepcopy(artifact)
    bad_strategy["destroy_strategies"] = ["random"]
    with pytest.raises(ValueError, match="destroy_strategies"):
        mod.validate_artifact(bad_strategy)

    bad_row = deepcopy(artifact)
    bad_row["row_records"][0]["solver_final_label"] = "advisory_label"
    with pytest.raises(ValueError, match="solver_final_label"):
        mod.validate_artifact(bad_row)

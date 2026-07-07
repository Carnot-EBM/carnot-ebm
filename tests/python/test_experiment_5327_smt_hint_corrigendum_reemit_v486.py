"""Tests for Exp 5327 SMT hint validation corrigendum re-emit.

Spec refs: REQ-VERIFY-5327, SCENARIO-VERIFY-5327.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5327_smt_hint_corrigendum_reemit_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_verify_5327_spec_declares_corrigendum_contract() -> None:
    """REQ-VERIFY-5327: OpenSpec anchors the corrigendum artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5327") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5327",
        "SCENARIO-VERIFY-5327",
        str(mod.RESULT_RELATIVE_PATH),
        "deterministic_smt_solver_protocol",
        "valid_hint_acceptance_rate",
        "unsound_hint_rejection_rate",
        "usefulness_rate",
        "methodology_duration_s",
        "compute_bound_marker_present=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5327_reuses_fixture_and_separates_usefulness() -> None:
    """SCENARIO-VERIFY-5327: useful accepted hints stay separate from fallback."""

    benchmark = mod.run_corrigendum_protocol()
    rows = benchmark["hint_validation_telemetry"]
    classes = {row["usefulness_class"] for row in rows}

    assert benchmark["valid_hint_acceptance_rate"] == pytest.approx(1.0)
    assert benchmark["unsound_hint_rejection_rate"] == pytest.approx(1.0)
    assert benchmark["usefulness_rate"] == pytest.approx(0.5)
    assert benchmark["solver_fallback_complete"] is True
    assert benchmark["completeness_preserved"] is True
    assert {"useful", "useless", "redundant", "unsound"} <= classes
    assert all(row["accepted"] is True for row in rows if row["solver_valid"])
    assert all(row["fallback_to_classical"] is True for row in rows if not row["solver_valid"])
    assert all(row["useful"] is False for row in rows if row["fallback_to_classical"])


def test_req_verify_5327_artifact_schema_and_marker_free_payload(tmp_path: Path) -> None:
    """REQ-VERIFY-5327: required fields are wrapped or bare as specified."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5327", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        methodology_duration_s=0.25,
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
    assert "0.029515s" in str(_value(artifact, "exp5318_flag_reason"))
    assert artifact["valid_hint_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["unsound_hint_rejection_rate"] == pytest.approx(1.0)
    assert artifact["usefulness_rate"] == pytest.approx(0.5)
    assert artifact["solver_fallback_complete"] is True
    assert artifact["methodology_duration_s"] == pytest.approx(0.25)
    assert artifact["compute_bound_marker_present"] is False
    assert artifact["smt_hint_protocol_clean"] is True
    assert _value(artifact, "tests_run") == tests_run
    assert mod.compute_bound_marker_present(artifact) is False
    serialized = json.dumps(artifact, sort_keys=True)
    assert all(marker not in serialized for marker in mod.COMPUTE_BOUND_MARKERS)


def test_scenario_verify_5327_blocks_untrusted_duration_without_padding() -> None:
    """SCENARIO-VERIFY-5327: sub-floor timing blocks instead of being padded."""

    artifact = mod.build_artifact(methodology_duration_s=0.0, tests_run=[])

    mod.validate_artifact(artifact)
    assert _value(artifact, "status") == "blocked"
    assert str(_value(artifact, "honest_verdict")).startswith(
        "blocked_methodology_duration_untrusted"
    )
    assert artifact["methodology_duration_s"] == pytest.approx(0.0)
    assert artifact["compute_bound_marker_present"] is False
    assert artifact["smt_hint_protocol_clean"] is False
    assert mod._honest_verdict(
        clean=False,
        duration_trusted=True,
        marker_present=True,
    ).startswith("blocked_runtime_marker_present")
    assert mod._honest_verdict(
        clean=False,
        duration_trusted=True,
        marker_present=False,
    ) == "blocked_smt_hint_protocol_not_clean"


def test_req_verify_5327_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5327: invalid substrate, marker, or metrics are rejected."""

    artifact = mod.build_artifact(methodology_duration_s=0.2, tests_run=[])

    missing = copy.deepcopy(artifact)
    missing.pop("methodology_duration_s")
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(missing)

    broken = copy.deepcopy(artifact)
    broken["solver_fallback_complete"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "live_llm_inference")
    with pytest.raises(AssertionError, match=mod.INFERENCE_SUBSTRATE):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["compute_bound_marker_present"] = True
    with pytest.raises(AssertionError, match="marker"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["unsound_hint_rejection_rate"] = 0.5
    with pytest.raises(AssertionError, match="unsound"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_verify_5327() -> None:
    """SCENARIO-VERIFY-5327: checked-in deliverable satisfies the V486 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith(("complete:", "blocked_"))
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["compute_bound_marker_present"] is False
    assert artifact["smt_hint_protocol_clean"] is True

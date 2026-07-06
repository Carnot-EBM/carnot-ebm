"""Tests for Exp 5318 deterministic SMT hint validation protocol.

Spec refs: REQ-VERIFY-5318, SCENARIO-VERIFY-5318.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5318_smt_hint_validation_protocol_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _row_by_hint(rows: list[dict[str, object]], hint_id: str) -> dict[str, object]:
    return next(row for row in rows if row["hint_id"] == hint_id)


def test_req_verify_5318_spec_declares_smt_hint_protocol() -> None:
    """REQ-VERIFY-5318: OpenSpec anchors the deterministic SMT hint protocol."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5318") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5318",
        "SCENARIO-VERIFY-5318",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "valid, useless, redundant, and unsound hints",
        "overwrite clauses",
        "Exp5309 runtime gate",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5318_builds_quantified_and_inductive_no_llm_fixtures() -> None:
    """REQ-VERIFY-5318: fixtures cover hint classes without an LLM proposer."""

    examples = mod.build_smt_hint_examples()
    hints = [hint for example in examples for hint in example.hints]

    assert {example.style for example in examples} == {
        "quantified_instantiation",
        "inductive_chain",
        "sat_overwrite_control",
    }
    assert {hint.expected_class for hint in hints} == {
        "valid",
        "useless",
        "redundant",
        "unsound",
    }
    assert {hint.hint_kind for hint in hints} == {
        "instantiation",
        "lemma",
        "conjecture",
    }
    assert all(hint.proposal_source == mod.PROPOSAL_SOURCE for hint in hints)
    assert all(mod.validation_context_status(example) == "sat" for example in examples)


def test_scenario_verify_5318_accepts_valid_but_not_unsound_hints() -> None:
    """SCENARIO-VERIFY-5318: solver entailment accepts only sound hints."""

    benchmark = mod.run_benchmark()
    rows = benchmark["hint_validation_telemetry"]

    assert benchmark["valid_hint_acceptance_rate"] == pytest.approx(1.0)
    assert benchmark["unsound_hint_rejection_rate"] == pytest.approx(1.0)
    assert benchmark["usefulness_rate"] == pytest.approx(0.5)

    for row in rows:
        if row["solver_valid"]:
            assert row["accepted"] is True
            assert row["fallback_to_classical"] is False
        else:
            assert row["accepted"] is False
            assert row["fallback_to_classical"] is True
            assert row["final_status"] == row["baseline_status"]

    useful = _row_by_hint(rows, "quantified_successor_instantiation")
    useless = _row_by_hint(rows, "quantified_tautology_useless")
    redundant = _row_by_hint(rows, "inductive_start_redundant")

    assert useful["solver_valid"] is True
    assert useful["useful"] is True
    assert useless["accepted"] is True
    assert useless["useful"] is False
    assert useless["usefulness_class"] == "useless"
    assert redundant["accepted"] is True
    assert redundant["redundant"] is True
    assert redundant["useful"] is False


def test_scenario_verify_5318_unsound_hint_fallback_preserves_sat_result() -> None:
    """SCENARIO-VERIFY-5318: unsound conjectures cannot override classical SMT."""

    example = next(
        row
        for row in mod.build_smt_hint_examples()
        if row.example_id == "sat_choice_overwrite"
    )
    hint = next(row for row in example.hints if row.hint_id == "sat_choice_unsound_b")
    row = mod.evaluate_hint(example, hint)

    assert row["solver_valid"] is False
    assert row["accepted"] is False
    assert row["blindly_added_status"] == "unsat"
    assert row["baseline_status"] == "sat"
    assert row["final_status"] == "sat"
    assert row["fallback_to_classical"] is True
    assert row["overwrite_count"] == 1
    assert row["overwrite_clauses"] == ["sat_choice_b"]
    assert row["completeness_preserved"] is True


def test_req_verify_5318_benchmark_reports_protocol_readiness() -> None:
    """REQ-VERIFY-5318: benchmark records fallback and completeness telemetry."""

    benchmark = mod.run_benchmark()

    assert benchmark["smt_hint_protocol_ready"] is True
    assert benchmark["solver_fallback_complete"] is True
    assert benchmark["completeness_preserved"] is True
    assert benchmark["future_llm_slot_gated_on_sota_runtime"] is True
    assert benchmark["llm_invoked"] is False
    assert benchmark["future_llm_slot"]["current_proposer"] == mod.PROPOSAL_SOURCE
    assert "Exp5309" in benchmark["future_llm_slot"]["gate"]
    assert "sota_runtime_unblocked=true" in benchmark["future_llm_slot"]["gate"]
    assert {row["hint_kind"] for row in benchmark["hint_validation_telemetry"]} == {
        "instantiation",
        "lemma",
        "conjecture",
    }


def test_req_verify_5318_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5318: artifact exposes principle fields and bare metrics."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5318", "outcome": "passed"}]
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
    assert _value(artifact, "fixture_path") == str(mod.FIXTURE_RELATIVE_PATH)
    assert artifact["smt_hint_protocol_ready"] is True
    assert artifact["valid_hint_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["unsound_hint_rejection_rate"] == pytest.approx(1.0)
    assert artifact["usefulness_rate"] == pytest.approx(0.5)
    assert artifact["solver_fallback_complete"] is True
    assert artifact["completeness_preserved"] is True
    assert artifact["future_llm_slot_gated_on_sota_runtime"] is True
    assert _value(artifact, "tests_run") == tests_run
    assert "REQ-VERIFY-5318" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5318_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5318: invalid readiness, substrate, or LLM gates fail."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5318", "outcome": "passed"}],
    )

    missing = copy.deepcopy(artifact)
    missing.pop("unsound_hint_rejection_rate")
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(missing)

    broken = copy.deepcopy(artifact)
    broken["smt_hint_protocol_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "live_llm_inference")
    with pytest.raises(AssertionError, match=mod.INFERENCE_SUBSTRATE):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["unsound_hint_rejection_rate"] = 0.5
    with pytest.raises(AssertionError, match="unsound"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["future_llm_slot_gated_on_sota_runtime"] = False
    with pytest.raises(AssertionError, match="Exp5309"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_verify_5318() -> None:
    """SCENARIO-VERIFY-5318: checked-in deliverable satisfies the V485 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["smt_hint_protocol_ready"] is True
    assert artifact["solver_fallback_complete"] is True
    assert artifact["completeness_preserved"] is True
    assert artifact["future_llm_slot_gated_on_sota_runtime"] is True

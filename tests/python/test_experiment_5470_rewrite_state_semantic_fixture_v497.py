"""Tests for Exp5470 deterministic rewrite-state semantic fixture.

Spec refs: REQ-SAFE-5470, SCENARIO-SAFE-5470.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5470_rewrite_state_semantic_fixture_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5470_rewrite_state_semantic_fixture_v497.py -q"
)


def _artifact() -> dict:
    return mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])


def _rows_by_id() -> dict[str, dict]:
    return {row["candidate_id"]: row for row in mod.evaluate_candidates(mod.build_candidates())}


def test_req_safe_5470_spec_declares_rewrite_state_semantic_fixture() -> None:
    """REQ-SAFE-5470: OpenSpec anchors the deterministic fixture contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5470") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5470",
        "SCENARIO-SAFE-5470",
        str(mod.RESULT_RELATIVE_PATH),
        "typed source-to-target state transition",
        "explicit license IDs",
        "valid rewrites",
        "hidden-premise mutations",
        "unlicensed state changes",
        "fabricated citation/evidence changes",
        "locally syntax-valid but semantically invalid outputs",
        "factual-distortion temptations",
        "arithmetic claims",
        "JSON constraints",
        "API preconditions",
        "fact anchors",
        "guided_decoding_quarantine_lifted",
        mod.INFERENCE_SUBSTRATE,
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_safe_5470_candidates_are_typed_licensed_transitions() -> None:
    """REQ-SAFE-5470: every fixture candidate is a typed licensed transition."""

    candidates = mod.build_candidates()
    assert len(candidates) == 8
    assert {candidate.case_type for candidate in candidates} == set(mod.REQUIRED_CASE_TYPES)
    assert {candidate.domain for candidate in candidates} == set(mod.REQUIRED_DOMAINS)

    for candidate in candidates:
        license_ids = {license_row.license_id for license_row in candidate.licenses}
        assert candidate.source_state.state_id
        assert candidate.target_state.state_id
        assert candidate.source_state.state_id != candidate.target_state.state_id
        assert license_ids
        assert all(atom.license_id for atom in candidate.source_state.facts.values())
        assert all(atom.license_id for atom in candidate.target_state.facts.values())
        assert all(citation.license_id for citation in candidate.target_state.citations)
        if candidate.expected_accept:
            assert mod.missing_or_bad_license_ids(candidate) == []

    hidden = mod.candidate_by_id(candidates, "5470-hidden-premise")
    assert hidden.case_type == "hidden_premise_mutation"
    assert hidden.target_state.facts["clinic.has_backup_generator"].license_id.startswith(
        "UNLICENSED:"
    )


def test_scenario_safe_5470_exact_validators_reject_each_trap() -> None:
    """SCENARIO-SAFE-5470: exact validators override syntax and LCD advice."""

    rows = _rows_by_id()

    assert rows["5470-valid-fact-paraphrase"]["exact_final_verdict"]["accepted"] is True
    assert rows["5470-valid-arithmetic"]["exact_final_verdict"]["accepted"] is True
    assert rows["5470-valid-fact-paraphrase"]["answer_set_atoms"]["fact_anchor"][
        "required"
    ] == ["fact_supported:clinic.opened_year"]

    hidden = rows["5470-hidden-premise"]
    assert hidden["local_syntax_valid"] is True
    assert hidden["lcd_advisory_accept"] is True
    assert hidden["exact_final_verdict"]["accepted"] is False
    assert "hidden_premise" in hidden["exact_final_verdict"]["violation_kinds"]
    assert "clinic.has_backup_generator" in hidden["license_result"]["hidden_premise_keys"]

    unlicensed = rows["5470-unlicensed-state-change"]
    assert "unlicensed_mutation" in unlicensed["exact_final_verdict"]["violation_kinds"]
    assert unlicensed["license_result"]["unlicensed_mutation_keys"] == ["order.locked"]

    fabricated = rows["5470-fabricated-citation"]
    assert "fabricated_evidence" in fabricated["exact_final_verdict"]["violation_kinds"]
    assert fabricated["license_result"]["fabricated_citation_ids"] == ["CITE:phantom-report"]

    json_trap = rows["5470-json-semantic-invalid"]
    assert json_trap["local_syntax_valid"] is True
    assert json_trap["lcd_advisory_accept"] is True
    assert "semantic_invalid" in json_trap["exact_final_verdict"]["violation_kinds"]
    assert "json_semantic_valid" in json_trap["semantic_result"]["missing_atoms"]

    api_trap = rows["5470-api-precondition-invalid"]
    assert api_trap["local_syntax_valid"] is True
    assert api_trap["lcd_advisory_accept"] is True
    assert "api_precondition_failed" in api_trap["exact_final_verdict"]["violation_kinds"]
    assert "api_preconditions_met" in api_trap["semantic_result"]["missing_atoms"]

    distortion = rows["5470-factual-distortion"]
    assert distortion["lcd_advisory_accept"] is True
    assert "factual_distortion" in distortion["exact_final_verdict"]["violation_kinds"]
    assert "fact_anchor_supported" in distortion["semantic_result"]["missing_atoms"]


def test_scenario_safe_5470_artifact_schema_metrics_and_write_path(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5470: terminal JSON exposes required bare fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=[TEST_COMMAND], write=True)
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fixture_count"] == 8
    assert artifact["transition_count"] == 8
    assert artifact["hidden_premise_catch_rate"] == pytest.approx(1.0)
    assert artifact["unlicensed_mutation_catch_rate"] == pytest.approx(1.0)
    assert artifact["semantic_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["factual_distortion_rate"] == pytest.approx(0.0)
    assert artifact["lcd_bias_probe_passed"] is True
    assert artifact["exact_validator_agreement"] == pytest.approx(1.0)
    assert artifact["rewrite_state_fixture_ready"] is True
    assert artifact["guided_decoding_quarantine_lifted"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_provenance_checksum"] == mod.row_provenance_checksum(
        artifact["row_results"]
    )
    assert artifact["research_conductor_modified"] is False
    assert artifact["tests_run"][0]["outcome"] == "recorded"
    assert mod.run(result_path=result_path, write=False)["rewrite_state_fixture_ready"] is True


def test_req_safe_5470_validation_fails_closed_on_schema_or_metric_drift() -> None:
    """REQ-SAFE-5470: schema, metric, and authority drift are rejected."""

    artifact = _artifact()

    missing = deepcopy(artifact)
    missing.pop("fixture_count")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_quarantine = deepcopy(artifact)
    bad_quarantine["guided_decoding_quarantine_lifted"] = True
    with pytest.raises(ValueError, match="quarantine"):
        mod.validate_artifact(bad_quarantine)

    bad_semantic_rate = deepcopy(artifact)
    bad_semantic_rate["semantic_false_accept_rate"] = 0.5
    with pytest.raises(ValueError, match="semantic_false_accept_rate"):
        mod.validate_artifact(bad_semantic_rate)

    bad_distortion_rate = deepcopy(artifact)
    bad_distortion_rate["factual_distortion_rate"] = 0.5
    with pytest.raises(ValueError, match="factual_distortion_rate"):
        mod.validate_artifact(bad_distortion_rate)

    bad_agreement = deepcopy(artifact)
    bad_agreement["exact_validator_agreement"] = 0.875
    with pytest.raises(ValueError, match="exact_validator_agreement"):
        mod.validate_artifact(bad_agreement)

    bad_ready = deepcopy(artifact)
    bad_ready["rewrite_state_fixture_ready"] = True
    bad_ready["lcd_bias_probe_passed"] = False
    with pytest.raises(ValueError, match="rewrite_state_fixture_ready"):
        mod.validate_artifact(bad_ready)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor.py"):
        mod.validate_artifact(bad_conductor)

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["row_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="row checksum"):
        mod.validate_artifact(bad_checksum)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)


def test_req_safe_5470_defensive_helpers_fail_closed() -> None:
    """REQ-SAFE-5470: defensive branches stay deterministic and exact."""

    artifact = _artifact()
    unknown = mod.TransitionCandidate(
        candidate_id="unknown",
        case_type="semantic_json_invalid",
        domain="new_domain",
        description="unknown domain",
        source_state=mod.TypedState(state_id="s", facts={}, citations=()),
        target_state=mod.TypedState(state_id="t", facts={}, citations=()),
        licenses=(),
        local_syntax_valid=True,
        lcd_advisory_accept=True,
        expected_accept=False,
        expected_violation_kinds=("semantic_invalid",),
    )

    exact = mod.exact_validate_candidate(unknown)
    assert exact["accepted"] is False
    assert "semantic_invalid" in exact["violation_kinds"]
    assert "unsupported_domain:new_domain" in exact["semantic_result"]["failure_reasons"]
    assert mod.missing_or_bad_license_ids(unknown) == []

    wrong_new_fact_license = mod.TransitionCandidate(
        candidate_id="wrong-new-license",
        case_type="hidden_premise_mutation",
        domain="fact_anchor",
        description="new fact uses a state license, not a problem-fact license",
        source_state=mod.TypedState(state_id="s", facts={}),
        target_state=mod.TypedState(
            state_id="t",
            facts={"order.locked": mod.StateAtom(True, "SF:order-o17-locked")},
        ),
        licenses=(
            mod.LicenseRecord("SF:order-o17-locked", "state_fact", "order.locked", True),
        ),
        local_syntax_valid=True,
        lcd_advisory_accept=True,
        expected_accept=False,
        expected_violation_kinds=("hidden_premise",),
    )
    assert mod.validate_license_transition(wrong_new_fact_license)[
        "hidden_premise_keys"
    ] == ["order.locked"]

    valid_json = mod.TransitionCandidate(
        candidate_id="valid-json",
        case_type="semantic_json_invalid",
        domain="json_constraints",
        description="valid JSON quantity",
        source_state=mod.TypedState(
            state_id="s",
            facts={"inventory.bolt.stock": mod.StateAtom(3, "PF:bolt-stock-3")},
        ),
        target_state=mod.TypedState(
            state_id="t",
            facts={"inventory.bolt.stock": mod.StateAtom(3, "PF:bolt-stock-3")},
            json_payload={"sku": "bolt", "quantity": 2},
        ),
        licenses=(),
        local_syntax_valid=True,
        lcd_advisory_accept=True,
        expected_accept=True,
        expected_violation_kinds=(),
    )
    assert "json_semantic_valid" in mod.answer_set_atoms(valid_json)["json_constraints"][
        "present"
    ]

    valid_api = mod.TransitionCandidate(
        candidate_id="valid-api",
        case_type="api_precondition_invalid",
        domain="api_preconditions",
        description="cancel is allowed when unlocked",
        source_state=mod.TypedState(
            state_id="s", facts={"order.locked": mod.StateAtom(False, "SF:order-o17-locked")}
        ),
        target_state=mod.TypedState(
            state_id="t",
            facts={"order.locked": mod.StateAtom(False, "SF:order-o17-locked")},
            api_call={"name": "cancel_order", "args": {"order_id": "O-17"}},
        ),
        licenses=(),
        local_syntax_valid=True,
        lcd_advisory_accept=True,
        expected_accept=True,
        expected_violation_kinds=(),
    )
    assert "api_preconditions_met" in mod.answer_set_atoms(valid_api)["api_preconditions"][
        "present"
    ]

    assert mod._expected_violations(
        {
            "exact_final_verdict": "bad",
            "expected_violation_kinds": ["hidden_premise"],
        }
    ) == ["hidden_premise"]

    bad_rows_type = deepcopy(artifact)
    bad_rows_type["row_results"] = "bad"
    assert "row_results must be a list" in "; ".join(mod.artifact_schema_errors(bad_rows_type))

    bad_field_principles = deepcopy(artifact)
    bad_field_principles["field_principles"] = {}
    assert "field_principles" in "; ".join(mod.artifact_schema_errors(bad_field_principles))

    no_lcd_probe = deepcopy(artifact)
    for row in no_lcd_probe["row_results"]:
        row["lcd_advisory_accept"] = False
        row["row_checksum"] = mod.row_checksum(row)
    no_lcd_probe["row_provenance_checksum"] = mod.row_provenance_checksum(
        no_lcd_probe["row_results"]
    )
    no_lcd_probe["metric_details"] = mod.derive_metrics(no_lcd_probe["row_results"])
    no_lcd_probe["lcd_bias_probe_passed"] = False
    assert "lcd_bias_probe_passed" in "; ".join(mod.artifact_schema_errors(no_lcd_probe))

    bad_seed = deepcopy(artifact)
    bad_seed["random_seed"] = 1
    assert "random_seed mismatch" in "; ".join(mod.artifact_schema_errors(bad_seed))

    bad_ready_type = deepcopy(artifact)
    bad_ready_type["rewrite_state_fixture_ready"] = "yes"
    assert "rewrite_state_fixture_ready must be boolean" in "; ".join(
        mod.artifact_schema_errors(bad_ready_type)
    )

    semantic_false_accept = deepcopy(artifact)
    semantic_row = next(
        row
        for row in semantic_false_accept["row_results"]
        if row["candidate_id"] == "5470-json-semantic-invalid"
    )
    semantic_row["exact_final_verdict"]["accepted"] = True
    semantic_row["row_checksum"] = mod.row_checksum(semantic_row)
    semantic_false_accept["semantic_false_accept_rate"] = mod.derive_metrics(
        semantic_false_accept["row_results"]
    )["semantic_false_accept_rate"]
    assert "rewrite_state_fixture_ready requires semantic_false_accept_rate=0.0" in "; ".join(
        mod.artifact_schema_errors(semantic_false_accept)
    )

    factual_false_accept = deepcopy(artifact)
    factual_row = next(
        row
        for row in factual_false_accept["row_results"]
        if row["candidate_id"] == "5470-factual-distortion"
    )
    factual_row["exact_final_verdict"]["accepted"] = True
    factual_row["row_checksum"] = mod.row_checksum(factual_row)
    factual_false_accept["factual_distortion_rate"] = mod.derive_metrics(
        factual_false_accept["row_results"]
    )["factual_distortion_rate"]
    assert "rewrite_state_fixture_ready requires factual_distortion_rate=0.0" in "; ".join(
        mod.artifact_schema_errors(factual_false_accept)
    )

    bad_case_type = deepcopy(artifact)
    bad_case_type["row_results"][0]["case_type"] = "unknown"
    assert "case_type is unknown" in "; ".join(mod.artifact_schema_errors(bad_case_type))

    bad_domain = deepcopy(artifact)
    bad_domain["row_results"][0]["domain"] = "unknown"
    assert "domain is unknown" in "; ".join(mod.artifact_schema_errors(bad_domain))

    bad_verdict_type = deepcopy(artifact)
    bad_verdict_type["row_results"][0]["exact_final_verdict"] = "bad"
    assert "exact_final_verdict must be a mapping" in "; ".join(
        mod.artifact_schema_errors(bad_verdict_type)
    )

    bad_authority = deepcopy(artifact)
    bad_authority["row_results"][0]["exact_final_verdict"]["final_authority"] = "model"
    assert "final authority must be exact validators" in "; ".join(
        mod.artifact_schema_errors(bad_authority)
    )

    bad_match = deepcopy(artifact)
    bad_match["row_results"][0]["exact_final_verdict"]["matches_expected"] = False
    assert "exact validator did not match expected label" in "; ".join(
        mod.artifact_schema_errors(bad_match)
    )

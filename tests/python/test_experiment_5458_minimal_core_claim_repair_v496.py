"""Tests for Exp5458 deterministic minimal-core claim repair.

Spec refs: REQ-VERIFY-5458, SCENARIO-VERIFY-5458.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5458_minimal_core_claim_repair_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5458_minimal_core_claim_repair_v496.py -q"
)


def _cases() -> list[dict[str, Any]]:
    return mod.select_repair_cases(mod.load_source_artifacts(REPO))


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in _cases() if case["case_id"] == case_id)


def test_req_verify_5458_spec_declares_minimal_core_contract() -> None:
    """REQ-VERIFY-5458: OpenSpec anchors deterministic core-guided repair."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5458") : spec.index("### REQ-VERIFY-5433")
    ]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5458",
        "SCENARIO-VERIFY-5458",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5443",
        "Exp5445",
        "stable constraint IDs",
        "non-minimal",
        mod.INFERENCE_SUBSTRATE,
        "exact verifier or AST/KB witness is the only acceptance authority",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5458_derives_stable_minimal_cores_from_both_sources() -> None:
    """REQ-VERIFY-5458: failed rows receive stable minimal repair-core IDs."""

    cases = _cases()
    assert [case["case_id"] for case in cases] == [
        "exp5443:5443-001",
        "exp5443:5443-003",
        "exp5443:5443-005",
        "exp5443:5443-007",
        "exp5445:fixture.nonexistent_json_method",
        "exp5445:fixture.wrong_module_alias",
        "exp5445:fixture.imported_symbol_missing",
        "exp5445:fixture.argument_intent_mismatch",
    ]
    assert {case["substrate"] for case in cases} == {
        "verifier_potential",
        "ast_kb_witness",
    }

    expected = {
        "exp5443:5443-001": (
            "vp:5443-001:schema:allowed_keys_only",
        ),
        "exp5443:5443-003": (
            "vp:5443-003:semantic:no_equal_object_negated_object",
        ),
        "exp5443:5443-005": (
            "vp:5443-005:arithmetic:sum_matches_operands",
        ),
        "exp5443:5443-007": (
            "vp:5443-007:api:nonce_fresh",
            "vp:5443-007:api:signature_matches",
        ),
        "exp5445:fixture.nonexistent_json_method": (
            "astkb:fixture.nonexistent_json_method:call_exists:json.parse",
        ),
        "exp5445:fixture.wrong_module_alias": (
            "astkb:fixture.wrong_module_alias:call_exists:statistics.loads",
        ),
        "exp5445:fixture.imported_symbol_missing": (
            "astkb:fixture.imported_symbol_missing:imported_symbol_exists:json.parse",
        ),
        "exp5445:fixture.argument_intent_mismatch": (
            "astkb:fixture.argument_intent_mismatch:intent_matches:parse_json_to_object",
        ),
    }

    for case in cases:
        assert tuple(case["minimal_core_ids"]) == expected[case["case_id"]]
        assert case["minimal_core_ids"] == sorted(case["minimal_core_ids"])
        assert len(case["minimal_core_ids"]) == len(set(case["minimal_core_ids"]))
        assert all(" " not in core_id for core_id in case["minimal_core_ids"])
        assert case["minimality_evidence"]
        assert all(
            item["accepted_without_constraint"] is False
            for item in case["minimality_evidence"]
        )
        assert mod.derive_minimal_core(case) == tuple(case["minimal_core_ids"])


def test_scenario_verify_5458_rejects_non_minimal_and_stale_core_ids() -> None:
    """SCENARIO-VERIFY-5458: only the exact minimal core can emit a repair."""

    api_case = _case("exp5443:5443-007")
    stale_case = _case("exp5445:fixture.argument_intent_mismatch")

    extra_core = [*api_case["minimal_core_ids"], api_case["satisfied_constraint_ids"][0]]
    with pytest.raises(ValueError, match="minimal core"):
        mod.generate_repair_hypothesis(api_case, extra_core)

    missing_core = api_case["minimal_core_ids"][:1]
    with pytest.raises(ValueError, match="minimal core"):
        mod.generate_repair_hypothesis(api_case, missing_core)

    stale_core = [stale_case["minimal_core_ids"][0]]
    with pytest.raises(ValueError, match="minimal core"):
        mod.generate_repair_hypothesis(api_case, stale_core)


def test_scenario_verify_5458_exact_recheck_is_only_acceptance_authority() -> None:
    """SCENARIO-VERIFY-5458: repaired-looking candidates still need exact recheck."""

    case = _case("exp5443:5443-007")
    original = mod.recheck_candidate(case, case["original_candidate"])
    hypothesis = mod.generate_repair_hypothesis(case, case["minimal_core_ids"])
    accepted = mod.summarize_repair_attempt(case, hypothesis)

    assert original["accepted"] is False
    assert original["authority"] == "exact_final_verifier"
    assert hypothesis["generated_from"] == "minimal_core_ids_only"
    assert accepted["accepted_after_exact_recheck"] is True
    assert accepted["exact_recheck"]["accepted"] is True
    assert accepted["exact_recheck"]["authority"] == "exact_final_verifier"

    claim_only = deepcopy(hypothesis)
    claim_only["candidate"] = deepcopy(case["original_candidate"])
    claim_only["claimed_accept_without_recheck"] = True
    rejected = mod.summarize_repair_attempt(case, claim_only)

    assert rejected["accepted_after_exact_recheck"] is False
    assert rejected["exact_recheck"]["accepted"] is False
    assert "api_witness_signature_failed" in rejected["exact_recheck"]["failure_reasons"]

    ast_case = _case("exp5445:fixture.nonexistent_json_method")
    ast_hypothesis = mod.generate_repair_hypothesis(ast_case, ast_case["minimal_core_ids"])
    ast_attempt = mod.summarize_repair_attempt(ast_case, ast_hypothesis)
    assert ast_attempt["exact_recheck"]["authority"] == "ast_kb_witness"
    assert ast_attempt["accepted_after_exact_recheck"] is True


def test_scenario_verify_5458_artifact_fields_rates_and_provenance(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5458: terminal artifact exposes required bare fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=[TEST_COMMAND], write=True)
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["source_artifacts"] == [str(path) for path in mod.SOURCE_ARTIFACTS]
    assert artifact["repair_case_count"] == len(artifact["repair_cases"]) == 8
    assert artifact["minimal_core_success_rate"] == pytest.approx(1.0)
    assert artifact["repaired_accept_rate_after_exact_recheck"] == pytest.approx(1.0)
    assert artifact["unrepaired_reject_rate"] == pytest.approx(1.0)
    assert artifact["core_constraint_id_count"] == len(
        {
            core_id
            for case in artifact["repair_cases"]
            for core_id in case["minimal_core_ids"]
        }
    )
    assert artifact["exact_final_authority"] is True
    assert artifact["minimal_core_repair_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_provenance_checksum"] == mod.row_provenance_checksum(
        artifact["repair_cases"]
    )
    assert artifact["research_conductor_modified"] is False


def test_req_verify_5458_validation_fails_closed_on_drift() -> None:
    """REQ-VERIFY-5458: schema, provenance, and authority drift are rejected."""

    artifact = mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])

    missing = deepcopy(artifact)
    missing.pop("source_artifacts")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_rate = deepcopy(artifact)
    bad_rate["repaired_accept_rate_after_exact_recheck"] = 0.5
    with pytest.raises(ValueError, match="repaired_accept_rate_after_exact_recheck"):
        mod.validate_artifact(bad_rate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["repair_cases"][0]["minimal_core_ids"] = []
    with pytest.raises(ValueError, match="row_provenance_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_ready = deepcopy(artifact)
    bad_ready["minimal_core_repair_ready"] = True
    bad_ready["unrepaired_reject_rate"] = 0.0
    with pytest.raises(ValueError, match="minimal_core_repair_ready"):
        mod.validate_artifact(bad_ready)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor.py"):
        mod.validate_artifact(bad_conductor)

    bad_cases = deepcopy(artifact)
    bad_cases["repair_cases"] = "not-a-list"
    with pytest.raises(ValueError, match="repair_cases"):
        mod.validate_artifact(bad_cases)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_authority = deepcopy(artifact)
    bad_authority["exact_final_authority"] = False
    with pytest.raises(ValueError, match="exact_final_authority"):
        mod.validate_artifact(bad_authority)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_ready_type = deepcopy(artifact)
    bad_ready_type["minimal_core_repair_ready"] = "yes"
    with pytest.raises(ValueError, match="minimal_core_repair_ready must be boolean"):
        mod.validate_artifact(bad_ready_type)

    bad_empty_ready = deepcopy(artifact)
    bad_empty_ready["repair_cases"] = []
    with pytest.raises(ValueError, match="minimal_core_repair_ready requires repair cases"):
        mod.validate_artifact(bad_empty_ready)

    bad_ready_authority = deepcopy(artifact)
    bad_ready_authority["exact_final_authority"] = False
    with pytest.raises(ValueError, match="minimal_core_repair_ready requires exact authority"):
        mod.validate_artifact(bad_ready_authority)

    bad_case_recheck = deepcopy(artifact)
    bad_case_recheck["repair_cases"][0]["unrepaired_exact_recheck"]["accepted"] = True
    with pytest.raises(ValueError, match="unrepaired exact recheck mismatch"):
        mod.validate_artifact(bad_case_recheck)


def test_req_verify_5458_minimizer_and_defensive_repair_branches() -> None:
    """REQ-VERIFY-5458: minimization removes redundant IDs and guards bad inputs."""

    redundant = _case("exp5445:fixture.nonexistent_json_method")
    redundant = deepcopy(redundant)
    for constraint in redundant["encoded_constraints"]:
        if ":intent_matches:" in constraint["constraint_id"]:
            constraint["repair_action"] = {"op": "rewrite_to_expected_call"}

    assert mod.derive_minimal_core(redundant) == (
        "astkb:fixture.nonexistent_json_method:intent_matches:parse_json_to_object",
    )

    missing_required_case = {
        "original_candidate": {"kind": "claim"},
        "recheck_context": {"required_keys": ["kind", "payload"]},
        "encoded_constraints": [
            {
                "constraint_id": "vp:probe:schema:required_keys_present",
                "satisfied": False,
                "repair_action": {"op": "fill_missing_required_keys"},
            }
        ],
    }
    assert mod._apply_core_repairs(  # noqa: SLF001
        missing_required_case,
        ["vp:probe:schema:required_keys_present"],
    ) == {"kind": "claim", "payload": {}}

    with pytest.raises(ValueError, match="no deterministic repair action"):
        mod._apply_core_repairs(_case("exp5443:5443-001"), ["missing:constraint"])  # noqa: SLF001

    source_rows = mod.load_source_artifacts(REPO)["verifier_potential"]["fixture_rows"]
    accepted_row = next(row for row in source_rows if row["row_id"] == "5443-008")
    with pytest.raises(ValueError, match="not failed"):
        mod._build_case(  # noqa: SLF001
            case_id="exp5443:5443-008",
            substrate="verifier_potential",
            source_artifact=str(mod.SOURCE_ARTIFACTS[0]),
            source_row=accepted_row,
            encoded_constraints=mod._encode_vp_constraints(accepted_row),  # noqa: SLF001
            original_candidate=accepted_row["final_output"],
            recheck_context=mod._vp_recheck_context(accepted_row),  # noqa: SLF001
            source_row_checksum=accepted_row["row_checksum"],
        )


def test_deliverable_file_validates_for_scenario_verify_5458() -> None:
    """SCENARIO-VERIFY-5458: checked-in deliverable satisfies the V496 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["minimal_core_repair_ready"] is True
    assert artifact["exact_final_authority"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

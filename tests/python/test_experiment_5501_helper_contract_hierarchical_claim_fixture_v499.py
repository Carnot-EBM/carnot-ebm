"""Tests for Exp5501 hierarchical helper-contract claim fixture.

Spec refs: REQ-VERIFY-5501, SCENARIO-VERIFY-5501.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5501_helper_contract_hierarchical_claim_fixture_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path(
    "tests/python/test_experiment_5501_helper_contract_hierarchical_claim_fixture_v499.py"
)


def _evaluated_fixture() -> dict:
    fixture = mod.build_fixture()
    mod.validate_fixture(fixture)
    return fixture


def _contract(report: dict, contract_id: str) -> dict:
    return next(row for row in report["contract_reports"] if row["contract_id"] == contract_id)


def test_req_verify_5501_spec_declares_helper_contract_fixture() -> None:
    """REQ-VERIFY-5501: OpenSpec anchors fields, paths, predicates, and negatives."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5501") : spec.index("### REQ-VERIFY-5462")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5501" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.FIXTURE_RELATIVE_PATH) in section
    assert "baseless helper claim" in normalized
    assert "contradicted helper claim" in normalized
    assert "overbroad soft-preference claim" in normalized
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5501_fixture_has_hierarchical_claim_evidence() -> None:
    """SCENARIO-VERIFY-5501: contracts expose local spans, evidence, and rollups."""

    fixture = _evaluated_fixture()
    contracts = fixture["helper_contracts"]

    assert fixture["source_artifact"] == mod.SOURCE_ARTIFACT_RELATIVE_PATH.as_posix()
    assert fixture["guided_decoding_used"] is False
    assert fixture["token_steering_used"] is False
    assert len(contracts) == 7
    assert {row["contract_kind"] for row in contracts} >= {
        "acceptance_rule",
        "repair_rule",
        "negative_control_baseless",
        "negative_control_contradicted",
        "negative_control_overbroad",
    }
    assert sum(row["expected_verdict"] == "refused_unsupported" for row in contracts) == 1
    for contract in contracts:
        assert contract["statement"]
        assert contract["claim_spans"]
        assert contract["evidence_map"]
        for claim in contract["claim_spans"]:
            assert (
                contract["statement"][claim["char_start"] : claim["char_end"]] == claim["span_text"]
            )
            assert claim["expected_label"] in mod.LOCAL_LABELS
            if claim["expected_label"] == "unsupported":
                assert claim["predicate"]["type"] == "unsupported"
            else:
                assert claim["predicate"]["type"] != "unsupported"


def test_req_verify_5501_executable_predicates_label_local_claims() -> None:
    """REQ-VERIFY-5501: exact predicates decide local labels before rollup."""

    fixture = _evaluated_fixture()
    report = mod.evaluate_fixture(fixture)

    assert report["local_claim_label_accuracy"] == pytest.approx(1.0)
    assert report["rolled_up_verdict_accuracy"] == pytest.approx(1.0)
    assert report["false_accept_rate"] == pytest.approx(0.0)
    assert report["unsupported_contract_count"] == 1
    assert report["helper_contract_fixture_ready"] is True

    support = _contract(report, "support_acceptance_contract")
    assert support["observed_verdict"] == "accepted"
    assert {claim["observed_label"] for claim in support["claim_reports"]} == {"entailed"}
    assert all(claim["compiled_to"] == "executable_predicate" for claim in support["claim_reports"])

    baseless = _contract(report, "baseless_confidence_contract")
    assert baseless["observed_verdict"] == "refused_unsupported"
    assert baseless["claim_reports"][0]["compiled_to"] == "unsupported_label"
    assert baseless["claim_reports"][0]["observed_label"] == "unsupported"

    contradicted = _contract(report, "contradicted_support_contract")
    assert contradicted["observed_verdict"] == "rejected_contradicted"
    assert any(claim["observed_label"] == "contradicted" for claim in contradicted["claim_reports"])

    overbroad = _contract(report, "overbroad_primary_contract")
    assert overbroad["observed_verdict"] == "rejected_overbroad"
    assert overbroad["claim_reports"][0]["observed_label"] == "overbroad"


def test_scenario_verify_5501_repairs_count_only_exact_reference_agreements() -> None:
    """SCENARIO-VERIFY-5501: useful repair rate uses Exp5499 exact references."""

    report = mod.evaluate_fixture(_evaluated_fixture())

    assert report["repair_attempt_count"] == 2
    assert report["useful_repair_count"] == 2
    assert report["useful_repair_rate"] == pytest.approx(1.0)
    support_repair = _contract(report, "support_soft_repair_contract")
    safety_repair = _contract(report, "safety_soft_repair_contract")
    for contract in (support_repair, safety_repair):
        assert contract["repair_attempted"] is True
        assert contract["useful_repair"] is True
        repair_claims = [
            row
            for row in contract["claim_reports"]
            if row["predicate_type"] == "repair_to_reference"
        ]
        assert len(repair_claims) == 1
        assert repair_claims[0]["predicate_result"]["before_reference_agreement"] is False
        assert repair_claims[0]["predicate_result"]["after_reference_agreement"] is True


def test_req_verify_5501_artifact_writes_required_deliverable_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5501: result JSON emits the required prompt schema fields."""

    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    fixture = json.loads((tmp_path / mod.FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert fixture == mod.build_fixture()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["helper_contract_fixture_path"] == mod.FIXTURE_RELATIVE_PATH.as_posix()
    assert artifact["executable_predicate_paths"] == list(mod.EXECUTABLE_PREDICATE_PATHS)
    assert artifact["test_paths"] == [TEST_PATH.as_posix()]
    assert artifact["num_helper_contracts"] == 7
    assert artifact["unsupported_contract_count"] == 1
    assert artifact["local_claim_label_accuracy"] == pytest.approx(1.0)
    assert artifact["rolled_up_verdict_accuracy"] == pytest.approx(1.0)
    assert artifact["useful_repair_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["helper_contract_fixture_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5501_validation_fails_closed_on_schema_or_metric_drift() -> None:
    """REQ-VERIFY-5501: artifact validation rejects drift and false accepts."""

    artifact = mod.build_artifact(tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}])
    mod.validate_artifact(artifact)

    missing = deepcopy(artifact)
    missing.pop("helper_contract_fixture_path")
    with pytest.raises(ValueError, match="helper_contract_fixture_path"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_metric = deepcopy(artifact)
    bad_metric["false_accept_rate"] = 1.0
    with pytest.raises(ValueError, match="false_accept_rate"):
        mod.validate_artifact(bad_metric)

    bad_expected = deepcopy(artifact)
    bad_expected["fixture"]["helper_contracts"][0]["claim_spans"][0]["expected_label"] = (
        "contradicted"
    )
    with pytest.raises(ValueError, match="local_claim_label_accuracy"):
        mod.validate_artifact(bad_expected)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5501_predicate_dispatch_is_total_for_fixture_and_defensive() -> None:
    """REQ-VERIFY-5501: predicate dispatch covers fixture predicates and bad types."""

    fixture = _evaluated_fixture()
    for contract in fixture["helper_contracts"]:
        for claim in contract["claim_spans"]:
            result = mod.evaluate_claim_span(fixture, contract, claim)
            assert result["observed_label"] in mod.LOCAL_LABELS
            assert result["expected_label"] == claim["expected_label"]

    unsupported = fixture["helper_contracts"][-3]["claim_spans"][0]
    assert (
        mod.evaluate_claim_span(fixture, fixture["helper_contracts"][-3], unsupported)[
            "compiled_to"
        ]
        == "unsupported_label"
    )

    bad_claim = deepcopy(fixture["helper_contracts"][0]["claim_spans"][0])
    bad_claim["predicate"] = {"type": "not_a_predicate"}
    with pytest.raises(ValueError, match="not_a_predicate"):
        mod.evaluate_claim_span(fixture, fixture["helper_contracts"][0], bad_claim)

    assert mod.honest_verdict(False, ["example_blocker"]).startswith(
        "blocked: helper_contract_fixture_not_ready_example_blocker"
    )

    missing_instance = deepcopy(fixture["helper_contracts"][0]["claim_spans"][0])
    missing_instance["predicate"]["instance_id"] = "missing_instance"
    with pytest.raises(ValueError, match="missing_instance"):
        mod.evaluate_claim_span(fixture, fixture["helper_contracts"][0], missing_instance)

    missing_candidate = deepcopy(fixture["helper_contracts"][0]["claim_spans"][0])
    missing_candidate["predicate"]["candidate_id"] = "missing_candidate"
    with pytest.raises(ValueError, match="missing_candidate"):
        mod.evaluate_claim_span(fixture, fixture["helper_contracts"][0], missing_candidate)

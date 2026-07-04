"""Tests for Exp 5218 verifier-authenticity remediation apply.

Spec refs: REQ-VERIFY-5218, SCENARIO-VERIFY-5218.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from carnot import experiment_5218_verifier_authenticity_remediation_apply_v477 as mod
from carnot.verify import and_composition_verifier as and_mod
from carnot.verify import claim_isolation_uncertainty_router as router_mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def test_req_verify_5218_spec_declares_apply_contract() -> None:
    """REQ-VERIFY-5218: OpenSpec declares the remediation-apply contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5218") :]

    for marker in (
        "REQ-VERIFY-5218",
        "SCENARIO-VERIFY-5218",
        str(mod.RESULT_RELATIVE_PATH),
        "AUTHENTICITY_REMEDIATION_TYPE",
        "HEADLINE_ELIGIBLE=False",
        "LIVE_ISOLATED_CLAIM_VERIFICATION=False",
        "code_and_doc_remediation",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5218_and_compose_reports_advisory_headline_ineligible() -> None:
    """REQ-VERIFY-5218: AND composition is flagged as advisory, not headline."""

    assert and_mod.AUTHENTICITY_REMEDIATION_TYPE == "registry_flag"
    assert and_mod.AUTHENTICITY_STATUS == "advisory_adapter_harness"
    assert and_mod.HEADLINE_ELIGIBLE is False
    assert "untrained" in and_mod.HEADLINE_INELIGIBLE_REASON.lower()
    assert "advisory" in and_mod.HEADLINE_INELIGIBLE_REASON.lower()
    assert "advisory" in (and_mod.__doc__ or "").lower()
    assert "production k=5" not in (and_mod.build_default_verifier_ensemble.__doc__ or "")

    ensemble = and_mod.build_default_verifier_ensemble()
    assert ensemble.headline_eligible is False
    assert ensemble.headline_ineligible_reason == and_mod.HEADLINE_INELIGIBLE_REASON


def test_scenario_verify_5218_claim_router_reports_artifact_only_no_live_call() -> None:
    """SCENARIO-VERIFY-5218: router is an artifact ledger, not a live verifier."""

    assert router_mod.AUTHENTICITY_REMEDIATION_TYPE == "registry_flag"
    assert router_mod.AUTHENTICITY_STATUS == "artifact_routing_ledger"
    assert router_mod.HEADLINE_ELIGIBLE is False
    assert router_mod.LIVE_ISOLATED_CLAIM_VERIFICATION is False
    assert "artifact" in router_mod.HEADLINE_INELIGIBLE_REASON.lower()
    assert "no live isolated-claim verifier call" in router_mod.HEADLINE_INELIGIBLE_REASON.lower()
    assert "artifact routing ledger" in (router_mod.__doc__ or "").lower()

    metadata = router_mod.authenticity_metadata()
    assert metadata["headline_eligible"] is False
    assert metadata["live_isolated_claim_verification"] is False


def test_scenario_verify_5218_writes_valid_remediation_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5218: artifact records reduced dishonest-naming risk."""

    result_path = tmp_path / "experiment_5218.json"
    artifact = mod.run_experiment(
        result_path=result_path,
        run_date="20260704",
        duration_s=0.25,
        tests_run=["unit fixture: PASS"],
    )

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["remediation_applied"]["value"] is True
    assert artifact["remediation_type"]["value"] == "registry_flag"
    assert artifact["headline_ineligible_until_real_verification"]["value"] is True
    assert artifact["specs_updated"]["value"] is True
    assert artifact["no_research_conductor_change"]["value"] is True
    assert artifact["inference_substrate"]["value"] == "code_and_doc_remediation"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "risk reduced" in artifact["honest_verdict"]["value"]
    assert artifact["remediated_modules"]["value"] == list(mod.REMEDIATED_MODULES)


def test_req_verify_5218_validation_rejects_unwrapped_or_wrong_values() -> None:
    """REQ-VERIFY-5218: artifact schema refuses hidden or dishonest fields."""

    artifact = mod.build_artifact(run_date="20260704", duration_s=0.0, tests_run=[])
    broken = dict(artifact)
    broken["remediation_type"] = "registry_flag"
    with pytest.raises(AssertionError, match="principle-wrapped"):
        mod.validate_artifact(broken)

    broken = mod.build_artifact(run_date="20260704", duration_s=0.0, tests_run=[])
    broken["inference_substrate"] = {
        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
        "value": "live_inference",
    }
    with pytest.raises(AssertionError, match="code_and_doc_remediation"):
        mod.validate_artifact(broken)


def test_req_verify_5218_blocked_when_flags_are_missing() -> None:
    """REQ-VERIFY-5218: missing quarantine flags block instead of pretending success."""

    fake_and = types.SimpleNamespace(
        AUTHENTICITY_REMEDIATION_TYPE="docs_only",
        AUTHENTICITY_STATUS="and_composition_verifier",
        HEADLINE_ELIGIBLE=True,
        HEADLINE_INELIGIBLE_REASON="",
    )
    fake_router = types.SimpleNamespace(
        AUTHENTICITY_REMEDIATION_TYPE="registry_flag",
        AUTHENTICITY_STATUS="artifact_routing_ledger",
        HEADLINE_ELIGIBLE=False,
        LIVE_ISOLATED_CLAIM_VERIFICATION=True,
        HEADLINE_INELIGIBLE_REASON="artifact routing only",
    )

    inspection = mod.inspect_remediation(and_module=fake_and, router_module=fake_router)
    artifact = mod.build_artifact(
        run_date="20260704",
        duration_s=0.0,
        tests_run=[],
        inspection=inspection,
    )

    assert inspection["and_composition_verifier"]["remediated"] is False
    assert inspection["claim_isolation_uncertainty_router"]["remediated"] is False
    assert artifact["remediation_applied"]["value"] is False
    assert artifact["remediation_type"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    mod.validate_artifact(artifact)

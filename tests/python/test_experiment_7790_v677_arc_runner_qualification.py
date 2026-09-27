"""REQ-REPORT-7790 and REQ-ARC-WMTE-7790 qualification controls."""

from __future__ import annotations

import pytest

from carnot.experiment_7790_v677_arc_runner_qualification import (
    classify_readiness,
    current_metadata,
    verify_selector_assertions,
)


def test_scenario_report_7790_validation_requires_every_receipt() -> None:
    """SCENARIO-REPORT-7790-VALIDATION: no omitted gate can open readiness."""
    required = ["format", "spec", "sdk", "terminal"]
    receipts = [
        {"name": name, "exit_code": 0, "passed": True, "timed_out": False} for name in required
    ]
    sdk = [
        {
            "arm": arm,
            "error": None,
            "counts": {"sdk_transitions": 1},
            "policy_entry": {"policy_class": "E3AgentPolicy"},
            "actions": [],
        }
        for arm in ("off", "total", "organic")
    ]
    assert classify_readiness(receipts, required, sdk) == (True, [])
    assert classify_readiness(receipts[:-1], required, sdk) == (False, ["terminal"])
    assert classify_readiness([*receipts, receipts[0]], required, sdk) == (False, ["format"])
    assert classify_readiness(receipts, required, sdk[:-1]) == (False, ["sdk_transport"])
    bad = [*sdk[:-1], dict(sdk[-1], actions=[{"induction_attempt_count": 1}])]
    assert classify_readiness(receipts, required, bad) == (False, ["sdk_transport"])


def test_scenario_report_7790_custody_metadata_is_prospective() -> None:
    """SCENARIO-REPORT-7790-CUSTODY: V677 owns its identity and no benefit claim."""
    value = current_metadata({"experiment_id": 7776, "milestone": "2026.09.676"}, "20260927")
    assert value["experiment_id"] == 7790
    assert value["milestone"] == "2026.09.677"
    assert value["run_date"] == "20260927"
    assert value["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert value["model_invocation_counts"]["generations"] == 0
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert value["organic_runner_ready_score"] == 0
    with pytest.raises(ValueError, match="run_date"):
        current_metadata({}, "20260926")


def test_scenario_arc_wmte_7790_ige_assertion_inventory() -> None:
    """SCENARIO-ARC-WMTE-7790-IGE: all historical selector tests keep assertions."""
    source = "def test_a():\n    assert True\n    assert False is False\n"
    assert verify_selector_assertions(source, expected_tests=1, expected_assertions=2)
    with pytest.raises(ValueError, match="selector_assertions_changed"):
        verify_selector_assertions(source, expected_tests=1, expected_assertions=3)

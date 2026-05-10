"""Tests for Exp 1669 LTLZinc CerCE continual-learning adapter.

Spec: REQ-LEARN-1669, SCENARIO-LEARN-1669a.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import cerce_ledger
from carnot.pipeline import ltlzinc_adapter as mod


def test_req_learn_1669_generates_minizinc_cases_and_replay_rows() -> None:
    """REQ-LEARN-1669-1/2/3: temporal rows become CerCE replay evidence."""

    cases = mod.generate_temporal_cases()
    update = mod.build_memory_update(cases)

    assert len(cases) == mod.DEFAULT_CASE_COUNT
    assert {case["temporal_operator"] for case in cases} == set(mod.SUPPORTED_OPERATORS)
    assert len(update.replay_cases) == len(cases)
    assert update.no_model_weight_mutation is True
    assert update.prior_memory_hash != update.updated_memory_hash
    for case, replay_case in zip(cases, update.replay_cases, strict=True):
        mod.validate_temporal_case(case)
        assert mod.verify_temporal_case(case) is bool(case["expected_satisfied"])
        assert str(case["minizinc_constraint"]).startswith("constraint ")
        assert str(case["ltl_formula"])
        assert replay_case.case_id == str(case["case_id"])
        assert replay_case.retained is True
        assert replay_case.replay_failed is False
        assert replay_case.pre_violation_bound == pytest.approx(0.0)
        assert replay_case.post_violation_bound == pytest.approx(0.0)
        assert replay_case.bound_worsened is False


def test_scenario_learn_1669a_cerce_reports_zero_forgetting() -> None:
    """SCENARIO-LEARN-1669a: verified temporal cases pass the CerCE gate."""

    artifact = mod.build_artifact(
        cases=mod.generate_temporal_cases(),
        project_root="/repo",
        run_date="20260510",
        tests_run=["tests/python/test_ltlzinc_adapter.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["ltlzinc_adapter_ready"] is True
    assert artifact["cerce_ledger_ready"] is True
    assert artifact["promotion_gate_passed"] is True
    assert artifact["temporal_cases_generated"] == mod.DEFAULT_CASE_COUNT
    assert artifact["temporal_cases_retained"] == mod.DEFAULT_CASE_COUNT
    assert artifact["forgetting_rate"] == pytest.approx(0.0)
    assert artifact["cerce_nonforgetting_rate"] == pytest.approx(1.0)
    assert artifact["replay_retention_rate"] == pytest.approx(1.0)
    assert artifact["accepted_violation_count"] == 0
    assert artifact["policy_certificates_evaluated"] == 1
    assert artifact["blockers"] == []
    assert artifact["tests_run"] == ["tests/python/test_ltlzinc_adapter.py"]
    assert artifact["ledger_artifact"]["schema"] == cerce_ledger.SCHEMA
    assert artifact["ledger_artifact"]["status"] == "complete"
    assert all(result["retained"] is True for result in artifact["case_results"])


def test_req_learn_1669_blocks_forgotten_replay_case() -> None:
    """REQ-LEARN-1669-3/4: a worsened temporal replay row is counted as forgetting."""

    cases = mod.generate_temporal_cases()
    forgotten_case_id = str(cases[0]["case_id"])

    artifact = mod.build_artifact(
        cases=cases,
        forgotten_case_ids=(forgotten_case_id,),
        project_root="/repo",
        run_date="20260510",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["ltlzinc_adapter_ready"] is False
    assert artifact["cerce_ledger_ready"] is False
    assert artifact["promotion_gate_passed"] is False
    assert artifact["forgetting_rate"] == pytest.approx(1 / len(cases))
    assert artifact["cerce_nonforgetting_rate"] == pytest.approx(0.0)
    assert artifact["accepted_violation_count"] == 1
    assert "bound_worsened" in artifact["blockers"]
    forgotten = [
        result for result in artifact["case_results"] if result["case_id"] == forgotten_case_id
    ]
    assert forgotten == [
        {
            "case_id": forgotten_case_id,
            "temporal_operator": "always",
            "expected_satisfied": True,
            "local_satisfied": True,
            "local_verifier_matches_expected": True,
            "retained": False,
            "pre_violation_bound": 0.0,
            "post_violation_bound": 1.0,
            "bound_worsened": True,
        }
    ]


def test_req_learn_1669_run_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1669-5: run_experiment writes the terminal JSON artifact."""

    output_path = tmp_path / "results" / mod.OUTPUT_FILE

    artifact = mod.run_experiment(
        output_path=output_path,
        project_root=tmp_path,
        run_date="20260510",
        tests_run=["pytest targeted"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["tests_run"] == ["pytest targeted"]
    assert artifact["honest_verdict"] == "complete: ltlzinc_cerce_nonforgetting_passed"


def test_req_learn_1669_validation_rejects_malformed_artifacts() -> None:
    """REQ-LEARN-1669-5: terminal schema validation catches impossible reports."""

    valid = mod.build_artifact(cases=mod.generate_temporal_cases(), project_root="/repo")
    until_never = mod.make_temporal_case(
        "until-never",
        "until",
        "released",
        [{"locked": True, "released": False}],
        False,
        guard_signal="locked",
    )
    assert mod.verify_temporal_case(until_never) is False

    missing = dict(valid)
    del missing["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="unsupported schema"):
        mod.validate_artifact(dict(valid, schema="wrong"))

    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(valid, status="in_progress"))

    with pytest.raises(AssertionError, match="forgetting_rate"):
        mod.validate_artifact(dict(valid, forgetting_rate=1.5))

    with pytest.raises(AssertionError, match="cerce_nonforgetting_rate"):
        mod.validate_artifact(dict(valid, cerce_nonforgetting_rate=-0.5))

    with pytest.raises(AssertionError, match="case_results"):
        mod.validate_artifact(dict(valid, case_results=[]))

    with pytest.raises(AssertionError, match="temporal_cases_retained"):
        mod.validate_artifact(dict(valid, temporal_cases_retained=0))

    with pytest.raises(AssertionError, match="policy certificate count"):
        mod.validate_artifact(dict(valid, policy_certificates_evaluated=99))

    with pytest.raises(AssertionError, match="complete artifact is invalid"):
        mod.validate_artifact(
            dict(
                valid,
                ltlzinc_adapter_ready=False,
                cerce_ledger_ready=False,
                promotion_gate_passed=False,
                forgetting_rate=0.25,
                cerce_nonforgetting_rate=0.75,
                blockers=["bound_worsened"],
            )
        )

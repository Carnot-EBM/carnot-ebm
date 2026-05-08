"""Tests for Exp 1542 ARM/EBT soft-value diagnostic.

Spec: REQ-VERIFY-1542, SCENARIO-VERIFY-1542.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import arm_ebm_soft_value_diagnostic as exp


def test_req_verify_1542_metric_computation_uses_supported_signals_only() -> None:
    """REQ-VERIFY-1542: correlations are computed only from available signals."""

    cases = [
        _case("a", deterministic_accept=True, energy=0.0, soft=-0.1, prefix=0.0),
        _case("b", deterministic_accept=True, energy=0.2, soft=-0.2, prefix=0.1),
        _case("c", deterministic_accept=False, energy=4.0, soft=-3.5, prefix=0.9),
        _case("d", deterministic_accept=False, energy=5.0, soft=-4.0, prefix=1.0),
    ]

    summary = exp.evaluate_diagnostic(cases, focused_tests_passed=True)

    assert summary["arm_ebm_diagnostic_ready"] is True
    assert summary["diagnostic_cases"] == 4
    assert summary["carnot_energy_available"] is True
    assert summary["logprob_available"] is True
    assert summary["energy_label_correlation"] > 0.9
    assert summary["soft_value_label_correlation"] < -0.9
    assert summary["routing_auc"] == pytest.approx(1.0)
    assert summary["deterministic_validators_final_authority"] is True
    assert summary["no_model_weight_mutation"] is True


def test_req_verify_1542_missing_logprobs_blocks_only_soft_value_metric() -> None:
    """REQ-VERIFY-1542: missing logprobs still allows the energy-only diagnostic."""

    cases = [
        _case("a", deterministic_accept=True, energy=0.0, soft=None, prefix=0.0),
        _case("b", deterministic_accept=False, energy=4.0, soft=None, prefix=0.8),
    ]

    summary = exp.evaluate_diagnostic(cases, focused_tests_passed=True)

    assert summary["arm_ebm_diagnostic_ready"] is True
    assert summary["logprob_available"] is False
    assert summary["soft_value_label_correlation"] is None
    assert summary["energy_label_correlation"] == pytest.approx(1.0)
    assert "soft_value_logprobs_unavailable_carnot_energy_only" in summary["blockers"]


def test_req_verify_1542_authority_boundary_ignores_soft_value_and_prefix_risk() -> None:
    """REQ-VERIFY-1542: soft values and BEAVER risk never override validators."""

    rejected_with_good_soft_value = _case(
        "reject",
        deterministic_accept=False,
        energy=0.0,
        soft=0.0,
        prefix=0.0,
    )
    accepted_with_high_risk = _case(
        "accept",
        deterministic_accept=True,
        energy=99.0,
        soft=-99.0,
        prefix=1.0,
    )

    assert exp.final_authority_accept(rejected_with_good_soft_value) is False
    assert exp.final_authority_accept(accepted_with_high_risk) is True


def test_req_verify_1542_empty_inputs_are_honest_not_ready() -> None:
    """REQ-VERIFY-1542: empty or unscored diagnostics report blockers."""

    summary = exp.evaluate_diagnostic([], focused_tests_passed=False)

    assert summary["arm_ebm_diagnostic_ready"] is False
    assert summary["energy_label_correlation"] is None
    assert summary["routing_auc"] is None
    assert "carnot_energy_unavailable" in summary["blockers"]
    assert "no_diagnostic_cases_loaded" in summary["blockers"]
    assert "focused_tests_not_passed" in summary["blockers"]


def test_req_verify_1542_case_builder_filters_bad_rows_and_extracts_soft_values() -> None:
    """REQ-VERIFY-1542: source adapters filter malformed rows and obey limits."""

    cases = exp.build_diagnostic_cases(
        satquest_rows=[
            {},
            _satquest_row("sat-soft", correct=True, energy=0.0, model_hf_id=exp.MODEL_SPECS[0])
            | {"token_logprob": -0.25},
            _satquest_row("sat-extra", correct=False, energy=50.0, model_hf_id=exp.MODEL_SPECS[0]),
        ],
        runtime_rows=[
            {"row_type": "summary"},
            {"row_type": "contract_case"},
            _runtime_row("contract-soft", final_accept=False, expected=False),
            _runtime_row("contract-extra", final_accept=True, expected=True),
        ],
        beaver_artifact={
            "high_risk_instances": [
                {},
                _beaver_instance("beaver-soft", deterministic_accept=False, risk=0.5)
                | {"topk_logprobs": [-1.0, -3.0]},
                _beaver_instance("beaver-extra", deterministic_accept=True, risk=0.0),
            ],
        },
        case_limit_per_source=1,
    )

    assert [case["source_kind"] for case in cases] == [
        "satquest",
        "runtime_contract",
        "beaver_prefix",
    ]
    assert cases[0]["soft_value_score"] == pytest.approx(-0.25)
    assert cases[2]["soft_value_score"] == pytest.approx(-2.0)


def test_req_verify_1542_validate_artifact_rejects_schema_and_authority_breaks() -> None:
    """REQ-VERIFY-1542: terminal artifacts enforce schema and authority invariants."""

    artifact = {
        "status": "complete",
        "milestone": ".118",
        "arm_ebm_diagnostic_ready": True,
        "model_specs": list(exp.MODEL_SPECS),
        "live_sota_model_inference_used": True,
        "diagnostic_cases": 2,
        "logprob_available": False,
        "carnot_energy_available": True,
        "energy_label_correlation": 1.0,
        "soft_value_label_correlation": None,
        "routing_auc": 1.0,
        "deterministic_validators_final_authority": True,
        "no_model_weight_mutation": True,
        "diagnostic_report_path": "results/report.jsonl",
        "focused_tests_passed": True,
        "honest_verdict": "complete: ready",
    }

    exp.validate_artifact(artifact)
    for mutation, message in [
        ({"honest_verdict": "blocked: nope"}, "allowed terminal prefix"),
        ({"focused_tests_passed": False}, "focused tests"),
        ({"carnot_energy_available": False}, "Carnot energy"),
        ({"deterministic_validators_final_authority": False}, "final authority"),
        ({"no_model_weight_mutation": False}, "model weights"),
    ]:
        broken = artifact | mutation
        with pytest.raises(AssertionError, match=message):
            exp.validate_artifact(broken)

    missing = dict(artifact)
    del missing["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact(missing)


def test_scenario_verify_1542_runner_writes_terminal_report(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1542: runner combines SATQuest, contract, and BEAVER rows."""

    satquest_manifest = tmp_path / "satquest.jsonl"
    runtime_manifest = tmp_path / "runtime.jsonl"
    satquest_artifact = tmp_path / "experiment_1536.json"
    beaver_artifact = tmp_path / "experiment_1537.json"
    output = tmp_path / "experiment_1542.json"
    report = tmp_path / "diagnostic.jsonl"

    _write_jsonl(
        satquest_manifest,
        [
            _satquest_row("sat-ok", correct=True, energy=0.0, model_hf_id=exp.MODEL_SPECS[0]),
            _satquest_row("sat-bad", correct=False, energy=50.0, model_hf_id=exp.MODEL_SPECS[0]),
        ],
    )
    _write_jsonl(
        runtime_manifest,
        [
            _runtime_row("contract-ok", final_accept=True, expected=True),
            _runtime_row("contract-bad", final_accept=False, expected=False),
        ],
    )
    _write_json(
        satquest_artifact,
        {
            "live_sota_model_inference_used": True,
            "models_used": [exp.MODEL_SPECS[0]],
        },
    )
    _write_json(
        beaver_artifact,
        {
            "live_sota_model_inference_used": True,
            "high_risk_instances": [
                _beaver_instance("contract-ok", deterministic_accept=True, risk=0.0),
                _beaver_instance("contract-bad", deterministic_accept=False, risk=1.0),
            ],
        },
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        satquest_artifact_path=satquest_artifact,
        satquest_manifest_path=satquest_manifest,
        runtime_manifest_path=runtime_manifest,
        beaver_artifact_path=beaver_artifact,
        output_path=output,
        diagnostic_report_path=report,
        focused_tests_passed=True,
    )
    report_rows = _read_jsonl(report)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["arm_ebm_diagnostic_ready"] is True
    assert artifact["model_specs"] == list(exp.MODEL_SPECS)
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["diagnostic_cases"] >= 4
    assert artifact["logprob_available"] is False
    assert artifact["carnot_energy_available"] is True
    assert artifact["deterministic_validators_final_authority"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["diagnostic_report_path"].endswith("diagnostic.jsonl")
    assert report_rows[-1]["row_type"] == "summary"
    assert all(row.get("soft_value_used_as_authority") is False for row in report_rows[:-1])


def _case(
    case_id: str,
    *,
    deterministic_accept: bool,
    energy: float,
    soft: float | None,
    prefix: float,
) -> dict[str, Any]:
    return {
        "diagnostic_case_id": case_id,
        "case_id": case_id,
        "source_kind": "fixture",
        "source_family": "fixture",
        "deterministic_accept": deterministic_accept,
        "deterministic_label": "accept" if deterministic_accept else "reject",
        "carnot_energy_score": energy,
        "soft_value_score": soft,
        "prefix_risk_score": prefix,
        "soft_value_used_as_authority": False,
    }


def _satquest_row(
    case_id: str,
    *,
    correct: bool,
    energy: float,
    model_hf_id: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": "fixture_sat",
        "model_hf_id": model_hf_id,
        "generation_source": "live_sota_llamacpp",
        "baseline": {
            "answer": "SAT" if correct else "UNSAT",
            "classification": "oracle_agreement" if correct else "wrong_label",
            "correct": correct,
            "energy": energy,
        },
        "solver_oracle": {"label": "SAT"},
        "parse_result": {"parse_ok": True},
    }


def _runtime_row(case_id: str, *, final_accept: bool, expected: bool) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_case_id": case_id,
        "source_family": "fixture_contract",
        "expected_label": expected,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
    }


def _beaver_instance(case_id: str, *, deterministic_accept: bool, risk: float) -> dict[str, Any]:
    return {
        "contract_case_id": case_id,
        "source_family": "fixture_contract",
        "decoder_mode": "fixture",
        "model_hf_id": exp.MODEL_SPECS[0],
        "risk_upper_bound": risk,
        "deterministic_validator_accept": deterministic_accept,
        "bound_used_as_authority": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

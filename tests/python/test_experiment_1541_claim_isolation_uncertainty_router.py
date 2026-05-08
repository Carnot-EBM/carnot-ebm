"""Tests for Exp 1541 claim-isolation uncertainty router.

Spec: REQ-VERIFY-1541, SCENARIO-VERIFY-1541.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import claim_isolation_uncertainty_router as exp


def test_req_verify_1541_extracts_claims_from_runtime_satquest_and_product_line() -> None:
    """REQ-VERIFY-1541: claim extraction preserves source and validator evidence."""

    cases = exp.build_bounded_case_set(
        runtime_rows=[_runtime_case("runtime-ok", expected_label=True, final_accept=True)],
        satquest_rows=[_satquest_case("sat-risk", solver_label="UNSAT", answer="SAT")],
        product_rows=[_product_case("product-ok")],
        case_limit_per_source=2,
    )

    assert [case["source_kind"] for case in cases] == [
        "runtime_contract",
        "satquest",
        "product_line",
    ]
    assert [case["router_case_id"] for case in cases] == [
        "runtime_contract:runtime-ok",
        "satquest:sat-risk",
        "product_line:product-ok",
    ]
    assert cases[0]["claims"][0] == {
        "claim_id": "runtime_contract:runtime-ok:claim:001",
        "claim_text": "runtime contract runtime-ok final deterministic decision is accept",
        "source_kind": "runtime_contract",
        "source_family": "safe_dsl",
        "deterministic_accept": True,
    }
    assert cases[1]["claims"][0]["claim_text"] == (
        "SATQuest answer SAT should match solver label UNSAT"
    )
    assert cases[1]["validator_disagreement"] is True
    assert cases[2]["claims"][0]["claim_text"] == (
        "product-line feature selection for product-ok should match the solver oracle"
    )


def test_req_verify_1541_routing_policy_is_deterministic() -> None:
    """REQ-VERIFY-1541: uncertainty, prefix risk, and disagreement route stably."""

    cases = exp.build_bounded_case_set(
        runtime_rows=[
            _runtime_case("runtime-prefix-risk", expected_label=True, final_accept=True),
            _runtime_case("runtime-low-risk", expected_label=True, final_accept=True),
        ],
        satquest_rows=[_satquest_case("sat-uncertain", solver_label="SAT", answer=None)],
        product_rows=[],
        case_limit_per_source=4,
    )
    risk = {"runtime-prefix-risk": 0.91}

    first = [exp.route_case(case, prefix_risk_by_case=risk) for case in cases]
    second = [exp.route_case(case, prefix_risk_by_case=risk) for case in cases]

    assert first == second
    assert first[0]["routed"] is True
    assert first[0]["routing_reasons"] == ["prefix_risk"]
    assert first[1]["routed"] is False
    assert first[1]["routing_reasons"] == ["low_risk_bypass"]
    assert first[2]["routed"] is True
    assert first[2]["routing_reasons"] == ["uncertainty", "validator_disagreement"]


def test_scenario_verify_1541_budget_accounting_reports_zero_false_accepts() -> None:
    """SCENARIO-VERIFY-1541: routed isolation reduces calls without false accepts."""

    cases = exp.build_bounded_case_set(
        runtime_rows=[_runtime_case("runtime-reject", expected_label=False, final_accept=False)],
        satquest_rows=[
            _satquest_case(
                "sat-self-false-accept",
                solver_label="UNSAT",
                answer="SAT",
                self_declared_accept=True,
            )
        ],
        product_rows=[_product_case("product-ok")],
        case_limit_per_source=2,
    )

    evaluation = exp.evaluate_routing(
        cases,
        prefix_risk_by_case={},
        focused_tests_passed=True,
    )
    summary = evaluation["summary"]

    assert summary["uncertainty_router_ready"] is True
    assert summary["cases_loaded"] == 3
    assert summary["claims_extracted"] == 3
    assert summary["routed_cases"] == 1
    assert summary["verifier_calls_full_context"] == 3
    assert summary["verifier_calls_claim_isolated"] == 1
    assert summary["budget_delta"] == -2
    assert summary["budget_improvement_claimed"] is True
    assert summary["full_context_accept_rate"] == pytest.approx(2 / 3)
    assert summary["claim_isolated_accept_rate"] == pytest.approx(0.0)
    assert summary["disagreements"] == 1
    assert summary["false_accept_count"] == 0
    assert summary["false_accept_rate"] == pytest.approx(0.0)
    assert evaluation["rows"][1]["final_accept"] is False
    assert evaluation["rows"][-1]["row_type"] == "summary"


def test_scenario_verify_1541_runner_writes_terminal_artifacts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1541: runner loads prior artifacts and writes the schema."""

    claim_artifact = tmp_path / "experiment_1525.json"
    claim_manifest = tmp_path / "march_claim_isolation_1525.jsonl"
    beaver_artifact = tmp_path / "experiment_1537.json"
    runtime_manifest = tmp_path / "runtime_contracts.jsonl"
    satquest_manifest = tmp_path / "satquest.jsonl"
    product_manifest = tmp_path / "product.jsonl"
    output = tmp_path / "experiment_1541.json"
    manifest = tmp_path / "router.jsonl"
    policy = tmp_path / "policy.json"
    _write_json(
        claim_artifact,
        {"live_sota_model_inference_used": True, "budget_delta": 3},
    )
    _write_jsonl(claim_manifest, [{"row_type": "summary"}])
    _write_json(
        beaver_artifact,
        {
            "live_sota_model_inference_used": True,
            "high_risk_instances": [
                [],
                {"contract_case_id": "", "risk_upper_bound": 1.0},
                {"contract_case_id": "runtime-risk", "risk_upper_bound": 0.91},
                {"contract_case_id": "runtime-risk", "risk_upper_bound": 0.25},
            ],
        },
    )
    _write_jsonl(
        runtime_manifest,
        [
            {"row_type": "summary"},
            {"row_type": "contract_case"},
            _runtime_case("runtime-risk", expected_label=True, final_accept=True),
            _runtime_case("runtime-low", expected_label=True, final_accept=True),
            _runtime_case("runtime-over-limit", expected_label=True, final_accept=True),
        ],
    )
    _write_jsonl(
        satquest_manifest,
        [
            {},
            _satquest_case(
                "sat-self-false-accept",
                solver_label="UNSAT",
                answer="SAT",
                self_declared_accept=True,
            ),
            _satquest_case("sat-over-limit", solver_label="SAT", answer="SAT"),
        ],
    )
    _write_jsonl(
        product_manifest,
        [
            {},
            _product_case("product-ok"),
            _product_case("product-over-limit"),
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        manifest_path=manifest,
        routing_policy_path=policy,
        claim_isolation_artifact_path=claim_artifact,
        claim_isolation_manifest_path=claim_manifest,
        beaver_artifact_path=beaver_artifact,
        runtime_manifest_path=runtime_manifest,
        satquest_manifest_path=satquest_manifest,
        product_manifest_path=product_manifest,
        focused_tests_passed=True,
        case_limit_per_source=2,
    )
    rows = _read_jsonl(manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert json.loads(policy.read_text(encoding="utf-8")) == exp.DEFAULT_ROUTING_POLICY
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["uncertainty_router_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cases_loaded"] == 6
    assert artifact["claims_extracted"] == 6
    assert artifact["routed_cases"] == 2
    assert artifact["budget_delta"] == -4
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["prior_claim_isolation_budget_delta"] == 3
    assert artifact["routing_policy_path"] == str(policy)
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "summary"
    exp.validate_artifact(artifact)
    assert exp.load_beaver_prefix_risk(
        {
            "high_risk_instances": [
                [],
                {"contract_case_id": ""},
                {"contract_case_id": "case-a", "risk_upper_bound": 0.7},
            ]
        }
    ) == {"case-a": 0.7}
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: nope"})
    bad_verdict = dict(artifact, honest_verdict="blocked")
    with pytest.raises(AssertionError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)
    with pytest.raises(AssertionError, match="focused tests"):
        exp.validate_artifact(dict(artifact, focused_tests_passed=False))
    with pytest.raises(AssertionError, match="zero false accepts"):
        exp.validate_artifact(dict(artifact, false_accept_rate=0.5))
    with pytest.raises(AssertionError, match="some but not all"):
        exp.validate_artifact(dict(artifact, routed_cases=artifact["cases_loaded"]))
    blocked = exp.run_experiment(
        project_root=tmp_path / "missing-sources",
        output_path=tmp_path / "blocked_1541.json",
        manifest_path=tmp_path / "blocked_router.jsonl",
        routing_policy_path=tmp_path / "blocked_policy.json",
        focused_tests_passed=False,
    )
    assert blocked["status"] == "blocked"
    assert "no_extractable_router_cases" in blocked["blockers"]


def _runtime_case(case_id: str, *, expected_label: bool | None, final_accept: bool) -> dict:
    return {
        "row_type": "contract_case",
        "contract_case_id": case_id,
        "source_family": "safe_dsl",
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "proposed_output": f"candidate output for {case_id}",
    }


def _satquest_case(
    case_id: str,
    *,
    solver_label: str,
    answer: str | None,
    self_declared_accept: bool | None = None,
) -> dict:
    parse_ok = answer is not None
    correct = answer == solver_label
    return {
        "case_id": case_id,
        "family": "unit_propagation_sat",
        "model_hf_id": exp.MODEL_SPECS[0],
        "baseline": {
            "answer": answer,
            "classification": "oracle_agreement" if correct else "wrong_label",
            "correct": correct,
            "energy": 0.0 if correct else 51.0,
            "parse_error": None if parse_ok else "no_json_object",
        },
        "parse_result": {
            "baseline_answer": answer,
            "model_declared_accept": self_declared_accept,
            "parse_ok": parse_ok,
            "parse_error": None if parse_ok else "no_json_object",
        },
        "solver_oracle": {"label": solver_label},
        "verifier": {"self_verifier_false_accept": bool(self_declared_accept and not correct)},
    }


def _product_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "model_id": "ProductFixture",
        "baseline_result": {
            "classification": "parse_failure",
            "oracle_agrees": False,
            "parse_ok": False,
        },
        "oracle_result": {
            "classification": "oracle_agreement",
            "oracle_agrees": True,
        },
        "policy_result": {"accepted": True, "false_accept": False},
        "verifier_result": {"accepted": True, "self_verifier_false_accept": False},
        "final_answer": {"selected_features": ["Core", "FeatureA"]},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

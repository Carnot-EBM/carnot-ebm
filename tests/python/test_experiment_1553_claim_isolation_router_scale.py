"""Tests for Exp 1553 claim-isolation router scale.

Spec: REQ-VERIFY-1553, SCENARIO-VERIFY-1553.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import claim_isolation_router_scale as exp


def test_req_verify_1553_threshold_policy_routes_only_risky_cases() -> None:
    """REQ-VERIFY-1553: threshold-risky and residual-drift cases route stably."""

    low = _case("runtime_contract", "low", uncertainty_score=0.499, prefix_risk=0.749)
    uncertainty_edge = _case("runtime_contract", "uncertain", uncertainty_score=0.5)
    prefix_edge = _case("runtime_contract", "prefix", prefix_risk=0.75)
    residual = _case("residual_drift", "drift", residual_failure_classification="satisfiable_drift")

    decisions = [
        exp.route_case(case, prefix_risk_by_case={case["case_id"]: case["prefix_risk"]})
        for case in (low, uncertainty_edge, prefix_edge, residual)
    ]

    assert decisions[0]["routed"] is False
    assert decisions[0]["routing_reasons"] == ["low_risk_bypass"]
    assert decisions[1]["routed"] is True
    assert decisions[1]["routing_reasons"] == ["uncertainty"]
    assert decisions[2]["routed"] is True
    assert decisions[2]["routing_reasons"] == ["prefix_risk"]
    assert decisions[3]["routed"] is True
    assert decisions[3]["routing_reasons"] == ["residual_drift"]
    assert decisions == [
        exp.route_case(case, prefix_risk_by_case={case["case_id"]: case["prefix_risk"]})
        for case in (low, uncertainty_edge, prefix_edge, residual)
    ]


def test_req_verify_1553_claim_extraction_schema_covers_four_sources() -> None:
    """REQ-VERIFY-1553: scaled claim rows preserve gate and validator evidence."""

    cases = exp.build_scaled_case_set(
        runtime_rows=[{"row_type": "summary"}, _runtime_row("runtime-ok")],
        satquest_rows=[_satquest_row("sat-hidden", answer="SAT", solver_label="UNSAT")],
        product_rows=[_product_row("product-ok")],
        residual_rows=[
            {"row_type": "residual_drift_repair_summary"},
            _residual_row(
                "drift-contradiction", accepted=True, failure_classification="true_contradiction"
            ),
        ],
        case_target=4,
    )

    assert [case["source_kind"] for case in cases] == [
        "runtime_contract",
        "satquest",
        "product_line",
        "residual_drift",
    ]
    for case in cases:
        assert case["unified_gate_checked"] is True
        assert case["deterministic_validator_final_authority"] is True
        assert len(case["claims"]) == 1
        assert set(case["claims"][0]) == {
            "claim_id",
            "claim_text",
            "source_kind",
            "source_family",
            "source_case_id",
            "deterministic_accept",
            "hidden_from_full_context",
        }
        assert case["claims"][0]["hidden_from_full_context"] is True


def test_scenario_verify_1553_hidden_deterministic_failure_is_rejected() -> None:
    """SCENARIO-VERIFY-1553: the unified gate blocks hidden false accepts."""

    hidden_failure = _case(
        "satquest",
        "hidden",
        deterministic_accept=False,
        full_context_accept=True,
        claim_isolated_accept=True,
        validator_disagreement=True,
    )

    evaluation = exp.evaluate_scaled_routing(
        [hidden_failure],
        prefix_risk_by_case={},
        unified_contract_gate_ready=True,
        focused_tests_passed=True,
    )
    row = evaluation["rows"][0]
    summary = evaluation["summary"]

    assert row["routed"] is True
    assert row["final_accept"] is False
    assert row["false_accept"] is False
    assert summary["missed_failure_count"] == 0
    assert summary["false_accept_rate"] == pytest.approx(0.0)


def test_scenario_verify_1553_runner_writes_ready_scale_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1553: runner writes 75+ safe budget-reduced metrics."""

    output = tmp_path / "experiment_1553.json"
    manifest = tmp_path / "router_scale.jsonl"
    policy = tmp_path / "policy.json"
    router_artifact = tmp_path / "experiment_1541.json"
    gate_artifact = tmp_path / "experiment_1551.json"
    runtime_manifest = tmp_path / "runtime.jsonl"
    satquest_manifest = tmp_path / "satquest.jsonl"
    product_manifest = tmp_path / "product.jsonl"
    residual_manifest = tmp_path / "residual.jsonl"

    _write_json(
        router_artifact,
        {
            "status": "complete",
            "uncertainty_router_ready": True,
            "live_sota_model_inference_used": True,
            "routing_policy_path": str(policy),
            "model_specs": list(exp.MODEL_SPECS),
            "high_risk_instances": [
                {"contract_case_id": "runtime-0", "risk_upper_bound": 0.91},
            ],
        },
    )
    _write_json(
        gate_artifact,
        {
            "status": "complete",
            "unified_contract_gate_ready": True,
            "live_sota_model_inference_used": True,
            "model_availability_blockers": [],
            "models_used": [exp.MODEL_SPECS[0]],
        },
    )
    _write_json(policy, exp.DEFAULT_ROUTING_POLICY)
    _write_jsonl(
        runtime_manifest,
        [{"row_type": "summary"}, *[_runtime_row(f"runtime-{idx}") for idx in range(40)]],
    )
    _write_jsonl(
        satquest_manifest,
        [
            _satquest_row("sat-declared", declared_accept=True),
            *[_satquest_row(f"sat-{idx}") for idx in range(17)],
        ],
    )
    _write_jsonl(product_manifest, [_product_row(f"product-{idx}") for idx in range(6)])
    _write_jsonl(
        residual_manifest,
        [
            {"row_type": "residual_drift_repair_summary"},
            _residual_row(
                "drift-contradiction", accepted=False, failure_classification="true_contradiction"
            ),
            *[_residual_row(f"drift-{idx}", accepted=idx % 2 == 0) for idx in range(29)],
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        manifest_path=manifest,
        router_policy_path=policy,
        router_artifact_path=router_artifact,
        unified_gate_artifact_path=gate_artifact,
        runtime_manifest_path=runtime_manifest,
        satquest_manifest_path=satquest_manifest,
        product_manifest_path=product_manifest,
        residual_manifest_path=residual_manifest,
        focused_tests_passed=True,
        case_target=75,
    )
    rows = _read_jsonl(manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["claim_isolation_router_scale_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cases_total"] == 75
    assert 0 < artifact["routed_cases"] < artifact["cases_total"]
    assert artifact["full_context_cases"] == 75
    assert artifact["claims_extracted"] == 75
    assert artifact["budget_delta"] < 0
    assert artifact["budget_reduced"] is True
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["missed_failure_count"] == 0
    assert artifact["router_policy_path"] == str(policy)
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "summary"
    exp.validate_artifact(artifact)

    blocked = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "blocked_1553.json",
        manifest_path=tmp_path / "blocked_router_scale.jsonl",
        router_policy_path=policy,
        router_artifact_path=router_artifact,
        unified_gate_artifact_path=gate_artifact,
        runtime_manifest_path=runtime_manifest,
        satquest_manifest_path=satquest_manifest,
        product_manifest_path=product_manifest,
        residual_manifest_path=residual_manifest,
        focused_tests_passed=False,
        case_target=1,
    )
    assert blocked["claim_isolation_router_scale_ready"] is False
    assert "focused_tests_not_passed" in blocked["blockers"]


def test_req_verify_1553_readiness_validator_fails_closed() -> None:
    """REQ-VERIFY-1553: ready artifacts require tests, safety, and savings."""

    artifact = exp.artifact_from_summary(
        status="complete",
        summary={
            "claim_isolation_router_scale_ready": True,
            "cases_total": 75,
            "routed_cases": 25,
            "full_context_cases": 75,
            "claims_extracted": 75,
            "budget_delta": -50,
            "budget_reduced": True,
            "false_accept_rate": 0.0,
            "missed_failure_count": 0,
            "source_kinds_loaded": [
                "product_line",
                "residual_drift",
                "runtime_contract",
                "satquest",
            ],
        },
        router_policy_path=Path("policy.json"),
        focused_tests_passed=True,
        live_sota_model_inference_used=True,
        blockers=[],
    )

    exp.validate_artifact(artifact)
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: nope"})
    with pytest.raises(AssertionError, match="honest_verdict"):
        exp.validate_artifact(dict(artifact, honest_verdict="blocked"))
    with pytest.raises(AssertionError, match="focused tests"):
        exp.validate_artifact(dict(artifact, focused_tests_passed=False))
    with pytest.raises(AssertionError, match="zero false accepts"):
        exp.validate_artifact(dict(artifact, false_accept_rate=0.1))
    with pytest.raises(AssertionError, match="budget reduction"):
        exp.validate_artifact(dict(artifact, budget_reduced=False))
    with pytest.raises(AssertionError, match="75 cases"):
        exp.validate_artifact(dict(artifact, cases_total=74))
    with pytest.raises(AssertionError, match="some but not all"):
        exp.validate_artifact(dict(artifact, routed_cases=75))
    assert exp._predecessor_blockers({}, {}) == [  # noqa: SLF001
        "exp1541_uncertainty_router_not_ready",
        "exp1551_unified_contract_gate_not_ready",
    ]


def _case(
    source_kind: str,
    case_id: str,
    *,
    deterministic_accept: bool = True,
    full_context_accept: bool = True,
    claim_isolated_accept: bool = True,
    uncertainty_score: float = 0.0,
    prefix_risk: float = 0.0,
    validator_disagreement: bool = False,
    residual_failure_classification: str | None = None,
) -> dict:
    return {
        "router_case_id": f"{source_kind}:{case_id}",
        "case_id": case_id,
        "source_kind": source_kind,
        "source_family": source_kind,
        "deterministic_accept": deterministic_accept,
        "full_context_accept": full_context_accept,
        "claim_isolated_accept": claim_isolated_accept,
        "uncertainty_score": uncertainty_score,
        "prefix_risk": prefix_risk,
        "validator_disagreement": validator_disagreement,
        "residual_failure_classification": residual_failure_classification,
        "claims": [
            {
                "claim_id": f"{source_kind}:{case_id}:claim:001",
                "claim_text": f"{source_kind} claim for {case_id}",
                "source_kind": source_kind,
                "source_family": source_kind,
                "source_case_id": case_id,
                "deterministic_accept": deterministic_accept,
                "hidden_from_full_context": True,
            }
        ],
    }


def _runtime_row(case_id: str) -> dict:
    return {
        "row_type": "contract_case",
        "contract_case_id": case_id,
        "source_family": "safe_dsl",
        "expected_label": True,
        "final_deterministic_accept": True,
        "final_deterministic_decision": "accept",
        "model_hf_id": exp.MODEL_SPECS[0],
    }


def _satquest_row(
    case_id: str,
    *,
    answer: str = "UNSAT",
    solver_label: str = "UNSAT",
    declared_accept: bool | None = None,
) -> dict:
    correct = answer == solver_label
    return {
        "case_id": case_id,
        "family": "unit_propagation_sat",
        "model_hf_id": exp.MODEL_SPECS[0],
        "baseline": {"answer": answer, "correct": correct, "energy": 51.0 if not correct else 0.0},
        "parse_result": {"parse_ok": True, "model_declared_accept": declared_accept},
        "solver_oracle": {"label": solver_label},
        "verifier": {"self_verifier_false_accept": not correct},
    }


def _product_row(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "model_id": "RetailCheckout",
        "model_hf_id": exp.MODEL_SPECS[0],
        "oracle_result": {"oracle_agrees": True},
        "policy_result": {"accepted": True, "false_accept": False},
        "verifier_result": {"accepted": True},
    }


def _residual_row(
    case_id: str,
    *,
    accepted: bool,
    failure_classification: str = "satisfiable_drift",
) -> dict:
    return {
        "row_type": "residual_drift_repair_case",
        "case_id": case_id,
        "source_domain": "satquest",
        "failure_classification": failure_classification,
        "accepted": accepted,
        "attempted": True,
        "replay_passed": accepted,
        "false_accept": False,
        "rejected_false_accept": False,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

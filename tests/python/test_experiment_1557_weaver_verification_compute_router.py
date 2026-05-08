"""Tests for Exp 1557 Weaver verification-compute router.

Spec: REQ-VERIFY-1557, SCENARIO-VERIFY-1557.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import verification_compute_router as exp


def test_req_verify_1557_route_selection_uses_cheap_path_and_fallback() -> None:
    """REQ-VERIFY-1557: weak signals choose compute, not final authority."""

    low_risk = _candidate(
        "satquest:low",
        source_kind="satquest",
        weak_signals={"claim_uncertainty_score": 0.1},
        deterministic_outcomes={"sat_solver": True, "unified_contract_gate": True},
    )
    high_risk = _candidate(
        "satquest:high",
        source_kind="satquest",
        weak_signals={"beaver_prefix_risk": 0.91},
        deterministic_outcomes={"sat_solver": True, "unified_contract_gate": True},
    )
    energy_parse_risk = _candidate(
        "mystery:energy",
        source_kind="mystery",
        weak_signals={
            "carnot_energy_score": 51.0,
            "parse_failure": True,
            "automata_reject": True,
            "telemetry_routing_score": True,
        },
        deterministic_outcomes={"custom_validator": True, "unified_contract_gate": True},
    )
    unknown_low_risk = _candidate(
        "mystery:cheap",
        source_kind="mystery",
        weak_signals={"claim_uncertainty_score": 0.0},
        deterministic_outcomes={"custom_validator": True, "unified_contract_gate": True},
    )

    cheap = exp.route_candidate(low_risk)
    fallback = exp.route_candidate(high_risk)
    fallback_with_unknown_primary = exp.route_candidate(energy_parse_risk)
    cheap_with_unknown_primary = exp.route_candidate(unknown_low_risk)

    assert cheap["route"] == "cheap_primary_validator"
    assert cheap["selected_deterministic_validators"] == ["sat_solver"]
    assert "claim_uncertainty" not in cheap["routing_reasons"]
    assert fallback["route"] == "deterministic_fallback"
    assert fallback["selected_deterministic_validators"] == [
        "sat_solver",
        "unified_contract_gate",
    ]
    assert fallback["routing_reasons"] == ["beaver_prefix_risk"]
    assert fallback_with_unknown_primary["selected_deterministic_validators"] == [
        "custom_validator",
        "unified_contract_gate",
    ]
    assert fallback_with_unknown_primary["routing_reasons"] == [
        "energy_high_risk",
        "parse_failure",
        "automata_reject",
    ]
    assert "energy_diagnostic" in fallback_with_unknown_primary["weak_verifiers"]
    assert cheap_with_unknown_primary["selected_deterministic_validators"] == ["custom_validator"]


def test_req_verify_1557_soft_signals_never_accept_without_validator() -> None:
    """REQ-VERIFY-1557: a soft accept cannot override deterministic rejection."""

    candidate = _candidate(
        "satquest:confident-wrong",
        source_kind="satquest",
        weak_signals={"claim_uncertainty_score": 0.0, "telemetry_routing_score": 0.1},
        soft_signals={"model_declared_accept": True, "mean_logprob": -0.001},
        deterministic_outcomes={"sat_solver": False, "unified_contract_gate": False},
    )

    result = exp.evaluate_routing([candidate], focused_tests_passed=True)
    row = result["rows"][0]
    summary = result["summary"]

    assert row["soft_signal_accept"] is True
    assert row["final_accept"] is False
    assert row["soft_signal_overrode_validator"] is False
    assert summary["false_accept_rate"] == pytest.approx(0.0)
    assert summary["missed_failure_count"] == 0


def test_scenario_verify_1557_high_risk_fallback_prevents_hidden_failure() -> None:
    """SCENARIO-VERIFY-1557: high-risk routing catches skipped-validator failures."""

    hidden_failure = _candidate(
        "runtime:hidden",
        source_kind="runtime_contract",
        weak_signals={"validator_disagreement": True},
        soft_signals={"full_context_accept": True},
        deterministic_outcomes={"runtime_contract_replay": True, "unified_contract_gate": False},
    )
    unsafe_low_risk = _candidate(
        "runtime:unsafe-low-risk",
        source_kind="runtime_contract",
        weak_signals={"claim_uncertainty_score": 0.0},
        soft_signals={"full_context_accept": True},
        deterministic_outcomes={"runtime_contract_replay": True, "unified_contract_gate": False},
    )

    protected = exp.evaluate_routing([hidden_failure], focused_tests_passed=True)
    unsafe = exp.evaluate_routing([unsafe_low_risk], focused_tests_passed=True)

    assert protected["rows"][0]["selected_deterministic_validators"] == [
        "runtime_contract_replay",
        "unified_contract_gate",
    ]
    assert protected["rows"][0]["final_accept"] is False
    assert protected["summary"]["missed_failure_count"] == 0
    assert unsafe["rows"][0]["selected_deterministic_validators"] == ["runtime_contract_replay"]
    assert unsafe["rows"][0]["final_accept"] is True
    assert unsafe["summary"]["missed_failure_count"] == 1


def test_req_verify_1557_builds_candidates_from_required_and_optional_sources() -> None:
    """REQ-VERIFY-1557: source rows become auditable routed candidates."""

    candidates = exp.build_candidate_set(
        satquest_rows=[
            {"family": "malformed"},
            _satquest_row("sat-extra", correct=False, energy=51.0),
        ],
        unified_gate_rows=[
            {"row_type": "summary"},
            _gate_row("gate-extra", source_family="runtime_contract"),
            _gate_row("sat-gate", source_family="satquest"),
            _gate_row("product-gate", source_family="product_line"),
            _gate_row("unknown-gate", source_family="unknown"),
        ],
        claim_router_rows=[
            {"row_type": "summary"},
            _claim_router_row(
                "sat-claim",
                source_kind="satquest",
                deterministic_accept=False,
                routing_reasons=["uncertainty", "validator_disagreement"],
            ),
        ],
        telemetry_rows=[
            {
                "row_type": "diagnostic_case",
                "case_id": "sat-claim",
                "deterministic_accept": False,
                "model_declared_accept": True,
                "routing_score": 52.0,
                "carnot_energy_score": 51.0,
            }
        ],
        beaver_rows=[
            {
                "row_type": "prefix_bound",
                "contract_case_id": "gate-extra",
                "unsafe_upper_bound": 0.88,
            }
        ],
        limit=6,
    )

    assert [case["candidate_id"] for case in candidates] == [
        "satquest:sat-claim",
        "satquest:sat-extra",
        "runtime_contract:gate-extra",
        "satquest:sat-gate",
        "product_line:product-gate",
        "runtime_contract:unknown-gate",
    ]
    assert candidates[0]["weak_signals"]["telemetry_routing_score"] == pytest.approx(52.0)
    assert candidates[0]["soft_signals"]["model_declared_accept"] is True
    assert candidates[2]["weak_signals"]["beaver_prefix_risk"] == pytest.approx(0.88)


def test_scenario_verify_1557_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1557: runner reports lower cost with zero missed failures."""

    output = tmp_path / "results" / "experiment_1557.json"
    manifest = tmp_path / "results" / "router_manifest.jsonl"
    policy = tmp_path / "results" / "policy.json"
    satquest_artifact = tmp_path / "results" / "experiment_1550.json"
    gate_artifact = tmp_path / "results" / "experiment_1551.json"
    claim_artifact = tmp_path / "results" / "experiment_1553.json"
    telemetry_artifact = tmp_path / "results" / "experiment_1556.json"
    beaver_artifact = tmp_path / "results" / "experiment_1537.json"
    satquest_manifest = tmp_path / "results" / "satquest.jsonl"
    gate_manifest = tmp_path / "results" / "gate.jsonl"
    claim_manifest = tmp_path / "results" / "claim.jsonl"
    telemetry_manifest = tmp_path / "results" / "telemetry.jsonl"
    beaver_manifest = tmp_path / "results" / "beaver.jsonl"

    _write_json(satquest_artifact, {"status": "complete", "satquest_sota_reeval_ready": True})
    _write_json(gate_artifact, {"status": "complete", "unified_contract_gate_ready": True})
    _write_json(claim_artifact, {"status": "complete", "claim_isolation_router_scale_ready": True})
    _write_json(telemetry_artifact, {"status": "complete", "arm_ebm_logprob_telemetry_ready": True})
    _write_json(beaver_artifact, {"status": "complete", "beaver_bound_ready": True})
    _write_jsonl(satquest_manifest, [_satquest_row("sat-optional", correct=True, energy=0.0)])
    _write_jsonl(gate_manifest, [_gate_row("gate-optional", source_family="runtime_contract")])
    _write_jsonl(
        claim_manifest,
        [
            _claim_router_row("runtime-low", source_kind="runtime_contract"),
            _claim_router_row(
                "sat-high",
                source_kind="satquest",
                deterministic_accept=False,
                final_accept=False,
                full_context_accept=True,
                routing_reasons=["uncertainty", "validator_disagreement"],
            ),
            _claim_router_row("product-low", source_kind="product_line"),
        ],
    )
    _write_jsonl(
        telemetry_manifest,
        [
            {"row_type": "diagnostic_case", "case_id": "sat-high", "routing_score": 55.0},
        ],
    )
    _write_jsonl(
        beaver_manifest,
        [
            {
                "row_type": "prefix_bound",
                "contract_case_id": "runtime-low",
                "unsafe_upper_bound": 0.25,
            }
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        manifest_path=manifest,
        router_policy_path=policy,
        satquest_artifact_path=satquest_artifact,
        unified_gate_artifact_path=gate_artifact,
        claim_router_artifact_path=claim_artifact,
        telemetry_artifact_path=telemetry_artifact,
        beaver_artifact_path=beaver_artifact,
        satquest_manifest_path=satquest_manifest,
        unified_gate_manifest_path=gate_manifest,
        claim_router_manifest_path=claim_manifest,
        telemetry_diagnostic_path=telemetry_manifest,
        beaver_manifest_path=beaver_manifest,
        focused_tests_passed=True,
        case_limit=3,
    )
    manifest_rows = _read_jsonl(manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert json.loads(policy.read_text(encoding="utf-8")) == exp.DEFAULT_ROUTER_POLICY
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["verification_compute_router_ready"] is True
    assert len(artifact["candidate_selection_cases"]) == 3
    assert artifact["verification_cost_router"] < artifact["verification_cost_baseline"]
    assert artifact["verification_cost_delta"] < 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["missed_failure_count"] == 0
    assert artifact["router_policy_path"] == str(policy)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert manifest_rows[-1]["row_type"] == "summary"
    exp.validate_artifact(artifact)

    blocked_missing = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_missing_1557.json",
        manifest_path=tmp_path / "results" / "blocked_missing_manifest.jsonl",
        router_policy_path=tmp_path / "results" / "blocked_missing_policy.json",
        satquest_artifact_path=tmp_path / "results" / "missing_1550.json",
        unified_gate_artifact_path=tmp_path / "results" / "missing_1551.json",
        claim_router_artifact_path=tmp_path / "results" / "missing_1553.json",
        telemetry_artifact_path=tmp_path / "results" / "missing_1556.json",
        beaver_artifact_path=tmp_path / "results" / "missing_1537.json",
        satquest_manifest_path=tmp_path / "results" / "missing_satquest.jsonl",
        unified_gate_manifest_path=tmp_path / "results" / "missing_gate.jsonl",
        claim_router_manifest_path=tmp_path / "results" / "missing_claim.jsonl",
        telemetry_diagnostic_path=tmp_path / "results" / "missing_telemetry.jsonl",
        beaver_manifest_path=tmp_path / "results" / "missing_beaver.jsonl",
        focused_tests_passed=False,
        case_limit=3,
    )
    assert blocked_missing["verification_compute_router_ready"] is False
    assert "focused_tests_not_passed" in blocked_missing["blockers"]
    assert "no_candidate_selection_cases" in blocked_missing["blockers"]
    assert any(
        blocker.startswith("missing_required_artifact:") for blocker in blocked_missing["blockers"]
    )

    _write_json(satquest_artifact, {"status": "complete", "satquest_sota_reeval_ready": False})
    _write_json(gate_artifact, {"status": "complete", "unified_contract_gate_ready": False})
    blocked_not_ready = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_not_ready_1557.json",
        manifest_path=tmp_path / "results" / "blocked_not_ready_manifest.jsonl",
        router_policy_path=policy,
        satquest_artifact_path=satquest_artifact,
        unified_gate_artifact_path=gate_artifact,
        claim_router_artifact_path=claim_artifact,
        telemetry_artifact_path=telemetry_artifact,
        beaver_artifact_path=beaver_artifact,
        satquest_manifest_path=satquest_manifest,
        unified_gate_manifest_path=gate_manifest,
        claim_router_manifest_path=claim_manifest,
        telemetry_diagnostic_path=telemetry_manifest,
        beaver_manifest_path=beaver_manifest,
        focused_tests_passed=True,
        case_limit=1,
    )
    assert "satquest_sota_reeval_not_ready" in blocked_not_ready["blockers"]
    assert "unified_contract_gate_not_ready" in blocked_not_ready["blockers"]


def test_req_verify_1557_validate_artifact_rejects_bad_ready_shape() -> None:
    """REQ-VERIFY-1557: ready artifacts must preserve deterministic authority."""

    artifact = {
        "status": "complete",
        "milestone": "20260508",
        "verification_compute_router_ready": True,
        "candidate_selection_cases": [{"candidate_id": "ok"}],
        "weak_verifiers_used": ["claim_router_uncertainty"],
        "deterministic_validators_used": ["sat_solver"],
        "soft_signals_used_for_routing_only": ["model_declared_accept"],
        "verification_cost_baseline": 10,
        "verification_cost_router": 6,
        "verification_cost_delta": -4,
        "false_accept_rate": 0.0,
        "missed_failure_count": 0,
        "router_policy_path": "results/policy.json",
        "focused_tests_passed": True,
        "honest_verdict": "complete: ready",
    }

    exp.validate_artifact(artifact)
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({key: value for key, value in artifact.items() if key != "status"})
    with pytest.raises(AssertionError, match="allowed terminal prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "blocked: no"})
    with pytest.raises(AssertionError, match="focused tests"):
        exp.validate_artifact(artifact | {"focused_tests_passed": False})
    with pytest.raises(AssertionError, match="zero false accepts"):
        exp.validate_artifact(artifact | {"false_accept_rate": 0.1})
    with pytest.raises(AssertionError, match="zero missed failures"):
        exp.validate_artifact(artifact | {"missed_failure_count": 1})
    with pytest.raises(AssertionError, match="lower routed cost"):
        exp.validate_artifact(artifact | {"verification_cost_delta": 0})
    with pytest.raises(AssertionError, match="candidate cases"):
        exp.validate_artifact(artifact | {"candidate_selection_cases": []})
    with pytest.raises(AssertionError, match="deterministic validators"):
        exp.validate_artifact(artifact | {"deterministic_validators_used": []})
    assert exp._unique(["duplicate", "duplicate"]) == ["duplicate"]  # noqa: SLF001


def _candidate(
    candidate_id: str,
    *,
    source_kind: str,
    weak_signals: dict[str, Any],
    deterministic_outcomes: dict[str, bool],
    soft_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "case_id": candidate_id.split(":", 1)[1],
        "source_kind": source_kind,
        "source_family": source_kind,
        "weak_signals": weak_signals,
        "soft_signals": soft_signals or {},
        "deterministic_outcomes": deterministic_outcomes,
    }


def _satquest_row(case_id: str, *, correct: bool, energy: float) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": "satquest",
        "baseline": {"correct": correct, "energy": energy, "parse_error": None},
        "parse_result": {"parse_ok": correct, "model_declared_accept": correct},
        "verifier": {"self_verifier_false_accept": False},
    }


def _gate_row(case_id: str, *, source_family: str) -> dict[str, Any]:
    return {
        "row_type": "gate_case",
        "case_id": case_id,
        "source_family": source_family,
        "final_accept": True,
        "stages": [
            {"stage": "automata_mask", "passed": False},
            {"stage": "semantic_repair", "passed": True},
            {"stage": "runtime_contracts", "deterministic_accept": True},
        ],
    }


def _claim_router_row(
    case_id: str,
    *,
    source_kind: str,
    deterministic_accept: bool = True,
    final_accept: bool = True,
    full_context_accept: bool = True,
    routing_reasons: list[str] | None = None,
) -> dict[str, Any]:
    reasons = routing_reasons or ["low_risk_bypass"]
    return {
        "row_type": "router_scale_case",
        "case_id": case_id,
        "router_case_id": f"{source_kind}:{case_id}",
        "source_kind": source_kind,
        "source_family": source_kind,
        "deterministic_accept": deterministic_accept,
        "final_accept": final_accept,
        "full_context_accept": full_context_accept,
        "claim_isolated_accept": deterministic_accept if "low_risk_bypass" not in reasons else None,
        "routed": "low_risk_bypass" not in reasons,
        "routing_reasons": reasons,
        "unified_gate_checked": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

"""Tests for Exp 1538 residual-drift commitment ledger.

Spec: REQ-VERIFY-1538, SCENARIO-VERIFY-1538.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import residual_drift_commitment_ledger as exp


def test_req_verify_1538_satquest_distinguishes_drift_from_contradiction() -> None:
    """REQ-VERIFY-1538: SAT oracle replay separates satisfiable drift from contradiction."""

    drift = exp.classify_satquest_case(
        _satquest_row(
            case_id="sat-drift",
            oracle_label="SAT",
            baseline_answer="UNSAT",
            classification="wrong_label",
            parse_ok=True,
            correct=False,
        )
    )
    contradiction = exp.classify_satquest_case(
        _satquest_row(
            case_id="sat-contradiction",
            oracle_label="UNSAT",
            baseline_answer="SAT",
            classification="wrong_label",
            parse_ok=True,
            correct=False,
            self_accept=True,
        )
    )
    blocker = exp.classify_satquest_case(
        _satquest_row(
            case_id="sat-parse-blocker",
            oracle_label="SAT",
            baseline_answer=None,
            classification="parse_failure",
            parse_ok=False,
            correct=False,
            parse_error="no_json_object",
        )
    )

    assert drift["failure_classification"] == exp.CLASS_SATISFIABLE_DRIFT
    assert drift["commitments"][0]["turn"] == 1
    assert drift["solver_oracle"]["satisfiable"] is True
    assert contradiction["failure_classification"] == exp.CLASS_TRUE_CONTRADICTION
    assert contradiction["solver_oracle"]["satisfiable"] is False
    assert blocker["failure_classification"] == exp.CLASS_OTHER_BLOCKER
    assert blocker["other_blocker"] == "parse_failure"


def test_req_verify_1538_product_line_repaired_drift_is_counted() -> None:
    """REQ-VERIFY-1538: product-line rescue rows count repaired satisfiable drift."""

    row = exp.classify_product_line_case(_product_line_row())

    assert row["failure_classification"] == exp.CLASS_SATISFIABLE_DRIFT
    assert row["repaired_drift"] is True
    assert row["deterministic_validator"]["oracle_agrees_after_repair"] is True
    assert [item["name"] for item in row["commitments"]] == [
        "feature_model_contract",
        "baseline_selection",
        "staged_feedback",
        "solver_oracle_validation",
    ]


def test_scenario_verify_1538_runner_writes_ready_artifact_and_ledger(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1538: runner writes bounded ledger rows and final metrics."""

    sat_manifest = tmp_path / "satquest.jsonl"
    product_manifest = tmp_path / "product.jsonl"
    runtime_manifest = tmp_path / "runtime.jsonl"
    output = tmp_path / "experiment_1538.json"
    ledger = tmp_path / "ledger_1538.jsonl"
    _write_jsonl(
        sat_manifest,
        [
            _satquest_row(
                case_id="sat-drift",
                oracle_label="SAT",
                baseline_answer="UNSAT",
                classification="wrong_label",
                parse_ok=True,
                correct=False,
            ),
            _satquest_row(
                case_id="sat-contradiction",
                oracle_label="UNSAT",
                baseline_answer="SAT",
                classification="wrong_label",
                parse_ok=True,
                correct=False,
                self_accept=True,
            ),
        ],
    )
    _write_jsonl(product_manifest, [_product_line_row()])
    _write_jsonl(runtime_manifest, [_runtime_structural_drift_row()])

    artifact = exp.run_experiment(
        project_root=tmp_path,
        satquest_manifest_path=sat_manifest,
        product_line_manifest_path=product_manifest,
        runtime_cdg_manifest_path=runtime_manifest,
        output_path=output,
        ledger_path=ledger,
        focused_tests_passed=True,
    )
    rows = _read_jsonl(ledger)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["residual_drift_ledger_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["multi_turn_cases"] == 4
    assert artifact["contradiction_cases"] == 1
    assert artifact["satisfiable_drift_cases"] == 3
    assert artifact["drift_rate"] == pytest.approx(0.75)
    assert artifact["repaired_drift_cases"] == 1
    assert artifact["solver_oracle_used"] is True
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "residual_drift_summary"
    assert rows[-1]["satisfiable_drift_cases"] == 3
    assert {row["source_domain"] for row in rows[:-1]} == {
        "satquest",
        "product_line",
        "runtime_contract",
    }


def test_req_verify_1538_runner_blocks_missing_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-1538: missing source manifests are concrete terminal blockers."""

    output = tmp_path / "experiment_1538.json"
    ledger = tmp_path / "ledger_1538.jsonl"

    artifact = exp.run_experiment(
        project_root=tmp_path,
        satquest_manifest_path=tmp_path / "missing_sat.jsonl",
        product_line_manifest_path=tmp_path / "missing_product.jsonl",
        runtime_cdg_manifest_path=tmp_path / "missing_runtime.jsonl",
        output_path=output,
        ledger_path=ledger,
        focused_tests_passed=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["residual_drift_ledger_ready"] is False
    assert artifact["multi_turn_cases"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert ledger.read_text(encoding="utf-8") == ""
    assert all(str(item).startswith("missing_") for item in artifact["blockers"])


def _satquest_row(
    *,
    case_id: str,
    oracle_label: str,
    baseline_answer: str | None,
    classification: str,
    parse_ok: bool,
    correct: bool,
    parse_error: str | None = None,
    self_accept: bool | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "instance_id": f"{case_id}-instance",
        "format_name": "narrative",
        "family": "unit_test",
        "n_vars": 1,
        "clauses": [[1]] if oracle_label == "SAT" else [[1], [-1]],
        "prompt": "bounded SATQuest prompt",
        "model_hf_id": exp.MANDATED_MODEL_SPECS[0],
        "model_name": "Qwen3.6-35B-A3B",
        "generation_source": "live_sota_llamacpp",
        "solver_oracle": {
            "backend": "exact_exhaustive_fallback",
            "label": oracle_label,
            "checked_assignments": 2,
            "satisfying_assignment": [True] if oracle_label == "SAT" else None,
        },
        "parse_result": {
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "baseline_answer": baseline_answer,
            "model_declared_accept": self_accept,
        },
        "baseline": {
            "answer": baseline_answer,
            "classification": classification,
            "correct": correct,
            "parse_error": parse_error,
        },
        "energy_ranked": {"correct": correct, "classification": classification},
        "repair_hint": {"correct": correct, "classification": classification},
        "verifier": {
            "self_declared_accept": self_accept,
            "self_verifier_false_accept": bool(self_accept and not correct),
        },
    }


def _product_line_row() -> dict[str, Any]:
    return {
        "case_id": "plc-drift",
        "model_id": "UnitProduct",
        "model_hf_id": exp.MANDATED_MODEL_SPECS[0],
        "model_name": "Qwen3.6-35B-A3B",
        "generation_source": "live_sota_llamacpp",
        "operation": {"kind": "max_value", "budget": 4, "include": ["Core"]},
        "baseline_result": {
            "parse_ok": True,
            "classification": "infeasible",
            "feasible": False,
            "oracle_agrees": False,
            "self_verifier_false_accept": False,
        },
        "stages": [
            {"stage": "syntax_parse_feedback", "status": "passed"},
            {"stage": "feature_model_consistency_feedback", "status": "repaired"},
            {"stage": "solver_feasibility_feedback", "status": "repaired"},
        ],
        "oracle_result": {
            "classification": "oracle_agreement",
            "feasible": True,
            "oracle_agrees": True,
            "optimal_features": ["Core", "Optimum"],
        },
        "verifier_result": {"accepted": True, "self_verifier_false_accept": False},
        "policy_result": {"accepted": True, "false_accept": False},
    }


def _runtime_structural_drift_row() -> dict[str, Any]:
    return {
        "row_type": "cdg_root_cause_case",
        "contract_case_id": "structural_contract:runtime-drift",
        "prompt_or_case_id": "runtime-drift",
        "source_family": "structural_contract",
        "failure_categories": ["structural_dependency", "final_accept"],
        "root_cause_category": "structural_dependency",
        "repair_ready": True,
        "candidate_repair_final_deterministic_accept": False,
        "deterministic_validator_accept": True,
        "false_accept": False,
        "contract_validation_row": {
            "row_type": "contract_case",
            "contract_case_id": "structural_contract:runtime-drift",
            "prompt_or_case_id": "runtime-drift",
            "source_family": "structural_contract",
            "expected_label": False,
            "final_deterministic_accept": False,
            "final_deterministic_decision": "reject",
            "structural_contract_result": {
                "linked": True,
                "expected_violation": True,
                "detected_violation": True,
                "contract_family": "graph_prerequisites",
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

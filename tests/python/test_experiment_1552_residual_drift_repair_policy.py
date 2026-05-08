"""Tests for Exp 1552 residual-drift local repair policy.

Spec: REQ-VERIFY-1552, SCENARIO-VERIFY-1552.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import residual_drift_repair_policy as exp


def test_req_verify_1552_localizes_drift_by_source_domain() -> None:
    """REQ-VERIFY-1552: repairs target only the violated commitment or validator span."""

    sat = exp.localize_drift(_satquest_drift_row())
    product = exp.localize_drift(_product_line_drift_row())
    runtime = exp.localize_drift(_runtime_drift_row())

    assert sat["localized_span"] == "commitments[1].evidence.answer"
    assert sat["repair_kind"] == "sat_answer_or_assignment_patch"
    assert product["localized_span"] == "commitments[1].evidence.selected_features"
    assert product["replacement_source"] == "solver_oracle_validation.optimal_features"
    assert runtime["localized_span"] == "commitments[1].evidence.root_cause_category"
    assert runtime["repair_kind"] == "runtime_contract_root_cause_patch"

    proposal = exp.propose_minimal_repair(_satquest_drift_row(), sat)
    hinted = exp.propose_minimal_repair(
        _satquest_drift_row(),
        sat,
        model_hint={"proposal_output_excerpt": "patch only the SAT answer"},
    )
    assert proposal["edit_scope"] == "localized"
    assert proposal["whole_answer_regenerated"] is False
    assert proposal["replacement"]["answer"] == "SAT"
    assert proposal["replacement"]["satisfying_assignment"] == [True]
    assert hinted["model_proposal_excerpt"] == "patch only the SAT answer"
    assert exp.propose_minimal_repair(
        {"source_domain": "unknown"},
        {"repair_kind": "unsupported_source"},
    )["replacement"] == {}


def test_req_verify_1552_contradictions_are_not_repaired() -> None:
    """REQ-VERIFY-1552: true contradictions are counted untouched, not patched."""

    result = exp.evaluate_repair(_satquest_contradiction_row())

    assert result.attempted is False
    assert result.accepted is False
    assert result.contradiction_untouched is True
    assert result.rejection_reason == "true_contradiction_untouched"
    assert result.proposal == {}


def test_scenario_verify_1552_replay_accepts_local_sat_repair() -> None:
    """SCENARIO-VERIFY-1552: SAT repairs are accepted only after oracle replay."""

    result = exp.evaluate_repair(_satquest_drift_row())

    assert result.attempted is True
    assert result.localized is True
    assert result.replay_passed is True
    assert result.accepted is True
    assert result.false_accept is False
    assert result.replay["validator"] == "sat_oracle"


def test_scenario_verify_1552_false_accept_candidate_is_rejected() -> None:
    """SCENARIO-VERIFY-1552: validator false accepts cannot reduce drift."""

    row = _product_line_drift_row()
    row["false_accept"] = True
    row["deterministic_validator"]["policy_false_accept"] = True

    result = exp.evaluate_repair(row)
    summary = exp.summarize_repair_results([row], [result])

    assert result.attempted is True
    assert result.replay_passed is False
    assert result.accepted is False
    assert result.rejected_false_accept is True
    assert summary["repaired_drift_cases"] == 0
    assert summary["false_accept_rate"] == pytest.approx(0.0)
    assert summary["rejected_false_accept_repairs"] == 1


def test_scenario_verify_1552_runner_writes_ready_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1552: runner writes replay-gated repair metrics."""

    ledger = tmp_path / "residual_ledger.jsonl"
    output = tmp_path / "experiment_1552.json"
    manifest = tmp_path / "repair_1552.jsonl"
    _write_jsonl(
        ledger,
        [
            _satquest_drift_row(),
            _product_line_drift_row(),
            _runtime_drift_row(),
            _satquest_contradiction_row(),
            {"row_type": "residual_drift_summary", "satisfiable_drift_cases": 3},
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        ledger_path=ledger,
        output_path=output,
        repair_manifest_path=manifest,
        model_probe_fn=lambda _root, _rows: {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": ["no_mandated_sota_gguf_runtime"],
            "legacy_small_models_excluded_from_headline_metrics": True,
        },
        focused_tests_passed=True,
    )
    rows = _read_jsonl(manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["residual_drift_repair_ready"] is True
    assert artifact["drift_cases_before"] == 3
    assert artifact["repair_attempts"] == 3
    assert artifact["localized_repairs_attempted"] == 3
    assert artifact["repaired_drift_cases"] == 3
    assert artifact["drift_reduction_delta"] == pytest.approx(1.0)
    assert artifact["contradiction_cases_untouched"] == 1
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["replay_pass_rate"] == pytest.approx(1.0)
    assert artifact["repair_policy_path"] == exp.REPAIR_POLICY_PATH
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "residual_drift_repair_summary"
    assert rows[-1]["repaired_drift_cases"] == 3


def test_req_verify_1552_blockers_and_helpers_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1552: missing ledgers, malformed rows, and model blockers fail closed."""

    output = tmp_path / "experiment_1552.json"
    manifest = tmp_path / "repair_1552.jsonl"
    blocked = exp.run_experiment(
        project_root=tmp_path,
        ledger_path=tmp_path / "missing.jsonl",
        output_path=output,
        repair_manifest_path=manifest,
        model_probe_fn=lambda _root, _rows: {"availability_blockers": []},
        focused_tests_passed=True,
    )

    assert blocked["status"] == "blocked"
    assert blocked["residual_drift_repair_ready"] is False
    assert blocked["drift_cases_before"] == 0
    assert blocked["honest_verdict"].startswith("complete:")
    assert manifest.read_text(encoding="utf-8") == ""

    assert exp.evaluate_repair({"source_case_id": "accepted", "failure_classification": "accepted"}).rejection_reason == "not_satisfiable_drift"
    assert exp.localize_drift({"source_domain": "unknown"})["repair_kind"] == "unsupported_source"
    assert exp.replay_candidate({"failure_classification": "accepted"}, {})["reason"] == "not_satisfiable_drift"
    assert exp.replay_candidate(
        {"failure_classification": "satisfiable_drift", "source_domain": "unknown"},
        {},
    )["reason"] == "unsupported_source"
    bad_answer = exp.replay_candidate(
        _satquest_drift_row(),
        {"replacement": {"answer": "UNSAT"}},
    )
    bad_assignment = exp.replay_candidate(
        _satquest_drift_row(),
        {"replacement": {"answer": "SAT", "satisfying_assignment": [False]}},
    )
    assert bad_answer["reason"] == "sat_answer_mismatch"
    assert bad_assignment["reason"] == "sat_assignment_invalid"
    assert exp._assignment_satisfies([1], [True]) is False  # noqa: SLF001
    assert exp._assignment_satisfies([["bad"]], [True]) is False  # noqa: SLF001
    assert exp._assignment_satisfies([[0]], [True]) is False  # noqa: SLF001
    assert exp._assignment_satisfies([[2]], [True]) is False  # noqa: SLF001
    assert exp._assignment_satisfies([[1]], [False]) is False  # noqa: SLF001
    assert exp._completion_text("plain") == "plain"  # noqa: SLF001
    assert exp._completion_text({"choices": [{"text": "ok"}]}) == "ok"  # noqa: SLF001
    assert exp._completion_text({"choices": []}) == ""  # noqa: SLF001
    assert exp._completion_text({"choices": ["bad"]}) == ""  # noqa: SLF001
    assert exp._completion_text(None) == ""  # noqa: SLF001
    assert exp._display_path(Path("/tmp/outside"), tmp_path) == "/tmp/outside"  # noqa: SLF001

    monkeypatch.setattr(exp, "_resolve_cached_gguf", lambda _hf_id: None)
    probe = exp.probe_headline_repair_model(tmp_path, [_satquest_drift_row()])
    assert probe["live_sota_model_inference_used"] is False
    assert "no_mandated_sota_gguf_runtime" in probe["availability_blockers"]

    monkeypatch.setattr(exp, "_resolve_cached_gguf", lambda _hf_id: "/tmp/model.gguf")
    cached_but_no_rows = exp.probe_headline_repair_model(tmp_path, [])
    assert cached_but_no_rows["availability_blockers"][0] == "no_satisfiable_drift_rows_for_model_proposal"

    contradiction_ledger = tmp_path / "contradictions_only.jsonl"
    _write_jsonl(contradiction_ledger, [_satquest_contradiction_row()])
    no_drift = exp.run_experiment(
        project_root=tmp_path,
        ledger_path=contradiction_ledger,
        output_path=tmp_path / "no_drift.json",
        repair_manifest_path=tmp_path / "no_drift.jsonl",
        model_probe_fn=lambda _root, _rows: {"availability_blockers": []},
        focused_tests_passed=False,
    )
    assert "focused_tests_not_passed" in no_drift["blockers"]
    assert "no_satisfiable_drift_cases" in no_drift["blockers"]


def _satquest_drift_row() -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_domain": "satquest",
        "source_case_id": "sat-drift",
        "failure_classification": "satisfiable_drift",
        "false_accept": False,
        "commitments": [
            {"turn": 1, "name": "cnf_constraints", "evidence": {"n_vars": 1, "clauses": [[1]]}},
            {"turn": 2, "name": "model_answer", "evidence": {"answer": "UNSAT", "parse_ok": True}},
            {
                "turn": 3,
                "name": "solver_oracle_validation",
                "evidence": {
                    "used": True,
                    "label": "SAT",
                    "satisfiable": True,
                    "satisfying_assignment": [True],
                },
            },
        ],
        "solver_oracle": {
            "used": True,
            "label": "SAT",
            "satisfiable": True,
            "satisfying_assignment": [True],
        },
        "deterministic_validator": {"classification": "wrong_label"},
    }


def _satquest_contradiction_row() -> dict[str, Any]:
    row = _satquest_drift_row()
    row["source_case_id"] = "sat-contradiction"
    row["failure_classification"] = "true_contradiction"
    row["commitments"][0]["evidence"] = {"n_vars": 1, "clauses": [[1], [-1]]}
    row["commitments"][1]["evidence"] = {"answer": "SAT", "parse_ok": True}
    row["commitments"][2]["evidence"] = {
        "used": True,
        "label": "UNSAT",
        "satisfiable": False,
        "satisfying_assignment": None,
    }
    row["solver_oracle"] = {
        "used": True,
        "label": "UNSAT",
        "satisfiable": False,
        "satisfying_assignment": None,
    }
    return row


def _product_line_drift_row() -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_domain": "product_line",
        "source_case_id": "pl-drift",
        "failure_classification": "satisfiable_drift",
        "false_accept": False,
        "commitments": [
            {
                "turn": 1,
                "name": "feature_model_contract",
                "evidence": {"model_id": "UnitProduct", "operation": {"budget": 3}},
            },
            {
                "turn": 2,
                "name": "baseline_selection",
                "evidence": {"parse_ok": True, "oracle_agrees": False},
            },
            {
                "turn": 3,
                "name": "solver_oracle_validation",
                "evidence": {
                    "oracle_agrees": True,
                    "optimal_features": ["Core", "FastPath"],
                },
            },
        ],
        "solver_oracle": {"used": True, "satisfiable": True, "oracle_agrees_after_repair": True},
        "deterministic_validator": {
            "baseline_oracle_agrees": False,
            "oracle_agrees_after_repair": True,
            "policy_false_accept": False,
        },
    }


def _runtime_drift_row() -> dict[str, Any]:
    return {
        "row_type": "residual_drift_case",
        "source_domain": "runtime_contract",
        "source_case_id": "runtime-drift",
        "failure_classification": "satisfiable_drift",
        "false_accept": False,
        "commitments": [
            {
                "turn": 1,
                "name": "runtime_contract_prompt",
                "evidence": {"prompt_or_case_id": "runtime-drift"},
            },
            {
                "turn": 2,
                "name": "cdg_failure_localization",
                "evidence": {
                    "root_cause_category": "structural_dependency",
                    "failure_categories": ["structural_dependency", "final_accept"],
                },
            },
            {
                "turn": 3,
                "name": "deterministic_contract_validation",
                "evidence": {
                    "expected_label": False,
                    "final_deterministic_accept": False,
                    "final_deterministic_decision": "reject",
                    "structural_contract_result": {
                        "detected_violation": True,
                        "expected_violation": True,
                    },
                },
            },
        ],
        "solver_oracle": {
            "used": True,
            "satisfiable": True,
            "root_cause_category": "structural_dependency",
        },
        "deterministic_validator": {
            "deterministic_validator_accept": True,
            "expected_label": False,
            "final_deterministic_accept": False,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

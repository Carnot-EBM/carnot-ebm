"""Tests for Exp 1324 certificate failure taxonomy.

Spec: REQ-VERIFY-1324,
      SCENARIO-VERIFY-1324
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import certificate_failure_taxonomy_formalizer_reality_check as mod


def _exp1311() -> dict[str, Any]:
    return {
        "status": "complete",
        "run_date": "20260505",
        "pysat_verified_rate": 0.5,
        "answer_stability_score": 0.9,
        "responses": [
            {
                "raw_output": "",
                "token_count": 1,
                "verified": False,
                "verifier_label": "SAT",
                "parsed_label": "ABSTAIN",
            },
            {
                "raw_output": "UNSAT",
                "token_count": 2,
                "verified": False,
                "verifier_label": "UNKNOWN",
                "parsed_label": "UNSAT",
            },
            {
                "raw_output": "UNKNOWN",
                "token_count": 4,
                "verified": True,
                "verifier_label": "UNKNOWN",
                "parsed_label": "UNKNOWN",
            },
        ],
    }


def _attempt(
    path: str,
    *,
    parseable: bool,
    truthful: bool,
    compact_encoding: bool = False,
    item_id: str = "case",
) -> dict[str, Any]:
    return {
        "path": path,
        "parseable": parseable,
        "truthful": truthful,
        "compact_encoding": compact_encoding,
        "item_id": item_id,
        "errors": [] if parseable else ["no_json_object"],
        "prompt_chars": 80,
    }


def _exp1312() -> dict[str, Any]:
    return {
        "status": "complete",
        "run_date": "20260505",
        "certificate_attempt_count": 6,
        "certificate_parse_rate": 0.666667,
        "certificate_truthfulness_rate": 0.5,
        "grammar_projection_tax_proxy": {
            "proxy": "extra_prompt_chars",
            "gbnf_mean_extra_prompt_chars": 40.0,
            "dccd_mean_extra_prompt_chars": -80.0,
            "repair_mean_extra_prompt_chars": 20.0,
        },
        "attempts": [
            _attempt("raw_trigger", parseable=False, truthful=False),
            _attempt("raw_trigger", parseable=False, truthful=False),
            _attempt("gbnf_constrained", parseable=True, truthful=False),
            _attempt("dccd_compact", parseable=True, truthful=True, compact_encoding=True),
            _attempt("repaired_certificate", parseable=True, truthful=True),
            _attempt(
                "gbnf_constrained",
                parseable=True,
                truthful=False,
                item_id="cb_unknown_missing_bound",
            ),
        ],
    }


def test_exp1324_direct_attempt_audit_classifies_required_failures() -> None:
    """REQ-VERIFY-1324-2/3/5/6: per-attempt records drive the taxonomy counts."""
    artifact = mod.build_failure_taxonomy_artifact(
        exp1311_artifact=_exp1311(),
        exp1312_artifact=_exp1312(),
        run_date="20260505",
        project_root="/repo",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["artifact_metadata"]["direct_exp1312_attempt_audit"] is True
    assert artifact["parser_failure_count"] == 2
    assert artifact["semantic_failure_count"] == 2
    assert artifact["undergeneration_failure_count"] == 1
    assert artifact["hardcoded_solution_leakage_rate"] == pytest.approx(2 / 6)
    assert artifact["solver_vs_certificate_delta"] == pytest.approx(0.0)
    assert artifact["minimum_gate_delta_needed"] == pytest.approx(0.083333)
    assert artifact["reasoning_token_overhead"]["gbnf_extra_token_proxy"] == pytest.approx(10.0)
    assert "parser repair" in artifact["parse_recovery_recommendation"]
    assert "solver-backed certificates" in artifact["literature_reality_check_summary"]
    assert artifact["formalizer_failure_modes"][0]["class"] == "undergeneration"
    assert artifact["honest_verdict"] == "diagnostic_complete_parse_gate_shortfall_parser_recovery_needed"


def test_exp1324_aggregate_only_fallback_records_limitation() -> None:
    """REQ-VERIFY-1324-2: aggregate-only Exp 1312 artifacts are labeled as limited."""
    exp1312 = _exp1312() | {"attempts": []}

    artifact = mod.build_failure_taxonomy_artifact(
        exp1311_artifact=_exp1311(),
        exp1312_artifact=exp1312,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["artifact_metadata"]["direct_exp1312_attempt_audit"] is False
    assert artifact["parser_failure_count"] == 2
    assert artifact["semantic_failure_count"] == 2
    assert artifact["hardcoded_solution_leakage_rate"] is None
    assert "aggregate-only" in artifact["source_data_limitations"]


def test_exp1324_run_experiment_writes_final_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1324-1/5: run_experiment writes the required deliverable JSON."""
    results = tmp_path / "results"
    results.mkdir()
    exp1311_path = results / "experiment_1311.json"
    exp1312_path = results / "experiment_1312.json"
    output_path = results / "experiment_1324.json"
    exp1311_path.write_text(json.dumps(_exp1311()), encoding="utf-8")
    exp1312_path.write_text(json.dumps(_exp1312()), encoding="utf-8")

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1311_path=exp1311_path,
        exp1312_path=exp1312_path,
        output_path=output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["minimum_gate_delta_needed"] == pytest.approx(0.083333)


def test_exp1324_handles_sparse_aggregate_inputs_and_gate_met() -> None:
    """REQ-VERIFY-1324-5/6: sparse aggregate records remain explicit and bounded."""
    gate_met = mod.build_failure_taxonomy_artifact(
        exp1311_artifact={"responses": [{"verified": True}, {"verified": False}]},
        exp1312_artifact={
            "certificate_attempt_count": 0,
            "certificate_parse_rate": 0.8,
            "certificate_truthfulness_rate": 0.75,
            "attempts": "aggregate-only",
            "grammar_projection_tax_proxy": {},
        },
        run_date="20260505",
        project_root="/repo",
    )

    assert gate_met["artifact_metadata"]["direct_exp1312_attempt_audit"] is False
    assert gate_met["minimum_gate_delta_needed"] == 0
    assert gate_met["honest_verdict"] == "diagnostic_complete_parse_gate_met_semantic_leakage_review_needed"
    assert gate_met["solver_vs_certificate_delta"] == pytest.approx(0.25)
    assert gate_met["reasoning_token_overhead"]["gbnf_extra_token_proxy"] is None
    assert gate_met["parse_recovery_recommendation"].startswith("No parse-rate gate delta")

    sparse = mod.build_failure_taxonomy_artifact(
        exp1311_artifact={"responses": "not saved"},
        exp1312_artifact={
            "certificate_attempt_count": 0,
            "certificate_parse_rate": 0.5,
            "certificate_truthfulness_rate": 0.0,
            "attempts": "aggregate-only",
        },
        run_date="20260505",
        project_root="/repo",
    )

    assert sparse["undergeneration_failure_count"] == 0
    assert sparse["solver_vs_certificate_delta"] == 0
    assert sparse["minimum_parseable_attempts_to_recover"] == 1

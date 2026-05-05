"""Tests for Exp 1341 HalluGuard certificate failure split.

Spec: REQ-VERIFY-1341,
      SCENARIO-VERIFY-1341
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import halluguard_certificate_failure_split as mod


def _exp1323() -> dict[str, Any]:
    return {
        "status": "complete",
        "artifact_metadata": {
            "project_root": "/home/ianblenke/github.com/ianblenke/carnot",
            "run_date": "20260505",
        },
        "empty_or_one_token_rate": 0.4,
        "min_tokens_recovered": True,
        "topk_logprob_available": True,
        "entropy_production_rate_available": True,
        "honest_verdict": "token_health_recovered_certificate_prompt_multitoken",
    }


def _exp1324() -> dict[str, Any]:
    return {
        "status": "complete",
        "parser_failure_count": 40,
        "undergeneration_failure_count": 25,
        "semantic_failure_count": 30,
        "unknown_state_mishandling_count": 4,
        "possible_hardcoded_solution_leakage_count": 35,
        "hardcoded_solution_leakage_rate": 0.251799,
        "solver_vs_certificate_delta": 0.17197,
        "minimum_parseable_attempts_to_recover": 6,
        "formalizer_failure_modes": [
            {"class": "undergeneration", "count": 25},
            {"class": "parser_schema_mismatch", "count": 40},
            {"class": "semantic_invalidity", "count": 30},
            {"class": "solver_disagreement", "count": 19},
            {"class": "unknown_state_mishandling", "count": 4},
            {"class": "possible_hardcoded_solution_leakage", "count": 35},
        ],
        "source_metrics": {
            "exp1312_certificate_attempt_count": 139,
            "exp1312_certificate_parse_rate": 0.71223,
            "exp1312_certificate_truthfulness_rate": 0.69697,
        },
        "honest_verdict": "diagnostic_complete_parse_gate_shortfall_parser_recovery_needed",
    }


def test_exp1341_builds_required_split_from_existing_failure_taxonomy() -> None:
    """REQ-VERIFY-1341: exp1324 failure classes map to explicit risk counts."""
    artifact = mod.build_halluguard_certificate_failure_split_artifact(
        exp1323_artifact=_exp1323(),
        exp1324_artifact=_exp1324(),
        exp1340_artifact=None,
        exp1340_limitation="exp1340_absent_or_unreadable_fallback_to_exp1324",
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["parser_schema_risk_count"] == 40
    assert artifact["undergeneration_risk_count"] == 25
    assert artifact["semantic_invalidity_count"] == 30
    assert artifact["unknown_mishandling_count"] == 4
    assert artifact["data_driven_risk_proxy"]["count"] == 35
    assert artifact["reasoning_driven_risk_proxy"]["count"] == 53
    assert artifact["reasoning_driven_risk_proxy"]["failure_types"] == {
        "semantic_invalidity": 30,
        "solver_disagreement": 19,
        "unknown_state_mishandling": 4,
    }
    assert artifact["universal_detector_claim_allowed"] is False
    assert artifact["honest_verdict"] == (
        "local_certificate_slice_diagnostic_exp1340_missing_no_universal_detector_claim"
    )
    assert artifact["artifact_metadata"]["run_date"] == "20260505"


def test_exp1341_repair_policy_names_concrete_actions() -> None:
    """REQ-VERIFY-1341: each failure class gets a concrete repair action."""
    artifact = mod.build_halluguard_certificate_failure_split_artifact(
        exp1323_artifact=_exp1323(),
        exp1324_artifact=_exp1324(),
        exp1340_artifact={"status": "complete"},
        run_date="20260505",
        project_root="/repo",
    )

    policy_text = json.dumps(artifact["repair_policy_by_failure_type"], sort_keys=True)
    for required_action in [
        "prompt retrieval",
        "reasoning budget",
        "grammar branch",
        "semantic validator",
        "UNKNOWN-preserving fallback",
    ]:
        assert required_action in policy_text
    assert artifact["source_cases_available"]["exp1340"] is True
    assert artifact["source_cases_available"]["limitations"] == []
    assert artifact["honest_verdict"] == (
        "local_certificate_slice_diagnostic_complete_no_universal_detector_claim"
    )
    assert mod._failure_mode_counts({"formalizer_failure_modes": "aggregate-only"}) == {}
    assert mod._failure_mode_counts(
        {
            "formalizer_failure_modes": [
                "malformed-row",
                {"class": "", "count": 99},
                {"class": "parser_schema_mismatch", "count": "not-an-int"},
            ]
        }
    ) == {"parser_schema_mismatch": 0}


def test_exp1341_run_experiment_writes_in_progress_then_final_with_missing_exp1340(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1341: missing Exp 1340 falls back to Exp 1324 replay."""
    results = tmp_path / "results"
    results.mkdir()
    exp1323_path = results / "experiment_1323.json"
    exp1324_path = results / "experiment_1324.json"
    output_path = results / "experiment_1341_halluguard_certificate_failure_split.json"
    exp1323_path.write_text(json.dumps(_exp1323()), encoding="utf-8")
    exp1324_path.write_text(json.dumps(_exp1324()), encoding="utf-8")

    writes: list[dict[str, Any]] = []
    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1323_path=exp1323_path,
        exp1324_path=exp1324_path,
        exp1340_path=results / "experiment_1340_missing.json",
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["source_cases_available"]["exp1323"] is True
    assert artifact["source_cases_available"]["exp1324"] is True
    assert artifact["source_cases_available"]["exp1340"] is False
    assert artifact["source_cases_available"]["limitations"] == [
        "exp1340_absent_or_unreadable_fallback_to_exp1324"
    ]
    assert artifact["universal_detector_claim_allowed"] is False


def test_exp1341_write_json_observer_is_optional(tmp_path: Path) -> None:
    """REQ-VERIFY-1341: artifact writes do not require a test observer hook."""
    output_path = tmp_path / "artifact.json"

    mod._write_json(output_path, {"status": "complete"})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"status": "complete"}

"""Tests for Exp 1344 failure-type governed memory policy.

Spec: REQ-LEARN-1344, SCENARIO-LEARN-1344.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import continuous_self_learning_failure_type_memory_policy as mod


def _exp1303() -> dict[str, Any]:
    return {
        "status": "complete",
        "self_learning_delta_overall": 0.4,
        "accepted_violation_delta": -0.2,
        "headline_result_allowed": False,
        "honest_verdict": "online_memory_policy_improved_non_headline",
    }


def _exp1315() -> dict[str, Any]:
    return {
        "status": "complete",
        "self_learning_delta_overall": 0.3,
        "nonforgetting_certificate_rate": 1.0,
        "memory_regression_count": 0,
        "accepted_violation_delta": -0.1,
        "promoted_memory_count": 2,
        "demoted_memory_count": 1,
        "replay_case_count": 5,
        "headline_result_allowed": False,
        "honest_verdict": "cerce_nonforgetting_preserved_improved_non_headline",
    }


def _exp1324() -> dict[str, Any]:
    return {
        "status": "complete",
        "formalizer_failure_modes": [
            {"class": "undergeneration", "count": 2},
            {"class": "parser_schema_mismatch", "count": 3},
            {"class": "semantic_invalidity", "count": 4},
            {"class": "solver_disagreement", "count": 1},
            {"class": "unknown_state_mishandling", "count": 1},
            {"class": "possible_hardcoded_solution_leakage", "count": 2},
        ],
        "parser_failure_count": 3,
        "semantic_failure_count": 4,
        "undergeneration_failure_count": 2,
        "unknown_state_mishandling_count": 1,
        "possible_hardcoded_solution_leakage_count": 2,
        "source_metrics": {"exp1312_certificate_attempt_count": 11},
        "honest_verdict": "diagnostic_complete_parse_gate_shortfall_parser_recovery_needed",
    }


def _exp1341(*, headline_cases: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "complete",
        "source_cases_available": {
            "exp1323": True,
            "exp1324": True,
            "exp1340": False,
            "limitations": ["exp1340_absent_or_unreadable_fallback_to_exp1324"],
        },
        "repair_policy_by_failure_type": {
            failure_type: {"next_actions": ["fixture action"]}
            for failure_type in (
                "undergeneration",
                "parser_schema_mismatch",
                "semantic_invalidity",
                "solver_disagreement",
                "unknown_state_mishandling",
                "possible_hardcoded_solution_leakage",
            )
        },
        "headline_certificate_cases": headline_cases or [],
        "honest_verdict": "local_certificate_slice_diagnostic_exp1340_missing_no_universal_detector_claim",
    }


def test_req_learn_1344_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1344-1: the workflow writes a durable in-progress artifact."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["artifact_metadata"]["run_date"] == "20260505"
    assert written["failure_type_policy"] == {}
    assert written["honest_verdict"] == "in_progress"


def test_scenario_learn_1344_maps_failure_types_and_allows_replay_dvi() -> None:
    """SCENARIO-LEARN-1344: replay evidence can allow non-headline DVI readiness."""

    artifact = mod.build_artifact(
        exp1303_artifact=_exp1303(),
        exp1315_artifact=_exp1315(),
        exp1324_artifact=_exp1324(),
        exp1341_artifact=_exp1341(),
        unavailable_inputs=["results/experiment_1303_online_memory_policy_v2.json"],
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["inputs_unavailable"] == [
        "results/experiment_1303_online_memory_policy_v2.json"
    ]
    assert artifact["self_learning_delta_overall"] == 0.3
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["memory_regression_count"] == 0
    assert artifact["accepted_violation_delta"] == -0.1
    assert artifact["promoted_memory_count"] == 6
    assert artifact["demoted_memory_count"] == 3
    assert artifact["replay_cases_used"] == 16
    assert artifact["headline_certificate_cases"] == []
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "failure_type_memory_policy_dvi_ready_replay_non_headline"

    policy = artifact["failure_type_policy"]
    assert set(policy) == {
        "undergeneration",
        "parser_schema_mismatch",
        "semantic_invalidity",
        "solver_disagreement",
        "unknown_state_mishandling",
        "possible_hardcoded_solution_leakage",
    }
    assert policy["semantic_invalidity"]["action"] == "promote"
    assert policy["possible_hardcoded_solution_leakage"]["action"] == "demote"
    assert policy["unknown_state_mishandling"]["action"] == "quarantine"
    assert policy["parser_schema_mismatch"]["action"] == "request_fresh_verifier"
    assert policy["undergeneration"]["action"] == "request_fresh_verifier"
    assert policy["solver_disagreement"]["action"] == "request_fresh_verifier"
    for entry in policy.values():
        assert entry["action"] in mod.POLICY_ACTIONS
        assert isinstance(entry["nonforgetting_check_required"], bool)
        assert isinstance(entry["certificate_tail_update_allowed"], bool)


def test_req_learn_1344_blocks_dvi_when_nonforgetting_regresses() -> None:
    """REQ-LEARN-1344-5: DVI readiness requires non-regressing non-forgetting."""

    regressing_1315 = dict(_exp1315(), memory_regression_count=1)

    artifact = mod.build_artifact(
        exp1303_artifact=_exp1303(),
        exp1315_artifact=regressing_1315,
        exp1324_artifact=_exp1324(),
        exp1341_artifact=_exp1341(headline_cases=["m104-cert-1"]),
        project_root="/repo",
    )

    assert artifact["dvi_ready"] is False
    assert artifact["headline_certificate_cases"] == ["m104-cert-1"]
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "failure_type_memory_policy_blocked_non_headline"


def test_req_learn_1344_run_records_requested_alias_fallbacks(tmp_path: Path) -> None:
    """REQ-LEARN-1344-2/6: requested aliases are recorded while fallbacks run."""

    results = tmp_path / "results"
    results.mkdir()
    (results / mod.EXP1303_FALLBACK_FILE).write_text(json.dumps(_exp1303()), encoding="utf-8")
    (results / mod.EXP1315_FALLBACK_FILE).write_text(json.dumps(_exp1315()), encoding="utf-8")
    (results / mod.EXP1324_FILE).write_text(json.dumps(_exp1324()), encoding="utf-8")
    (results / mod.EXP1341_FILE).write_text(json.dumps(_exp1341()), encoding="utf-8")
    out_path = results / mod.OUTPUT_FILE

    artifact = mod.run(
        results_dir=results,
        out_path=out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["inputs_unavailable"] == [
        f"results/{mod.EXP1303_REQUESTED_FILE}",
        f"results/{mod.EXP1315_REQUESTED_FILE}",
    ]
    assert artifact["input_resolution"]["exp1303"]["used"] == (
        f"results/{mod.EXP1303_FALLBACK_FILE}"
    )
    assert artifact["input_resolution"]["exp1315"]["used"] == (
        f"results/{mod.EXP1315_FALLBACK_FILE}"
    )
    assert artifact["headline_result_allowed"] is False


def test_req_learn_1344_missing_all_aliases_and_fallback_failure_classes(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1344-2/3: missing inputs and split-only classes stay auditable."""

    payloads, unavailable, resolution, sources = mod.load_inputs(tmp_path / "empty-results")

    assert payloads == {}
    assert sources == []
    assert f"results/{mod.EXP1303_REQUESTED_FILE}" in unavailable
    assert f"results/{mod.EXP1303_FALLBACK_FILE}" in unavailable
    assert resolution["exp1303"]["used"] is None

    counts = mod.failure_type_counts(
        {"formalizer_failure_modes": ["skip-me"], "certificate_attempt_count": 7},
        {
            "source_cases_available": {
                "source_failure_classes": ["solver_disagreement", "new_unmapped_failure"]
            },
            "parser_schema_risk_count": 3,
            "undergeneration_risk_count": 2,
            "semantic_invalidity_count": 1,
            "unknown_mishandling_count": 1,
        },
    )

    assert counts["parser_schema_mismatch"] == 3
    assert counts["solver_disagreement"] == 0
    assert counts["new_unmapped_failure"] == 0


def test_req_learn_1344_headline_requires_current_104_case_and_dvi_ready() -> None:
    """REQ-LEARN-1344-6: headline claims require current .104 certificate cases."""

    artifact = mod.build_artifact(
        exp1303_artifact=_exp1303(),
        exp1315_artifact=_exp1315(),
        exp1324_artifact={
            **_exp1324(),
            "source_metrics": {},
            "certificate_attempt_count": 9,
            "headline_certificate_cases": [{"milestone": ".104", "case_id": "cert-a"}],
        },
        exp1341_artifact=_exp1341(headline_cases=["old-cert"]),
        project_root="/repo",
    )

    assert artifact["replay_cases_used"] == 14
    assert artifact["headline_certificate_cases"] == [{"milestone": ".104", "case_id": "cert-a"}]
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "failure_type_memory_policy_dvi_ready_headline_eligible"


def test_req_learn_1344_validation_rejects_malformed_artifacts() -> None:
    """REQ-LEARN-1344-7: malformed final artifacts fail schema validation."""

    artifact = mod.build_artifact(
        exp1303_artifact=_exp1303(),
        exp1315_artifact=_exp1315(),
        exp1324_artifact=_exp1324(),
        exp1341_artifact=_exp1341(),
        project_root="/repo",
    )

    def check(mutated: dict[str, Any], message: str) -> None:
        try:
            mod.validate_artifact(mutated)
        except AssertionError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - failure branch for the assertion helper
            raise AssertionError(f"expected validation failure containing {message!r}")

    missing = dict(artifact)
    missing.pop("status")
    check(missing, "missing required fields")

    invalid_rate = dict(artifact, nonforgetting_certificate_rate=1.5)
    check(invalid_rate, "nonforgetting_certificate_rate")

    invalid_policy = dict(artifact, failure_type_policy="not-a-mapping")
    check(invalid_policy, "failure_type_policy")

    bad_entry = dict(artifact, failure_type_policy={"semantic_invalidity": "bad"})
    check(bad_entry, "policy for semantic_invalidity")

    bad_action = dict(
        artifact,
        failure_type_policy={
            "semantic_invalidity": {
                "action": "store_anyway",
                "nonforgetting_check_required": True,
                "certificate_tail_update_allowed": True,
            }
        },
    )
    check(bad_action, "unsupported policy")

    bad_nonforgetting_flag = dict(
        artifact,
        failure_type_policy={
            "semantic_invalidity": {
                "action": "promote",
                "nonforgetting_check_required": "yes",
                "certificate_tail_update_allowed": True,
            }
        },
    )
    check(bad_nonforgetting_flag, "nonforgetting_check_required")

    bad_tail_flag = dict(
        artifact,
        failure_type_policy={
            "semantic_invalidity": {
                "action": "promote",
                "nonforgetting_check_required": True,
                "certificate_tail_update_allowed": "yes",
            }
        },
    )
    check(bad_tail_flag, "certificate_tail_update_allowed")


def test_req_learn_1344_private_coercion_helpers_are_defensive() -> None:
    """REQ-LEARN-1344-4: malformed numeric fields coerce to safe defaults."""

    assert mod._float(None, 1.25) == 1.25
    assert mod._float("not-a-number", 0.75) == 0.75
    assert mod._int(None) == 0
    assert mod._int("not-an-int") == 0


def test_req_learn_1344_headline_requires_dvi_and_current_104_case() -> None:
    """REQ-LEARN-1344-6: headline labels require DVI readiness and current .104 cases."""

    artifact = mod.build_artifact(
        exp1303_artifact=dict(_exp1303(), self_learning_delta_overall="bad-float"),
        exp1315_artifact=dict(_exp1315(), self_learning_delta_overall=None),
        exp1324_artifact={
            "status": "complete",
            "certificate_attempt_count": "7",
            "formalizer_failure_modes": [
                "skip-non-record",
                {"class": "novel_failure", "count": "bad-int"},
            ],
            "headline_certificate_cases": [
                {"case_id": "stale-cert", "source_milestone": ".103"},
            ],
        },
        exp1341_artifact={
            "status": "complete",
            "source_cases_available": {
                "source_failure_classes": ["novel_failure", "semantic_invalidity"],
            },
            "parser_schema_risk_count": 1,
            "undergeneration_risk_count": 1,
            "semantic_invalidity_count": 2,
            "unknown_mishandling_count": 1,
            "repair_policy_by_failure_type": {
                "semantic_invalidity": {"next_actions": "malformed action list"},
            },
            "headline_certificate_cases": [
                {"case_id": "current-cert", "source_milestone": ".104"},
            ],
        },
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["self_learning_delta_overall"] == 0.0
    assert artifact["replay_cases_used"] == 12
    assert artifact["headline_certificate_cases"] == [
        {"case_id": "current-cert", "source_milestone": ".104"}
    ]
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "failure_type_memory_policy_dvi_ready_headline_eligible"
    assert artifact["failure_type_policy"]["novel_failure"]["action"] == "quarantine"
    assert artifact["failure_type_policy"]["semantic_invalidity"]["source_next_actions"] == []


def test_req_learn_1344_load_inputs_records_missing_fallbacks(tmp_path: Path) -> None:
    """REQ-LEARN-1344-2: fully absent artifacts are recorded as unavailable."""

    results = tmp_path / "results"
    results.mkdir()

    payloads, unavailable, resolution, sources = mod.load_inputs(results)

    assert payloads == {}
    assert sources == []
    assert resolution["exp1303"]["used"] is None
    assert f"results/{mod.EXP1303_REQUESTED_FILE}" in unavailable
    assert f"results/{mod.EXP1303_FALLBACK_FILE}" in unavailable
    assert f"results/{mod.EXP1315_REQUESTED_FILE}" in unavailable
    assert f"results/{mod.EXP1315_FALLBACK_FILE}" in unavailable
    assert f"results/{mod.EXP1324_FILE}" in unavailable
    assert f"results/{mod.EXP1341_FILE}" in unavailable


def test_req_learn_1344_validation_rejects_bad_artifacts() -> None:
    """REQ-LEARN-1344-7: downstream schema validation rejects unsafe artifacts."""

    artifact = mod.build_artifact(
        exp1303_artifact=_exp1303(),
        exp1315_artifact=_exp1315(),
        exp1324_artifact=_exp1324(),
        exp1341_artifact=_exp1341(),
        project_root="/repo",
    )

    missing = dict(artifact)
    del missing["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_rate = dict(artifact, nonforgetting_certificate_rate=1.5)
    with pytest.raises(AssertionError, match="between 0 and 1"):
        mod.validate_artifact(bad_rate)

    bad_policy_root = dict(artifact, failure_type_policy=[])
    with pytest.raises(AssertionError, match="must be a mapping"):
        mod.validate_artifact(bad_policy_root)

    bad_policy_entry = dict(artifact, failure_type_policy={"semantic_invalidity": []})
    with pytest.raises(AssertionError, match="must be a mapping"):
        mod.validate_artifact(bad_policy_entry)

    unsupported = dict(
        artifact,
        failure_type_policy={
            "semantic_invalidity": {
                "action": "accept",
                "nonforgetting_check_required": True,
                "certificate_tail_update_allowed": False,
            }
        },
    )
    with pytest.raises(AssertionError, match="unsupported policy"):
        mod.validate_artifact(unsupported)

    missing_nonforgetting_flag = dict(
        artifact,
        failure_type_policy={
            "semantic_invalidity": {
                "action": "promote",
                "certificate_tail_update_allowed": True,
            }
        },
    )
    with pytest.raises(AssertionError, match="nonforgetting_check_required"):
        mod.validate_artifact(missing_nonforgetting_flag)

    missing_tail_flag = dict(
        artifact,
        failure_type_policy={
            "semantic_invalidity": {
                "action": "promote",
                "nonforgetting_check_required": True,
            }
        },
    )
    with pytest.raises(AssertionError, match="certificate_tail_update_allowed"):
        mod.validate_artifact(missing_tail_flag)

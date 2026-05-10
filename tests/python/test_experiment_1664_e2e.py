"""Tests for Exp 1664 EBRM/SMGI E2E plan updates.

Spec: REQ-LEARN-1664, SCENARIO-LEARN-1664, SCENARIO-LEARN-1665.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_1664_e2e as mod


def _base_plan() -> str:
    return """# Carnot — E2E Test Plan

**Last Updated:** 2026-04-12

### E2E-005: Packaged Code Verification Generate-Verify-Repair

**Objective:** Existing packaged verifier coverage.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _source_artifacts(tmp_path: Path, *, smgi_status: str = "complete") -> dict[str, Path]:
    return {
        "ebrm_scorer": _write_json(
            tmp_path / "experiment_1656_ebrm_trace_scorer.json",
            {
                "status": "complete",
                "experiment_id": 1656,
                "ebrm_trace_scorer_ready": True,
                "continuous_energy_used": True,
                "score_accuracy": 1.0,
                "spec_traces": ["REQ-VERIFY-1656", "SCENARIO-VERIFY-1656"],
            },
        ),
        "kv260_binding": _write_json(
            tmp_path / "experiment_1657_kv260_ebrm_binding.json",
            {
                "status": "complete",
                "experiment_id": 1657,
                "kv260_ebrm_binding_ready": True,
                "continuous_energy_used": True,
                "potts_q_states": 3,
                "score_accuracy": 1.0,
                "spec_traces": ["REQ-VERIFY-1657", "SCENARIO-VERIFY-1657"],
            },
        ),
        "hw_eval": _write_json(
            tmp_path / "experiment_1658_hw_eval.json",
            {
                "status": "complete",
                "experiment_id": 1658,
                "cases_total": 2,
                "max_score_delta": 0.0,
                "scoring_delta_within_tolerance": True,
                "cpu_score_accuracy": 1.0,
                "kv260_score_accuracy": 1.0,
                "case_scores": [
                    {
                        "case_id": "case-1",
                        "cpu_energy": 0.0,
                        "kv260_energy": 0.0,
                        "score_delta": 0.0,
                        "potts_q_states": 3,
                    }
                ],
                "spec_traces": ["REQ-VERIFY-1658", "SCENARIO-VERIFY-1658"],
            },
        ),
        "smgi": _write_json(
            tmp_path / "experiment_1659_smgi_certified_updates.json",
            {
                "status": smgi_status,
                "experiment_id": 1659,
                "continuous_self_learning_task": True,
                "smgi_certified_update_ready": smgi_status == "complete",
                "certified_update_success": smgi_status == "complete",
                "cerce_ledger_ready": smgi_status == "complete",
                "accepted_violation_count": 0,
                "false_accept_delta": 0,
                "soundness_mistakes": 0,
                "nonforgetting_certificate_rate": 1.0,
                "certified_updates": [
                    {
                        "policy_update_id": "policy:unit",
                        "prior_memory_hash": "a" * 64,
                        "updated_memory_hash": "b" * 64,
                        "replay_case_count": 1,
                        "retained_case_count": 1,
                        "replay_failure_count": 0,
                        "no_model_weight_mutation": True,
                        "gates": {
                            "cerce_certificate_match": True,
                            "memory_hashes_present": True,
                            "memory_hash_changed": True,
                            "retention_replay_passed": True,
                            "no_model_weight_mutation": True,
                        },
                        "provenance": ["results/experiment_1594_cerce_ledger.json"],
                    }
                ]
                if smgi_status == "complete"
                else [],
                "spec_traces": [
                    "REQ-LEARN-1659",
                    "SCENARIO-LEARN-1659",
                    "SCENARIO-LEARN-1660",
                ],
            },
        ),
    }


def _run(tmp_path: Path, plan_path: Path, **overrides: Any) -> dict[str, Any]:
    sources = _source_artifacts(tmp_path, **overrides)
    return mod.run_experiment(
        plan_path=plan_path,
        output_path=tmp_path / "experiment_1664_e2e_plan.json",
        ebrm_scorer_artifact_path=sources["ebrm_scorer"],
        kv260_binding_artifact_path=sources["kv260_binding"],
        hw_eval_artifact_path=sources["hw_eval"],
        smgi_artifact_path=sources["smgi"],
        tests_run=["tests/python/test_experiment_1664_e2e.py"],
    )


def test_scenario_learn_1664_adds_ebrm_and_smgi_e2e_sections(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1664: the plan receives EBRM and SMGI coverage entries."""

    plan_path = tmp_path / "e2e-test-plan.md"
    plan_path.write_text(_base_plan(), encoding="utf-8")

    artifact = _run(tmp_path, plan_path)
    plan_text = plan_path.read_text(encoding="utf-8")

    assert json.loads((tmp_path / "experiment_1664_e2e_plan.json").read_text()) == artifact
    assert artifact["status"] == "complete"
    assert artifact["plan_updated"] is True
    assert artifact["e2e_sections_added"] == ["E2E-006", "E2E-007"]
    assert artifact["e2e_section_ids"] == ["E2E-006", "E2E-007"]
    assert artifact["ebrm_e2e_ready"] is True
    assert artifact["smgi_e2e_ready"] is True
    assert artifact["source_artifacts"]["exp1658"]["ready"] is True
    assert artifact["source_artifacts"]["exp1659"]["ready"] is True
    assert artifact["spec_traces"] == ["REQ-LEARN-1664", "SCENARIO-LEARN-1664"]
    assert "### E2E-006: EBRM Trace Scorer CPU/KV260 Verification" in plan_text
    assert "### E2E-007: SMGI Certified Update Verification" in plan_text
    assert plan_text.count("### E2E-006:") == 1
    assert plan_text.count("### E2E-007:") == 1
    assert "REQ-VERIFY-1656" in plan_text
    assert "REQ-VERIFY-1658" in plan_text
    assert "REQ-LEARN-1659" in plan_text
    assert "no_model_weight_mutation=true" in plan_text
    mod.validate_artifact(artifact)


def test_req_learn_1664_update_is_idempotent(tmp_path: Path) -> None:
    """REQ-LEARN-1664-4: reruns do not duplicate existing E2E plan sections."""

    plan_path = tmp_path / "e2e-test-plan.md"
    plan_path.write_text(_base_plan(), encoding="utf-8")

    first = _run(tmp_path, plan_path)
    plan_after_first = plan_path.read_text(encoding="utf-8")
    second = _run(tmp_path, plan_path)
    plan_after_second = plan_path.read_text(encoding="utf-8")

    assert first["plan_updated"] is True
    assert second["status"] == "complete"
    assert second["plan_updated"] is False
    assert second["e2e_sections_added"] == []
    assert second["plan_hash_before"] == second["plan_hash_after"]
    assert plan_after_first == plan_after_second
    assert plan_after_second.count("### E2E-006:") == 1
    assert plan_after_second.count("### E2E-007:") == 1


def test_scenario_learn_1665_blocks_missing_or_incomplete_source_evidence(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1665: incomplete SMGI evidence blocks completion."""

    plan_path = tmp_path / "e2e-test-plan.md"
    plan_path.write_text(_base_plan(), encoding="utf-8")

    artifact = _run(tmp_path, plan_path, smgi_status="blocked")

    assert artifact["status"] == "blocked"
    assert artifact["ebrm_e2e_ready"] is True
    assert artifact["smgi_e2e_ready"] is False
    assert artifact["source_artifacts"]["exp1659"]["ready"] is False
    assert any("Exp 1659" in blocker for blocker in artifact["blockers"])
    assert "### E2E-007: SMGI Certified Update Verification" in plan_path.read_text(
        encoding="utf-8"
    )
    mod.validate_artifact(artifact)


def test_req_learn_1664_validation_catches_schema_edges(tmp_path: Path) -> None:
    """REQ-LEARN-1664-5: artifact validation catches missing and impossible fields."""

    plan_path = tmp_path / "e2e-test-plan.md"
    plan_path.write_text(_base_plan(), encoding="utf-8")
    artifact = _run(tmp_path, plan_path)

    missing = dict(artifact)
    del missing["schema"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="schema"):
        mod.validate_artifact(dict(artifact, schema="wrong"))

    with pytest.raises(AssertionError, match="status"):
        mod.validate_artifact(dict(artifact, status="unknown"))

    with pytest.raises(AssertionError, match="experiment_id"):
        mod.validate_artifact(dict(artifact, experiment_id=0))

    with pytest.raises(AssertionError, match="e2e_section_ids"):
        mod.validate_artifact(dict(artifact, e2e_section_ids=[]))

    with pytest.raises(AssertionError, match="spec_traces"):
        mod.validate_artifact(dict(artifact, spec_traces=[]))

    with pytest.raises(AssertionError, match="source evidence"):
        mod.validate_artifact(dict(artifact, ebrm_e2e_ready=False))

    with pytest.raises(AssertionError, match="blockers"):
        mod.validate_artifact(dict(artifact, blockers=["unexpected"]))

    with pytest.raises(AssertionError, match="blocked artifact requires blockers"):
        mod.validate_artifact(dict(artifact, status="blocked", blockers=[]))


def test_req_learn_1664_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-1664-3: malformed evidence helpers fail closed."""

    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")

    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._read_json(array_json) == {}
    assert mod._certified_update_ready([]) is False
    assert mod._certified_update_ready({"gates": []}) is False
    assert mod._float("not-a-number") == 0.0
    assert mod._int("not-a-number") == 0

"""Tests for Exp 1472 online verifier asymmetric mistake-budget audit.

Spec: REQ-LEARN-1472, SCENARIO-LEARN-1473, SCENARIO-LEARN-1474.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import online_verifier_asymmetric_mistake_budget as mod


def _exp1471(
    *,
    status: str = "complete",
    headline_result_allowed: bool = True,
    pivot_preserved: bool = True,
    soundness_mistakes: int = 0,
    completeness_mistakes: int = 3,
) -> dict[str, Any]:
    return {
        "experiment": "1471_fr11_v8_verified_memory_growth_pivot",
        "status": status,
        "headline_result_allowed": headline_result_allowed,
        "pivot_preserved": pivot_preserved,
        "pivot_retired": not pivot_preserved,
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "nonforgetting_rate": 1.0,
        "self_learning_delta_overall": 2 if headline_result_allowed else 0,
        "memory_updates": {
            "promoted": ["dvi_v8:verified:case_a", "dvi_v8:verified:case_b"],
            "demoted": [f"dvi_v8:verified:demoted_{index}" for index in range(3)],
            "promoted_memory_count": 2,
            "demoted_memory_count": 3,
            "rejection_reason_counts": {"verifier_rejection": 3},
        },
        "honest_verdict": "fr11_v8_positive_verified_memory_growth_persisted_without_forgetting",
    }


def test_req_learn_1472_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1472-1/5: bootstrap artifact exposes required fields first."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "in_progress"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["self_learning_claim_preserved"] is False
    assert artifact["self_learning_claim_retired"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_learn_1473_zero_soundness_preserves_narrow_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1473: zero dangerous mistakes preserves the narrow claim."""

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(soundness_mistakes=0, completeness_mistakes=3),
        audit_note_path=tmp_path / "audit.md",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 3
    assert artifact["asymmetric_cost_weights"]["soundness"] > artifact[
        "asymmetric_cost_weights"
    ]["completeness"]
    assert artifact["asymmetric_cost_score"] == 3.0
    assert artifact["self_learning_claim_preserved"] is True
    assert artifact["self_learning_claim_retired"] is False
    assert artifact["pareto_decision"] == mod.PARETO_PRESERVE
    assert artifact["ledger_summary"]["demoted_memory_count"] == 3
    assert artifact["ledger_summary"]["verifier_rejection_count"] == 3


def test_scenario_learn_1474_soundness_risk_retires_claim(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1474: a dangerous missed error retires the claim."""

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(soundness_mistakes=1, completeness_mistakes=0),
        audit_note_path=tmp_path / "audit.md",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["asymmetric_cost_score"] == 10.0
    assert artifact["self_learning_claim_preserved"] is False
    assert artifact["self_learning_claim_retired"] is True
    assert artifact["pareto_decision"] == mod.PARETO_RETIRE_SOUNDNESS
    assert artifact["honest_verdict"] == "self_learning_claim_retired_soundness_risk"


def test_req_learn_1472_source_gate_failure_retires_claim(tmp_path: Path) -> None:
    """REQ-LEARN-1472-4: source headline failure blocks preservation."""

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(headline_result_allowed=False, pivot_preserved=False),
        audit_note_path=tmp_path / "audit.md",
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["self_learning_claim_preserved"] is False
    assert artifact["self_learning_claim_retired"] is True
    assert artifact["pareto_decision"] == mod.PARETO_RETIRE_SOURCE
    assert artifact["honest_verdict"] == "self_learning_claim_retired_source_gate_failed"


def test_req_learn_1472_run_writes_artifact_and_note(tmp_path: Path) -> None:
    """REQ-LEARN-1472-2/6: run loads Exp 1471 and writes both outputs."""

    results_dir = tmp_path / "results"
    exp1471_path = results_dir / "experiment_1471.json"
    out_path = results_dir / mod.OUTPUT_FILE
    note_path = tmp_path / "docs" / "research-notes" / "fr11_v8_audit.md"
    exp1471_path.parent.mkdir(parents=True, exist_ok=True)
    exp1471_path.write_text(json.dumps(_exp1471()), encoding="utf-8")

    artifact = mod.run(
        exp1471_path=exp1471_path,
        out_path=out_path,
        audit_note_path=note_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert note_path.exists()
    assert "asymmetric mistake-budget audit" in note_path.read_text(encoding="utf-8")
    assert artifact["commands_run"] == ["pytest targeted"]


def test_req_learn_1472_validation_rejects_bad_contract(tmp_path: Path) -> None:
    """REQ-LEARN-1472-5: validation enforces cost and decision invariants."""

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(),
        audit_note_path=tmp_path / "audit.md",
        project_root="/repo",
    )

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "status"})

    bad_weights = dict(artifact)
    bad_weights["asymmetric_cost_weights"] = {"soundness": 1.0, "completeness": 1.0}
    with pytest.raises(AssertionError, match="soundness cost weight"):
        mod.validate_artifact(bad_weights)

    bad_score = dict(artifact)
    bad_score["asymmetric_cost_score"] = 999.0
    with pytest.raises(AssertionError, match="asymmetric_cost_score"):
        mod.validate_artifact(bad_score)

    bad_flags = dict(artifact)
    bad_flags["self_learning_claim_retired"] = True
    with pytest.raises(AssertionError, match="exactly one"):
        mod.validate_artifact(bad_flags)

    malformed_ledger = _exp1471()
    malformed_ledger["memory_updates"] = {
        "promoted": "not-a-list",
        "demoted": "not-a-list",
        "rejection_reason_counts": "not-a-dict",
    }
    edge_artifact = mod.build_artifact(
        exp1471_artifact=malformed_ledger,
        audit_note_path=tmp_path / "audit.md",
        project_root="/repo",
    )
    assert edge_artifact["ledger_summary"]["promoted_ledger_count"] == 0
    assert edge_artifact["ledger_summary"]["demoted_ledger_count"] == 0

"""Tests for Exp 3129 FR-11 constraint-memory retention and drift audit.

Spec refs: REQ-LEARN-3129, SCENARIO-LEARN-3129,
SCENARIO-LEARN-3129-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fragment_time_monitor_satisfiable_drift_audit_v1 as monitor
from carnot.eval import fr11_constraint_memory_retention_drift_audit_v1 as mod
from carnot.eval import fr11_evoenv_verifiable_environment_synthesis_v1 as evo


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    target = root / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _monitor_events() -> list[dict[str, Any]]:
    return [
        {
            "event_type": "candidate_final_answer",
            "fixture_id": "valid-a",
            "payload": {
                "has_returned_answer": True,
                "final_answer_consistent_with_ledger": True,
            },
        },
        {
            "event_type": "drift_classification",
            "fixture_id": "valid-a",
            "payload": {"failure_mechanism": "no_failure"},
        },
        {
            "event_type": "candidate_final_answer",
            "fixture_id": "valid-b",
            "payload": {
                "has_returned_answer": True,
                "final_answer_consistent_with_ledger": True,
            },
        },
        {
            "event_type": "drift_classification",
            "fixture_id": "valid-b",
            "payload": {"failure_mechanism": "no_failure"},
        },
    ]


def _exp3128_payload() -> dict[str, Any]:
    summary = evo.evaluate_admission(evo.sample_candidate_environments(seed=3128))
    return {
        "artifact": "experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1",
        "fr11_evoenv_pilot_v1_ready": True,
        "admitted_environment_count": summary.admitted_count,
        "admission_records": [record.to_dict() for record in summary.records],
        "admitted_environments": evo.admitted_environment_rows(summary),
        "soundness_errors": summary.soundness_errors,
        "completeness_errors": summary.completeness_errors,
        "retention_delta": 0.0,
        "no_weight_update_claim": True,
        "inference_substrate": {"model_weight_mutation": False},
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake verifier claims\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3129\nSCENARIO-LEARN-3129\nSCENARIO-LEARN-3129-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3116_REL_PATH,
        {
            "fr11_unsolvable_curriculum_ready": True,
            "hard_family_count": 2,
            "unsolvable_detection_summary": {"hard_families": ["arith", "json"]},
            "guarded_decisions": [
                {
                    "fixture_id": "prior-1",
                    "controller_decision": "accept",
                    "target_action": "accept",
                    "decision_label": "correct",
                },
                {
                    "fixture_id": "prior-2",
                    "controller_decision": "reject",
                    "target_action": "reject",
                    "decision_label": "correct",
                },
            ],
            "no_weight_update_claim": True,
        },
    )
    _write_json(root, mod.EXP3128_REL_PATH, _exp3128_payload())
    replay = monitor.replay_monitor_events(_monitor_events())
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "fragment_time_monitor_v1_ready": True,
            "monitor_events": _monitor_events(),
            "satisfiable_drift_count": replay["satisfiable_drift_count"],
            "ledger_consistency_rate": replay["ledger_consistency_rate"],
            "inference_substrate": {"fresh_live_inference_calls": 0},
        },
    )


def test_req_learn_3129_spec_anchor_exists() -> None:
    """REQ-LEARN-3129: OpenSpec declares the retention and drift audit."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3129" in spec
    assert "SCENARIO-LEARN-3129" in spec
    assert "SCENARIO-LEARN-3129-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_constraint_memory_audit_v1_ready" in spec
    assert "novelty_retention_delta" in spec
    assert "satisfiable_drift_count" in spec
    assert "no_weight_update_claim" in spec


def test_req_learn_3129_replays_admitted_environment_exactly() -> None:
    """REQ-LEARN-3129-1/2/4: admitted environments replay by exact enumeration."""

    exp3128 = _exp3128_payload()
    summary = mod.replay_admitted_environments(exp3128)
    first = mod.environment_from_row(exp3128["admitted_environments"][0])

    assert summary["admitted_environment_count"] == 3
    assert summary["family_count"] == 3
    assert summary["baseline_success_rate"] == pytest.approx(1.0)
    assert summary["post_replay_success_rate"] == pytest.approx(1.0)
    assert summary["novelty_retention_delta"] == pytest.approx(0.0)
    assert summary["soundness_errors"] == 0
    assert summary["completeness_errors"] == 0
    assert first.score_response(first.compute_reference().canonical_assignment).accepted is True


def test_scenario_learn_3129_builds_complete_audit_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3129: audit separates reusable memory from model weights."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-LEARN-3129 focused"],
    )

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_constraint_memory_audit_v1_ready"] is True
    assert artifact["admitted_environment_count"] == 3
    assert artifact["replay_family_count"] == 5
    assert artifact["prior_retention_delta"] == pytest.approx(0.0)
    assert artifact["novelty_retention_delta"] == pytest.approx(0.0)
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["satisfiable_drift_count"] == 0
    assert artifact["ledger_consistency_rate"] == pytest.approx(1.0)
    assert artifact["promotion_recommendation"] == "promote_controller_environment_memory_only"
    assert artifact["no_weight_update_claim"] is True
    assert artifact["tests_run"] == ["REQ-LEARN-3129 focused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "artifact_only_constraint_memory_audit"
    assert substrate["controller_environment_memory_only"] is True
    assert substrate["fresh_live_inference_calls"] == 0
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    mod.validate_artifact(artifact)


def test_scenario_learn_3129_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3129-BLOCKED: missing sources fail closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_constraint_memory_audit_v1_ready"] is False
    assert artifact["admitted_environment_count"] == 0
    assert artifact["replay_family_count"] == 0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["novelty_retention_delta"] == 0.0
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["satisfiable_drift_count"] == 0
    assert artifact["ledger_consistency_rate"] == 0.0
    assert artifact["promotion_recommendation"].startswith("block_")
    assert artifact["no_weight_update_claim"] is True
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    assert mod.precondition_blocker({}, {}, {}) == "exp3128_evoenv_artifact_missing_or_not_ready"
    assert (
        mod.precondition_blocker(
            {"fr11_evoenv_pilot_v1_ready": True},
            {},
            {},
        )
        == "exp3116_retention_guard_missing_or_not_ready"
    )
    assert (
        mod.precondition_blocker(
            {"fr11_evoenv_pilot_v1_ready": True},
            {"fr11_unsolvable_curriculum_ready": True},
            {},
        )
        == "exp3126_drift_monitor_missing_or_not_ready"
    )
    mod.validate_artifact(artifact)


def test_req_learn_3129_writes_artifact_and_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-LEARN-3129-3/5/6: validation blocks regressions and overclaims."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        started_s=3.0,
        now_s=4.0,
        tests_run=["write-check"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["fr11_constraint_memory_audit_v1_ready"] is True
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rate(1, 0) == 0.0
    assert mod.prior_retention_summary({})["prior_retention_delta"] == 0.0
    malformed_prior = mod.prior_retention_summary(
        {"hard_family_count": 0, "unsolvable_detection_summary": {"hard_families": "bad"}}
    )
    assert malformed_prior["family_count"] == 0
    assert mod.drift_summary({"monitor_events": []})["ledger_consistency_rate"] == 0.0
    fallback_drift = mod.drift_summary(
        {
            "satisfiable_drift_count": 2,
            "ledger_consistency_rate": 0.5,
            "contradiction_count": 1,
            "monitor_violation_count": 3,
        }
    )
    assert fallback_drift["satisfiable_drift_count"] == 2
    assert mod.promotion_recommendation(True, 0, 0, 1, 0.0, 0.0, 1.0).startswith("block_")
    assert mod.promotion_recommendation(True, 1, 0, 0, 0.0, 0.0, 1.0).startswith("block_")
    assert mod.promotion_recommendation(True, 0, 0, 0, -0.1, 0.0, 1.0).startswith("block_")
    assert mod.promotion_recommendation(True, 0, 0, 0, 0.0, -0.1, 1.0).startswith("block_")
    assert "ledger_consistency" in mod.promotion_recommendation(True, 0, 0, 0, 0.0, 0.0, 0.5)
    assert mod.promotion_recommendation(False, 0, 0, 0, 0.0, 0.0, 1.0).startswith("block_")
    assert mod.honest_verdict(False, "block").startswith("blocked_precondition_failed")
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="admitted_environment_count"):
        mod.validate_artifact(artifact | {"admitted_environment_count": 99})
    with pytest.raises(ValueError, match="no_weight_update_claim"):
        mod.validate_artifact(artifact | {"no_weight_update_claim": False})
    with pytest.raises(ValueError, match="model_weight_mutation"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            }
        )
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"fresh_live_inference_calls": 1}
            }
        )
    with pytest.raises(ValueError, match="ledger_consistency_rate"):
        mod.validate_artifact(artifact | {"ledger_consistency_rate": 1.5})
    with pytest.raises(ValueError, match="soundness_errors"):
        mod.validate_artifact(artifact | {"soundness_errors": 1})
    with pytest.raises(ValueError, match="completeness_errors"):
        mod.validate_artifact(artifact | {"completeness_errors": 1})
    with pytest.raises(ValueError, match="satisfiable_drift_count"):
        mod.validate_artifact(artifact | {"satisfiable_drift_count": 1})
    with pytest.raises(ValueError, match="prior_retention_delta"):
        mod.validate_artifact(artifact | {"prior_retention_delta": -0.1})
    with pytest.raises(ValueError, match="novelty_retention_delta"):
        mod.validate_artifact(artifact | {"novelty_retention_delta": -0.1})
    with pytest.raises(ValueError, match="source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})

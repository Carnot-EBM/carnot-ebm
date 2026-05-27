"""Tests for Exp 3200 FR-11 VeriFY-style trace-memory controller.

Spec refs: REQ-LEARN-3200, SCENARIO-LEARN-3200,
SCENARIO-LEARN-3200-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_verify_trace_memory_controller_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(
    row_id: str,
    fixture_family: str = "modular_balance",
    *,
    expected_action: str = "accept",
    observed_action: str = "accept",
    exact_label: str = "unknown",
    source_artifact: str = "results/unit_source.json",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_family": fixture_family,
        "panel_category": "admitted_environment",
        "source_artifact": source_artifact,
        "expected_action": expected_action,
        "ledger_action": expected_action,
        "observed_action": observed_action,
        "routing_decision": "normal",
        "mismatch_class": "",
        "exact_label": exact_label,
        "exact_authority": True,
    }


def _exp3186_payload(*, promotion_allowed: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3186_fr11_controller_memory_promotion_pack_v1",
        "fr11_controller_memory_promotion_pack_v1_ready": True,
        "continuous_self_learning_task": True,
        "promotion_allowed": promotion_allowed,
        "no_model_weight_update_claimed": True,
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "uses_checked_in_artifacts_only": True,
            "executes_live_model_inference": False,
            "fresh_live_inference_calls": 0,
            "model_weight_learning": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
            "kan_model_weight_training": False,
            "hidden_state_mutation_claimed": False,
        },
        "promotion_manifest": {
            "update_id": "fr11-controller-memory-exp3172-row-exact-v1",
            "update_type": "controller_memory_exact_row_action_override",
            "activation_predicate": {
                "mode": "exact_row_id_controller_memory_override",
                "row_action_overrides": {"resyn-3084-arith-003": "reject"},
            },
            "evidence": {
                "before_ledger_consistency_rate": 0.857143,
                "after_ledger_consistency_rate": 1.0,
                "heldout_consistency_rate": 1.0,
                "negative_control_regression_count": 0,
            },
        },
        "honest_verdict": "complete: unit promotion pack",
    }


def _exp3187_payload(
    *,
    promotion_allowed: bool = True,
    negative_control_regression_count: int = 0,
) -> dict[str, Any]:
    heldout_rows = [
        _row("safe-accept"),
        _row(
            "known-reject",
            "arithmetic_code_assertions",
            expected_action="reject",
            observed_action="reject",
            exact_label="INVALID",
        ),
    ]
    drift_rows = [_row("safe-accept")]
    negative_rows = [_row("safe-accept")]
    return {
        "artifact": "experiment_3187_fr11_cross_environment_drift_replay_v1",
        "fr11_cross_environment_drift_replay_v1_ready": True,
        "continuous_self_learning_task": True,
        "promotion_allowed": promotion_allowed,
        "replay_mode_only": True,
        "no_model_weight_update_claimed": True,
        "heldout_row_count": len(heldout_rows),
        "cross_environment_row_count": len(drift_rows),
        "negative_control_regression_count": negative_control_regression_count,
        "negative_control_regressions": [{"row_id": "safe-accept", "reason": "unit regression"}]
        if negative_control_regression_count
        else [],
        "drift_cases": [],
        "rollback_triggered": negative_control_regression_count > 0,
        "before_after_consistency": {
            "source": {"before_rate": 0.857143, "after_rate": 1.0, "lift": 0.142857},
            "heldout": {"row_count": len(heldout_rows), "before_rate": 1.0, "after_rate": 1.0},
            "cross_environment": {
                "row_count": len(drift_rows),
                "before_rate": 1.0,
                "after_rate": 1.0,
            },
            "negative_control": {
                "row_count": len(negative_rows),
                "before_rate": 1.0,
                "after_rate": 1.0,
                "regression_count": negative_control_regression_count,
            },
        },
        "row_selection": {
            "heldout_rows": heldout_rows,
            "cross_environment_rows": drift_rows,
            "negative_control_rows": negative_rows,
        },
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "uses_checked_in_artifacts_only": True,
            "executes_live_model_inference": False,
            "fresh_live_inference_calls": 0,
            "model_weight_learning": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
            "kan_model_weight_training": False,
            "hidden_state_mutation_claimed": False,
        },
        "honest_verdict": "complete: unit drift replay",
    }


def _write_sources(
    root: Path,
    *,
    exp3186_promotion_allowed: bool = True,
    exp3187_promotion_allowed: bool = True,
    negative_control_regression_count: int = 0,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no model-weight update claims\n", encoding="utf-8")
    (root / "research-program.md").write_text("VeriFY trace memory\n", encoding="utf-8")
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3200\nSCENARIO-LEARN-3200\nSCENARIO-LEARN-3200-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3186_REL_PATH,
        _exp3186_payload(promotion_allowed=exp3186_promotion_allowed),
    )
    _write_json(
        root,
        mod.EXP3187_REL_PATH,
        _exp3187_payload(
            promotion_allowed=exp3187_promotion_allowed,
            negative_control_regression_count=negative_control_regression_count,
        ),
    )


def test_req_learn_3200_spec_anchor_exists() -> None:
    """REQ-LEARN-3200: OpenSpec declares the trace-memory controller artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3200" in spec
    assert "SCENARIO-LEARN-3200" in spec
    assert "SCENARIO-LEARN-3200-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "trace_schema" in spec
    assert "redundant_check_suppression_count" in spec
    assert "model_weight_update_performed" in spec


def test_req_learn_3200_trace_schema_and_experience_pool() -> None:
    """REQ-LEARN-3200-1/2/3: exact trace records drive redundant-check suppression."""

    replay = _exp3187_payload()
    materialized = mod.materialize_trace_memory(replay)
    records = materialized["trace_records"]
    by_role = {(row["row_id"], row["replay_role"]): row for row in records}

    assert set(mod.TRACE_SCHEMA_FIELDS) <= set(mod.trace_schema()["fields"])
    assert len(records) == 4
    assert materialized["heldout_row_count"] == 2
    assert materialized["drift_row_count"] == 1
    assert materialized["negative_control_row_count"] == 1
    assert materialized["redundant_check_suppression_count"] == 2
    assert by_role[("safe-accept", "heldout")]["redundant_check_suppressed"] is False
    assert by_role[("safe-accept", "drift")]["redundant_check_suppressed"] is True
    assert by_role[("safe-accept", "negative_control")]["routing_outcome"] == (
        "skip_redundant_recheck"
    )
    assert by_role[("safe-accept", "heldout")]["exact_label"] == "EXACT_ACCEPT"
    assert by_role[("safe-accept", "heldout")]["answer_abstain_decision"] == "answer"
    assert by_role[("known-reject", "heldout")]["answer_abstain_decision"] == "abstain"
    assert by_role[("known-reject", "heldout")]["routing_outcome"] == "abstain_or_escalate"


def test_scenario_learn_3200_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3200: complete artifact promotes controller memory only."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-LEARN-3200 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["trace_count"] == 4
    assert artifact["heldout_row_count"] == 2
    assert artifact["drift_row_count"] == 1
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["redundant_check_suppression_count"] == 2
    assert artifact["routing_accuracy_delta"] == pytest.approx(0.142857)
    assert artifact["heldout_drift_accuracy_delta"] == pytest.approx(0.0)
    assert artifact["model_weight_update_performed"] is False
    assert artifact["promotion_allowed"] is True
    assert artifact["tests_run"] == ["REQ-LEARN-3200 focused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3200_blocks_unsafe_sources_and_regressions(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3200-BLOCKED: unsafe source or negative control denies promotion."""

    missing = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["trace_count"] == 0
    assert missing["promotion_allowed"] is False
    assert missing["blocked_reason"] == "exp3186_missing_or_not_ready"
    assert missing["honest_verdict"].startswith("complete:")
    mod.validate_artifact(missing)

    _write_sources(tmp_path, exp3186_promotion_allowed=False)
    not_allowed = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert not_allowed["blocked_reason"] == "exp3186_promotion_not_allowed"
    assert not_allowed["promotion_allowed"] is False
    mod.validate_artifact(not_allowed)

    _write_sources(tmp_path, negative_control_regression_count=1)
    regressed = mod.build_artifact(tmp_path, started_s=3.0, now_s=4.0)
    assert regressed["negative_control_regression_count"] == 1
    assert regressed["promotion_allowed"] is False
    assert regressed["promotion_blockers"] == ["negative_control_regression"]
    assert regressed["model_weight_update_performed"] is False
    mod.validate_artifact(regressed)


def test_req_learn_3200_validation_and_helper_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3200-3/4/5/6: validation rejects overclaims and unsafe promotion."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.0)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rows_from_selection({"bad": "rows"}, "bad") == []
    assert mod.rows_from_selection({"rows": [None, {"row_id": "ok"}]}, "rows") == [{"row_id": "ok"}]
    assert mod.rate(1, 0) == 0.0
    assert mod.rate(1, 2) == pytest.approx(0.5)
    assert mod.sha256_file(tmp_path / "absent.txt") is None
    assert mod.routing_accuracy_delta({}) is None
    assert mod.heldout_drift_accuracy_delta({"before_after_consistency": []}) is None
    assert mod.heldout_drift_accuracy_delta({"before_after_consistency": {"heldout": []}}) is None
    assert mod.exact_label_for({"expected_action": "reject"}) == "EXACT_REJECT"
    assert mod.exact_label_for({}) == "EXACT_UNKNOWN"
    assert mod.answer_decision_for("inconsistent", "accept") == "abstain"
    assert (
        mod.consistency_judgment_for(
            {"expected_action": "accept", "ledger_action": "accept", "observed_action": "reject"}
        )
        == "inconsistent"
    )
    assert mod.source_blocker({"exp3186": _exp3186_payload(), "exp3187": {}}) == (
        "exp3187_missing_or_not_ready"
    )
    bad_weight = _exp3186_payload()
    bad_weight["no_model_weight_update_claimed"] = False
    assert mod.source_blocker({"exp3186": bad_weight, "exp3187": _exp3187_payload()}) == (
        "exp3186_model_weight_update_claimed"
    )
    bad_substrate = _exp3187_payload()
    bad_substrate["inference_substrate"]["fresh_live_inference_calls"] = 1
    assert mod.source_blocker({"exp3186": _exp3186_payload(), "exp3187": bad_substrate}) == (
        "exp3187_live_inference_or_weight_update_claimed"
    )
    bad_exp3186_substrate = _exp3186_payload()
    bad_exp3186_substrate["inference_substrate"]["fresh_live_inference_calls"] = 1
    assert mod.source_blocker(
        {"exp3186": bad_exp3186_substrate, "exp3187": _exp3187_payload()}
    ) == ("exp3186_live_inference_or_weight_update_claimed")
    assert mod.source_claims_live_or_mutation({"inference_substrate": []}) is True
    assert mod.detected_model_weight_update({"bad": []}) is False
    assert mod.detected_model_weight_update({"bad": {"no_model_weight_update_claimed": False}})
    assert mod.detected_model_weight_update(
        {"bad": {"inference_substrate": {"model_weight_training": True}}}
    )
    blockers = mod.promotion_blockers(
        {"promotion_allowed": False},
        {
            "promotion_allowed": False,
            "negative_control_regression_count": 1,
            "before_after_consistency": [],
        },
        {"trace_count": 0, "heldout_row_count": 0, "drift_row_count": 0},
    )
    assert blockers == [
        "exp3186_promotion_not_allowed",
        "exp3187_promotion_not_allowed",
        "empty_trace_memory",
        "missing_heldout_replay",
        "missing_drift_replay",
        "negative_control_regression",
        "missing_heldout_drift_accuracy",
    ]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="model_weight_update_performed"):
        mod.validate_artifact(artifact | {"model_weight_update_performed": True})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": []})
    live_substrate = dict(artifact["inference_substrate"])
    live_substrate["fresh_live_inference_calls"] = 1
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(artifact | {"inference_substrate": live_substrate})
    mutation_substrate = dict(artifact["inference_substrate"])
    mutation_substrate["model_weight_training"] = True
    with pytest.raises(ValueError, match="model mutation flags"):
        mod.validate_artifact(artifact | {"inference_substrate": mutation_substrate})
    with pytest.raises(ValueError, match="trace_schema"):
        mod.validate_artifact(artifact | {"trace_schema": {"fields": []}})
    with pytest.raises(ValueError, match="trace_count"):
        mod.validate_artifact(artifact | {"trace_count": 0, "promotion_allowed": True})
    with pytest.raises(ValueError, match="heldout_row_count"):
        mod.validate_artifact(artifact | {"heldout_row_count": 0, "promotion_allowed": True})
    with pytest.raises(ValueError, match="drift_row_count"):
        mod.validate_artifact(artifact | {"drift_row_count": 0, "promotion_allowed": True})
    with pytest.raises(ValueError, match="negative-control"):
        mod.validate_artifact(
            artifact
            | {
                "promotion_allowed": True,
                "negative_control_regression_count": 1,
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})

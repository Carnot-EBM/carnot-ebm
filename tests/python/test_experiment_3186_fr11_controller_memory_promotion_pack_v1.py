"""Tests for Exp 3186 FR-11 controller-memory promotion pack v1.

Spec refs: REQ-LEARN-3186, SCENARIO-LEARN-3186,
SCENARIO-LEARN-3186-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_controller_memory_promotion_pack_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _updated_row(row_id: str, family: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_family": family,
        "update_type": "row_exact_action_override",
        "before_observed_action": "accept",
        "after_observed_action": "reject",
        "bounded_to_training_row_id": True,
        "model_weight_update": False,
    }


def _exp3172_payload(*, promotion_allowed: bool = True) -> dict[str, Any]:
    rows = [
        _updated_row("resyn-3084-arith-003", "arithmetic_code_assertions"),
        _updated_row("resyn-3084-smt-000", "smt_constraints"),
    ]
    return {
        "artifact": "experiment_3172_fr11_nonforgetting_self_learning_pilot_v2",
        "fr11_nonforgetting_self_learning_pilot_v2_ready": True,
        "continuous_self_learning_task": True,
        "controller_memory_update_applied": True,
        "model_weight_update_claimed": False,
        "before_ledger_consistency_rate": 0.857143,
        "after_ledger_consistency_rate": 1.0,
        "heldout_consistency_rate": 1.0,
        "negative_control_regression_count": 0,
        "negative_control_regressions": [],
        "nonforgetting_passed": True,
        "promotion_allowed": promotion_allowed,
        "promotion_recommendation": "promote_controller_memory_update_only"
        if promotion_allowed
        else "block_fr11_promotion_nonforgetting_failed",
        "controller_memory_update": {
            "update_policy": "exact_row_id_controller_memory_override_from_training_rows_only",
            "updated_row_count": len(rows),
            "updated_rows": rows,
            "row_action_overrides": {
                "resyn-3084-arith-003": "reject",
                "resyn-3084-smt-000": "reject",
            },
            "heldout_rows_used_for_update": False,
            "negative_control_rows_used_for_update": False,
            "model_weight_update": False,
        },
        "training_replay_rows": [
            {
                "row_id": row["row_id"],
                "fixture_family": row["fixture_family"],
                "observed_action": "reject",
                "expected_action": "reject",
                "consistent": True,
            }
            for row in rows
        ],
        "heldout_replay_rows": [{"row_id": "holdout-a", "consistent": True}],
        "negative_control_replay_rows": [{"row_id": "control-a", "consistent": True}],
        "inference_substrate": {
            "fresh_live_inference_calls": 0,
            "executes_live_model_inference": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
            "kan_model_weight_training": False,
            "hidden_state_mutation_claimed": False,
        },
        "honest_verdict": "complete: unit exp3172 source",
    }


def _exp3171_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3171_fr11_ledger_counterexample_isolation_v1",
        "fr11_ledger_counterexample_isolation_ready": True,
        "continuous_self_learning_task": True,
        "isolated_counterexample_families": [
            {
                "fixture_family": "arithmetic_code_assertions",
                "failing_row_ids": ["resyn-3084-arith-003"],
            },
            {"fixture_family": "smt_constraints", "failing_row_ids": ["resyn-3084-smt-000"]},
        ],
        "environment_variant_split": {
            "training_update_rows": [
                {"row_id": "resyn-3084-arith-003"},
                {"row_id": "resyn-3084-smt-000"},
            ],
            "held_out_replay_rows": [{"row_id": "holdout-a"}],
            "negative_control_rows": [{"row_id": "control-a"}],
        },
        "promotion_allowed": False,
        "no_model_weight_update_claimed": True,
        "honest_verdict": "complete: unit exp3171 source",
    }


def _write_sources(root: Path, *, promotion_allowed: bool = True) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller-only FR-11 memory\n", encoding="utf-8")
    (root / "research-program.md").write_text(
        "Tier 1: Online weights\n"
        "Tier 2: Constraint Memory / Trace2Skill\n"
        "Cache verified facts across sessions, not model weights.\n",
        encoding="utf-8",
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3186\nSCENARIO-LEARN-3186\nSCENARIO-LEARN-3186-BLOCKED\n"
        "results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3171_REL_PATH, _exp3171_payload())
    _write_json(root, mod.EXP3172_REL_PATH, _exp3172_payload(promotion_allowed=promotion_allowed))
    _write_json(
        root,
        mod.EXP3175_REL_PATH,
        {
            "artifact": "experiment_3175_cross_corpus_matrix_v28",
            "fr11_status": "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update",
        },
    )
    _write_json(
        root,
        mod.EXP3176_REL_PATH,
        {
            "artifact": "experiment_3176_capstone_v294",
            "fr11_self_learning_status": (
                "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
            ),
            "fr11_model_weight_update_claimed": False,
        },
    )


def test_req_learn_3186_spec_anchor_exists() -> None:
    """REQ-LEARN-3186: OpenSpec declares the promotion pack artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3186" in spec
    assert "SCENARIO-LEARN-3186" in spec
    assert "SCENARIO-LEARN-3186-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "promotion_manifest" in spec
    assert "rollback_policy" in spec
    assert "no_model_weight_update_claimed" in spec


def test_req_learn_3186_extracts_manifest_and_replay_contract(tmp_path: Path) -> None:
    """REQ-LEARN-3186-2/3/4/5: Exp 3172 update becomes auditable manifest."""

    _write_sources(tmp_path)
    sources = mod.load_sources(tmp_path)
    pack = mod.build_promotion_pack(tmp_path, sources)
    manifest = pack["promotion_manifest"]

    assert pack["learning_tier"].startswith("Tier 2:")
    assert "controller-memory" in pack["learning_tier"]
    assert "online weight" not in pack["learning_tier"]
    assert pack["source_update_artifact"] == mod.EXP3172_REL_PATH.as_posix()
    assert pack["promotion_allowed"] is True
    assert manifest["update_id"] == "fr11-controller-memory-exp3172-row-exact-v1"
    assert manifest["update_type"] == "controller_memory_exact_row_action_override"
    assert manifest["source_counterexample_families"] == [
        "arithmetic_code_assertions",
        "smt_constraints",
    ]
    assert manifest["evidence"]["before_ledger_consistency_rate"] == pytest.approx(0.857143)
    assert manifest["evidence"]["after_ledger_consistency_rate"] == 1.0
    assert manifest["evidence"]["heldout_consistency_rate"] == 1.0
    assert manifest["evidence"]["negative_control_regression_count"] == 0
    assert manifest["activation_predicate"]["allowed_row_ids"] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    assert manifest["activation_predicate"]["row_action_overrides"] == {
        "resyn-3084-arith-003": "reject",
        "resyn-3084-smt-000": "reject",
    }
    assert manifest["owner_artifact_paths"]["drift_replay_artifact"] == (
        mod.EXP3187_REL_PATH.as_posix()
    )
    assert {req["id"] for req in pack["replay_requirements"]} == {
        "exp3172_training_and_heldout_replay",
        "negative_control_replay",
        "exp3187_cross_environment_drift_replay",
        "ops_documentation_reconciliation",
    }
    assert {trigger["trigger"] for trigger in pack["rollback_policy"]["triggers"]} == {
        "negative_control_regression",
        "stale_ledger_evidence",
        "drift_replay_failure",
        "exact_authority_conflict",
    }


def test_scenario_learn_3186_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3186: ready source writes promotion-allowed pack."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-LEARN-3186 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["fr11_controller_memory_promotion_pack_v1_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_update_claimed"] is True
    assert artifact["promotion_allowed"] is True
    assert artifact["promotion_manifest"]["promotion_decision"] == "promote_controller_memory_only"
    assert artifact["rollback_policy"]["rollback_action"] == "remove_exact_row_overrides"
    assert artifact["drift_replay_contract"]["target_artifact"] == mod.EXP3187_REL_PATH.as_posix()
    assert artifact["ops_reconciliation"]["status"] == "pending_conductor_reconciliation"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["REQ-LEARN-3186 focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["executes_live_model_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3186_blocked_without_promotable_source(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3186-BLOCKED: missing or unsafe source fails closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["fr11_controller_memory_promotion_pack_v1_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_update_claimed"] is True
    assert artifact["source_update_artifact"] == mod.EXP3172_REL_PATH.as_posix()
    assert artifact["promotion_allowed"] is False
    assert artifact["promotion_manifest"]["promotion_decision"] == "blocked"
    assert artifact["rollback_policy"]["rollback_action"] == "no_active_update"
    assert artifact["blocked_reason"] == "exp3172_missing_or_not_ready"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed:")
    mod.validate_artifact(artifact)

    _write_sources(tmp_path, promotion_allowed=False)
    blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert blocked["blocked_reason"] == "exp3172_promotion_not_allowed"
    assert blocked["promotion_allowed"] is False
    mod.validate_artifact(blocked)


def test_req_learn_3186_validation_and_helper_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3186-1/6: validation rejects overclaims and malformed inputs."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=5.0)
    mod.validate_artifact(artifact)

    not_ready = _exp3172_payload()
    not_ready["fr11_nonforgetting_self_learning_pilot_v2_ready"] = False
    assert mod.source_blocker({"exp3172": not_ready}) == "exp3172_missing_or_not_ready"
    weight_claim = _exp3172_payload()
    weight_claim["model_weight_update_claimed"] = True
    assert mod.source_blocker({"exp3172": weight_claim}) == "exp3172_model_weight_update_claimed"
    regression = _exp3172_payload()
    regression["negative_control_regression_count"] = 1
    assert mod.source_blocker({"exp3172": regression}) == (
        "exp3172_negative_control_regression_present"
    )
    no_update = _exp3172_payload()
    no_update["controller_memory_update"] = {"updated_rows": []}
    assert mod.source_blocker({"exp3172": no_update}) == (
        "exp3172_controller_memory_update_missing"
    )

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.updated_rows({"updated_rows": "bad"}) == []
    assert mod.updated_rows({"updated_rows": [None, {"row_id": "ok"}]}) == [{"row_id": "ok"}]
    assert mod.row_overrides({"row_action_overrides": "bad"}) == {}
    assert mod.sha256_file(tmp_path / "absent.txt") is None

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="no_model_weight_update_claimed"):
        mod.validate_artifact(artifact | {"no_model_weight_update_claimed": False})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": []})
    bad_substrate = dict(artifact["inference_substrate"])
    bad_substrate["fresh_live_inference_calls"] = 1
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(artifact | {"inference_substrate": bad_substrate})
    mutation_substrate = dict(artifact["inference_substrate"])
    mutation_substrate["model_weight_training"] = True
    with pytest.raises(ValueError, match="mutation flags"):
        mod.validate_artifact(artifact | {"inference_substrate": mutation_substrate})
    with pytest.raises(ValueError, match="promotion_manifest"):
        mod.validate_artifact(artifact | {"promotion_manifest": []})
    bad_manifest_shape = dict(artifact["promotion_manifest"])
    bad_manifest_shape["evidence"] = []
    with pytest.raises(ValueError, match="promotion evidence"):
        mod.validate_artifact(artifact | {"promotion_manifest": bad_manifest_shape})
    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(artifact | {"promotion_allowed": False})
    bad_manifest = dict(artifact["promotion_manifest"])
    bad_manifest["evidence"] = dict(bad_manifest["evidence"])
    bad_manifest["evidence"]["heldout_consistency_rate"] = 0.5
    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(artifact | {"promotion_manifest": bad_manifest})
    with pytest.raises(ValueError, match="required source_artifacts"):
        mod.validate_artifact(
            artifact | {"source_artifacts": [{"required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "not_done"})

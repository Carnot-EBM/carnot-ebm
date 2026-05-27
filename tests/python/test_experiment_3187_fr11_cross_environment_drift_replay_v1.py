"""Tests for Exp 3187 FR-11 cross-environment drift replay v1.

Spec refs: REQ-LEARN-3187, SCENARIO-LEARN-3187,
SCENARIO-LEARN-3187-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_cross_environment_drift_replay_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(
    row_id: str,
    fixture_family: str,
    panel_category: str,
    *,
    expected_action: str = "accept",
    observed_action: str = "accept",
    source_artifact: str = "results/unit_source.json",
    exact_label: str = "VALID",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_family": fixture_family,
        "panel_category": panel_category,
        "source_artifact": source_artifact,
        "expected_action": expected_action,
        "ledger_action": expected_action,
        "observed_action": observed_action,
        "routing_decision": "normal",
        "mismatch_class": "",
        "exact_label": exact_label,
    }


def _exp3186_payload(
    *,
    promotion_allowed: bool = True,
    row_action_overrides: Mapping[str, str] | None = None,
    include_manifest: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3186_fr11_controller_memory_promotion_pack_v1",
        "fr11_controller_memory_promotion_pack_v1_ready": True,
        "continuous_self_learning_task": True,
        "no_model_weight_update_claimed": True,
        "promotion_allowed": promotion_allowed,
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
        "honest_verdict": "complete: unit promotion pack",
    }
    if include_manifest:
        overrides = dict(
            row_action_overrides
            or {
                "resyn-3084-arith-003": "reject",
                "resyn-3084-smt-000": "reject",
            }
        )
        payload["promotion_manifest"] = {
            "update_id": "fr11-controller-memory-exp3172-row-exact-v1",
            "update_type": "controller_memory_exact_row_action_override",
            "promotion_decision": "promote_controller_memory_only"
            if promotion_allowed
            else "blocked",
            "source_update_artifact": mod.EXP3172_REL_PATH.as_posix(),
            "source_counterexample_artifact": mod.EXP3171_REL_PATH.as_posix(),
            "source_counterexample_families": [
                "arithmetic_code_assertions",
                "smt_constraints",
            ],
            "activation_predicate": {
                "mode": "exact_row_id_controller_memory_override",
                "allowed_row_ids": sorted(overrides),
                "row_action_overrides": overrides,
                "requires_exact_authority_consensus": True,
                "scope": "training_row_ids_only",
            },
            "rollback_predicate": {
                "rollback_if_any": [
                    "negative_control_regression_count > 0",
                    "cross_environment_drift_failure_count > 0",
                ]
            },
            "evidence": {
                "before_ledger_consistency_rate": 0.857143,
                "after_ledger_consistency_rate": 1.0,
                "heldout_consistency_rate": 1.0,
                "negative_control_regression_count": 0,
                "updated_row_count": len(overrides),
            },
        }
    return payload


def _exp3172_payload() -> dict[str, Any]:
    heldout = [
        _row(
            "graph_coloring-3128-0",
            "graph_coloring",
            "admitted_environment",
            source_artifact="results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json",
        ),
        _row(
            "interval_order-3128-0",
            "interval_order",
            "admitted_environment",
            source_artifact="results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json",
        ),
        _row(
            "resyn-3084-arith-001",
            "arithmetic_code_assertions",
            "historical_false_accept_family",
            expected_action="reject",
            observed_action="reject",
            source_artifact="results/experiment_3136_false_accept_root_cause_autopsy_v1.json",
            exact_label="INVALID",
        ),
    ]
    negative_controls = heldout[:2]
    return {
        "artifact": "experiment_3172_fr11_nonforgetting_self_learning_pilot_v2",
        "fr11_nonforgetting_self_learning_pilot_v2_ready": True,
        "continuous_self_learning_task": True,
        "model_weight_update_claimed": False,
        "promotion_allowed": True,
        "before_ledger_consistency_rate": 0.857143,
        "after_ledger_consistency_rate": 1.0,
        "heldout_consistency_rate": 1.0,
        "negative_control_regression_count": 0,
        "controller_memory_update": {
            "row_action_overrides": {
                "resyn-3084-arith-003": "reject",
                "resyn-3084-smt-000": "reject",
            },
            "model_weight_update": False,
        },
        "heldout_replay_rows": heldout,
        "negative_control_replay_rows": negative_controls,
        "inference_substrate": {
            "fresh_live_inference_calls": 0,
            "executes_live_model_inference": False,
            "model_weight_mutation": False,
        },
        "honest_verdict": "complete: unit nonforgetting pilot",
    }


def _exp3171_payload() -> dict[str, Any]:
    environment_rows = [
        _row(
            "graph_coloring-3128-0",
            "graph_coloring",
            "admitted_environment",
            source_artifact="results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json",
        ),
        _row(
            "interval_order-3128-0",
            "interval_order",
            "admitted_environment",
            source_artifact="results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json",
        ),
    ]
    variant_rows = [
        _row(
            "graph_coloring-3128-0-vera-h-2",
            "graph_coloring_vera_hardened",
            "hardened_variant",
            source_artifact="results/experiment_3142_fr11_vera_evoenv_hardening_v2.json",
        ),
        _row(
            "resyn-3084-arith-001",
            "arithmetic_code_assertions",
            "historical_false_accept_family",
            expected_action="reject",
            observed_action="reject",
            source_artifact="results/experiment_3136_false_accept_root_cause_autopsy_v1.json",
            exact_label="INVALID",
        ),
    ]
    return {
        "artifact": "experiment_3171_fr11_ledger_counterexample_isolation_v1",
        "fr11_ledger_counterexample_isolation_ready": True,
        "continuous_self_learning_task": True,
        "promotion_allowed": False,
        "no_model_weight_update_claimed": True,
        "environment_variant_split": {
            "training_update_rows": [
                _row(
                    "resyn-3084-arith-003",
                    "arithmetic_code_assertions",
                    "historical_false_accept_family",
                    expected_action="reject",
                    observed_action="accept",
                    exact_label="INVALID",
                )
            ],
            "held_out_replay_rows": environment_rows + variant_rows,
            "environment_rows": environment_rows,
            "variant_rows": variant_rows,
            "negative_control_rows": environment_rows,
        },
        "honest_verdict": "complete: unit counterexample isolation",
    }


def _write_sources(
    root: Path,
    *,
    promotion_allowed: bool = True,
    row_action_overrides: Mapping[str, str] | None = None,
    include_manifest: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller-only FR-11 memory\n", encoding="utf-8")
    (root / "research-program.md").write_text(
        "Tier 2: Constraint Memory / Trace2Skill\n",
        encoding="utf-8",
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3187\nSCENARIO-LEARN-3187\nSCENARIO-LEARN-3187-BLOCKED\n"
        "results/experiment_3187_fr11_cross_environment_drift_replay_v1.json\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3186_REL_PATH,
        _exp3186_payload(
            promotion_allowed=promotion_allowed,
            row_action_overrides=row_action_overrides,
            include_manifest=include_manifest,
        ),
    )
    _write_json(root, mod.EXP3172_REL_PATH, _exp3172_payload())
    _write_json(root, mod.EXP3171_REL_PATH, _exp3171_payload())


def test_req_learn_3187_spec_anchor_exists() -> None:
    """REQ-LEARN-3187: OpenSpec declares the drift replay artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3187" in spec
    assert "SCENARIO-LEARN-3187" in spec
    assert "SCENARIO-LEARN-3187-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "cross_environment_row_count" in spec
    assert "negative_control_regression_count" in spec
    assert "rollback_triggered" in spec


def test_req_learn_3187_selects_exact_cross_environment_rows(tmp_path: Path) -> None:
    """REQ-LEARN-3187-2/4: replay selects exact non-counterexample environments."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.0)

    selected = artifact["row_selection"]
    cross_rows = selected["cross_environment_rows"]
    before_after = artifact["before_after_consistency"]

    assert artifact["fr11_cross_environment_drift_replay_v1_ready"] is True
    assert artifact["heldout_row_count"] == 3
    assert artifact["cross_environment_row_count"] == 3
    assert {row["fixture_family"] for row in cross_rows} == {
        "graph_coloring",
        "graph_coloring_vera_hardened",
        "interval_order",
    }
    assert all(row["exact_authority"] for row in cross_rows)
    assert all(
        row["fixture_family"] not in selected["source_counterexample_families"]
        for row in cross_rows
    )
    assert before_after["source"]["before_rate"] == pytest.approx(0.857143)
    assert before_after["source"]["after_rate"] == 1.0
    assert before_after["source"]["lift"] == pytest.approx(0.142857)
    assert before_after["heldout"]["after_rate"] == 1.0
    assert before_after["cross_environment"]["after_rate"] == 1.0
    assert before_after["negative_control"]["regression_count"] == 0
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["drift_cases"] == []
    assert artifact["rollback_triggered"] is False
    assert artifact["promotion_allowed"] is True
    mod.validate_artifact(artifact)


def test_scenario_learn_3187_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3187: complete replay writes matrix-v29 source artifact."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=3.5,
        tests_run=["REQ-LEARN-3187 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["fr11_cross_environment_drift_replay_v1_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["replay_mode_only"] is True
    assert artifact["no_model_weight_update_claimed"] is True
    assert artifact["promotion_allowed"] is True
    assert artifact["rollback_triggered"] is False
    assert artifact["matrix_v29_input"]["ready_for_matrix_v29"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["REQ-LEARN-3187 focused"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["executes_live_model_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3187_blocked_without_promotable_manifest(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3187-BLOCKED: missing or unsafe manifest fails closed."""

    missing = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= set(missing)
    assert missing["fr11_cross_environment_drift_replay_v1_ready"] is False
    assert missing["replay_mode_only"] is True
    assert missing["no_model_weight_update_claimed"] is True
    assert missing["heldout_row_count"] == 0
    assert missing["cross_environment_row_count"] == 0
    assert missing["negative_control_regression_count"] == 0
    assert missing["rollback_triggered"] is True
    assert missing["promotion_allowed"] is False
    assert missing["blocked_reason"] == "exp3186_missing_or_not_ready"
    assert missing["replay_rows"] == []
    assert missing["honest_verdict"].startswith("blocked_precondition_failed:")
    mod.validate_artifact(missing)

    _write_sources(tmp_path, promotion_allowed=False)
    not_allowed = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert not_allowed["blocked_reason"] == "exp3186_promotion_not_allowed"
    assert not_allowed["promotion_allowed"] is False
    mod.validate_artifact(not_allowed)

    _write_sources(tmp_path, include_manifest=False)
    no_manifest = mod.build_artifact(tmp_path, started_s=3.0, now_s=4.0)
    assert no_manifest["blocked_reason"] == "exp3186_promotion_manifest_missing"
    assert no_manifest["promotion_allowed"] is False
    mod.validate_artifact(no_manifest)


def test_req_learn_3187_drift_and_negative_control_trigger_rollback(tmp_path: Path) -> None:
    """REQ-LEARN-3187-5/6: drift or negative-control regression blocks promotion."""

    _write_sources(
        tmp_path,
        row_action_overrides={
            "resyn-3084-arith-003": "reject",
            "graph_coloring-3128-0": "reject",
        },
    )
    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.5)

    assert artifact["fr11_cross_environment_drift_replay_v1_ready"] is True
    assert artifact["promotion_allowed"] is False
    assert artifact["rollback_triggered"] is True
    assert artifact["negative_control_regression_count"] == 1
    assert artifact["before_after_consistency"]["cross_environment"]["after_rate"] < 1.0
    assert artifact["before_after_consistency"]["negative_control"]["regression_count"] == 1
    assert {case["row_id"] for case in artifact["drift_cases"]} == {"graph_coloring-3128-0"}
    assert artifact["rollback_triggers"] == [
        "negative_control_regression",
        "cross_environment_drift_failure",
    ]
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_learn_3187_validation_and_helper_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3187-1/3/6: helpers fail closed and reject overclaims."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=7.0)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rows_from_payload({"rows": "bad"}, "rows") == []
    assert mod.rows_from_payload({"rows": [None, {"row_id": "ok"}]}, "rows") == [{"row_id": "ok"}]
    assert mod.row_overrides({"activation_predicate": {"row_action_overrides": "bad"}}) == {}
    assert mod.row_overrides({"activation_predicate": {"row_action_overrides": {"x": ""}}}) == {
        "x": "unknown"
    }
    assert (
        mod.has_exact_authority({"expected_action": "accept", "ledger_action": "reject"}) is False
    )
    assert mod.sha256_file(tmp_path / "absent.txt") is None
    assert mod.rate(1, 0) == 0.0

    bad_weight = _exp3186_payload()
    bad_weight["no_model_weight_update_claimed"] = False
    assert mod.source_blocker({"exp3186": bad_weight}) == "exp3186_model_weight_update_claimed"
    bad_substrate = _exp3186_payload()
    bad_substrate["inference_substrate"]["fresh_live_inference_calls"] = 1
    assert mod.source_blocker({"exp3186": bad_substrate}) == "exp3186_live_inference_claimed"
    bad_manifest = _exp3186_payload()
    bad_manifest["promotion_manifest"]["activation_predicate"] = []
    assert mod.source_blocker({"exp3186": bad_manifest}) == "exp3186_activation_predicate_missing"

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="replay_mode_only"):
        mod.validate_artifact(artifact | {"replay_mode_only": False})
    with pytest.raises(ValueError, match="no_model_weight_update_claimed"):
        mod.validate_artifact(artifact | {"no_model_weight_update_claimed": False})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": []})
    live_substrate = dict(artifact["inference_substrate"])
    live_substrate["fresh_live_inference_calls"] = 1
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(artifact | {"inference_substrate": live_substrate})
    mutation_substrate = dict(artifact["inference_substrate"])
    mutation_substrate["model_weight_training"] = True
    with pytest.raises(ValueError, match="mutation flags"):
        mod.validate_artifact(artifact | {"inference_substrate": mutation_substrate})
    with pytest.raises(ValueError, match="before_after_consistency"):
        mod.validate_artifact(artifact | {"before_after_consistency": []})
    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(
            artifact | {"promotion_allowed": True, "drift_cases": [{"row_id": "x"}]}
        )
    with pytest.raises(ValueError, match="rollback_triggered"):
        mod.validate_artifact(
            artifact
            | {
                "promotion_allowed": False,
                "rollback_triggered": False,
                "negative_control_regression_count": 1,
            }
        )
    with pytest.raises(ValueError, match="heldout_row_count"):
        mod.validate_artifact(artifact | {"heldout_row_count": 0})
    with pytest.raises(ValueError, match="cross_environment_row_count"):
        mod.validate_artifact(artifact | {"cross_environment_row_count": 0})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "not_done"})

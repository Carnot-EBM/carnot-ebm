"""Tests for Exp 3171 FR-11 ledger counterexample isolation.

Spec refs: REQ-LEARN-3171, SCENARIO-LEARN-3171,
SCENARIO-LEARN-3171-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_ledger_counterexample_isolation_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    target = root / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(
    row_id: str,
    family: str,
    category: str,
    *,
    expected: str,
    observed: str,
    consistent: bool,
    exact_label: str = "VALID",
    routing: str = "normal",
    mismatch_class: str = "",
    source: str = "unit-source.json",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_family": family,
        "panel_category": category,
        "source_artifact": source,
        "expected_action": expected,
        "ledger_action": expected,
        "observed_action": observed,
        "routing_decision": routing,
        "consistent": consistent,
        "mismatch_class": mismatch_class,
        "exact_label": exact_label,
        "soundness_errors": 0,
        "completeness_errors": 0,
    }


def _exp3156_payload() -> dict[str, Any]:
    rows = [
        _row(
            "env-safe",
            "modular_balance",
            "admitted_environment",
            expected="accept",
            observed="accept",
            consistent=True,
            source=mod.EXP3128_REL_PATH.as_posix(),
        ),
        _row(
            "variant-safe",
            "modular_balance_vera_equivalent",
            "equivalent_variant",
            expected="accept",
            observed="accept",
            consistent=True,
            source=mod.EXP3142_REL_PATH.as_posix(),
        ),
        _row(
            "arith-pass",
            "arithmetic_code_assertions",
            "historical_false_accept_family",
            expected="reject",
            observed="reject",
            consistent=True,
            exact_label="INVALID",
            routing="escalate",
            source=mod.EXP3136_REL_PATH.as_posix(),
        ),
        _row(
            "arith-fail",
            "arithmetic_code_assertions",
            "historical_false_accept_family",
            expected="reject",
            observed="accept",
            consistent=False,
            exact_label="INVALID",
            routing="escalate",
            mismatch_class="contradictory_memory",
            source=mod.EXP3136_REL_PATH.as_posix(),
        ),
        _row(
            "smt-fail",
            "smt_constraints",
            "historical_false_accept_family",
            expected="reject",
            observed="accept",
            consistent=False,
            exact_label="UNSAT",
            routing="escalate",
            mismatch_class="contradictory_memory",
            source=mod.EXP3136_REL_PATH.as_posix(),
        ),
    ]
    return {
        "artifact": "experiment_3156_fr11_ledger_consistency_closure_v1",
        "fr11_ledger_consistency_closure_v1_ready": True,
        "continuous_self_learning_targeted": True,
        "replay_panel_count": len(rows),
        "ledger_consistency_rate": 0.6,
        "ledger_consistent_count": 3,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "residual_mismatch_rows": [row for row in rows if not row["consistent"]],
        "replay_panel_rows": rows,
        "category_counts": {
            "admitted_environment": 1,
            "equivalent_variant": 1,
            "historical_false_accept_family": 3,
        },
        "promotion_recommendation": (
            "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
        ),
        "no_weight_update_claim": True,
    }


def _exp3157_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3157_fr11_attractor_residual_memory_audit_v1",
        "fr11_attractor_residual_memory_audit_v1_ready": True,
        "ledger_consistency_rate": 0.6,
        "risky_families": ["arithmetic_code_assertions", "smt_constraints"],
        "promotion_recommendation": (
            "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
        ),
        "no_weight_update_claim": True,
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller-only FR-11 memory\n", encoding="utf-8")
    (root / "research-program.md").write_text("continuous self-learning\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3171\nSCENARIO-LEARN-3171\nSCENARIO-LEARN-3171-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3156_REL_PATH, _exp3156_payload())
    _write_json(root, mod.EXP3157_REL_PATH, _exp3157_payload())
    _write_json(
        root,
        mod.EXP3128_REL_PATH,
        {"artifact": "exp3128", "fr11_evoenv_pilot_v1_ready": True},
    )
    _write_json(
        root,
        mod.EXP3129_REL_PATH,
        {"artifact": "exp3129", "fr11_constraint_memory_audit_v1_ready": True},
    )
    _write_json(
        root,
        mod.EXP3142_REL_PATH,
        {"artifact": "exp3142", "fr11_vera_evoenv_v2_ready": True},
    )
    _write_json(
        root,
        mod.EXP3143_REL_PATH,
        {"artifact": "exp3143", "fr11_experience_verifier_memory_v1_ready": True},
    )
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {"artifact": "exp3136", "false_accept_autopsy_v1_ready": True},
    )


def test_req_learn_3171_spec_anchor_exists() -> None:
    """REQ-LEARN-3171: OpenSpec declares the isolation artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3171" in spec
    assert "SCENARIO-LEARN-3171" in spec
    assert "SCENARIO-LEARN-3171-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "isolated_counterexample_families" in spec
    assert "environment_variant_split" in spec
    assert "negative_control_rows" in spec
    assert "no_model_weight_update_claimed" in spec


def test_req_learn_3171_isolates_failing_families_and_split() -> None:
    """REQ-LEARN-3171-1/2/3/4/5: row evidence drives the split."""

    isolation = mod.isolate_counterexamples(
        {"exp3156": _exp3156_payload(), "exp3157": _exp3157_payload()}
    )
    by_family = {item["fixture_family"]: item for item in isolation["isolated_counterexample_families"]}
    split = isolation["environment_variant_split"]
    training_ids = {row["row_id"] for row in split["training_update_rows"]}
    heldout_ids = {row["row_id"] for row in split["held_out_replay_rows"]}
    negative_ids = {row["row_id"] for row in isolation["negative_control_rows"]}
    mode_by_name = {row["mode"]: row for row in isolation["suspected_failure_modes"]}

    assert isolation["prior_ledger_consistency_rate"] == pytest.approx(0.6)
    assert isolation["replay_panel_count"] == 5
    assert isolation["ledger_consistent_count"] == 3
    assert sorted(by_family) == ["arithmetic_code_assertions", "smt_constraints"]
    assert by_family["arithmetic_code_assertions"]["failing_row_ids"] == ["arith-fail"]
    assert by_family["smt_constraints"]["failing_row_ids"] == ["smt-fail"]
    assert {row["observed_action"] for row in by_family["smt_constraints"]["rows"]} == {"accept"}
    assert {item["fixture_family"] for item in isolation["passing_families"]} >= {
        "arithmetic_code_assertions",
        "modular_balance",
        "modular_balance_vera_equivalent",
    }
    assert training_ids == {"arith-fail", "smt-fail"}
    assert "arith-pass" in heldout_ids
    assert {"env-safe", "variant-safe"} <= negative_ids
    assert training_ids.isdisjoint(heldout_ids)
    assert training_ids.isdisjoint(negative_ids)
    assert split["split_counts"] == {
        "training_update_rows": 2,
        "held_out_replay_rows": 3,
        "negative_control_rows": 2,
    }
    assert mode_by_name["controller_observed_decision_contradicts_exact_reject"]["applies"] is True
    assert mode_by_name["stale_memory"]["applies"] is False
    assert mode_by_name["environment_mismatch"]["applies"] is False
    assert mode_by_name["threshold_drift"]["applies"] is False
    assert mode_by_name["missing_exact_label"]["applies"] is False
    assert mode_by_name["aggregation_schema_mismatch"]["applies"] is False


def test_scenario_learn_3171_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3171: artifact is complete and promotion remains blocked."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["REQ-LEARN-3171 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_ledger_counterexample_isolation_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["prior_ledger_consistency_rate"] == pytest.approx(0.6)
    assert len(artifact["isolated_counterexample_families"]) == 2
    assert artifact["environment_variant_split"]["training_update_rows"]
    assert artifact["negative_control_rows"]
    assert artifact["promotion_allowed"] is False
    assert artifact["no_model_weight_update_claimed"] is True
    assert artifact["tests_run"] == ["REQ-LEARN-3171 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["mode"] == "controller_memory_replay"
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3171_blocked_without_sources(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3171-BLOCKED: missing source evidence fails closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_ledger_counterexample_isolation_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["prior_ledger_consistency_rate"] == 0.0
    assert artifact["isolated_counterexample_families"] == []
    assert artifact["passing_families"] == []
    assert artifact["environment_variant_split"]["training_update_rows"] == []
    assert artifact["negative_control_rows"] == []
    assert artifact["promotion_allowed"] is False
    assert artifact["no_model_weight_update_claimed"] is True
    assert artifact["blocked_reason"] == "exp3156_ledger_closure_missing_or_not_ready"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    mod.validate_artifact(artifact)


def test_req_learn_3171_validation_and_edge_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3171-5/6: validation rejects overclaims and malformed sources."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    mod.validate_artifact(artifact)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.replay_rows({"replay_panel_rows": "not-rows"}) == []
    assert mod.aggregation_schema_mismatch([], {"ledger_consistency_rate": 1.0}) is False
    assert mod.aggregation_schema_mismatch([{"consistent": True}], {}) is False
    assert (
        mod.precondition_blocker(
            {
                "exp3156": {"fr11_ledger_consistency_closure_v1_ready": True},
                "exp3157": {"fr11_attractor_residual_memory_audit_v1_ready": True},
                "exp3128": {"fr11_evoenv_pilot_v1_ready": True},
                "exp3129": {"fr11_constraint_memory_audit_v1_ready": True},
                "exp3136": {"false_accept_autopsy_v1_ready": True},
                "exp3142": {"fr11_vera_evoenv_v2_ready": True},
                "exp3143": {"fr11_experience_verifier_memory_v1_ready": True},
            }
        )
        == "exp3156_replay_panel_rows_missing"
    )
    assert mod.round_float(1 / 3) == pytest.approx(0.333333)
    assert mod.rate(1, 0) == 0.0
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.row_id_set("not-rows") == set()
    assert mod.honest_verdict(False, 0.0, [], False).startswith("blocked_precondition_failed")

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="no_model_weight_update_claimed"):
        mod.validate_artifact(artifact | {"no_model_weight_update_claimed": False})
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
    with pytest.raises(ValueError, match="prior_ledger_consistency_rate"):
        mod.validate_artifact(artifact | {"prior_ledger_consistency_rate": 1.5})
    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(
            artifact
            | {
                "prior_ledger_consistency_rate": 0.6,
                "promotion_allowed": True,
            }
        )
    with pytest.raises(ValueError, match="imperfect ledgers"):
        mod.validate_artifact(artifact | {"isolated_counterexample_families": []})
    with pytest.raises(ValueError, match="required source_artifacts"):
        mod.validate_artifact(
            artifact
            | {
                "source_artifacts": [
                    {"path": "missing", "required": True, "exists": False}
                ]
            }
        )
    with pytest.raises(ValueError, match="environment_variant_split"):
        mod.validate_artifact(artifact | {"environment_variant_split": "not-a-split"})
    with pytest.raises(ValueError, match="split overlap"):
        bad_split = artifact["environment_variant_split"] | {
            "held_out_replay_rows": artifact["environment_variant_split"]["training_update_rows"]
        }
        mod.validate_artifact(artifact | {"environment_variant_split": bad_split})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})

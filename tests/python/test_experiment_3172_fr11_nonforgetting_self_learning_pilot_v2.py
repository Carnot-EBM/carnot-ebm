"""Tests for Exp 3172 FR-11 nonforgetting self-learning pilot v2.

Spec refs: REQ-LEARN-3172, SCENARIO-LEARN-3172,
SCENARIO-LEARN-3172-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_nonforgetting_self_learning_pilot_v2 as mod


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
    split_role: str = "",
    exact_label: str = "VALID",
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
        "routing_decision": "escalate" if category == "historical_false_accept_family" else "normal",
        "mismatch_class": "" if expected == observed else "contradictory_memory",
        "exact_label": exact_label,
        "split_role": split_role,
    }


def _exp3171_payload() -> dict[str, Any]:
    training = [
        _row(
            "train-arith",
            "arithmetic_code_assertions",
            "historical_false_accept_family",
            expected="reject",
            observed="accept",
            split_role="controller_memory_update",
            exact_label="INVALID",
            source=mod.EXP3136_REL_PATH.as_posix(),
        ),
        _row(
            "train-smt",
            "smt_constraints",
            "historical_false_accept_family",
            expected="reject",
            observed="accept",
            split_role="controller_memory_update",
            exact_label="UNSAT",
            source=mod.EXP3136_REL_PATH.as_posix(),
        ),
    ]
    heldout = [
        _row(
            "hold-env",
            "modular_balance",
            "admitted_environment",
            expected="accept",
            observed="accept",
            split_role="held_out_nonforgetting_replay",
            source=mod.EXP3128_REL_PATH.as_posix(),
        ),
        _row(
            "hold-variant",
            "modular_balance_vera_equivalent",
            "equivalent_variant",
            expected="accept",
            observed="accept",
            split_role="held_out_nonforgetting_replay",
            source=mod.EXP3142_REL_PATH.as_posix(),
        ),
        _row(
            "hold-reject",
            "arithmetic_code_assertions",
            "historical_false_accept_family",
            expected="reject",
            observed="reject",
            split_role="held_out_nonforgetting_replay",
            exact_label="INVALID",
            source=mod.EXP3136_REL_PATH.as_posix(),
        ),
    ]
    negative_controls = [
        heldout[0] | {"control_role": "environment_variant_nonforgetting"},
        heldout[1] | {"control_role": "environment_variant_nonforgetting"},
    ]
    return {
        "artifact": "experiment_3171_fr11_ledger_counterexample_isolation_v1",
        "fr11_ledger_counterexample_isolation_ready": True,
        "continuous_self_learning_task": True,
        "prior_ledger_consistency_rate": 0.6,
        "isolated_counterexample_families": [
            {"fixture_family": "arithmetic_code_assertions", "failing_row_ids": ["train-arith"]},
            {"fixture_family": "smt_constraints", "failing_row_ids": ["train-smt"]},
        ],
        "passing_families": [{"fixture_family": "modular_balance", "passing_row_ids": ["hold-env"]}],
        "environment_variant_split": {
            "split_policy": "unit split",
            "training_update_rows": training,
            "held_out_replay_rows": heldout,
            "negative_control_rows": negative_controls,
            "split_counts": {
                "training_update_rows": len(training),
                "held_out_replay_rows": len(heldout),
                "negative_control_rows": len(negative_controls),
            },
        },
        "negative_control_rows": negative_controls,
        "promotion_allowed": False,
        "no_model_weight_update_claimed": True,
        "inference_substrate": {"fresh_live_inference_calls": 0},
        "honest_verdict": "complete: unit exp3171 split",
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller-only FR-11 memory\n", encoding="utf-8")
    (root / "research-program.md").write_text("continuous self-learning\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3172\nSCENARIO-LEARN-3172\nSCENARIO-LEARN-3172-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3156_REL_PATH, {"artifact": "exp3156", "ledger_consistency_rate": 0.857143})
    _write_json(root, mod.EXP3157_REL_PATH, {"artifact": "exp3157", "unsafe_skip_count": 0})
    _write_json(root, mod.EXP3171_REL_PATH, _exp3171_payload())


def test_req_learn_3172_spec_anchor_exists() -> None:
    """REQ-LEARN-3172: OpenSpec declares the pilot artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3172" in spec
    assert "SCENARIO-LEARN-3172" in spec
    assert "SCENARIO-LEARN-3172-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "controller_memory_update_applied" in spec
    assert "negative_control_regression_count" in spec
    assert "model_weight_update_claimed" in spec


def test_req_learn_3172_applies_bounded_controller_memory_update() -> None:
    """REQ-LEARN-3172-2/3/4/5: only training rows construct update memory."""

    pilot = mod.run_pilot(_exp3171_payload())
    update_ids = {row["row_id"] for row in pilot["controller_memory_update"]["updated_rows"]}

    assert update_ids == {"train-arith", "train-smt"}
    assert pilot["controller_memory_update"]["updated_row_count"] == 2
    assert pilot["before_ledger_consistency_rate"] == pytest.approx(3 / 5)
    assert pilot["after_ledger_consistency_rate"] == 1.0
    assert pilot["heldout_consistency_rate"] == 1.0
    assert pilot["negative_control_regression_count"] == 0
    assert pilot["nonforgetting_passed"] is True
    assert pilot["promotion_allowed"] is True
    assert pilot["training_replay_rows"][0]["controller_update_applied"] is True
    assert {row["controller_update_applied"] for row in pilot["heldout_replay_rows"]} == {False}
    assert {row["controller_update_applied"] for row in pilot["negative_control_replay_rows"]} == {False}


def test_scenario_learn_3172_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3172: complete artifact allows controller-only promotion."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=13.25,
        tests_run=["REQ-LEARN-3172 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_nonforgetting_self_learning_pilot_v2_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["controller_memory_update_applied"] is True
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["before_ledger_consistency_rate"] == pytest.approx(3 / 5)
    assert artifact["after_ledger_consistency_rate"] == 1.0
    assert artifact["heldout_consistency_rate"] == 1.0
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["nonforgetting_passed"] is True
    assert artifact["promotion_allowed"] is True
    assert artifact["promotion_recommendation"] == "promote_controller_memory_update_only"
    assert artifact["tests_run"] == ["REQ-LEARN-3172 focused"]
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["mode"] == "controller_memory_nonforgetting_replay"
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3172_blocked_without_split(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3172-BLOCKED: missing Exp 3171 split fails closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_nonforgetting_self_learning_pilot_v2_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["controller_memory_update_applied"] is False
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["before_ledger_consistency_rate"] == 0.0
    assert artifact["after_ledger_consistency_rate"] == 0.0
    assert artifact["heldout_consistency_rate"] == 0.0
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["nonforgetting_passed"] is False
    assert artifact["promotion_allowed"] is False
    assert artifact["blocked_reason"] == "exp3171_counterexample_split_missing_or_not_ready"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    mod.validate_artifact(artifact)


def test_req_learn_3172_validation_and_edge_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3172-1/5/6: validation rejects overclaims and bad sources."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.0)
    mod.validate_artifact(artifact)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rows_from_split({"training_update_rows": "not-rows"}, "training_update_rows") == []
    assert mod.rows_from_split({"training_update_rows": [None]}, "training_update_rows") == []
    assert mod.split_blocker({"fr11_ledger_counterexample_isolation_ready": True}) == (
        "exp3171_environment_variant_split_missing"
    )
    no_training = _exp3171_payload()
    no_training["environment_variant_split"]["training_update_rows"] = []
    assert mod.split_blocker(no_training) == "exp3171_training_update_rows_missing"
    no_heldout = _exp3171_payload()
    no_heldout["environment_variant_split"]["held_out_replay_rows"] = []
    assert mod.split_blocker(no_heldout) == "exp3171_held_out_replay_rows_missing"
    no_controls = _exp3171_payload()
    no_controls["environment_variant_split"]["negative_control_rows"] = []
    assert mod.split_blocker(no_controls) == "exp3171_negative_control_rows_missing"
    overlap = _exp3171_payload()
    overlap["environment_variant_split"]["held_out_replay_rows"].append(
        overlap["environment_variant_split"]["training_update_rows"][0]
    )
    assert mod.split_blocker(overlap) == "exp3171_training_rows_overlap_evaluation_controls"
    assert mod.rate(1, 0) == 0.0
    assert mod.round_float(1 / 3) == pytest.approx(0.333333)
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.negative_control_regressions(
        [
            {
                "row_id": "control-regressed",
                "consistent": True,
                "observed_action": "accept",
            }
        ],
        [
            {
                "row_id": "control-regressed",
                "consistent": False,
                "observed_action": "reject",
            }
        ],
    ) == [
        {
            "row_id": "control-regressed",
            "before_observed_action": "accept",
            "after_observed_action": "reject",
        }
    ]
    assert mod.promotion_recommendation(False, 1.0, 1.0, 0, True, False).startswith("block")
    assert mod.promotion_recommendation(True, 0.9, 1.0, 0, True, False).startswith("block")
    assert mod.promotion_recommendation(True, 1.0, 0.9, 0, True, False).startswith("block")
    assert mod.promotion_recommendation(True, 1.0, 1.0, 1, False, False).startswith("block")
    assert mod.promotion_recommendation(True, 1.0, 1.0, 0, True, True).startswith("block")
    assert mod.honest_verdict(False, False, 0.0, False).startswith("blocked_precondition_failed")

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="model_weight_update_claimed"):
        mod.validate_artifact(artifact | {"model_weight_update_claimed": True})
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
    with pytest.raises(ValueError, match="before_ledger_consistency_rate"):
        mod.validate_artifact(artifact | {"before_ledger_consistency_rate": 1.5})
    with pytest.raises(ValueError, match="controller_memory_update_applied"):
        mod.validate_artifact(artifact | {"controller_memory_update_applied": False})
    with pytest.raises(ValueError, match="promotion_allowed"):
        mod.validate_artifact(
            artifact
            | {
                "after_ledger_consistency_rate": 0.5,
                "promotion_allowed": True,
            }
        )
    with pytest.raises(ValueError, match="required source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})

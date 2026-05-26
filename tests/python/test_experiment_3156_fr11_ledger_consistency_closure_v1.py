"""Tests for Exp 3156 FR-11 ledger consistency closure replay.

Spec refs: REQ-LEARN-3156, SCENARIO-LEARN-3156,
SCENARIO-LEARN-3156-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_evoenv_verifiable_environment_synthesis_v1 as evo
from carnot.eval import fr11_ledger_consistency_closure_v1 as mod
from carnot.eval import fr11_vera_evoenv_hardening_v2 as vera


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    target = root / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_event(
    fixture_id: str,
    *,
    expected_action: str,
    ledger_action: str,
    live_decision: str,
    exact_label: str = "VALID",
    returned: bool = True,
    stored_consistent: bool | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "constraint_ledger",
            "fixture_id": fixture_id,
            "payload": {"ledger_action": ledger_action},
        },
        {
            "event_type": "exact_test_z3_result",
            "fixture_id": fixture_id,
            "payload": {
                "expected_action": expected_action,
                "exact_label": exact_label,
                "exact_authority_available": True,
            },
        },
        {
            "event_type": "candidate_final_answer",
            "fixture_id": fixture_id,
            "payload": {
                "expected_action": expected_action,
                "ledger_action": ledger_action,
                "live_decision": live_decision,
                "has_returned_answer": returned,
                # REQ-LEARN-3156-3: this value is deliberately wrong on
                # inconsistent fixtures; replay must recompute from actions.
                "final_answer_consistent_with_ledger": stored_consistent,
            },
        },
    ]


def _verifier_row(
    row_id: str,
    family: str,
    *,
    expected_action: str,
    live_decision: str,
    exact_label: str,
    answer_format: str = "validity_token",
    difficulty: list[str] | None = None,
    mechanism: str = "no_failure",
    routing_decision: str = "normal",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_family": family,
        "difficulty_buckets": difficulty or ["easy"],
        "answer_extraction_format": answer_format,
        "failure_mechanism_from_exp3124": mechanism,
        "expected_action": expected_action,
        "live_decision": live_decision,
        "exact_label": exact_label,
        "routing_decision": routing_decision,
        "monitor_events": _candidate_event(
            row_id,
            expected_action=expected_action,
            ledger_action=expected_action,
            live_decision=live_decision,
            exact_label=exact_label,
            returned=live_decision != "missing",
            stored_consistent=True,
        ),
    }


def _exp3128_payload() -> dict[str, Any]:
    summary = evo.evaluate_admission(evo.sample_candidate_environments(seed=3128))
    return {
        "artifact": "experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1",
        "fr11_evoenv_pilot_v1_ready": True,
        "admitted_environment_count": summary.admitted_count,
        "admitted_environments": evo.admitted_environment_rows(summary),
        "admission_records": [record.to_dict() for record in summary.records],
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
    }


def _variant_records(exp3128: dict[str, Any]) -> list[dict[str, Any]]:
    admitted = vera.load_admitted_environments(exp3128)
    return [record.to_dict() for record in vera.generate_and_validate_variants(admitted).records]


def _exp3136_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
        "false_accept_autopsy_v1_ready": True,
        "false_accept_row_ids": ["unsafe-invalid"],
        "false_accept_mechanism_counts": {"contradiction miss": 1},
        "source_false_accept_count": 1,
        "source_live_row_count": 3,
        "verifier_rows": [
            _verifier_row(
                "unsafe-valid",
                "unsafe_math",
                expected_action="accept",
                live_decision="accept",
                exact_label="VALID",
            ),
            _verifier_row(
                "unsafe-invalid",
                "unsafe_math",
                expected_action="reject",
                live_decision="accept",
                exact_label="INVALID",
                mechanism="contradiction",
            ),
            _verifier_row(
                "safe-repair",
                "repairable_json",
                expected_action="reject",
                live_decision="reject",
                exact_label="REPAIRABLE",
                answer_format="repairability_token",
                difficulty=["hard"],
            ),
        ],
    }


def _exp3126_payload() -> dict[str, Any]:
    events = _candidate_event(
        "residual-monitor",
        expected_action="reject",
        ledger_action="reject",
        live_decision="accept",
        exact_label="INVALID",
        stored_consistent=True,
    )
    return {
        "artifact": "experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1",
        "fragment_time_monitor_v1_ready": True,
        "monitor_events": events,
        "ledger_consistency_rate": 1.0,
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake verifier claims\n", encoding="utf-8")
    (root / "research-program.md").write_text("continuous self-learning\n", encoding="utf-8")
    (root / "research-references.md").write_text("FR-11 references\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3156\nSCENARIO-LEARN-3156\nSCENARIO-LEARN-3156-BLOCKED\n",
        encoding="utf-8",
    )

    exp3128 = _exp3128_payload()
    exp3142 = {
        "artifact": "experiment_3142_fr11_vera_evoenv_hardening_v2",
        "fr11_vera_evoenv_v2_ready": True,
        "variant_records": _variant_records(exp3128),
        "ledger_consistency_rate": 0.75,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
    }
    exp3143 = {
        "artifact": "experiment_3143_fr11_experience_driven_verifier_memory_v1",
        "fr11_experience_verifier_memory_v1_ready": True,
        "routing_rows": [
            {
                "row_id": "unsafe-valid",
                "routing_decision": "escalate",
                "ledger_consistent": False,
            },
            {
                "row_id": "unsafe-invalid",
                "routing_decision": "escalate",
                "ledger_consistent": True,
            },
        ],
        "ledger_consistency_rate": 0.75,
        "no_weight_update_claim": True,
    }
    _write_json(root, mod.EXP3128_REL_PATH, exp3128)
    _write_json(
        root,
        mod.EXP3129_REL_PATH,
        {
            "artifact": "experiment_3129_fr11_constraint_memory_retention_drift_audit_v1",
            "fr11_constraint_memory_audit_v1_ready": True,
            "ledger_consistency_rate": 0.5,
            "soundness_errors": 0,
            "completeness_errors": 0,
            "no_weight_update_claim": True,
        },
    )
    _write_json(root, mod.EXP3126_REL_PATH, _exp3126_payload())
    _write_json(root, mod.EXP3136_REL_PATH, _exp3136_payload())
    _write_json(root, mod.EXP3142_REL_PATH, exp3142)
    _write_json(root, mod.EXP3143_REL_PATH, exp3143)


def test_req_learn_3156_spec_anchor_exists() -> None:
    """REQ-LEARN-3156: OpenSpec declares the closure artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3156" in spec
    assert "SCENARIO-LEARN-3156" in spec
    assert "SCENARIO-LEARN-3156-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_ledger_consistency_closure_v1_ready" in spec
    assert "replay_panel_count" in spec
    assert "residual_mismatch_rows" in spec
    assert "no_weight_update_claim" in spec


def test_req_learn_3156_replay_panel_ignores_tautological_fields(tmp_path: Path) -> None:
    """REQ-LEARN-3156-1/2/3/4: exact replay builds the measured denominator."""

    _write_sources(tmp_path)
    sources = mod.load_sources(tmp_path)
    panel = mod.build_replay_panel(sources)
    by_id = {row["row_id"]: row for row in panel["rows"]}

    assert panel["replay_panel_count"] == 12
    assert panel["ledger_consistency_rate"] == pytest.approx(10 / 12)
    assert panel["soundness_errors"] == 0
    assert panel["completeness_errors"] == 0
    assert panel["category_counts"] == {
        "admitted_environment": 3,
        "equivalent_variant": 3,
        "hardened_variant": 3,
        "historical_false_accept_family": 2,
        "residual_monitor_inconsistent": 1,
    }
    assert by_id["unsafe-invalid"]["consistent"] is False
    assert by_id["unsafe-invalid"]["mismatch_class"] == "contradictory_memory"
    assert by_id["unsafe-invalid"]["tautological_consistency_fields_ignored"] is True
    assert by_id["residual-monitor"]["consistent"] is False
    assert by_id["residual-monitor"]["mismatch_class"] == "contradictory_memory"
    assert all(row["consistent"] for row in panel["rows"] if row["row_id"].endswith("vera-e-0"))


def test_scenario_learn_3156_writes_complete_artifact_preserving_gap(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3156: non-perfect replay blocks promotion without weight claims."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-LEARN-3156 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_ledger_consistency_closure_v1_ready"] is True
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["replay_panel_count"] == 12
    assert artifact["ledger_consistency_rate"] == pytest.approx(10 / 12)
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert len(artifact["residual_mismatch_rows"]) == 2
    assert {row["mismatch_class"] for row in artifact["residual_mismatch_rows"]} == {
        "contradictory_memory"
    }
    assert artifact["promotion_recommendation"] == (
        "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
    )
    assert artifact["no_weight_update_claim"] is True
    assert artifact["methodology_complete"] is True
    assert artifact["tests_run"] == ["REQ-LEARN-3156 focused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["mode"] == "solver_only_memory_ledger_replay"
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["model_weight_training"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3156_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3156-BLOCKED: missing closure sources fail closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_ledger_consistency_closure_v1_ready"] is False
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["replay_panel_count"] == 0
    assert artifact["ledger_consistency_rate"] == 0.0
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["residual_mismatch_rows"] == []
    assert artifact["promotion_recommendation"] == (
        "block_fr11_ledger_consistency_closure_missing_source_evidence"
    )
    assert artifact["no_weight_update_claim"] is True
    assert artifact["methodology_complete"] is False
    assert artifact["blocked_reason"] == "exp3128_evoenv_missing_or_not_ready"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    assert mod.precondition_blocker(
        {
            "fr11_evoenv_pilot_v1_ready": True,
            "no_weight_update_claim": True,
        },
        {},
        {},
        {},
        {},
    ) == "exp3129_constraint_memory_missing_or_not_ready"
    mod.validate_artifact(artifact)


def test_req_learn_3156_classification_and_validation_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3156-3/4/5: mismatch classes and overclaim guards are enforced."""

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
    assert mod.admitted_environment_rows({"admitted_environments": [None]}) == []
    assert mod.variant_rows({"variant_records": [None, {}]}) == []
    assert mod.residual_monitor_rows(
        _exp3126_payload(),
        represented_ids={"residual-monitor"},
    ) == []
    fallback_actions = mod.actions_from_row(
        {
            "expected_action": "accept",
            "live_decision": "accept",
            "exact_label": "VALID",
            "monitor_events": [],
        }
    )
    assert fallback_actions["expected_action"] == "accept"
    assert fallback_actions["observed_action"] == "accept"
    assert fallback_actions["exact_label"] == "VALID"
    assert mod.actions_from_events("not-events")["observed_action"] == "missing"
    assert mod.actions_from_events([None])["expected_action"] == "unknown"
    assert mod.actions_from_events(
        [
            {
                "event_type": "exact_test_z3_result",
                "payload": {"expected_action": "reject", "exact_label": "INVALID"},
            }
        ]
    )["ledger_action"] == "reject"
    assert mod.grouped_monitor_events("not-events") == {}
    assert mod.rate(1, 0) == 0.0
    assert mod.round_float(1 / 3) == pytest.approx(0.333333)
    assert mod.promotion_recommendation(True, 1.0, 0, 0) == (
        "promote_controller_environment_memory_only"
    )
    assert mod.promotion_recommendation(True, 0.9, 0, 0).startswith("block_fr11")
    assert mod.promotion_recommendation(True, 1.0, 1, 0).startswith("block_fr11")
    assert mod.promotion_recommendation(False, 1.0, 0, 0).startswith("block_fr11")
    assert mod.honest_verdict(True, 1.0, "promote_controller_environment_memory_only").startswith(
        "complete:"
    )

    assert mod.classify_mismatch(
        {
            "panel_category": "historical_false_accept_family",
            "observed_action": "missing",
            "expected_action": "accept",
            "ledger_action": "accept",
            "routing_decision": "normal",
        }
    ) == "missing_label"
    assert mod.classify_mismatch(
        {
            "panel_category": "historical_false_accept_family",
            "observed_action": "accept",
            "expected_action": "reject",
            "ledger_action": "reject",
            "routing_decision": "suppress",
        }
    ) == "stale_memory"
    assert mod.classify_mismatch(
        {
            "panel_category": "equivalent_variant",
            "observed_action": "reject",
            "expected_action": "accept",
            "ledger_action": "accept",
            "routing_decision": "normal",
        }
    ) == "variant_generation_error"
    assert mod.classify_mismatch(
        {
            "panel_category": "residual_monitor_inconsistent",
            "observed_action": "accept",
            "expected_action": "accept",
            "ledger_action": "reject",
            "routing_decision": "normal",
        }
    ) == "monitor_replay_error"
    assert mod.classify_mismatch(
        {
            "panel_category": "historical_false_accept_family",
            "observed_action": "accept",
            "expected_action": "accept",
            "ledger_action": "accept",
            "routing_decision": "normal",
        }
    ) == "monitor_replay_error"
    assert mod.precondition_blocker(
        {"fr11_evoenv_pilot_v1_ready": True},
        {"fr11_constraint_memory_audit_v1_ready": True},
        {},
        {},
        {},
    ) == "exp3136_false_accept_autopsy_missing_or_not_ready"
    assert mod.precondition_blocker(
        {"fr11_evoenv_pilot_v1_ready": True},
        {"fr11_constraint_memory_audit_v1_ready": True},
        {"false_accept_autopsy_v1_ready": True},
        {},
        {},
    ) == "exp3142_vera_evoenv_missing_or_not_ready"
    assert mod.precondition_blocker(
        {"fr11_evoenv_pilot_v1_ready": True},
        {"fr11_constraint_memory_audit_v1_ready": True},
        {"false_accept_autopsy_v1_ready": True},
        {"fr11_vera_evoenv_v2_ready": True},
        {},
    ) == "exp3143_experience_memory_missing_or_not_ready"
    assert mod.precondition_blocker(
        {"fr11_evoenv_pilot_v1_ready": True},
        {"fr11_constraint_memory_audit_v1_ready": True},
        {"false_accept_autopsy_v1_ready": True},
        {"fr11_vera_evoenv_v2_ready": True},
        {"fr11_experience_verifier_memory_v1_ready": True},
        {},
    ) == "exp3126_fragment_time_monitor_missing_or_not_ready"
    assert mod.honest_verdict(False, 0.0, "block").startswith("blocked_precondition_failed")
    assert mod.sha256_file(tmp_path / "nope.txt") is None

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
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
    with pytest.raises(ValueError, match="replay_panel_count"):
        mod.validate_artifact(artifact | {"replay_panel_count": 0})
    with pytest.raises(ValueError, match="promotion_recommendation"):
        mod.validate_artifact(
            artifact
            | {
                "ledger_consistency_rate": 0.5,
                "promotion_recommendation": "promote_controller_environment_memory_only",
            }
        )
    with pytest.raises(ValueError, match="source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})

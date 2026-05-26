"""Tests for Exp 3157 FR-11 attractor residual memory audit.

Spec refs: REQ-LEARN-3157, SCENARIO-LEARN-3157,
SCENARIO-LEARN-3157-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_attractor_residual_memory_audit_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    target = root / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _closure_row(
    row_id: str,
    family: str,
    *,
    expected: str,
    observed: str,
    consistent: bool,
    routing: str = "normal",
    mismatch_class: str = "",
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "panel_category": "historical_false_accept_family"
        if family.startswith("risky")
        else "equivalent_variant",
        "source_artifact": "unit-fixture.json",
        "fixture_family": family,
        "expected_action": expected,
        "ledger_action": expected,
        "observed_action": observed,
        "routing_decision": routing,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "consistent": consistent,
        "mismatch_class": mismatch_class,
    }


def _exp3156_payload(*, ledger_rate: float = 0.75) -> dict[str, Any]:
    return {
        "artifact": "experiment_3156_fr11_ledger_consistency_closure_v1",
        "fr11_ledger_consistency_closure_v1_ready": True,
        "continuous_self_learning_targeted": True,
        "replay_panel_count": 3,
        "ledger_consistency_rate": ledger_rate,
        "no_weight_update_claim": True,
        "replay_panel_rows": [
            _closure_row(
                "safe-variant",
                "safe_variant",
                expected="accept",
                observed="accept",
                consistent=True,
            ),
            _closure_row(
                "risky-valid",
                "risky_math",
                expected="accept",
                observed="accept",
                consistent=True,
                routing="escalate",
            ),
            _closure_row(
                "risky-invalid",
                "risky_math",
                expected="reject",
                observed="accept",
                consistent=False,
                routing="escalate",
                mismatch_class="contradictory_memory",
            ),
        ],
    }


def _exp3143_payload(*, unsafe_suppression: bool = False) -> dict[str, Any]:
    risky_route = "suppress" if unsafe_suppression else "escalate"
    return {
        "artifact": "experiment_3143_fr11_experience_driven_verifier_memory_v1",
        "fr11_experience_verifier_memory_v1_ready": True,
        "no_weight_update_claim": True,
        "routing_rows": [
            {"row_id": "safe-variant", "routing_decision": "suppress"},
            {"row_id": "risky-valid", "routing_decision": risky_route},
            {"row_id": "risky-invalid", "routing_decision": "escalate"},
        ],
    }


def _exp3136_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
        "false_accept_autopsy_v1_ready": True,
        "false_accept_row_ids": ["risky-invalid"],
        "verifier_rows": [
            {
                "row_id": "risky-valid",
                "fixture_family": "risky_math",
                "expected_action": "accept",
                "live_decision": "accept",
            },
            {
                "row_id": "risky-invalid",
                "fixture_family": "risky_math",
                "expected_action": "reject",
                "live_decision": "accept",
            },
        ],
    }


def _write_sources(root: Path, *, unsafe_suppression: bool = False) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller diagnostics only\n", encoding="utf-8")
    (root / "research-program.md").write_text("FR-11 continuous self-learning\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "attractor residual references\n", encoding="utf-8"
    )
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3157\nSCENARIO-LEARN-3157\nSCENARIO-LEARN-3157-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3156_REL_PATH, _exp3156_payload())
    _write_json(root, mod.EXP3143_REL_PATH, _exp3143_payload(unsafe_suppression=unsafe_suppression))
    _write_json(root, mod.EXP3136_REL_PATH, _exp3136_payload())


def test_req_learn_3157_spec_anchor_exists() -> None:
    """REQ-LEARN-3157: OpenSpec declares residual audit fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3157" in spec
    assert "SCENARIO-LEARN-3157" in spec
    assert "SCENARIO-LEARN-3157-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "residual_signal_definitions" in spec
    assert "risky_family_escalation_rate" in spec
    assert "redundant_check_suppression_rate" in spec
    assert "unsafe_skip_count" in spec


def test_req_learn_3157_residual_signals_route_safe_and_risky_rows() -> None:
    """REQ-LEARN-3157-1/2/3/4/5: residual signals drive safe routing."""

    sources = {
        "exp3156": _exp3156_payload(),
        "exp3143": _exp3143_payload(),
        "exp3136": _exp3136_payload(),
    }
    audit = mod.audit_residual_memory(sources)
    by_id = {row["row_id"]: row for row in audit["residual_memory_rows"]}

    assert {item["signal_id"] for item in mod.residual_signal_definitions()} == {
        "repeated_mismatch_count",
        "stable_verdict_convergence",
        "contradiction_core_stability",
        "memory_routing_entropy",
    }
    assert audit["replay_panel_count"] == 3
    assert audit["risky_family_escalation_rate"] == pytest.approx(1.0)
    assert audit["redundant_check_suppression_rate"] == pytest.approx(1 / 3)
    assert audit["unsafe_skip_count"] == 0
    assert by_id["safe-variant"]["residual_policy_route"] == "suppress"
    assert by_id["safe-variant"]["stable_verdict_convergence"] is True
    assert by_id["safe-variant"]["repeated_mismatch_count"] == 0
    assert by_id["risky-invalid"]["residual_policy_route"] == "escalate"
    assert by_id["risky-invalid"]["stable_verdict_convergence"] is False
    assert by_id["risky-invalid"]["contradiction_core_stability"] is False
    assert by_id["risky-invalid"]["repeated_mismatch_count"] == 1


def test_req_learn_3157_unsafe_suppression_blocks_even_with_perfect_ledger() -> None:
    """REQ-LEARN-3157-5: unsafe skips block promotion independently."""

    sources = {
        "exp3156": _exp3156_payload(ledger_rate=1.0),
        "exp3143": _exp3143_payload(unsafe_suppression=True),
        "exp3136": _exp3136_payload(),
    }
    audit = mod.audit_residual_memory(sources)
    recommendation = mod.promotion_recommendation(
        ready=True,
        ledger_consistency_rate=1.0,
        unsafe_skip_count=audit["unsafe_skip_count"],
    )

    assert audit["unsafe_skip_count"] == 1
    assert recommendation == "block_fr11_residual_memory_unsafe_skip_detected"


def test_scenario_learn_3157_writes_complete_blocked_promotion_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3157: imperfect ledger blocks promotion without weight claims."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=20.0,
        now_s=21.5,
        tests_run=["REQ-LEARN-3157 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_attractor_residual_memory_audit_v1_ready"] is True
    assert artifact["continuous_self_learning_targeted"] is True
    assert len(artifact["residual_signal_definitions"]) == 4
    assert artifact["replay_panel_count"] == 3
    assert artifact["risky_family_escalation_rate"] == pytest.approx(1.0)
    assert artifact["redundant_check_suppression_rate"] == pytest.approx(1 / 3)
    assert artifact["unsafe_skip_count"] == 0
    assert artifact["promotion_recommendation"] == (
        "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
    )
    assert artifact["no_weight_update_claim"] is True
    assert artifact["tests_run"] == ["REQ-LEARN-3157 focused"]
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["mode"] == "exact_replay_residual_controller_memory"
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["model_weight_training"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3157_blocked_without_sources(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3157-BLOCKED: missing source artifacts fail closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_attractor_residual_memory_audit_v1_ready"] is False
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["residual_signal_definitions"] == mod.residual_signal_definitions()
    assert artifact["replay_panel_count"] == 0
    assert artifact["risky_family_escalation_rate"] == 0.0
    assert artifact["redundant_check_suppression_rate"] == 0.0
    assert artifact["unsafe_skip_count"] == 0
    assert (
        artifact["promotion_recommendation"] == "block_fr11_residual_memory_missing_source_evidence"
    )
    assert artifact["no_weight_update_claim"] is True
    assert artifact["blocked_reason"] == "exp3156_ledger_closure_missing_or_not_ready"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    assert (
        mod.precondition_blocker(
            {"fr11_ledger_consistency_closure_v1_ready": True},
            {},
            {},
        )
        == "exp3143_experience_memory_missing_or_not_ready"
    )
    assert (
        mod.precondition_blocker(
            {"fr11_ledger_consistency_closure_v1_ready": True},
            {"fr11_experience_verifier_memory_v1_ready": True},
            {},
        )
        == "exp3136_false_accept_autopsy_missing_or_not_ready"
    )
    mod.validate_artifact(artifact)


def test_req_learn_3157_validation_and_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-3157-6: validation rejects overclaims and bad metrics."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=5.0)
    mod.validate_artifact(artifact)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.row_lookup(
        {"routing_rows": ["bad", {"row_id": "x", "routing_decision": "suppress"}]}
    ) == {"x": {"row_id": "x", "routing_decision": "suppress"}}
    assert mod.routing_entropy([]) == 0.0
    assert mod.routing_entropy(["suppress", "suppress"]) == 0.0
    late_false_accept = {
        "false_accept_row_ids": ["late-false-accept"],
        "verifier_rows": [
            "bad",
            {"row_id": "late-false-accept", "fixture_family": "late_family"},
        ],
    }
    assert mod.risky_family_set([], late_false_accept) == {"late_family"}
    assert mod.mismatch_count_by_family([], late_false_accept)["late_family"] == 1
    assert (
        mod.residual_policy_route(
            source_route="escalate",
            risky_family=False,
            repeated_mismatch_count=0,
            stable_verdict=True,
            contradiction_core_stable=True,
        )
        == "escalate"
    )
    assert (
        mod.residual_policy_route(
            source_route="normal",
            risky_family=False,
            repeated_mismatch_count=0,
            stable_verdict=True,
            contradiction_core_stable=True,
        )
        == "normal"
    )
    assert mod.rate(1, 0) == 0.0
    assert mod.round_float(1 / 3) == pytest.approx(0.333333)
    assert mod.normalize_action("") == "unknown"
    assert mod.sha256_file(tmp_path / "nope.txt") is None
    assert mod.promotion_recommendation(True, 1.0, 0) == (
        "promote_controller_residual_memory_diagnostics_only"
    )
    assert mod.promotion_recommendation(False, 1.0, 0).startswith("block_fr11")
    assert mod.honest_verdict(False, "block").startswith("blocked_precondition_failed")

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
    with pytest.raises(ValueError, match="risky_family_escalation_rate"):
        mod.validate_artifact(artifact | {"risky_family_escalation_rate": 1.2})
    with pytest.raises(ValueError, match="replay_panel_count"):
        mod.validate_artifact(artifact | {"replay_panel_count": 0})
    with pytest.raises(ValueError, match="promotion_recommendation"):
        mod.validate_artifact(
            artifact
            | {
                "unsafe_skip_count": 1,
                "promotion_recommendation": "promote_controller_residual_memory_diagnostics_only",
            }
        )
    with pytest.raises(ValueError, match="promotion_recommendation"):
        mod.validate_artifact(
            artifact
            | {
                "promotion_recommendation": "promote_controller_residual_memory_diagnostics_only",
            }
        )
    with pytest.raises(ValueError, match="source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="residual_signal_definitions"):
        mod.validate_artifact(artifact | {"residual_signal_definitions": []})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})

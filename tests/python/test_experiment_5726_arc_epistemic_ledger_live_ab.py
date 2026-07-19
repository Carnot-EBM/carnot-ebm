"""Tests for Exp5726 ARC epistemic-ledger matched live A/B.

Spec refs: REQ-ARC-WMTE-5726,
SCENARIO-ARC-WMTE-5726-GATED-MATCHED-AB,
SCENARIO-ARC-WMTE-5726-CONTROLS-AND-OVERHEAD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5726_arc_epistemic_ledger_live_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5726_spec_declares_live_ab_contract() -> None:
    """REQ-ARC-WMTE-5726: OpenSpec anchors the matched ledger A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5726") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        "SCENARIO-ARC-WMTE-5726-GATED-MATCHED-AB",
        "SCENARIO-ARC-WMTE-5726-CONTROLS-AND-OVERHEAD",
        mod.RESULT_RELATIVE_PATH,
        "control arm SHALL be the unchanged submitted full stack",
        "ledger/commitment mechanism explicitly disabled",
        "arc_epistemic_live_ab_ready_score",
        "Null is terminal",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def _ok_preconditions(root=mod.REPO_ROOT):
    return {
        "registry_exists": True,
        "e3_policy_importable": True,
        "offline_arcade_importable": True,
        "exp5725_ready": True,
        "exp5725_live_reachable": True,
        "exp5725_zero_leakage": True,
        "ok": True,
    }


def _row(
    game: str,
    *,
    arm: str,
    levels: int,
    actions: int,
    first_action: int,
    order_changes: int = 0,
    commitments: int = 0,
    unsafe: int = 0,
    verification_calls: int | None = None,
    noops: int = 0,
) -> dict:
    return {
        "game": game,
        "arm": arm,
        "seed": mod.RANDOM_SEEDS[0],
        "start_level": 0,
        "reached_level": levels,
        "levels": levels,
        "actions": actions,
        "actions_to_first_levelup": actions if levels else None,
        "frontier_expansions": actions // 4,
        "legal_proposal_count": actions * 2,
        "first_decision": [first_action, None],
        "ledger_operation_counts": {
            "observe_state": commitments + order_changes,
            "observe_transition": commitments,
            "rank_candidates": order_changes,
            "commitment_checks": verification_calls or actions * 2,
        },
        "hypothesis_revision_count": commitments * 2,
        "open_question_resolution_count": commitments,
        "action_order_change_count": order_changes,
        "commitment_count": commitments,
        "unsafe_commit_count": unsafe,
        "invalid_actions": 0,
        "noop_count": noops,
        "verification_calls": verification_calls if verification_calls is not None else actions * 2,
        "redundant_verification_count": noops,
        "evidence_lost_count": 0,
        "cpu_time_s": 0.10 + (0.01 if arm == mod.TREATMENT_ARM else 0.0),
        "ledger_memory_bytes": 640 if arm == mod.TREATMENT_ARM else 0,
        "failed_reason": None,
    }


def _fake_successful_pairs(saved_actions: int = 30) -> dict:
    pairs = []
    for index, game in enumerate(("tu93", "sp80", "lp85")):
        pairs.append(
            {
                "game": game,
                "seed": mod.RANDOM_SEEDS[0],
                mod.CONTROL_ARM: _row(
                    game,
                    arm=mod.CONTROL_ARM,
                    levels=1,
                    actions=100,
                    first_action=1,
                    noops=20,
                    verification_calls=220,
                ),
                mod.TREATMENT_ARM: _row(
                    game,
                    arm=mod.TREATMENT_ARM,
                    levels=1,
                    actions=100 - saved_actions,
                    first_action=2 if index == 0 else 1,
                    order_changes=1,
                    commitments=2,
                    noops=10,
                    verification_calls=160,
                ),
                "failed_reason": None,
            }
        )
    return {"pairs": pairs, "duration_s": 1.5}


def test_req_arc_wmte_5726_blocked_gate_stops_before_pairs(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5726: failed Exp5725 gates block before matched pairs run."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {"exp5725_ready": False, "ok": False},
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_matched_pairs must not run after a failed gate")

    monkeypatch.setattr(mod, "run_matched_pairs", _fail_if_called)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "blocked: exp5725_ready"
    assert artifact["successful_pair_count"] == 0
    assert artifact["failed_pair_reasons"] == [{"reason": "exp5725_ready"}]
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["registry_updated"] is False
    assert artifact["new_levels_claimed"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_scenario_arc_wmte_5726_ready_score_requires_interval_gain(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5726-GATED-MATCHED-AB: promotion needs paired utility."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_matched_pairs", lambda *_a, **_kw: _fake_successful_pairs())

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["successful_pair_count"] == 3
    assert artifact["levels_reproduced_by_arm"] == {mod.CONTROL_ARM: 3, mod.TREATMENT_ARM: 3}
    assert artifact["environment_actions_by_arm"][mod.TREATMENT_ARM] < artifact[
        "environment_actions_by_arm"
    ][mod.CONTROL_ARM]
    assert artifact["paired_intervals"]["actions_saved_per_reproduced_level"]["ci95_low"] > 0
    assert artifact["known_level_regression_count"] == 0
    assert artifact["action_order_change_count"][mod.TREATMENT_ARM] == 3
    assert artifact["first_decision_divergence"]["count"] == 1
    assert artifact["verification_calls_by_arm"][mod.TREATMENT_ARM] < artifact[
        "verification_calls_by_arm"
    ][mod.CONTROL_ARM]
    assert artifact["redundant_verification_delta"]["verification_calls_avoided"] > 0
    assert artifact["ledger_cpu_overhead"]["over_cap"] is False
    assert artifact["ledger_memory_overhead"]["over_cap"] is False
    assert artifact["unsafe_commit_count"] == 0
    assert artifact["negative_controls_passed"] is True
    assert artifact["arc_epistemic_live_ab_ready_score"] == 1.0
    mod.validate_artifact(artifact)


def test_req_arc_wmte_5726_level_regression_blocks_promotion(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5726: retained-level regression blocks the ready score."""

    data = _fake_successful_pairs()
    data["pairs"][0][mod.TREATMENT_ARM] = {
        **data["pairs"][0][mod.TREATMENT_ARM],
        "levels": 0,
        "reached_level": 0,
        "actions": 100,
        "actions_to_first_levelup": None,
    }
    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_matched_pairs", lambda *_a, **_kw: data)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["known_level_regression_count"] == 1
    assert artifact["arc_epistemic_live_ab_ready_score"] == 0.0
    assert artifact["honest_verdict"] == "complete: epistemic_ledger_live_ab_null_with_regression"


def test_scenario_arc_wmte_5726_controls_exercise_and_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-5726-CONTROLS-AND-OVERHEAD: controls are real."""

    controls = mod.run_ledger_controls()
    by_name = {row["name"]: row for row in controls}

    assert set(by_name) == {
        "ledger_disabled",
        "shuffled_stale_ledger",
        "corrupted_links",
        "always_commit",
        "never_commit",
        "budget_matched_inert",
    }
    assert by_name["ledger_disabled"]["safe_fallback"] is True
    assert by_name["shuffled_stale_ledger"]["safe_fallback"] is True
    assert by_name["corrupted_links"]["safe_fallback"] is True
    assert by_name["always_commit"]["unsafe_detected"] is True
    assert by_name["never_commit"]["commitment_count"] == 0
    assert by_name["budget_matched_inert"]["budget_matched"] is True
    assert mod.entry_propagation_recovery_metrics(controls)["fallback_exercised"] is True


def test_req_arc_wmte_5726_validator_rejects_bad_scope(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5726: schema validation protects no-solve/no-registry scope."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_matched_pairs", lambda *_a, **_kw: _fake_successful_pairs())
    artifact = mod.build_artifact(root=tmp_path)

    with pytest.raises(ValueError, match="registry_updated"):
        mod.validate_artifact({**artifact, "registry_updated": True})

    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})


def test_req_arc_wmte_5726_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5726: checked-in artifact is the stable live A/B receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["new_levels_claimed"] == 0
    assert artifact["registry_updated"] is False
    assert artifact["inference_substrate"] == "matched_arc_live_policy_epistemic_ledger_no_llm"
    assert artifact["honest_verdict"].startswith(("complete:", "blocked:"))
    assert len(artifact["reproducibility_checksum"].removeprefix("sha256:")) == 64
    mod.validate_artifact(artifact)

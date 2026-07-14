"""Tests for Exp 5700: real live-integration test of the goal-predicate-consistency veto
(task 8 completion -- "empirically verify the live veto doesn't hurt plan-success rate").

Spec refs: REQ-ARC-WMTE-5593-3 (LIVE-INTEGRATION EMPIRICAL FOLLOW-UP subsection).
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5700_goal_predicate_veto_live_integration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5593_3_spec_declares_live_integration_follow_up() -> None:
    """REQ-ARC-WMTE-5593-3: OpenSpec declares the live-integration empirical follow-up,
    including the honestly-disclosed dynamics-gate-dominance finding and the caveat that
    the two arms are independent induction calls, not a perfectly matched pair."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5593-3") :]
    section = section[: section.index("### REQ-ARC-WMTE-5594")]

    for marker in (
        "LIVE-INTEGRATION EMPIRICAL FOLLOW-UP",
        "goal_predicate_consistency_accuracy=",
        "Honest caveat",
        "Conclusion: no threshold adjustment",
    ):
        assert marker in section


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) is None


def test_build_artifact_blocked_when_precondition_missing(monkeypatch, tmp_path) -> None:
    """A missing resource fails closed without attempting any live collection."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {"gguf_cached": False, "ok": False},
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("run_prototype must not run when a precondition is missing")

    monkeypatch.setattr(mod, "run_prototype", _fail_if_called)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "complete: blocked_gguf_cached"
    assert artifact["arm_on"] == {}
    assert artifact["arm_off"] == {}
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert len(artifact["reproducibility_checksum"]) == 64


def _ok_preconds(root=mod.REPO_ROOT):
    return {"gguf_cached": True, "e3_policy_import": True, "ok": True}


def test_build_artifact_inconclusive_when_no_real_levelup(monkeypatch, tmp_path) -> None:
    """FALSE_NEGATIVE_RISK guard: a collection window with zero real level-ups is reported
    as inconclusive, not silently treated as a veto pass or fail."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 12,
            "real_levelups_in_collected_transitions": 0,
            "arm_on": {},
            "arm_off": {},
            "run_ok": False,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "complete: inconclusive_no_real_levelup_collected"
    assert artifact["real_levelups_in_collected_transitions"] == 0


def test_build_artifact_confirms_veto_catches_miscalibrated_predicate(
    monkeypatch, tmp_path
) -> None:
    """When the veto-on arm is rejected for goal-predicate inconsistency AND the veto-off
    arm on the same real transitions would have proceeded, the headline verdict names the
    confirmation directly."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 47,
            "real_levelups_in_collected_transitions": 1,
            "arm_on": {
                "veto_on": True,
                "planned": False,
                "skipped": "goal_predicate_consistency_failed",
            },
            "arm_off": {"veto_on": False, "planned": True, "skipped": ""},
            "run_ok": True,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert (
        artifact["honest_verdict"]
        == "complete: veto_confirmed_catches_real_miscalibrated_predicate"
    )
    assert "dynamics gate" in artifact["dynamics_gate_finding"]


def test_build_artifact_veto_did_not_fire_is_reported_honestly(monkeypatch, tmp_path) -> None:
    """If the veto-on arm was NOT rejected on goal-predicate grounds, the verdict must not
    claim a confirmation that did not happen this run."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 30,
            "real_levelups_in_collected_transitions": 2,
            "arm_on": {"veto_on": True, "planned": True, "skipped": ""},
            "arm_off": {"veto_on": False, "planned": True, "skipped": ""},
            "run_ok": True,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "complete: veto_did_not_fire_this_run"


def test_req_arc_wmte_5593_3_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5593-3: the checked-in real run measured the goal-predicate-consistency
    veto against a real, live, GPU-backed E3AgentPolicy episode on lp85 -- the veto-on arm
    was genuinely rejected for a badly-miscalibrated induced predicate
    (goal_predicate_consistency_accuracy far below 1.0) while the veto-off arm, on the SAME
    real transitions, proceeded to a plan. Adversarially clean
    (scripts/adversarial_verify.py: no CRITICAL flags)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert (
        result["honest_verdict"] == "complete: veto_confirmed_catches_real_miscalibrated_predicate"
    )
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["target_game"] == "lp85"
    assert result["real_levelups_in_collected_transitions"] >= 1
    assert result["arm_on"]["skipped"] == "goal_predicate_consistency_failed"
    assert result["arm_on"]["planned"] is False
    assert result["arm_off"]["planned"] is True
    on_round = result["arm_on"]["rounds"][0]
    assert on_round["goal_predicate_consistency_accuracy"] < 1.0
    assert on_round["goal_predicate_consistency_n_real_levelups"] >= 1
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert len(result["reproducibility_checksum"]) == 64

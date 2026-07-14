"""Tests for Exp 5704: live A/B testing whether the min_heldout_accuracy=1.0
dynamics gate is too strict to be useful (task 13 completion).

Spec refs: REQ-ARC-WMTE-5593-5 (extends REQ-ARC-WMTE-5593-4's calibration question).
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5704_dynamics_gate_relaxed_threshold_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5593_5_spec_declares_relaxed_threshold_ab() -> None:
    """REQ-ARC-WMTE-5593-5: OpenSpec declares the live A/B, including the honest
    inconclusive outcome (no attempt landed in the relaxed-only band)."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5593-5") :]

    for marker in (
        "SCENARIO-ARC-WMTE-5593-5",
        "inconclusive_no_attempt_in_relaxed_only_band",
        "does NOT presuppose",
    ):
        assert marker in section


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) is None


def _ok_preconds(root=mod.REPO_ROOT):
    return {"gguf_cached": True, "e3_policy_import": True, "ok": True}


def test_build_artifact_blocked_when_precondition_missing(monkeypatch, tmp_path) -> None:
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
    assert artifact["attempts"] == []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert len(artifact["reproducibility_checksum"]) == 64


def test_build_artifact_inconclusive_when_no_real_levelup(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 10,
            "real_levelups_in_collected_transitions": 0,
            "attempts": [],
            "run_ok": False,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "complete: inconclusive_no_real_levelup_collected"


def _attempt(heldout_accuracy, planned=True, plan_reaches_goal=True):
    return {
        "attempt_index": 0,
        "duration_s": 10.0,
        "heldout_accuracy": heldout_accuracy,
        "accepted_by_strict": heldout_accuracy >= mod.STRICT_THRESHOLD,
        "accepted_by_relaxed": heldout_accuracy >= mod.RELAXED_THRESHOLD,
        "planned": planned,
        "goal_predicate_satisfiable": True,
        "plan_reaches_goal": plan_reaches_goal,
        "skipped": "",
    }


def test_build_artifact_inconclusive_when_all_attempts_outside_relaxed_band(
    monkeypatch, tmp_path
) -> None:
    """Mirrors the real checked-in result: every attempt scores 0.0, so nothing lands
    in the [relaxed, strict) band -- must report inconclusive, not force a conclusion."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 47,
            "real_levelups_in_collected_transitions": 1,
            "attempts": [_attempt(0.0), _attempt(0.0), _attempt(0.0)],
            "run_ok": True,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["n_relaxed_only_accepts"] == 0
    assert artifact["honest_verdict"] == "complete: inconclusive_no_attempt_in_relaxed_only_band"


def test_build_artifact_relaxed_unlocks_a_good_plan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 40,
            "real_levelups_in_collected_transitions": 1,
            "attempts": [_attempt(0.85, plan_reaches_goal=True), _attempt(0.0)],
            "run_ok": True,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["n_relaxed_only_accepts"] == 1
    assert artifact["relaxed_only_accepts_with_good_plan"] == 1
    assert "relaxed_threshold_unlocks_1_of_1_good_plans" in artifact["honest_verdict"]


def test_build_artifact_relaxed_accepts_but_no_good_plan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transitions_collected": 40,
            "real_levelups_in_collected_transitions": 1,
            "attempts": [_attempt(0.85, plan_reaches_goal=False)],
            "run_ok": True,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["n_relaxed_only_accepts"] == 1
    assert artifact["relaxed_only_accepts_with_good_plan"] == 0
    assert "strict_gate_correctly_protective" in artifact["honest_verdict"]


def test_req_arc_wmte_5593_5_repository_artifact_is_a_real_measured_result() -> None:
    """The checked-in real run collected real transitions on lp85 (1 real level-up)
    and made 3 fresh real induction attempts, all landing at heldout_accuracy=0.0 --
    an honest inconclusive result for the relaxed-threshold question, not a forced
    conclusion in either direction. Adversarially clean."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"] == "complete: inconclusive_no_attempt_in_relaxed_only_band"
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["target_game"] == "lp85"
    assert result["real_levelups_in_collected_transitions"] >= 1
    assert result["n_attempts"] == len(result["attempts"])
    assert result["n_relaxed_only_accepts"] == 0
    for attempt in result["attempts"]:
        assert attempt["heldout_accuracy"] < mod.RELAXED_THRESHOLD
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert len(result["reproducibility_checksum"]) == 64

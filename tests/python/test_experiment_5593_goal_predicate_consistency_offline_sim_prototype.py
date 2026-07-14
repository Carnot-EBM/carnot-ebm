"""Tests for Exp 5593 goal-predicate consistency offline-sim prototype.

Spec refs: REQ-ARC-WMTE-5593, SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR,
SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5593_goal_predicate_consistency_offline_sim_prototype as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5593_spec_declares_prototype_contract() -> None:
    """REQ-ARC-WMTE-5593: OpenSpec declares the goal-predicate consistency contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5593") :]

    for marker in (
        "REQ-ARC-WMTE-5593",
        "SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR",
        "SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT",
        "score_goal_predicate_consistency",
        "context",
    ):
        assert marker in section


def test_scenario_arc_wmte_5593_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any game."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_policy_import": True,
            "goal_predicate_consistency_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": False,
        },
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("run_prototype must not run when a precondition is missing")

    monkeypatch.setattr(mod, "run_prototype", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["transition_count"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5593_synthetic_correct_predictor(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR: a correctly-scored induced predicate is
    classified as a correct-predicate outcome."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "goal_predicate_consistency_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transition_count": 10,
            "induce_transition_count": 3,
            "real_levelup_present_in_sample": True,
            "goal_predicate_accuracy": 1.0,
            "goal_predicate_n_correct": 3,
            "goal_predicate_n": 3,
            "goal_predicate_n_real_levelups": 1,
            "goal_predicate_n_real_noops": 2,
            "goal_predicate_mismatches": [],
            "induction_ok": True,
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: goal_predicate_consistency_prototype_induced_predicate_correct"
    )
    assert artifact["goal_predicate_accuracy"] == 1.0


def test_scenario_arc_wmte_5593_synthetic_broken_predictor_caught(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT: a miscalibrated induced predicate
    is honestly reported, not silently treated as correct."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "goal_predicate_consistency_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transition_count": 10,
            "induce_transition_count": 3,
            "real_levelup_present_in_sample": True,
            "goal_predicate_accuracy": 0.5,
            "goal_predicate_n_correct": 1,
            "goal_predicate_n": 2,
            "goal_predicate_n_real_levelups": 1,
            "goal_predicate_n_real_noops": 1,
            "goal_predicate_mismatches": [{"i": 0, "real_levelup": True, "claimed": False}],
            "induction_ok": True,
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: goal_predicate_consistency_prototype_induced_predicate_miscalibrated"
    )
    assert artifact["goal_predicate_mismatches"] != []


def test_scenario_arc_wmte_5593_no_real_levelup_is_inconclusive(monkeypatch) -> None:
    """A real induced predicate scored against a sample with no genuine level-up is
    honestly reported as inconclusive, not misread as a correct-predicate result (the
    CLAUDE.md FALSE_NEGATIVE_RISK discipline applied to this new consistency check)."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "goal_predicate_consistency_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transition_count": 10,
            "induce_transition_count": 10,
            "real_levelup_present_in_sample": False,
            "goal_predicate_accuracy": 1.0,
            "goal_predicate_n_correct": 10,
            "goal_predicate_n": 10,
            "goal_predicate_n_real_levelups": 0,
            "goal_predicate_n_real_noops": 10,
            "goal_predicate_mismatches": [],
            "induction_ok": True,
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: goal_predicate_consistency_prototype_no_real_levelup_inconclusive"
    )


def test_scenario_arc_wmte_5593_context_overflow_classified_distinctly(monkeypatch) -> None:
    """An induction failure whose detail mentions context-size overflow is classified
    distinctly from a generic induction failure, when that detail is available."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "goal_predicate_consistency_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_prototype",
        lambda **_kwargs: {
            "transition_count": 37,
            "induce_transition_count": 8,
            "real_levelup_present_in_sample": True,
            "goal_predicate_accuracy": None,
            "goal_predicate_mismatches": [],
            "induction_failure_detail": (
                '{"error":{"code":400,"message":"request (18355 tokens) exceeds the '
                'available context size (16384 tokens)","type":"exceed_context_size_error"}}'
            ),
            "induction_ok": False,
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: goal_predicate_consistency_prototype_induction_context_overflow"
    )


def test_req_arc_wmte_5593_repository_artifact_is_a_real_positive_control_result() -> None:
    """REQ-ARC-WMTE-5593 + REQ-ARC-WMTE-5593-2: the checked-in real run is now the real
    positive-control demo the induce_prompt scalability fix (REQ-ARC-WMTE-5593-2) enabled --
    a genuinely real induction on lp85's 8-transition window (no context overflow) scored by
    score_goal_predicate_consistency against real observed transitions. The induced predicate
    is imperfect (0.75 accuracy, 2 real-level-up misses) -- an honest finding about induction
    QUALITY on lp85, distinct from the check working correctly (also validated by
    test_arc_goal_predicate_consistency.py's 5 direct unit tests on synthetic data)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    detail = result.get("goal_predicate_detail", {})

    assert result["real_levelup_present_in_sample"] is True
    assert detail.get("induction_ok") is True
    assert detail.get("induce_transition_count") == 8
    assert result["goal_predicate_accuracy"] == 0.75
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["solve_provenance"] == "development_proxy"
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
